"""
Layout state machine for dynamic video layout switching based on face detection.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from app.layout.face_detector import FaceDetector
from app.settings import settings


class LayoutState(Enum):
    """Layout states for video rendering."""
    VERT_FOCUS = "vert_focus"  # Face-centered vertical crop
    BG_BLUR_RECT = "bg_blur_rect"  # Blurred background with foreground rectangle
    GAMEPLAY = "gameplay"  # Gameplay background with overlay
    PODCAST = "podcast"  # Podcast mode with background video split screen
    COVER = "cover"  # Simple cover crop


class FaceTrack:
    """Face track data structure."""
    
    def __init__(self, track_id: int, bbox: List[int], confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence


class LayoutStateMachine:
    """State machine for dynamic layout selection based on face detection."""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.current_state = LayoutState.COVER
        self.state_history: List[Dict[str, Any]] = []
        self.frame_history: List[int] = []
        self.primary_face: Optional[FaceTrack] = None
        self.face_detection_count = 0
        self.no_face_count = 0
        
    def update_state(self, frame: np.ndarray, frame_number: int, 
                    background_mode: str = "blur") -> LayoutState:
        """
        Update state machine based on current frame.
        
        Args:
            frame: Current video frame
            frame_number: Current frame number
            background_mode: "blur" or "gameplay"
            
        Returns:
            Current layout state
        """
        # Get frame dimensions for main subject detection
        frame_height, frame_width = frame.shape[:2]
        
        # Track faces in current frame
        tracked_faces = self.face_detector.track_faces(frame, frame_number)
        
        # Get primary face (main subject only, filters out corner webcams)
        primary_face_data = self.face_detector.get_primary_face(
            tracked_faces, 
            frame_width=frame_width, 
            frame_height=frame_height
        )
        
        if primary_face_data:
            # Main subject face detected (not corner webcam)
            self.primary_face = FaceTrack(
                track_id=primary_face_data["track_id"],
                bbox=primary_face_data["bbox"],
                confidence=primary_face_data["confidence"]
            )
            self.face_detection_count += 1
            self.no_face_count = 0
        else:
            # No main subject face (might have corner webcam, but not main subject)
            self.primary_face = None
            self.no_face_count += 1
            self.face_detection_count = max(0, self.face_detection_count - 1)
        
        # Determine state based on face detection and background mode
        if self.face_detection_count >= settings.min_face_detections:
            # Face detected consistently, use face-centered layout
            if background_mode == "gameplay":
                new_state = LayoutState.GAMEPLAY
            else:
                new_state = LayoutState.VERT_FOCUS
        elif self.no_face_count > settings.state_transition_threshold:
            # No face for a while, use simple cover crop (for gameplay, non-human content, etc.)
            new_state = LayoutState.COVER
        else:
            # Transition period, keep current state or use cover
            if self.current_state == LayoutState.COVER:
                new_state = LayoutState.COVER  # Stay in cover during transition
            else:
                new_state = self.current_state
        
        # Update state with hysteresis to avoid rapid switching
        if new_state != self.current_state:
            # Only switch if we've been in the new state condition for a while
            if self._should_transition(new_state):
                self.current_state = new_state
        
        # Record state history
        self.state_history.append({
            "frame": frame_number,
            "state": self.current_state.value,
            "primary_face": self.primary_face
        })
        self.frame_history.append(frame_number)
        
        return self.current_state
    
    def _should_transition(self, new_state: LayoutState) -> bool:
        """Check if we should transition to a new state (hysteresis)."""
        # Simple hysteresis: only transition if we have enough evidence
        if new_state == LayoutState.VERT_FOCUS:
            return self.face_detection_count >= settings.min_face_detections
        elif new_state == LayoutState.COVER:
            # Transition to cover when no face detected for threshold frames
            return self.no_face_count >= settings.state_transition_threshold
        elif new_state == LayoutState.GAMEPLAY:
            # For gameplay mode, transition based on face detection
            if self.face_detection_count >= settings.min_face_detections:
                return True
            return self.no_face_count >= settings.state_transition_threshold
        return True
    
    def get_current_layout_config(self) -> Dict[str, Any]:
        """Get current layout configuration."""
        config = {
            "state": self.current_state.value,
            "primary_face": None
        }
        
        if self.primary_face:
            config["primary_face"] = {
                "track_id": self.primary_face.track_id,
                "bbox": self.primary_face.bbox,
                "confidence": self.primary_face.confidence
            }
        
        return config
    
    def get_state_timeline(self) -> List[Dict[str, Any]]:
        """Get state timeline for manifest."""
        return self.state_history.copy()
    
    def reset(self):
        """Reset state machine (useful when processing a new video)."""
        self.current_state = LayoutState.COVER
        self.state_history = []
        self.frame_history = []
        self.primary_face = None
        self.face_detection_count = 0
        self.no_face_count = 0
        self.face_detector.reset()

