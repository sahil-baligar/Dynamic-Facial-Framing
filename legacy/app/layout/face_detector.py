"""
Face detection and tracking using MediaPipe or OpenCV.
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from app.settings import settings

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available, using OpenCV DNN face detector")


class FaceDetector:
    """Face detection and tracking."""
    
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.face_tracks: Dict[int, Dict[str, Any]] = {}  # track_id -> face data
        self.next_track_id = 0
        
        if settings.face_detection_model == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            self._init_opencv_dnn()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detection."""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=settings.mediapipe_model_complexity,
            min_detection_confidence=settings.mediapipe_min_detection_confidence
        )
        print("✅ Initialized MediaPipe face detector")
    
    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN face detector."""
        # Load OpenCV DNN face detector
        try:
            # Try to load the face detector model files
            # These should be in the project directory or downloaded
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            prototxt_path = os.path.join(project_root, "models", "deploy.prototxt")
            model_path = os.path.join(project_root, "models", "res10_300x300_ssd_iter_140000.caffemodel")
            
            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                # Use a simpler Haar Cascade as fallback
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.detector = cv2.CascadeClassifier(cascade_path)
                    self.detector_type = "haar"
                    print("✅ Initialized OpenCV Haar Cascade face detector")
                else:
                    print("⚠️ No face detector model found, face detection will be disabled")
                    self.detector = None
                    self.detector_type = None
            else:
                self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.detector_type = "dnn"
                print("✅ Initialized OpenCV DNN face detector")
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenCV DNN: {e}")
            # Fallback to Haar Cascade
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                self.detector_type = "haar"
                print("✅ Initialized OpenCV Haar Cascade face detector (fallback)")
            except Exception as e2:
                print(f"❌ Failed to initialize any face detector: {e2}")
                self.detector = None
                self.detector_type = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of face detections with bbox, confidence, and landmarks
        """
        if self.detector is None:
            return []
        
        faces = []
        h, w = frame.shape[:2]
        
        if settings.face_detection_model == "mediapipe" and MEDIAPIPE_AVAILABLE:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure bbox is within frame bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width >= settings.min_face_size and height >= settings.min_face_size:
                        faces.append({
                            "bbox": [x, y, width, height],
                            "confidence": detection.score[0],
                            "landmarks": None  # MediaPipe provides landmarks but we don't need them for cropping
                        })
        
        elif self.detector_type == "dnn":
            # OpenCV DNN face detector
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), [104, 117, 123]
            )
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > settings.face_detection_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype(int)
                    width = x2 - x
                    height = y2 - y
                    
                    # Ensure bbox is within frame bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width >= settings.min_face_size and height >= settings.min_face_size:
                        faces.append({
                            "bbox": [x, y, width, height],
                            "confidence": float(confidence),
                            "landmarks": None
                        })
        
        elif self.detector_type == "haar":
            # OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(settings.min_face_size, settings.min_face_size)
            )
            
            for (x, y, w, h) in detections:
                faces.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.8,  # Haar doesn't provide confidence
                    "landmarks": None
                })
        
        return faces
    
    def track_faces(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """
        Detect and track faces across frames.
        
        Args:
            frame: Input frame
            frame_number: Current frame number
            
        Returns:
            List of tracked faces with track_id, bbox, and confidence
        """
        # Detect faces in current frame
        detections = self.detect_faces(frame)
        
        if not detections:
            # No faces detected, decay existing tracks
            tracks_to_remove = []
            for track_id, track_data in self.face_tracks.items():
                track_data["missed_frames"] = track_data.get("missed_frames", 0) + 1
                if track_data["missed_frames"] > 10:  # Remove track after 10 missed frames
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.face_tracks[track_id]
            
            return []
        
        # Simple tracking: match detections to existing tracks using IoU
        tracked_faces = []
        used_track_ids = set()
        
        for detection in detections:
            det_bbox = detection["bbox"]
            det_center = (det_bbox[0] + det_bbox[2] // 2, det_bbox[1] + det_bbox[3] // 2)
            best_match = None
            best_iou = 0.3  # Minimum IoU threshold
            
            # Find best matching existing track
            for track_id, track_data in self.face_tracks.items():
                if track_id in used_track_ids:
                    continue
                
                track_bbox = track_data["bbox"]
                iou = self._calculate_iou(det_bbox, track_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            if best_match is not None:
                # Update existing track
                track_data = self.face_tracks[best_match]
                # Smooth the bbox update
                smoothing = settings.face_tracking_smoothing
                old_bbox = track_data["bbox"]
                new_bbox = [
                    int(old_bbox[0] * (1 - smoothing) + det_bbox[0] * smoothing),
                    int(old_bbox[1] * (1 - smoothing) + det_bbox[1] * smoothing),
                    int(old_bbox[2] * (1 - smoothing) + det_bbox[2] * smoothing),
                    int(old_bbox[3] * (1 - smoothing) + det_bbox[3] * smoothing),
                ]
                
                self.face_tracks[best_match] = {
                    "track_id": best_match,
                    "bbox": new_bbox,
                    "confidence": detection["confidence"],
                    "frame": frame_number,
                    "missed_frames": 0
                }
                
                tracked_faces.append(self.face_tracks[best_match])
                used_track_ids.add(best_match)
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.face_tracks[track_id] = {
                    "track_id": track_id,
                    "bbox": det_bbox,
                    "confidence": detection["confidence"],
                    "frame": frame_number,
                    "missed_frames": 0
                }
                
                tracked_faces.append(self.face_tracks[track_id])
                used_track_ids.add(track_id)
        
        # Mark unused tracks as missed
        for track_id, track_data in self.face_tracks.items():
            if track_id not in used_track_ids:
                track_data["missed_frames"] = track_data.get("missed_frames", 0) + 1
        
        return tracked_faces
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def is_main_subject_face(self, face: Dict[str, Any], frame_width: int, frame_height: int) -> bool:
        """
        Check if a face is the main subject (not a corner webcam).
        
        Criteria:
        1. Face must be large enough (>= min_face_area_percentage of frame)
        2. Face must not be in corner regions
        
        Args:
            face: Face detection with bbox
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            True if face is main subject, False if corner webcam
        """
        bbox = face["bbox"]
        x, y, w, h = bbox
        
        # Calculate face area as percentage of frame
        face_area = w * h
        frame_area = frame_width * frame_height
        face_area_percentage = (face_area / frame_area) * 100 if frame_area > 0 else 0
        
        # Check if face is large enough to be main subject
        if face_area_percentage < settings.min_face_area_percentage:
            return False  # Too small, likely a webcam
        
        # Calculate face center
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        # Define corner regions (outer edges of frame)
        corner_threshold_x = frame_width * settings.corner_webcam_threshold
        corner_threshold_y = frame_height * settings.corner_webcam_threshold
        
        # Check if face is in a corner (must be in both horizontal AND vertical corner regions)
        is_in_left_corner = face_center_x < corner_threshold_x
        is_in_right_corner = face_center_x > (frame_width - corner_threshold_x)
        is_in_top_corner = face_center_y < corner_threshold_y
        is_in_bottom_corner = face_center_y > (frame_height - corner_threshold_y)
        
        # If face is in both horizontal and vertical corner regions, it's a corner webcam
        is_corner_webcam = (is_in_left_corner or is_in_right_corner) and (is_in_top_corner or is_in_bottom_corner)
        
        if is_corner_webcam:
            return False  # Corner webcam, not main subject
        
        # If face is large enough and not in corner, it's the main subject
        return True
    
    def get_primary_face(self, tracked_faces: List[Dict[str, Any]], 
                        frame_width: int = None, frame_height: int = None) -> Optional[Dict[str, Any]]:
        """
        Get the primary (main subject) face from tracked faces.
        Filters out corner webcams and only returns main subject faces.
        
        Args:
            tracked_faces: List of tracked faces
            frame_width: Frame width (optional, for main subject detection)
            frame_height: Frame height (optional, for main subject detection)
            
        Returns:
            Primary main subject face or None
        """
        if not tracked_faces:
            return None
        
        # If frame dimensions provided, filter for main subject faces only
        if frame_width and frame_height:
            main_subject_faces = [
                f for f in tracked_faces 
                if self.is_main_subject_face(f, frame_width, frame_height)
            ]
            
            if main_subject_faces:
                # Return the largest main subject face
                primary = max(main_subject_faces, key=lambda f: f["confidence"] * f["bbox"][2] * f["bbox"][3])
                return primary
            else:
                # No main subject faces found (only corner webcams)
                return None
        
        # Fallback: return the face with highest confidence and size (old behavior)
        primary = max(tracked_faces, key=lambda f: f["confidence"] * f["bbox"][2] * f["bbox"][3])
        return primary
    
    def reset(self):
        """Reset face tracks (useful when processing a new video)."""
        self.face_tracks = {}
        self.next_track_id = 0

