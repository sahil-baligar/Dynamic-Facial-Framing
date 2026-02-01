"""
Settings and configuration for Facial Framing.
"""
import os
from typing import Optional

class Settings:
    """Application settings."""
    
    # Video dimensions
    vertical_width: int = 1080
    vertical_height: int = 1920
    
    # Face detection settings
    face_detection_model: str = "mediapipe"  # "mediapipe" or "opencv_dnn"
    face_detection_confidence: float = 0.5
    face_tracking_smoothing: float = 0.7  # Smoothing factor (0-1), higher = smoother
    
    # Crop settings
    crop_smoothing_factor: float = 0.05  # How quickly crop follows face (0-1), lower = smoother
    min_face_size: int = 50  # Minimum face size in pixels (absolute) - lowered for better detection
    min_face_area_percentage: float = 2.0  # Minimum face area as % of frame (for main subject detection) - lowered for better detection
    corner_webcam_threshold: float = 0.15  # Faces in outer 15% of frame edges are considered corner webcams
    center_region_percentage: float = 0.70  # Center region where faces are considered main subject (70% of frame)
    crop_padding: float = 1.5  # Padding around face (1.0 = no padding, 1.5 = 50% padding)
    
    # Podcast detection settings
    podcast_face_detection_threshold: float = 70.0  # Minimum % of frames with faces to detect as podcast
    podcast_background_video_path: str = "BackgroundVideo.mp4"  # Default background video for podcasts
    
    # Fast teleport settings (for podcasts)
    fast_teleport_enabled: bool = True  # Enable fast teleport to speaker for podcasts
    teleport_threshold_frames: int = 5  # Frames to wait before teleporting to new speaker
    
    # Smaller dynamic frame settings
    smaller_frame_enabled: bool = False  # Enable smaller frame mode with vertical tracking
    smaller_frame_width: int = 864  # Smaller frame width (80% of 1080)
    smaller_frame_height: int = 1536  # Smaller frame height (80% of 1920)
    smaller_frame_smoothing: float = 0.08  # Slightly faster smoothing for smaller frame mode
    
    # Layout transition settings
    state_transition_threshold: int = 15  # Frames before switching to cover (when no face detected)
    min_face_detections: int = 10  # Minimum frames with face to use face tracking
    min_face_detection_percentage: float = 15.0  # Minimum % of frames with faces to use dynamic framing (lowered for mixed content)
    segment_analysis_duration: float = 5.0  # Analyze video in segments for mode switching (seconds)
    
    # FFmpeg settings
    ffmpeg_preset: str = "medium"  # Balanced quality and speed
    ffmpeg_crf: int = 16  # Good quality (lower number = better quality)
    output_fps: int = None  # None = preserve source FPS for smoother motion
    segment_duration: float = 0.05  # Optimal segment size for smooth transitions (seconds)
    
    # MediaPipe specific settings
    mediapipe_model_complexity: int = 1  # 0, 1, or 2
    mediapipe_min_detection_confidence: float = 0.3  # Lowered for better detection (was 0.5)
    mediapipe_min_tracking_confidence: float = 0.3  # Lowered for better tracking (was 0.5)

# Global settings instance
settings = Settings()
