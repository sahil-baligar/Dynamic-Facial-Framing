"""Dynamic Facial Framing: horizontal video in, vertical (9:16) video out, with the camera following faces like an operator would.

    from dynamic_facial_framing import reframe
    info = reframe("podcast.mp4", "short.mp4", start=312.0, duration=42.0)   # mode='auto' classifies the footage

Lower level: analyze_video -> focus_timeline -> plan_crop -> render_crop (see README).
"""
from .api import reframe, analyze_range
from .analyze import analyze_video
from .speaker import focus_timeline, expression_change, mouth_activity
from .framing import plan_crop, CropPlan
from .layouts import classify_content, Rect
from .render import render_crop, render_split, render_blur_fill
from .detect import FaceDetector

__all__ = ["reframe", "analyze_range", "analyze_video", "focus_timeline", "expression_change", "mouth_activity", "plan_crop", "CropPlan",
           "classify_content", "Rect", "render_crop", "render_split", "render_blur_fill", "FaceDetector"]
__version__ = "2.0.0"
