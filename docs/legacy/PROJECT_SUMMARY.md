# Dynamic Facial Framing - Project Summary

## Overview

This project implements a dynamic facial framing system that converts horizontal videos to vertical (9:16) format while intelligently tracking and centering faces as they move. The system is designed for creating TikTok/Reel-style vertical videos from longer horizontal source videos.

## Key Features

### 1. **Dynamic Face Tracking**
- Automatically detects faces using MediaPipe (recommended) or OpenCV
- Tracks faces across frames using IoU (Intersection over Union) matching
- Handles multiple faces and selects the primary speaker
- Smooth interpolation between face detections

### 2. **Smooth Crop Transitions**
- Exponential moving average for smooth crop position transitions
- Configurable smoothing factor to balance responsiveness vs. stability
- Bounds checking to ensure crop stays within video frame
- Interpolation for frames without face detections

### 3. **Segment-Based Processing**
- Processes video in small segments (0.33 seconds) for true dynamic cropping
- Each segment uses the appropriate crop position for that time
- Segments are concatenated to create smooth final video
- Maintains audio sync throughout processing

### 4. **Multiple Layout Modes**
- **VERT_FOCUS**: Face-centered vertical crop (when face detected)
- **BG_BLUR_RECT**: Blurred background with foreground rectangle (no face)
- **GAMEPLAY**: Gameplay background overlay (gaming content)
- **COVER**: Simple cover crop (fallback)

### 5. **State Machine**
- Intelligent layout switching based on face detection
- Hysteresis to prevent rapid state changes
- Tracks state history for analysis
- Configurable thresholds for state transitions

## Architecture

### Core Components

1. **`app/vertical.py`**
   - Main video processing functions
   - Dynamic facial framing implementation
   - Segment-based cropping logic
   - FFmpeg integration

2. **`app/layout/face_detector.py`**
   - Face detection using MediaPipe or OpenCV
   - Face tracking across frames
   - IoU-based matching
   - Primary face selection

3. **`app/layout/state_machine.py`**
   - Layout state machine
   - State transition logic
   - Face tracking integration
   - Layout configuration

4. **`app/settings.py`**
   - Configuration settings
   - Face detection parameters
   - Crop settings
   - Output parameters

## How It Works

### Processing Pipeline

1. **Video Analysis**
   - Opens video and extracts frame information
   - Calculates FPS and total frames
   - Determines source video dimensions

2. **Face Detection & Tracking**
   - Processes frames to detect faces
   - Tracks faces across frames using IoU matching
   - Identifies primary face (largest/most confident)
   - Smooths face positions using exponential moving average

3. **Crop Position Calculation**
   - For each frame, calculates optimal crop position to center face
   - Applies bounds checking to keep crop within frame
   - Interpolates positions for frames without face detections
   - Smooths crop transitions using exponential moving average

4. **Segment Processing**
   - Divides video into small segments (0.33 seconds)
   - Each segment uses crop position calculated for that time
   - Processes segments with FFmpeg crop filter
   - Concatenates segments to create final video

5. **Audio Sync**
   - Extracts audio from original video
   - Merges audio with processed video
   - Ensures audio-video synchronization

### Face Tracking Algorithm

1. **Detection**: Uses MediaPipe or OpenCV to detect faces in each frame
2. **Matching**: Matches detections to existing tracks using IoU
3. **Update**: Updates track positions with smoothing
4. **Creation**: Creates new tracks for unmatched detections
5. **Removal**: Removes tracks that haven't been detected for multiple frames

### Crop Smoothing

- Uses exponential moving average: `crop_pos = last_pos * (1 - α) + target_pos * α`
- Smoothing factor `α` controls responsiveness (default: 0.15)
- Lower values = smoother but slower response
- Higher values = faster but potentially jittery

## Configuration

### Key Settings

- **`face_detection_model`**: "mediapipe" or "opencv_dnn"
- **`crop_smoothing_factor`**: 0.15 (how quickly crop follows face)
- **`face_tracking_smoothing`**: 0.7 (face position smoothing)
- **`min_face_size`**: 100 (minimum face size in pixels)
- **`vertical_width`**: 1080 (output video width)
- **`vertical_height`**: 1920 (output video height)

### Performance Tuning

- **Segment Duration**: Smaller segments (0.2s) = smoother but slower
- **FFmpeg Preset**: "veryfast" = faster, "slow" = better quality
- **Face Detection**: MediaPipe is faster and more accurate than OpenCV

## Usage Examples

### Basic Usage

```python
from app.vertical import extract_dynamic_layout
from app.layout.state_machine import LayoutStateMachine

state_machine = LayoutStateMachine()
result = extract_dynamic_layout(
    src="input.mp4",
    dst="output.mp4",
    start=0.0,
    duration=60.0,
    state_machine=state_machine,
    background_mode="blur"
)
```

### Advanced Usage

```python
from app.vertical import extract_dynamic_facial_framing
from app.vertical import get_video_dimensions

width, height = get_video_dimensions("input.mp4")
face_tracks = [...]  # Your face track data

extract_dynamic_facial_framing(
    src="input.mp4",
    dst="output.mp4",
    start=0.0,
    duration=30.0,
    face_tracks=face_tracks,
    source_width=width,
    source_height=height
)
```

## Limitations & Future Improvements

### Current Limitations

1. **Processing Speed**: Segment-based processing can be slow for long videos
2. **Face Detection**: Requires visible faces for best results
3. **Multiple Faces**: Currently focuses on primary face only
4. **Audio Sync**: May have minor sync issues with very long videos

### Potential Improvements

1. **GPU Acceleration**: Use GPU for face detection and video processing
2. **Parallel Processing**: Process segments in parallel
3. **Better Interpolation**: Use more sophisticated interpolation algorithms
4. **Multi-Face Support**: Track and frame multiple faces
5. **Real-time Processing**: Optimize for real-time video processing
6. **Advanced Cropping**: Use zoom/pan effects for smoother transitions

## Dependencies

- **OpenCV**: Computer vision and video processing
- **MediaPipe**: Face detection (recommended)
- **NumPy**: Numerical operations
- **FFmpeg**: Video encoding/decoding (external dependency)

## File Structure

```
.
├── app/
│   ├── __init__.py
│   ├── settings.py           # Configuration
│   ├── vertical.py           # Main processing
│   └── layout/
│       ├── __init__.py
│       ├── face_detector.py  # Face detection
│       └── state_machine.py  # State machine
├── example.py                # Usage examples
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── QUICKSTART.md            # Quick start guide
└── .gitignore               # Git ignore rules
```

## Testing

To test the system:

1. Place a test video as `input.mp4`
2. Run: `python example.py --dynamic`
3. Check output video for face tracking accuracy
4. Adjust settings in `app/settings.py` as needed

## Conclusion

This system provides a robust solution for converting horizontal videos to vertical format with intelligent face tracking. The segment-based approach ensures smooth, dynamic framing that follows the speaker as they move, creating professional-looking vertical videos suitable for TikTok, Instagram Reels, and other short-form video platforms.

