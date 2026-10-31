# Quick Start Guide

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg:**
   - Windows: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

3. **Verify installation:**
   ```bash
   ffmpeg -version
   python -c "import cv2, mediapipe; print('✅ All dependencies installed')"
   ```

## Basic Usage

### 1. Simple Vertical Conversion (No Face Detection)

```python
from app.vertical import extract_vertical_clip

extract_vertical_clip(
    src="input.mp4",
    dst="output.mp4",
    start=0.0,
    duration=30.0,
    mode="cover"
)
```

### 2. Dynamic Facial Framing (Recommended)

```python
from app.vertical import extract_dynamic_layout
from app.layout.state_machine import LayoutStateMachine

# Initialize state machine for face tracking
state_machine = LayoutStateMachine()

# Process video with dynamic facial framing
result = extract_dynamic_layout(
    src="input.mp4",
    dst="output.mp4",
    start=0.0,
    duration=60.0,
    state_machine=state_machine,
    background_mode="blur"  # or "gameplay"
)

if result["success"]:
    print(f"✅ Processed successfully!")
    print(f"Face tracks: {len(result.get('face_tracks', []))}")
```

### 3. Run Example Script

```bash
# Place your input video as "input.mp4" in the project root
python example.py --dynamic
```

## Configuration

Edit `app/settings.py` to customize:

- **Face detection**: Change `face_detection_model` to "mediapipe" (recommended) or "opencv_dnn"
- **Smoothing**: Adjust `crop_smoothing_factor` (0.1 = fast tracking, 0.3 = smoother)
- **Output dimensions**: Modify `vertical_width` and `vertical_height`

## Troubleshooting

**FFmpeg not found:**
- Make sure FFmpeg is in your PATH
- Test with: `ffmpeg -version`

**Face detection not working:**
- Install MediaPipe: `pip install mediapipe`
- Check that faces are visible and large enough
- Adjust `face_detection_confidence` in settings

**Video processing is slow:**
- Reduce video duration for testing
- Use faster FFmpeg preset in settings
- Process shorter clips first

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example.py](example.py) for more usage examples
- Customize settings in `app/settings.py` for your needs

