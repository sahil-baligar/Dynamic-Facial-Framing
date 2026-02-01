# Testing Dynamic Facial Framing

## Quick Start

### Step 1: Place Your Test Video

Place your test video in the project root directory and name it **`input.mp4`**

Alternatively, you can use any video file and specify the path when running the test.

```
FacialFraming/
├── input.mp4          ← Place your test video here
├── test_facial_framing.py
├── app/
└── ...
```

### Step 2: Run the Test

```bash
# Basic test (uses input.mp4)
python test_facial_framing.py

# Or specify custom input/output
python test_facial_framing.py --input your_video.mp4 --output result.mp4

# Test a specific segment (first 30 seconds)
python test_facial_framing.py --input your_video.mp4 --start 0 --duration 30
```

### Step 3: Check the Output

The output video will be saved as `output_dynamic.mp4` (or your specified output path).

## What to Expect

### Successful Output

When the test runs successfully, you should see:

1. **Video Information**: Dimensions, duration, etc.
2. **Settings Display**: Current configuration
3. **Processing Progress**: Real-time updates
4. **Results Summary**:
   - Face tracks detected
   - Layout state distribution
   - Face detection rate
   - Average face confidence

### Output Video Characteristics

- **Format**: MP4 (H.264)
- **Dimensions**: 1080x1920 (9:16 vertical)
- **FPS**: 30 fps (configurable)
- **Face Tracking**: Smooth, dynamic framing that follows the speaker
- **Background**: Blurred background with foreground subject (when face detected)

## Troubleshooting

### No Face Detected

If faces aren't being detected:

1. **Check Video Quality**:
   - Ensure faces are clearly visible
   - Good lighting helps
   - Face should be large enough (at least 100x100 pixels)

2. **Adjust Settings** (in `app/settings.py`):
   ```python
   face_detection_confidence = 0.3  # Lower threshold (default: 0.5)
   min_face_size = 50  # Smaller minimum size (default: 100)
   ```

3. **Try Different Model**:
   ```python
   face_detection_model = "mediapipe"  # Recommended
   # or
   face_detection_model = "opencv_dnn"  # Alternative
   ```

### Jittery or Jumpy Tracking

If the crop position is jittery:

1. **Increase Smoothing** (in `app/settings.py`):
   ```python
   crop_smoothing_factor = 0.1  # Slower, smoother (default: 0.15)
   face_tracking_smoothing = 0.8  # Higher smoothing (default: 0.7)
   ```

### Face Not Centered

If the face isn't staying centered:

1. **Check Crop Settings**:
   - Verify `vertical_width` and `vertical_height` in settings
   - Ensure source video has enough resolution

2. **Adjust Padding** (if needed):
   ```python
   crop_padding = 2.0  # More padding around face (default: 1.5)
   ```

### Processing is Slow

To speed up processing:

1. **Test with Shorter Clips**:
   ```bash
   python test_facial_framing.py --duration 10  # Test 10 seconds first
   ```

2. **Use Faster Preset** (in `app/settings.py`):
   ```python
   ffmpeg_preset = "veryfast"  # Faster processing
   ```

3. **Reduce Segment Duration** (in `app/vertical.py`):
   - Change `segment_duration = 0.5` to `0.33` (already optimized)

## Example Test Scenarios

### Test 1: Basic Face Tracking
```bash
python test_facial_framing.py --input talking_head.mp4 --output test1.mp4
```
**Expected**: Face should be centered and tracked smoothly

### Test 2: Moving Speaker
```bash
python test_facial_framing.py --input moving_speaker.mp4 --output test2.mp4
```
**Expected**: Crop should follow the speaker as they move

### Test 3: Multiple Faces
```bash
python test_facial_framing.py --input interview.mp4 --output test3.mp4
```
**Expected**: System should focus on the primary/largest face

### Test 4: No Face (Background Mode)
```bash
python test_facial_framing.py --input landscape.mp4 --output test4.mp4
```
**Expected**: Should fall back to blurred background layout

## Verifying Results

### Visual Check

1. **Play the output video** and verify:
   - Face is centered in frame
   - Crop follows face movement smoothly
   - No jittery movements
   - Background is appropriately blurred

2. **Compare with input**:
   - Face should remain centered even when moving
   - Vertical format (9:16) should be maintained
   - Audio should be synchronized

### Technical Check

The test script will output:
- Face detection rate (should be > 50% for good tracking)
- Average face confidence (should be > 0.7)
- Layout state distribution (VERT_FOCUS should dominate when face detected)

## Advanced Testing

### Test with Custom Settings

Create a test script with custom settings:

```python
from app.settings import settings
from app.vertical import extract_dynamic_layout
from app.layout.state_machine import LayoutStateMachine

# Customize settings
settings.crop_smoothing_factor = 0.1  # Smoother
settings.face_detection_confidence = 0.4  # More sensitive

# Run test
state_machine = LayoutStateMachine()
result = extract_dynamic_layout(
    src="input.mp4",
    dst="output.mp4",
    start=0.0,
    duration=30.0,
    state_machine=state_machine,
    background_mode="blur"
)
```

### Batch Testing

Test multiple videos:

```bash
# Windows PowerShell
foreach ($video in Get-ChildItem *.mp4) {
    python test_facial_framing.py --input $video.Name --output "output_$($video.Name)"
}

# Linux/Mac
for video in *.mp4; do
    python test_facial_framing.py --input "$video" --output "output_$video"
done
```

## Next Steps

After testing:

1. **Review Results**: Check if face tracking meets your expectations
2. **Adjust Settings**: Tune parameters in `app/settings.py` for your use case
3. **Optimize**: Adjust segment duration and processing settings for speed/quality balance
4. **Scale Up**: Process longer videos or batch process multiple videos

## Support

If you encounter issues:

1. Check the error messages in the console output
2. Verify FFmpeg is installed and in PATH
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check video file format (MP4 is recommended)
5. Verify video has audio track (required for proper processing)

