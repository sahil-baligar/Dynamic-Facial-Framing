# Testing Your Video

## Where to Place Your Test Video

### Option 1: Default Location (Recommended)

Place your test video in the **project root directory** and name it **`input.mp4`**:

```
FacialFraming/
├── input.mp4          ← Place your test video here
├── test_facial_framing.py
├── app/
└── ...
```

Then simply run:
```bash
python test_facial_framing.py
```

### Option 2: Custom Location

You can place your video anywhere and specify the path:

```bash
python test_facial_framing.py --input "path/to/your/video.mp4" --output "result.mp4"
```

### Option 3: Test Directory (Optional)

Create a `test_videos/` directory for organization:

```
FacialFraming/
├── test_videos/
│   ├── test1.mp4
│   ├── test2.mp4
│   └── ...
├── test_facial_framing.py
└── ...
```

Then run:
```bash
python test_facial_framing.py --input test_videos/test1.mp4
```

## Quick Test Command

```bash
# Basic test (uses input.mp4 if it exists)
python test_facial_framing.py

# Test with custom video
python test_facial_framing.py --input your_video.mp4

# Test first 30 seconds only (faster for testing)
python test_facial_framing.py --input your_video.mp4 --duration 30

# Test specific segment (e.g., seconds 10-40)
python test_facial_framing.py --input your_video.mp4 --start 10 --duration 30
```

## What Happens During Testing

1. **Video Analysis**: System reads video metadata (dimensions, FPS, duration)
2. **Face Detection**: Processes frames to detect and track faces
3. **Crop Calculation**: Calculates optimal crop positions for each frame
4. **Segment Processing**: Processes video in small segments with dynamic cropping
5. **Output Generation**: Creates final vertical video with smooth face tracking

## Expected Output

The test will create `output_dynamic.mp4` (or your specified output name) with:
- **Vertical format**: 1080x1920 (9:16 aspect ratio)
- **Face-centered**: Speaker's face should be centered in frame
- **Smooth tracking**: Crop follows face movement smoothly
- **Blurred background**: Background is blurred when face is detected

## Verifying Results

After processing, check the output video:

1. **Face Centering**: Face should be centered in the vertical frame
2. **Smooth Movement**: Crop should follow face movement without jitter
3. **Background**: Background should be appropriately blurred
4. **Audio Sync**: Audio should be synchronized with video

## Troubleshooting

### Video Not Found
```
❌ Input video not found: input.mp4
```
**Solution**: Make sure the video file exists in the specified location

### No Face Detected
```
⚠️ No face tracks detected
```
**Solution**: 
- Check that faces are clearly visible in the video
- Adjust `face_detection_confidence` in `app/settings.py` (try 0.3)
- Ensure good lighting in the video

### Processing Too Slow
**Solution**: 
- Test with shorter duration: `--duration 10`
- Use faster preset in settings
- Process shorter clips first

## Video Requirements

### Recommended Specifications
- **Format**: MP4 (H.264)
- **Resolution**: 1920x1080 or higher (horizontal)
- **Frame Rate**: 30 fps (or any standard frame rate)
- **Audio**: Should have audio track
- **Length**: Any length (test with short clips first)

### Best Results
- **Lighting**: Good, even lighting on faces
- **Face Size**: Face should be at least 100x100 pixels
- **Clarity**: Clear, in-focus faces
- **Background**: Any background (will be blurred)

## Example Workflow

1. **Prepare Video**: Ensure your test video is ready
2. **Place Video**: Copy to project root as `input.mp4`
3. **Run Test**: `python test_facial_framing.py`
4. **Check Output**: Review `output_dynamic.mp4`
5. **Adjust Settings**: Tune parameters in `app/settings.py` if needed
6. **Re-test**: Run again with adjusted settings

## Need Help?

See `TEST_INSTRUCTIONS.md` for detailed troubleshooting and advanced testing options.

