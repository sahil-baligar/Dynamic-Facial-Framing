# How to Test Your Video

## 🎬 Quick Start (3 Steps)

### Step 1: Place Your Video

**Put your test video in the project root folder and name it `input.mp4`**

```
FacialFraming/
├── input.mp4          ← Your video goes here!
├── test_facial_framing.py
├── app/
└── ...
```

### Step 2: Run the Test

```bash
python test_facial_framing.py
```

### Step 3: Check the Result

The output will be saved as `output_dynamic.mp4` in the same folder.

## 📝 What the Test Does

1. **Detects faces** in your video
2. **Tracks the speaker** as they move
3. **Dynamically crops** to keep the face centered
4. **Creates a vertical video** (1080x1920) with smooth tracking

## 🎯 Expected Result

Based on your screenshot example, the system should:
- ✅ Keep the speaker centered in the vertical frame
- ✅ Follow the speaker as they move (smooth, dynamic tracking)
- ✅ Blur the background when a face is detected
- ✅ Create an engaging, TikTok/Reel-style vertical video

## ⚙️ Custom Options

### Test a Specific Segment
```bash
# Test first 30 seconds only (faster)
python test_facial_framing.py --duration 30

# Test a specific time range (e.g., 10-40 seconds)
python test_facial_framing.py --start 10 --duration 30
```

### Use a Different Video File
```bash
python test_facial_framing.py --input "path/to/your/video.mp4"
```

### Custom Output Name
```bash
python test_facial_framing.py --input your_video.mp4 --output my_result.mp4
```

## 🔧 Troubleshooting

### Video Not Found?
- Make sure the file is named `input.mp4` (or use `--input` to specify path)
- Check that the file is in the project root directory

### No Face Detected?
- Ensure the speaker's face is clearly visible
- Good lighting helps detection
- Try adjusting settings in `app/settings.py`:
  ```python
  face_detection_confidence = 0.3  # Lower = more sensitive
  ```

### Tracking Not Smooth?
- Adjust smoothing in `app/settings.py`:
  ```python
  crop_smoothing_factor = 0.1  # Lower = smoother (but slower response)
  ```

## 📊 What You'll See

The test script will show:
- Video information (dimensions, duration)
- Processing progress
- Face detection statistics
- Layout state distribution
- Success/failure status

## 💡 Tips for Best Results

1. **Start Small**: Test with a 10-30 second clip first
2. **Good Lighting**: Ensure faces are well-lit
3. **Clear Face**: Speaker's face should be clearly visible
4. **Check Output**: Review the output video to verify tracking
5. **Adjust Settings**: Tune parameters based on your video characteristics

## 🚀 Next Steps

After testing:
1. Review the output video quality
2. Adjust settings in `app/settings.py` if needed
3. Process your full video when satisfied
4. Batch process multiple videos if needed

## 📖 More Information

- See `TEST_INSTRUCTIONS.md` for detailed troubleshooting
- See `README.md` for full documentation
- See `README_TESTING.md` for testing details

