# Quick Test Guide for Mixed Content Video

## 🎬 Step 1: Place Your Video

**Put your video in the project root folder and name it `input.mp4`**

Or keep your current name and use the `--input` flag.

## 🚀 Step 2: Run the Test

### Test the Full Video
```bash
python test_facial_framing.py --input "your_video.mp4" --output "output_mixed.mp4"
```

### Test Gameplay Section (No Face)
```bash
# Adjust start time based on your video
python test_facial_framing.py --input "your_video.mp4" --start 0 --duration 30 --output "output_gameplay.mp4"
```

### Test Face Section (With Face)
```bash
# Adjust start time to where face appears
python test_facial_framing.py --input "your_video.mp4" --start 60 --duration 30 --output "output_face.mp4"
```

## 📊 What You'll See

The system will automatically:
- **Analyze all frames** for face detection
- **Calculate face detection percentage**
- **Choose the appropriate mode**:
  - ≥20% faces → Dynamic facial framing
  - <20% faces → Simple vertical cover crop

## 🎯 Expected Behavior

### If Face Appears in ≥20% of Frames
```
👤 Face detection rate: 60.0% (600/1000 frames)
👤 Human detected! Using dynamic facial framing
```
→ Uses smooth face tracking for entire video

### If Face Appears in <20% of Frames  
```
👤 Face detection rate: 15.0% (150/1000 frames)
🎮 No human detected (15.0% face detection)
📐 Using simple vertical cover crop
```
→ Uses simple cover crop for entire video

## 💡 Tips

1. **Check the output** to see which mode was selected
2. **Test different segments** to see each mode in action
3. **Adjust threshold** in `app/settings.py` if needed:
   ```python
   min_face_detection_percentage: float = 20.0  # Change this if needed
   ```

