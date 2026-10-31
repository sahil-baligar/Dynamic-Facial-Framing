# Testing Mixed Content (Gameplay + Face)

## 📁 Where to Put Your Video

### Option 1: Default Location (Easiest)

**Place your video in the project root and name it `input.mp4`:**

```
FacialFraming/
├── input.mp4          ← Your mixed content video here!
├── test_facial_framing.py
├── app/
└── ...
```

### Option 2: Custom Location

Place your video anywhere and specify the path:

```bash
python test_facial_framing.py --input "path/to/your/mixed_video.mp4"
```

## 🎬 How to Test

### Basic Test (Full Video)

```bash
# If named input.mp4
python test_facial_framing.py

# Or with custom path
python test_facial_framing.py --input "your_mixed_video.mp4" --output "output_mixed.mp4"
```

### Test Specific Segments

Since your video has both gameplay and face, test different segments:

```bash
# Test the gameplay part (should use simple cover crop)
python test_facial_framing.py --input "your_video.mp4" --start 0 --duration 30 --output "output_gameplay.mp4"

# Test the face part (should use dynamic facial framing)
python test_facial_framing.py --input "your_video.mp4" --start 60 --duration 30 --output "output_face.mp4"

# Test the transition area
python test_facial_framing.py --input "your_video.mp4" --start 30 --duration 30 --output "output_transition.mp4"
```

## 🎯 What to Expect

### Overall Mode Selection

The system analyzes the **entire clip** and chooses the dominant mode:

- **If face appears in ≥20% of frames** → Uses **dynamic facial framing** for the whole video
- **If face appears in <20% of frames** → Uses **simple cover crop** for the whole video

### Example Scenarios

#### Scenario 1: Face in Most of Video (≥20%)
```
Video: 2 minutes gameplay + 3 minutes face = 5 minutes total
Face detection: 60% of frames
Result: Uses dynamic facial framing for entire video
```

#### Scenario 2: Mostly Gameplay (<20% face)
```
Video: 4 minutes gameplay + 1 minute face = 5 minutes total  
Face detection: 20% of frames
Result: Uses simple cover crop for entire video
```

#### Scenario 3: Balanced (Close to 20%)
```
Video: 2.5 minutes gameplay + 2.5 minutes face = 5 minutes total
Face detection: 50% of frames
Result: Uses dynamic facial framing for entire video
```

## 📊 Understanding the Output

The test will show you:

```
🎯 Dominant layout state: vert_focus (591 frames)
👤 Face detection rate: 60.0% (600/1000 frames)
👤 Human detected! Using dynamic facial framing with 600 face detections
```

Or for mostly gameplay:

```
🎯 Dominant layout state: cover (800 frames)
👤 Face detection rate: 15.0% (200/1000 frames)
🎮 No human detected (15.0% face detection)
📐 Using simple vertical cover crop
```

## 🔧 Testing Different Segments

To see how each mode works:

### 1. Test Gameplay Segment
```bash
# Find where gameplay starts (e.g., 0-30 seconds)
python test_facial_framing.py --input "your_video.mp4" --start 0 --duration 30 --output "test_gameplay.mp4"
```
**Expected**: Simple vertical cover crop (no face tracking)

### 2. Test Face Segment  
```bash
# Find where face appears (e.g., 60-90 seconds)
python test_facial_framing.py --input "your_video.mp4" --start 60 --duration 30 --output "test_face.mp4"
```
**Expected**: Dynamic facial framing (smooth face tracking)

### 3. Test Full Video
```bash
# Test entire video
python test_facial_framing.py --input "your_video.mp4" --output "test_full.mp4"
```
**Expected**: Mode based on overall face detection percentage

## 💡 Tips

1. **Check Detection Rate**: Look at the face detection percentage in the output
2. **Adjust Threshold**: If needed, adjust `min_face_detection_percentage` in `app/settings.py`
3. **Test Segments**: Test different time segments to see each mode in action
4. **Review Output**: Check the output video to verify the mode selection

## ⚙️ Adjusting Sensitivity

If the system isn't detecting the transition correctly:

### Make it More Sensitive (Detect Face More Easily)
```python
# In app/settings.py
min_face_detection_percentage: float = 15.0  # Lower threshold
min_face_detections: int = 5  # Fewer frames needed
```

### Make it Less Sensitive (Require More Face Detection)
```python
# In app/settings.py
min_face_detection_percentage: float = 30.0  # Higher threshold
min_face_detections: int = 20  # More frames needed
```

## 🎮 Quick Test Commands

```bash
# Replace "your_video.mp4" with your actual video filename

# Test full video
python test_facial_framing.py --input "your_video.mp4"

# Test first 30 seconds (likely gameplay)
python test_facial_framing.py --input "your_video.mp4" --start 0 --duration 30

# Test from 1 minute to 2 minutes (might have face)
python test_facial_framing.py --input "your_video.mp4" --start 60 --duration 60
```

## 📝 Notes

- The system analyzes the **entire specified segment** and chooses one mode
- For true per-segment switching within a single video, you'd need to process in chunks
- The 20% threshold works well for most mixed content
- You can always process segments separately if you want different modes for different parts

