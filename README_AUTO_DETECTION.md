# Automatic Human Detection & Layout Switching

The system now automatically detects when a human/speaker is present in the video and switches between two modes:

## 🎯 Automatic Mode Detection

### 👤 **Human/Speaker Detected** → Dynamic Facial Framing
- When faces are detected in ≥20% of frames
- Uses smooth, dynamic facial tracking
- Crop follows the speaker as they move
- Perfect for talking head videos, podcasts, interviews

### 🎮 **No Human Detected** → Simple Vertical Cover
- When faces are detected in <20% of frames
- Uses the original horizontal-to-vertical conversion
- Simple cover crop (no face tracking)
- Perfect for gameplay videos, landscape footage, non-human content

## 📊 How It Works

1. **Frame Analysis**: System analyzes frames to detect faces
2. **Detection Rate**: Calculates percentage of frames with faces
3. **Automatic Selection**:
   - **≥20% face detection** → Dynamic facial framing
   - **<20% face detection** → Simple vertical cover crop
4. **Seamless Processing**: All handled automatically - no manual mode selection needed

## ⚙️ Configuration

### Face Detection Thresholds

In `app/settings.py`:

```python
# Minimum % of frames with faces to use dynamic framing
min_face_detection_percentage: float = 20.0  # 20% threshold

# Minimum number of frames with faces
min_face_detections: int = 10

# Frames before switching to cover (when no face detected)
state_transition_threshold: int = 15
```

### Adjusting Thresholds

**More sensitive (detect humans more easily):**
```python
min_face_detection_percentage: float = 15.0  # Lower threshold
min_face_detections: int = 5  # Fewer frames needed
```

**Less sensitive (require more consistent face detection):**
```python
min_face_detection_percentage: float = 30.0  # Higher threshold
min_face_detections: int = 20  # More frames needed
```

## 🎬 Usage Examples

### Example 1: Talking Head Video (Human Present)
```python
from app.vertical import extract_dynamic_layout
from app.layout.state_machine import LayoutStateMachine

state_machine = LayoutStateMachine()
result = extract_dynamic_layout(
    src="talking_head.mp4",
    dst="output.mp4",
    start=0.0,
    duration=60.0,
    state_machine=state_machine,
    background_mode="blur"
)

# Result: Uses dynamic facial framing automatically
# Face detection: 95% → Dynamic facial framing
```

### Example 2: Gameplay Video (No Human)
```python
state_machine = LayoutStateMachine()
result = extract_dynamic_layout(
    src="gameplay.mp4",
    dst="output.mp4",
    start=0.0,
    duration=60.0,
    state_machine=state_machine,
    background_mode="blur"
)

# Result: Uses simple vertical cover automatically
# Face detection: 2% → Simple vertical cover crop
```

### Example 3: Mixed Content (Human + Gameplay)
```python
state_machine = LayoutStateMachine()
result = extract_dynamic_layout(
    src="mixed_content.mp4",
    dst="output.mp4",
    start=0.0,
    duration=120.0,
    state_machine=state_machine,
    background_mode="blur"
)

# Result: Automatically uses appropriate mode for each segment
# Segments with humans → Dynamic facial framing
# Segments without humans → Simple vertical cover
```

## 📈 Detection Statistics

The system provides detailed statistics:

```
🎯 Dominant layout state: vert_focus (591 frames)
👤 Face detection rate: 100.0% (600/600 frames)
👤 Human detected! Using dynamic facial framing with 600 face detections
```

Or for non-human content:

```
🎯 Dominant layout state: cover (600 frames)
👤 Face detection rate: 2.5% (15/600 frames)
🎮 No human detected (2.5% face detection)
📐 Using simple vertical cover crop (original horizontal-to-vertical conversion)
```

## 🔧 Manual Override

If you want to force a specific mode, use the `extract_vertical_clip` function:

```python
from app.vertical import extract_vertical_clip

# Force simple cover (no face detection)
extract_vertical_clip(
    src="video.mp4",
    dst="output.mp4",
    start=0.0,
    duration=60.0,
    mode="cover"
)

# Force dynamic facial framing (with face detection)
extract_vertical_clip(
    src="video.mp4",
    dst="output.mp4",
    start=0.0,
    duration=60.0,
    mode="dynamic"
)
```

## 💡 Tips

1. **Gameplay Videos**: System automatically detects no human and uses simple cover crop
2. **Talking Head Videos**: System automatically detects human and uses dynamic facial framing
3. **Mixed Content**: System handles transitions automatically based on face detection
4. **Threshold Tuning**: Adjust `min_face_detection_percentage` if detection is too sensitive/not sensitive enough
5. **Processing Speed**: Simple cover crop is faster than dynamic facial framing

## 🎮 Use Cases

### ✅ Automatic Mode Works Great For:
- **Gameplay videos** (no human) → Simple cover crop
- **Talking head videos** (human present) → Dynamic facial framing
- **Podcast videos** (human present) → Dynamic facial framing
- **Landscape videos** (no human) → Simple cover crop
- **Mixed content** (human + gameplay) → Automatic switching

### ⚙️ Manual Mode Recommended For:
- Very specific requirements
- Testing different modes
- Fine-tuning for specific content types

## 📝 Notes

- The 20% threshold is a good default for most content
- System analyzes every frame for accurate detection
- Transitions between modes are smooth and automatic
- No manual intervention required for most videos

