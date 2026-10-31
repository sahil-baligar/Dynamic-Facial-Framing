# Dynamic Facial Framing

Horizontal video in, vertical 9:16 video out, with the camera behaving like a human operator: it follows faces, goes to whoever is speaking, snaps to a strong reaction, and puts gameplay on top of the facecam for gaming footage.

```python
from dynamic_facial_framing import reframe

info = reframe("podcast.mp4", "short.mp4", start=312.0, duration=42.0)   # mode="auto"
print(info["mode"], info["events"])
```

```bash
python -m dynamic_facial_framing podcast.mp4 short.mp4 --start 312 --duration 42        # or: dff ...
python -m dynamic_facial_framing game.mp4 --classify                                    # what is this footage?
```

## What it does

| Footage | What you get |
|---|---|
| `talking_head` | Smooth face tracking: dead zone, critically damped spring, zoom to a consistent head size, eyes on a fixed line. Shot cuts in the source reset the camera instead of gliding. |
| `podcast` (2+ faces) | Active-speaker framing. The speaker is the face whose mouth moves while the audio is active; the camera pans over in about 0.15 s. A strong expression change (laugh, raised brows, wide eyes) on someone who is not talking sends the camera to them quickly and holds about a second. |
| `gaming` (facecam overlay) | Webcam rectangle found automatically. Gameplay crop on top (kept clear of the webcam), the webcam below with its own face tracking. |
| `other` | Whole frame fitted to the width over a blurred, darkened copy of itself. |

`mode="auto"` samples frames across the range and classifies. You can force any mode.

## How it works

1. Decode the range once with ffmpeg (analysis and render see identical frames). Shot cuts are found on every frame (histogram plus pixel difference).
2. Detect faces every ~3 frames. Hybrid detector: OpenCV YuNet finds faces of any size, then MediaPipe FaceLandmarker reads each face crop for 478 landmarks, jaw opening and 52 blendshapes. Without MediaPipe it falls back to YuNet boxes (no speaker or expression cues).
3. Link faces into per-shot tracks.
4. Decide the focus (`speaker.py`): mouth activity (mean absolute change of jaw opening) gated by an energy VAD on the audio, with dwell time and hysteresis so it does not flicker. Expression change is the weighted distance of the smile / brow / eye-wide / squint / frown groups from the track's own 2.5 s baseline.
5. Plan the crop (`framing.py`): per-frame x, y and zoom. Calm spring (0.55 s) normally, quick spring (0.13 s) for speaker and expression snaps.
6. Render (`render.py`): sub-pixel crop and scale with OpenCV, piped to ffmpeg (x264, source audio, light loudness normalisation).

Models download on first use to `~/.cache/dff` (override with `DFF_MODEL_DIR`).

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Python 3.9+ and ffmpeg/ffprobe on PATH. MediaPipe is optional but recommended.

## Python API

* `reframe(src, dst, start, duration, mode="auto", size=(1080,1920))` returns a manifest (mode, classification, webcam rectangle, focus events).
* `classify_content(src, start, duration)` returns `{type, confidence, evidence, webcam}`.
* Lower level: `analyze_video` -> `focus_timeline` -> `plan_crop` -> `render_crop`, or `analyze_range` for the first three.

## Tests

```bash
python -m pytest tests
```

Offline unit tests on synthetic tracks: speaker switching, expression snaps, crop bounds, VAD, cut detection.

## Legacy

The original implementation (ffmpeg crop filters and a layout state machine) is kept in `legacy/` and `docs/legacy/`. The new package does not import it.

## Used by

RRender's "YouTube link to viral shorts" pipeline calls this package as a library.
