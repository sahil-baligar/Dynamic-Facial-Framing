"""Model files. Downloaded once into ~/.cache/dff (override with DFF_MODEL_DIR)."""
from __future__ import annotations
import os, urllib.request
from pathlib import Path

URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def model_dir() -> Path:
    d = Path(os.environ.get("DFF_MODEL_DIR") or Path.home() / ".cache" / "dff")
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(name: str) -> Path:
    p = model_dir() / name
    if not p.exists() or p.stat().st_size < 10_000:
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(URLS[name], tmp)
        tmp.replace(p)
    return p
