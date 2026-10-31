from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Blendshapes used for the expression vector. jawOpen is kept apart (it is mostly speech).
EXPR_KEYS = [
    "mouthSmileLeft", "mouthSmileRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "browDownLeft", "browDownRight", "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "mouthFrownLeft", "mouthFrownRight", "cheekSquintLeft", "cheekSquintRight", "mouthPucker",
]


@dataclass
class Face:
    """One detected face in pixel coordinates of the analysed frame."""
    x: float; y: float; w: float; h: float        # bounding box (forehead to chin)
    score: float = 1.0
    eye: tuple = (0.0, 0.0)                         # midpoint between the eyes
    mouth: float = 0.0                              # jaw/mouth opening, 0..1 (blendshape or landmark ratio)
    expr: Optional[np.ndarray] = None               # len(EXPR_KEYS) blendshape vector, None for non-landmark detectors

    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2


@dataclass
class Track:
    id: int
    shot: int
    t: list = field(default_factory=list)          # seconds, relative to the analysed range
    faces: list = field(default_factory=list)

    def size(self):
        return float(np.median([f.h for f in self.faces])) if self.faces else 0.0

    def series(self, attr):
        return np.array([getattr(f, attr) for f in self.faces], dtype=float)


@dataclass
class Analysis:
    path: str
    start: float
    duration: float
    fps: float
    width: int
    height: int
    shots: list                                     # [(first_frame, last_frame_exclusive)]
    tracks: list                                    # list[Track]
    n_frames: int
    sample_every: int
    speech: Optional[np.ndarray] = None             # bool per 20 ms hop, None when audio was not analysed
    speech_hop: float = 0.02
