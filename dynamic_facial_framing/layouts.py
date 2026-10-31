"""Layout helpers: webcam region detection for gameplay + facecam footage, and content-type classification."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, cv2
from .media import probe, read_frames
from .detect import FaceDetector


@dataclass
class Rect:
    x: int; y: int; w: int; h: int
    def as_list(self): return [int(self.x), int(self.y), int(self.w), int(self.h)]


def sample_frames(path, start, duration, n=24, scale_w=960):
    """n frames spread evenly over the range (one ffmpeg seek each: cheap and independent of length)."""
    out = []
    for k in range(n):
        t = start + duration * (k + 0.5) / n
        for _, fr in read_frames(path, t, 0.2, scale_w=scale_w):
            out.append((t, fr.copy())); break
    return out


def classify_content(path, start=0.0, duration=None, n=24, detector=None, frames=None):
    """Return {'type': talking_head|podcast|gaming|other, 'confidence', 'evidence', 'webcam': Rect|None}.
    gaming  = one small face that sits in the same place in most samples while the rest of the frame is not a person
    podcast = two or more sizeable faces at once (or in alternating shots) for a good share of the samples
    talking_head = a sizeable face in most samples, rarely two
    """
    info = probe(path); duration = duration or info["duration"] - start
    own = detector is None
    det = detector or FaceDetector(video=False)
    frames = frames or sample_frames(path, start, duration, n)
    H = frames[0][1].shape[0]; W = frames[0][1].shape[1]
    per = []
    for t, fr in frames:
        per.append([f for f in det.detect(fr, t) if f.h > 0.035 * H])
    if own: det.close()
    N = len(per)
    multi = sum(1 for fs in per if len([f for f in fs if f.h > 0.10 * H]) >= 2) / N
    any_big = sum(1 for fs in per if any(f.h > 0.14 * H for f in fs)) / N
    small = [(i, f) for i, fs in enumerate(per) for f in fs if f.h < 0.30 * H and f.h > 0.04 * H]
    ev = {"samples": N, "multi_face_ratio": round(multi, 2), "big_face_ratio": round(any_big, 2)}
    # webcam: cluster small faces by position
    cam = None
    if len(small) >= 0.35 * N:
        pts = np.array([[f.cx / W, f.cy / H] for _, f in small])
        c = np.median(pts, axis=0)
        near = [(i, f) for (i, f), p in zip(small, pts) if abs(p[0] - c[0]) < 0.05 and abs(p[1] - c[1]) < 0.07]
        frames_with = len({i for i, _ in near}) / N
        corner = (c[0] < 0.30 or c[0] > 0.70) and (c[1] < 0.35 or c[1] > 0.65)
        ev.update(cam_ratio=round(frames_with, 2), cam_center=[round(float(c[0]), 2), round(float(c[1]), 2)], cam_corner=bool(corner))
        if frames_with >= 0.5 and corner and any_big < 0.3:
            cam = _webcam_rect([f for _, f in near], [fr for _, fr in frames], W, H)
    if cam is not None:
        s = info["width"] / W
        return {"type": "gaming", "confidence": round(min(1.0, 0.55 + ev["cam_ratio"] / 2), 2), "evidence": ev,
                "webcam": Rect(cam.x * s, cam.y * s, cam.w * s, cam.h * s)}
    if multi >= 0.2: return {"type": "podcast", "confidence": round(min(1.0, 0.5 + multi), 2), "evidence": ev, "webcam": None}
    if any_big >= 0.5: return {"type": "talking_head", "confidence": round(any_big, 2), "evidence": ev, "webcam": None}
    return {"type": "other", "confidence": round(1 - any_big, 2), "evidence": ev, "webcam": None}


def _webcam_rect(faces, frames, W, H) -> Rect:
    """Face box -> overlay rectangle. Edges are the strongest steps in the median image around the face; near-frame-edge sides snap out to the frame."""
    fx = float(np.median([f.cx for f in faces])); fy = float(np.median([f.cy for f in faces])); fh = float(np.median([f.h for f in faces])); fw = float(np.median([f.w for f in faces]))
    med = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:: max(1, len(frames) // 12)]]), axis=0).astype(np.float32)
    gx = np.abs(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)); gy = np.abs(cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    y0, y1 = int(max(0, fy - fh * 0.7)), int(min(H, fy + fh * 0.7)); x0, x1 = int(max(0, fx - fw * 0.7)), int(min(W, fx + fw * 0.7))
    def edge(prof, lo, hi):
        lo, hi = int(max(0, lo)), int(min(len(prof) - 1, hi))
        if hi <= lo: return None
        seg = prof[lo:hi + 1]; i = int(np.argmax(seg))
        return lo + i if seg[i] > 1.8 * (np.median(prof[lo:hi + 1]) + 1e-3) else None
    colp = gx[y0:y1].mean(0); rowp = gy[:, x0:x1].mean(1)
    L = edge(colp, fx - fw * 3.0, fx - fw * 0.65); R = edge(colp, fx + fw * 0.65, fx + fw * 3.0)
    T = edge(rowp, fy - fh * 3.0, fy - fh * 0.6); B = edge(rowp, fy + fh * 0.6, fy + fh * 3.0)
    pad = 0.08
    left = L if L is not None else fx - fw * 1.2; right = R if R is not None else fx + fw * 1.2
    top = T if T is not None else fy - fh * 1.2; bot = B if B is not None else fy + fh * 1.2
    if left < pad * W: left = 0
    if right > (1 - pad) * W: right = W
    if top < pad * H: top = 0
    if bot > (1 - pad) * H: bot = H
    return Rect(int(left), int(top), int(right - left), int(bot - top))
