"""Pass 1: decode the range once, find shot cuts on every frame, detect faces every few frames, link them into tracks."""
from __future__ import annotations
import numpy as np, cv2
from .types import Analysis, Track, Face
from .media import probe, read_frames, read_pcm, speech_activity
from .detect import FaceDetector


def _sig(frame):
    g = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32), cv2.calcHist([g], [0], None, [32], [0, 256]).ravel() / g.size


def is_cut(prev, cur):
    (pg, ph), (cg, ch) = prev, cur
    hist_d = 0.5 * np.abs(ph - ch).sum()          # 0..1
    pix_d = float(np.abs(pg - cg).mean()) / 255.0
    return (hist_d > 0.28 and pix_d > 0.07) or pix_d > 0.30


def _iou_dist(a: Face, b: Face):
    cx, cy = a.cx - b.cx, a.cy - b.cy
    return np.hypot(cx, cy) / max(1.0, max(a.h, b.h))


def analyze_video(path, start=0.0, duration=None, sample_fps=6.0, detector=None, audio=True, analysis_width=960, progress=None) -> Analysis:
    info = probe(path)
    duration = min(duration or info["duration"], info["duration"] - start)
    fps = info["fps"]
    every = max(1, int(round(fps / sample_fps)))
    own = detector is None
    det = detector or FaceDetector()
    sw = min(analysis_width, info["width"])
    scale = sw / info["width"]
    shots = []; shot_start = 0; shot_id = 0
    tracks: list[Track] = []; active: list[Track] = []
    prev = None; n = 0
    total = max(1, int(duration * fps))
    for i, frame in read_frames(path, start, duration, scale_w=sw):
        n = i + 1
        s = _sig(frame)
        if prev is not None and is_cut(prev, s):
            shots.append((shot_start, i)); shot_start = i; shot_id += 1; active = []
            if hasattr(det, "_last_ms"): pass
        prev = s
        if i % every == 0:
            t = i / fps
            faces = det.detect(frame, t)
            for f in faces:                            # back to source pixel space
                f.x /= scale; f.y /= scale; f.w /= scale; f.h /= scale; f.eye = (f.eye[0] / scale, f.eye[1] / scale)
            taken = set()
            for f in sorted(faces, key=lambda q: -q.h):
                best, bd = None, 1e9
                for tr in active:
                    if id(tr) in taken or t - tr.t[-1] > 1.0: continue
                    d = _iou_dist(f, tr.faces[-1])
                    if d < bd: best, bd = tr, d
                if best is not None and bd < 1.2:
                    best.t.append(t); best.faces.append(f); taken.add(id(best))
                else:
                    tr = Track(len(tracks), shot_id, [t], [f]); tracks.append(tr); active.append(tr); taken.add(id(tr))
        if progress and i % 90 == 0: progress(min(1.0, i / total))
    if prev is not None: shots.append((shot_start, n))
    if own: det.close()
    sp = None
    if audio:
        try: sp = speech_activity(read_pcm(path, start, duration))
        except Exception: sp = None
    tracks = [t for t in tracks if len(t.t) >= 3]
    return Analysis(str(path), start, duration, fps, info["width"], info["height"], shots, tracks, n, every, sp)
