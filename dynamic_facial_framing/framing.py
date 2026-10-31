"""Pass 2: turn tracks + focus decisions into a per-frame crop window with camera-operator style smoothing."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .types import Analysis
from .speaker import Focus


@dataclass
class CropPlan:
    x: np.ndarray; y: np.ndarray; w: np.ndarray; h: np.ndarray        # per frame, source pixels
    events: list = field(default_factory=list)                        # [(t, kind, detail)] for debugging and manifests
    src_w: int = 0; src_h: int = 0; fps: float = 30.0


def smooth_damp(cur, target, vel, smooth, dt):
    omega = 2.0 / max(smooth, 1e-3); x = omega * dt
    e = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x ** 3)
    change = cur - target; temp = (vel + omega * change) * dt
    return target + (change + temp) * e, (vel - omega * temp) * e


def _track_at(tr, t, attr):
    return float(np.interp(t, tr.t, tr.series(attr)))


def _median3(v):
    if len(v) < 3: return v
    p = np.r_[v[0], v, v[-1]]
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def plan_crop(an: Analysis, focus: dict, aspect=9 / 16, face_fill=0.27, max_zoom=1.4, eye_line=0.42,
              calm=0.55, quick=0.13, deadzone=0.05, zoom_dead=0.12) -> CropPlan:
    """face_fill: face height as a fraction of the crop height we aim for. eye_line: where the eyes sit from the top of the crop."""
    W, H, fps = an.width, an.height, an.fps
    n = an.n_frames; dt = 1.0 / fps
    max_ch = min(H, W / aspect)
    min_ch = max_ch / max_zoom
    X = np.zeros(n); Y = np.zeros(n); CW = np.zeros(n); CH = np.zeros(n)
    events = []
    # smoothed per-track series (median of 3 removes single-detection jitter)
    sm = {}
    for tr in an.tracks:
        cx = _median3(np.array([f.cx for f in tr.faces])); ey = _median3(np.array([f.eye[1] if f.eye[1] else f.y + 0.4 * f.h for f in tr.faces]))
        fh = _median3(tr.series("h"))
        sm[tr.id] = (np.array(tr.t), cx, ey, fh)
    for si, (f0, f1) in enumerate(an.shots):
        trs = [tr for tr in an.tracks if tr.shot == si]
        evs = focus.get(si, [])
        if not trs or not evs:                         # nothing to follow: hold a centred crop
            for f in range(f0, f1): X[f], Y[f], CW[f], CH[f] = W / 2 - max_ch * aspect / 2, H / 2 - max_ch / 2, max_ch * aspect, max_ch
            events.append((f0 / fps, "no-face", si)); continue
        cur = None; vel = np.zeros(3); fast_until = -1; ei = 0; held = None; last_focus = evs[0].track
        for f in range(f0, f1):
            t = f / fps
            while ei + 1 < len(evs) and evs[ei + 1].t <= t:
                ei += 1; e = evs[ei]; last_focus = e.track
                if e.reason in ("speaker", "expression"): fast_until = t + 0.6
                events.append((t, e.reason, e.track))
            tid = evs[ei].track
            tt, cx, ey, fh = sm[tid]
            q = min(max(t, tt[0]), tt[-1])
            tgt_x = float(np.interp(q, tt, cx)); tgt_e = float(np.interp(q, tt, ey)); tgt_h = float(np.interp(q, tt, fh))
            tgt_ch = float(np.clip(tgt_h / face_fill, min_ch, max_ch))
            tgt_y = tgt_e - eye_line * tgt_ch + 0.0
            if cur is None:                            # first frame of a shot: no glide, a cut is a cut
                cur = np.array([tgt_x, tgt_y + tgt_ch / 2, tgt_ch]); held = cur.copy()
            else:
                cw_now = cur[2] * aspect
                if abs(tgt_x - held[0]) > deadzone * cw_now or fast_until > t: held[0] = tgt_x
                if abs(tgt_y + tgt_ch / 2 - held[1]) > deadzone * cur[2] or fast_until > t: held[1] = tgt_y + tgt_ch / 2
                if abs(tgt_ch - held[2]) > zoom_dead * cur[2] or fast_until > t: held[2] = tgt_ch
                s = quick if fast_until > t else calm
                for k in range(3):
                    cur[k], vel[k] = smooth_damp(cur[k], held[k], vel[k], s if k < 2 else calm * 1.6, dt)
            ch = float(np.clip(cur[2], min_ch, max_ch)); cw = ch * aspect
            cx_ = float(np.clip(cur[0], cw / 2, W - cw / 2)); cy_ = float(np.clip(cur[1], ch / 2, H - ch / 2))
            X[f], Y[f], CW[f], CH[f] = cx_ - cw / 2, cy_ - ch / 2, cw, ch
    # frames past the last shot (rounding): repeat the last crop
    if n and (CW == 0).any():
        idx = np.where(CW > 0)[0]
        for f in np.where(CW == 0)[0]:
            j = idx[np.argmin(abs(idx - f))]; X[f], Y[f], CW[f], CH[f] = X[j], Y[j], CW[j], CH[j]
    return CropPlan(X, Y, CW, CH, events, W, H, fps)
