"""Frame compositing and encoding. Frames are decoded with ffmpeg, cropped/composed with OpenCV,
encoded by a second ffmpeg that also muxes the source audio."""
from __future__ import annotations
import subprocess
import numpy as np, cv2
from .media import ffmpeg_bin, read_frames, probe, NOWIN
from .detect import FaceDetector
from .framing import CropPlan, smooth_damp

OUT_W, OUT_H = 1080, 1920


class Encoder:
    def __init__(self, dst, size, fps, src=None, start=0.0, duration=None, audio=True, crf=17, preset="veryfast"):
        w, h = size
        cmd = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.5f}", "-i", "-"]
        if audio and src:
            cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "160k",
                    "-af", "afade=t=in:d=0.04,loudnorm=I=-15:TP=-1.5:LRA=11"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(dst)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=NOWIN)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc:
            raise RuntimeError(f"ffmpeg encoder exited {rc}")


def crop_resize(frame, x, y, w, h, size):
    """Sub-pixel crop of (x,y,w,h) scaled to size=(W,H). Area resampling when shrinking, bicubic when growing."""
    ow, oh = size
    s = ow / w
    if s < 0.75:
        x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
        return cv2.resize(frame[y0:int(round(y + h)), x0:int(round(x + w))], size, interpolation=cv2.INTER_AREA)
    M = np.array([[s, 0, -x * s], [0, oh / h, -y * oh / h]], np.float32)
    return cv2.warpAffine(frame, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def render_crop(src, dst, start, duration, plan: CropPlan, size=(OUT_W, OUT_H), progress=None, **enc):
    e = Encoder(dst, size, plan.fps, src, start, duration, **enc)
    try:
        for i, fr in read_frames(src, start, duration):
            j = min(i, len(plan.x) - 1)
            e.write(crop_resize(fr, plan.x[j], plan.y[j], plan.w[j], plan.h[j], size))
            if progress and i % 60 == 0:
                progress(i / max(1, len(plan.x)))
    finally:
        e.close()


def render_blur_fill(src, dst, start, duration, fps, size=(OUT_W, OUT_H), progress=None, **enc):
    """Whole frame fitted to the width, over a blurred zoomed copy of itself. For footage with no face to follow."""
    info = probe(src)
    W, H = info["width"], info["height"]
    ow, oh = size
    fh = int(round(ow * H / W / 2) * 2)
    top = (oh - fh) // 2
    e = Encoder(dst, size, fps, src, start, duration, **enc)
    try:
        for i, fr in read_frames(src, start, duration):
            sc = oh / H
            bw = int(W * sc)
            x0 = max(0, (bw - ow) // 2)
            bg = cv2.resize(fr, (bw, oh), interpolation=cv2.INTER_LINEAR)[:, x0:x0 + ow]
            bg = cv2.GaussianBlur(cv2.resize(bg, (ow // 4, oh // 4), interpolation=cv2.INTER_AREA), (0, 0), 6)
            bg = (cv2.resize(bg, (ow, oh), interpolation=cv2.INTER_LINEAR) * 0.55).astype(np.uint8)
            bg[top:top + fh] = cv2.resize(fr, (ow, fh), interpolation=cv2.INTER_AREA if W > ow else cv2.INTER_CUBIC)
            e.write(bg)
    finally:
        e.close()


def plan_cam(src, start, duration, cam, fps, out_aspect, detector=None):
    """Follow the face inside the webcam rectangle. Returns per-frame crop origin (x,y arrays) and size (w,h) in full-frame pixels."""
    cx0, cy0, cw0, ch0 = cam
    ch = min(ch0, cw0 / out_aspect)
    cw = ch * out_aspect
    own = detector is None
    det = detector or FaceDetector()
    n = int(duration * fps) + 2
    X = np.zeros(n)
    Y = np.zeros(n)
    tgt = np.array([cx0 + cw0 / 2, cy0 + ch0 / 2])
    cur = tgt.copy()
    vel = np.zeros(2)
    scale = max(1.0, 480 / max(8, cw0))
    last = 0
    for i, fr in read_frames(src, start, duration, crop=(cx0, cy0, cw0, ch0)):
        if i >= n:
            break
        if i % 3 == 0:
            small = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else fr
            fs = det.detect(small, i / fps)
            if fs:
                f = max(fs, key=lambda q: q.w * q.h)
                tgt = np.array([cx0 + f.cx / scale, cy0 + (f.eye[1] if f.eye[1] else f.cy) / scale])
        for k in range(2):
            cur[k], vel[k] = smooth_damp(cur[k], tgt[k], vel[k], 0.5, 1 / fps)
        X[i] = np.clip(cur[0] - cw / 2, cx0, cx0 + cw0 - cw)
        Y[i] = np.clip(cur[1] - ch * 0.42, cy0, cy0 + ch0 - ch)
        last = i
    X[last:] = X[last]
    Y[last:] = Y[last]
    if own:
        det.close()
    return X, Y, cw, ch


def game_crop(frame_w, frame_h, aspect, cam=None):
    """Static cover crop of the gameplay for a slot of w/h = aspect, kept clear of the webcam where possible."""
    ch = float(frame_h)
    cw = ch * aspect
    if cw > frame_w:
        cw = float(frame_w)
        ch = cw / aspect
    x = (frame_w - cw) / 2
    y = (frame_h - ch) / 2
    if cam is not None:
        if cam[0] + cam[2] / 2 > frame_w / 2:
            x = min(x, max(0.0, cam[0] - cw - 4))
        else:
            x = max(x, min(frame_w - cw, cam[0] + cam[2] + 4))
        x = float(np.clip(x, 0, frame_w - cw))
    return x, y, cw, ch


def _hide(top, a, b, c, d):
    """Pixelate-blur the part of the webcam that still lies inside the gameplay slot."""
    reg = top[b:d, a:c]
    if reg.shape[0] < 10 or reg.shape[1] < 10:
        return
    small = cv2.resize(reg, (max(2, reg.shape[1] // 12), max(2, reg.shape[0] // 12)), interpolation=cv2.INTER_AREA)
    top[b:d, a:c] = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5), (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_LINEAR)


def render_split(src, dst, start, duration, fps, cam, split=0.5, size=(OUT_W, OUT_H), progress=None, detector=None, **enc):
    """Gameplay on top, webcam below (face-followed). cam = (x,y,w,h) in source pixels."""
    info = probe(src)
    W, H = info["width"], info["height"]
    ow, oh = size
    th = int(round(oh * split / 2) * 2)
    bh = oh - th
    cx, cy, cwid, chei = plan_cam(src, start, duration, cam, fps, ow / bh, detector)
    gx, gy, gw, gh = game_crop(W, H, ow / th, cam)
    ox0, oy0 = max(cam[0], gx), max(cam[1], gy)
    ox1, oy1 = min(cam[0] + cam[2], gx + gw), min(cam[1] + cam[3], gy + gh)
    overlap = (ox1 - ox0 > 4 and oy1 - oy0 > 4)
    e = Encoder(dst, size, fps, src, start, duration, **enc)
    try:
        for i, fr in read_frames(src, start, duration):
            j = min(i, len(cx) - 1)
            top = fr[int(gy):int(gy + gh), int(gx):int(gx + gw)].copy()
            if overlap:
                _hide(top, int(ox0 - gx), int(oy0 - gy), int(ox1 - gx), int(oy1 - gy))
            canvas = np.empty((oh, ow, 3), np.uint8)
            canvas[:th] = cv2.resize(top, (ow, th), interpolation=cv2.INTER_AREA if gw > ow else cv2.INTER_CUBIC)
            canvas[th:] = crop_resize(fr, cx[j], cy[j], cwid, chei, (ow, bh))
            e.write(canvas)
            if progress and i % 60 == 0:
                progress(i / max(1.0, duration * fps))
    finally:
        e.close()
