from __future__ import annotations
from .media import probe
from .analyze import analyze_video
from .speaker import focus_timeline
from .framing import plan_crop
from .layouts import classify_content
from .render import render_crop, render_split, render_blur_fill, OUT_W, OUT_H
from .detect import FaceDetector

MODES = ("auto", "talking_head", "podcast", "gaming", "other", "cover")


def analyze_range(src, start=0.0, duration=None, mode="podcast", detector=None, progress=None, **kw):
    """Analysis + focus decisions for a range, without rendering. Returns (Analysis, focus, CropPlan)."""
    an = analyze_video(src, start, duration, detector=detector, progress=progress)
    focus = focus_timeline(an, expressions=(mode == "podcast"), speaker=(mode in ("podcast", "talking_head")))
    return an, focus, plan_crop(an, focus, **kw)


def reframe(src, dst, start=0.0, duration=None, mode="auto", size=(OUT_W, OUT_H), audio=True, progress=None, detector=None, content=None, **enc):
    """Render [start, start+duration) of `src` as a vertical video at `dst`. Returns a manifest dict.
    mode: auto | talking_head | podcast | gaming | other.  `content` can carry a prior classify_content() result."""
    info = probe(src)
    duration = min(duration or info["duration"], info["duration"] - start)
    own = detector is None
    det = detector or FaceDetector()
    p = (lambda a, b: (lambda x: progress and progress(a + (b - a) * x)))
    try:
        cls = content
        if mode == "auto":
            cls = cls or classify_content(src, start, duration, detector=FaceDetector(video=False))
            mode = cls["type"]
        out = {"mode": mode, "classification": {k: v for k, v in (cls or {}).items() if k != "webcam"}, "src": str(src), "start": start, "duration": duration}
        if mode == "gaming":
            cam = (cls or {}).get("webcam") or classify_content(src, start, duration, detector=FaceDetector(video=False)).get("webcam")
            if cam is None:
                mode = out["mode"] = "other"
            else:
                out["webcam"] = cam.as_list()
                render_split(src, dst, start, duration, info["fps"], cam.as_list(), size=size, progress=p(0, 1), detector=det, audio=audio, **enc)
                return out
        if mode in ("talking_head", "podcast"):
            an, focus, plan = analyze_range(src, start, duration, mode, det, p(0, 0.4))
            if an.tracks:
                out["faces"] = len(an.tracks); out["shots"] = len(an.shots)
                out["events"] = [{"t": round(t, 2), "kind": k, "track": d} for t, k, d in plan.events if k != "start"]
                render_crop(src, dst, start, duration, plan, size, p(0.4, 1), audio=audio, **enc)
                return out
            mode = out["mode"] = "other"
        render_blur_fill(src, dst, start, duration, info["fps"], size, p(0, 1), audio=audio, **enc)
        return out
    finally:
        if own: det.close()
