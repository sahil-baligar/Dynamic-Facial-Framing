"""Face detection. MediaPipe FaceLandmarker (478 landmarks + blendshapes) when available, OpenCV YuNet otherwise."""
from __future__ import annotations
import numpy as np, cv2
from .types import Face, EXPR_KEYS
from .models import model_path

_MP_OK = None


def mediapipe_available() -> bool:
    global _MP_OK
    if _MP_OK is None:
        try:
            import mediapipe  # noqa: F401
            from mediapipe.tasks.python import vision  # noqa: F401
            _MP_OK = True
        except Exception:
            _MP_OK = False
    return _MP_OK


class FaceDetector:
    """detect(frame_bgr, t_seconds) -> list[Face]. Faces are in the pixel space of the frame passed in.
    backend: 'auto' | 'hybrid' | 'mediapipe' | 'yunet'.
    hybrid (the auto choice when mediapipe is installed): YuNet finds faces of any size, MediaPipe FaceLandmarker then reads each face crop
    for landmarks, mouth opening and the 52 blendshapes. mediapipe alone only sees reasonably large faces; yunet alone has no expression data."""

    def __init__(self, backend="auto", max_faces=4, min_conf=0.45, video=True):
        self.max_faces = max_faces
        self.backend = backend if backend != "auto" else ("hybrid" if mediapipe_available() else "yunet")
        self._last_ms = -1
        self.min_conf = min_conf
        if self.backend in ("mediapipe", "hybrid"):
            from mediapipe.tasks.python import vision, BaseOptions
            self.video = video and self.backend == "mediapipe"
            mode = vision.RunningMode.VIDEO if self.video else vision.RunningMode.IMAGE
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("face_landmarker.task"))),
                running_mode=mode, num_faces=max_faces if self.backend == "mediapipe" else 1, output_face_blendshapes=True,
                min_face_detection_confidence=min_conf, min_face_presence_confidence=min_conf, min_tracking_confidence=0.4)
            self._lm = vision.FaceLandmarker.create_from_options(opts)
        if self.backend in ("yunet", "hybrid"):
            self._yn = cv2.FaceDetectorYN.create(str(model_path("yunet.onnx")), "", (320, 320), max(min_conf, 0.6), 0.3, 50)

    def detect(self, frame, t=0.0):
        if self.backend == "mediapipe": return self._mp(frame, t)
        if self.backend == "yunet": return self._yunet(frame)
        return self._hybrid(frame, t)

    def _hybrid(self, frame, t):
        H, W = frame.shape[:2]
        out = []
        for f in sorted(self._yunet(frame), key=lambda q: -q.w * q.h)[: self.max_faces]:
            side = max(f.w, f.h) * 1.7
            cx, cy = f.cx, f.cy
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2)); x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            if x1 - x0 < 16 or y1 - y0 < 16: out.append(f); continue
            crop = frame[y0:y1, x0:x1]
            sc = 320.0 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
            r = self._mp(crop, t)
            if not r: out.append(f); continue
            g = max(r, key=lambda q: q.w * q.h)
            g.x = g.x / sc + x0; g.y = g.y / sc + y0; g.w /= sc; g.h /= sc; g.eye = (g.eye[0] / sc + x0, g.eye[1] / sc + y0); g.score = f.score
            out.append(g)
        return out

    # -- MediaPipe
    def _mp(self, frame, t):
        import mediapipe as mp
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.video:
            ms = max(int(t * 1000), self._last_ms + 1); self._last_ms = ms
            r = self._lm.detect_for_video(img, ms)
        else:
            r = self._lm.detect(img)
        out = []
        for i, lms in enumerate(r.face_landmarks):
            xs = np.fromiter((p.x for p in lms), float, len(lms)) * w
            ys = np.fromiter((p.y for p in lms), float, len(lms)) * h
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            bs = {c.category_name: c.score for c in r.face_blendshapes[i]} if r.face_blendshapes else {}
            expr = np.array([bs.get(k, 0.0) for k in EXPR_KEYS], dtype=np.float32) if bs else None
            if "jawOpen" in bs:
                mouth = float(bs["jawOpen"])
            else:
                mouth = float(abs(ys[14] - ys[13]) / max(1.0, y1 - y0)) * 4
            eye = ((xs[33] + xs[263]) / 2, (ys[33] + ys[263]) / 2)
            out.append(Face(x0, y0, x1 - x0, y1 - y0, 1.0, eye, mouth, expr))
        return out

    # -- OpenCV YuNet
    def _yunet(self, frame):
        h, w = frame.shape[:2]
        self._yn.setInputSize((w, h))
        _, res = self._yn.detect(frame)
        out = []
        for r in (res if res is not None else [])[: self.max_faces]:
            x, y, fw, fh = r[:4]
            # YuNet's box is eyes-to-chin tight; grow it upward to approximate a forehead-to-chin box
            y2 = y - 0.25 * fh
            eye = ((r[4] + r[6]) / 2, (r[5] + r[7]) / 2)
            out.append(Face(float(x), float(y2), float(fw), float(fh * 1.25), float(r[14]), eye, 0.0, None))
        return out

    def close(self):
        if self.backend in ("mediapipe", "hybrid"):
            try: self._lm.close()
            except Exception: pass
