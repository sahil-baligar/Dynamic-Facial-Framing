"""ffmpeg/ffprobe helpers: probing, exact raw-frame reading, speech activity from audio."""
from __future__ import annotations
import json, subprocess, shutil, queue, threading
import numpy as np

NOWIN = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0


def ffmpeg_bin(): return shutil.which("ffmpeg") or "ffmpeg"


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, creationflags=NOWIN)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    fps = float(n) / float(d or 1)
    dur = float(s.get("duration") or j["format"]["duration"])
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "duration": dur}


def read_frames(path, start, duration, fps=None, scale_w=None, crop=None):
    """Yield (index, bgr ndarray) for [start, start+duration). Decodes with ffmpeg so analysis and render see identical frames."""
    info = probe(path)
    w, h = info["width"], info["height"]
    vf = []
    if crop:
        cx, cy, cw, ch = [int(v) // 2 * 2 for v in crop]
        vf.append(f'crop={cw}:{ch}:{cx}:{cy}'); w, h = cw, ch
    if fps: vf.append(f"fps={fps}")
    if scale_w and scale_w < w:
        sw = scale_w - scale_w % 2; sh = int(round(h * sw / w / 2) * 2); vf.append(f"scale={sw}:{sh}:flags=area"); w, h = sw, sh
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path), "-an"]
    if vf: cmd += ["-vf", ",".join(vf)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NOWIN, bufsize=1 << 24)
    size = w * h * 3
    q: "queue.Queue" = queue.Queue(maxsize=8)
    stop = threading.Event()

    def pump():                                   # decode ahead of the consumer so ffmpeg and OpenCV overlap
        try:
            while not stop.is_set():
                buf = p.stdout.read(size)
                if len(buf) < size: break
                q.put(buf)
        finally:
            q.put(None)
    threading.Thread(target=pump, daemon=True).start()
    n = 0
    try:
        while True:
            buf = q.get()
            if buf is None: break
            yield n, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        stop.set()
        try:
            while not q.empty(): q.get_nowait()
            p.kill(); p.stdout.close()
        except Exception: pass


def read_pcm(path, start, duration, sr=16000):
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=NOWIN).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def speech_activity(samples, sr=16000, hop=0.02):
    """Adaptive energy VAD. Returns a bool array, one value per `hop` seconds. Good enough to tell 'someone is talking' from silence
    in clean podcast audio; words from a transcript can replace it where available."""
    n = int(sr * hop)
    m = len(samples) // n
    if m == 0: return np.zeros(0, bool)
    rms = np.sqrt((samples[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    floor = np.percentile(db, 10); top = np.percentile(db, 95)
    thr = floor + max(6.0, 0.35 * (top - floor))
    act = db > thr
    k = int(0.15 / hop)                         # bridge gaps shorter than 150 ms, drop blips shorter than 100 ms
    if k > 1:
        kern = np.ones(k)
        act = np.convolve(act.astype(float), kern, "same") >= 1
        act = np.convolve(act.astype(float), np.ones(max(2, k - 1)), "same") >= (k - 1) * 0.99
    return act
