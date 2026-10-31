"""Offline unit tests: no models, no video files."""
import numpy as np
from dynamic_facial_framing.framing import smooth_damp, plan_crop
from dynamic_facial_framing.speaker import focus_timeline, expression_change, mouth_activity
from dynamic_facial_framing.types import Analysis, Track, Face, EXPR_KEYS
from dynamic_facial_framing.media import speech_activity
from dynamic_facial_framing.analyze import is_cut


def face(x, mouth=0.0, smile=0.0):
    e = np.zeros(len(EXPR_KEYS), np.float32); e[0] = e[1] = smile
    return Face(x, 100, 120, 160, 1.0, (x + 60, 160), mouth, e)


def two_person_analysis(secs=10, fps=30, step=3):
    ts = np.arange(0, secs, step / fps)
    a, b = Track(0, 0), Track(1, 0)
    for t in ts:
        a.t.append(t); b.t.append(t)
        a.faces.append(face(200, mouth=0.4 * abs(np.sin(t * 14)) if t < 5 else 0.0))   # A talks first half
        b.faces.append(face(1200, mouth=0.4 * abs(np.sin(t * 14)) if t >= 5 else 0.0, smile=0.0))
    return Analysis("x", 0, secs, fps, 1920, 1080, [(0, int(secs * fps))], [a, b], int(secs * fps), step)


def test_smooth_damp_converges():
    x, v = 0.0, 0.0
    for _ in range(120): x, v = smooth_damp(x, 100.0, v, 0.3, 1 / 30)
    assert abs(x - 100) < 0.5


def test_speaker_switch():
    an = two_person_analysis()
    f = focus_timeline(an, expressions=False)[0]
    assert f[0].track == 0 and any(e.track == 1 and e.reason == "speaker" for e in f)


def test_expression_snap_on_listener_smile():
    an = two_person_analysis()
    for fc, t in zip(an.tracks[1].faces, an.tracks[1].t):   # B (silent until 5 s) smiles at 2 s
        if 2.0 <= t < 3.2: fc.expr[0] = fc.expr[1] = 0.9
    f = focus_timeline(an, expressions=True)[0]
    assert any(e.track == 1 and e.reason == "expression" and 1.9 < e.t < 3.0 for e in f)


def test_plan_stays_inside_frame():
    an = two_person_analysis()
    plan = plan_crop(an, focus_timeline(an))
    assert (plan.x >= -1e-6).all() and (plan.x + plan.w <= 1920 + 1e-6).all() and (plan.y + plan.h <= 1080 + 1e-6).all()
    assert abs(plan.w[0] / plan.h[0] - 9 / 16) < 1e-6


def test_camera_moves_fast_to_new_speaker():
    an = two_person_analysis()
    plan = plan_crop(an, focus_timeline(an, expressions=False))
    i5 = int(5.2 * 30); i7 = int(7.0 * 30)
    assert plan.x[i7] > plan.x[i5 - 20] + 600      # panned across to B within ~2 s


def test_vad_finds_speech():
    sr = 16000; t = np.arange(sr * 2) / sr
    x = np.zeros_like(t, np.float32); x[sr // 2: sr + sr // 2] = 0.3 * np.sin(2 * np.pi * 220 * t[sr // 2: sr + sr // 2]); x += 0.001 * np.random.randn(len(x))
    a = speech_activity(x.astype(np.float32))
    assert a[int(0.9 / 0.02)] and not a[int(0.1 / 0.02)] and not a[int(1.9 / 0.02)]


def test_cut_detector():
    g = np.zeros((36, 64), np.float32); h = np.full((36, 64), 200, np.float32)
    hist = lambda im: np.histogram(im, 32, (0, 256))[0] / im.size
    assert is_cut((g, hist(g)), (h, hist(h))) and not is_cut((g, hist(g)), (g + 3, hist(g + 3)))
