"""Who should the frame be on? Active speaker from mouth motion gated by audio, plus quick moves to strong expression changes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .types import Analysis, Track

W = {"smile": 1.0, "brow_up": 1.0, "brow_down": 0.8, "eye_wide": 1.0, "squint": 0.5, "frown": 0.6, "cheek": 0.7, "pucker": 0.4}


def _groups(e: np.ndarray):
    # order = EXPR_KEYS
    return np.array([(e[0] + e[1]) / 2, max(e[2], (e[3] + e[4]) / 2), (e[5] + e[6]) / 2, (e[7] + e[8]) / 2,
                     (e[9] + e[10]) / 2, (e[11] + e[12]) / 2, (e[13] + e[14]) / 2, e[15]])


GW = np.array(list(W.values()))


def expression_change(track: Track, base_win=2.5, gap=0.3):
    """Per sample: weighted distance of the expression vector from the track's own recent baseline (median of the last base_win s).
    Speech opens the jaw but barely moves these groups, so a smile, a raised brow or wide eyes stand out."""
    t = np.array(track.t)
    if not track.faces or track.faces[0].expr is None:
        return t, np.zeros(len(t))
    rows = []; prev = np.zeros(len(GW))
    for f in track.faces:                          # a face the landmarker missed repeats the last reading
        if f.expr is not None: prev = _groups(f.expr)
        rows.append(prev)
    E = np.array(rows)
    d = np.zeros(len(t))
    for i in range(len(t)):
        lo = np.searchsorted(t, t[i] - base_win); hi = np.searchsorted(t, t[i] - gap)
        base = np.median(E[lo:hi], axis=0) if hi - lo >= 3 else E[max(0, i - 3):i + 1].mean(0)
        d[i] = float(np.sqrt((((E[i] - base) * GW) ** 2).sum()))
    # one-sample blips (blink, detector noise) are not expressions
    d = np.minimum(d, np.maximum(np.r_[d[1:], 0], np.r_[0, d[:-1]]) * 1.6 + 0.02)
    return t, d


def mouth_activity(track: Track, win=0.9):
    """Mean absolute change of mouth opening over a window: high while talking, near zero for a closed or held-open mouth."""
    t = np.array(track.t); m = track.series("mouth")
    dm = np.abs(np.diff(m, prepend=m[:1]))
    out = np.zeros(len(t))
    for i in range(len(t)):
        lo = np.searchsorted(t, t[i] - win)
        out[i] = dm[lo:i + 1].mean() if i + 1 > lo else 0
    return t, out


@dataclass
class Focus:
    t: float          # seconds from range start, when the focus moves here
    track: int        # Track.id
    reason: str       # 'start' | 'speaker' | 'expression' | 'largest'
    strength: float = 0.0


def _interp(t, v, q):
    return float(np.interp(q, t, v, left=0.0, right=0.0))


def focus_timeline(an: Analysis, expressions=True, speaker=True, expr_thr=0.28, hold=1.3, dwell=0.9, speech_gate=True):
    """Return per-shot lists of Focus events: shot index -> list[Focus] (times in seconds, range-relative).
    Every shot starts on its best face; inside a shot the focus follows the speaker and jumps to strong reactions."""
    fps = an.fps
    out: dict[int, list[Focus]] = {}
    mouth = {tr.id: mouth_activity(tr) for tr in an.tracks}
    expr = {tr.id: expression_change(tr) for tr in an.tracks}
    for si, (f0, f1) in enumerate(an.shots):
        t0, t1 = f0 / fps, f1 / fps
        trs = [tr for tr in an.tracks if tr.shot == si]
        if trs:                                    # posters on the wall and far-away bystanders are not candidates
            big = max(tr.size() for tr in trs); trs = [tr for tr in trs if tr.size() >= 0.45 * big]
        if not trs: continue
        grid = np.arange(t0, t1, 1.0 / (fps / an.sample_every))
        if len(trs) == 1:
            out[si] = [Focus(t0, trs[0].id, "start")]; continue
        sz = {tr.id: tr.size() for tr in trs}
        cur = max(trs, key=lambda q: (len(q.t) * q.size())).id
        events = [Focus(t0, cur, "start")]; last_switch = t0; lock_until = t0
        for t in grid:
            alive = [tr for tr in trs if tr.t[0] - 0.4 <= t <= tr.t[-1] + 0.4]
            if len(alive) < 2: continue
            sp = {tr.id: _interp(*mouth[tr.id], t) for tr in alive}
            ex = {tr.id: _interp(*expr[tr.id], t) for tr in alive}
            talking = True
            if speech_gate and an.speech is not None and len(an.speech):
                a = int(t / an.speech_hop); talking = bool(an.speech[max(0, a - 15):a + 15].any())
            if cur not in sp: cur = max(sp, key=sp.get); events.append(Focus(t, cur, "largest")); last_switch = t; continue
            if expressions:
                cand = max((k for k in ex if k != cur), key=lambda k: ex[k], default=None)
                if cand is not None and ex[cand] > expr_thr and ex[cand] > 1.25 * ex[cur] + 0.05 and t - last_switch > 0.5:
                    cur = cand; events.append(Focus(t, cur, "expression", ex[cand])); last_switch = t; lock_until = t + hold; continue
            if speaker and talking and t >= lock_until and t - last_switch > dwell:
                best = max(sp, key=sp.get)
                if best != cur and sp[best] > 1.5 * sp[cur] + 0.012:
                    cur = best; events.append(Focus(t, cur, "speaker", sp[best])); last_switch = t
        out[si] = events
    return out
