"""
rula.py — full RULA score (1–7), action level (1–4) and plain-language text.

Input : the angle dict from angles.compute_angles()  (same for 2D, 3D and GT)
Output: one RULA result per model arm, for both the image and webcam modes.

RULA tables A, B, C from McAtamney & Corlett (1993).
Segment scores come from angles.rula_segment_scores(), so thresholds live in one place.

What a webcam cannot measure is fixed in DEFAULTS and applied identically to every
arm and to the ground truth. State these assumptions in the report.
"""

from collections import deque

import numpy as np

from angles import ANGLE_NAMES, rula_segment_scores

# ---------------------------------------------------------------------------
# Inputs a body-pose model cannot see (seated desk work)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "wrist": 1,        # wrist neutral (no hand keypoints)
    "wrist_twist": 1,  # mid-range twist
    "legs": 1,         # seated, feet supported
    "muscle_use": 1,   # posture mainly static (held > 1 min), as in desk work
    "load": 0,         # no load or force
}

# ---------------------------------------------------------------------------
# RULA tables
# ---------------------------------------------------------------------------
# Table A [upper_arm 1-6][lower_arm 1-3] -> 8 columns:
#   (wrist1,twist1) (wrist1,twist2) (wrist2,t1) (wrist2,t2) ... (wrist4,t2)
TABLE_A = [
    [[1, 2, 2, 2, 2, 3, 3, 3], [2, 2, 2, 2, 3, 3, 3, 3], [2, 3, 3, 3, 3, 3, 4, 4]],
    [[2, 3, 3, 3, 3, 4, 4, 4], [3, 3, 3, 3, 3, 4, 4, 4], [3, 4, 4, 4, 4, 4, 5, 5]],
    [[3, 3, 4, 4, 4, 4, 5, 5], [3, 4, 4, 4, 4, 4, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5]],
    [[4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 5, 5, 5, 6, 6]],
    [[5, 5, 5, 5, 5, 6, 6, 7], [5, 6, 6, 6, 6, 7, 7, 7], [6, 6, 6, 7, 7, 7, 7, 8]],
    [[7, 7, 7, 7, 7, 8, 8, 9], [8, 8, 8, 8, 8, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9]],
]

# Table B [neck 1-6] -> 12 columns: (trunk1,legs1) (trunk1,legs2) ... (trunk6,legs2)
TABLE_B = [
    [1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
    [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
    [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
    [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
    [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9],
]

# Table C [score C 1-8+] x [score D 1-7+]
TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5],
    [2, 2, 3, 4, 4, 5, 5],
    [3, 3, 3, 4, 4, 5, 6],
    [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7],
    [4, 4, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7],
]

LEVELS = {
    1: ("Acceptable", "posture is fine"),
    2: ("Investigate", "posture may need changes"),
    3: ("Change soon", "posture should be corrected soon"),
    4: ("Change now", "correct posture immediately"),
}

# Plain-language message per segment score (only scores above 1 get a message)
MESSAGES = {
    "trunk": {2: "Leaning slightly", 3: "Leaning forward", 4: "Leaning far forward"},
    "neck": {2: "Neck slightly bent", 3: "Neck bent forward too much"},
    "upper_arm": {2: "arm slightly raised", 3: "arm raised high", 4: "arm above shoulder level"},
    "lower_arm": {2: "elbow angle outside the comfortable range"},
}


def _table_a(ua, la, wrist, twist):
    return TABLE_A[ua - 1][la - 1][(wrist - 1) * 2 + (twist - 1)]


def _table_b(neck, trunk, legs):
    return TABLE_B[neck - 1][(trunk - 1) * 2 + (legs - 1)]


def _table_c(c, d):
    return TABLE_C[min(c, 8) - 1][min(d, 7) - 1]


def action_level(grand):
    if grand <= 2:
        return 1
    if grand <= 4:
        return 2
    if grand <= 6:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def rula_from_angles(angles, defaults=DEFAULTS):
    """
    angles: dict from compute_angles (NaN = joint not reliable).
    Returns dict:
        grand (1-7), level (1-4), label, text (main view),
        messages (list), segments (dict), missing (list), complete (bool)
    Missing segments are scored as neutral (1) and listed in 'missing'; the result
    is then marked incomplete instead of silently looking fine.
    """
    seg = rula_segment_scores(angles)
    missing = [k for k, v in seg.items() if np.isnan(v)]

    def s(k):
        return 1 if np.isnan(seg[k]) else int(seg[k])

    d = defaults
    # Group A per side; RULA scores the worse side
    side_a = {}
    for side in ("l", "r"):
        ua, la = s(f"{side}_upper_arm"), s(f"{side}_lower_arm")
        side_a[side] = _table_a(ua, la, d["wrist"], d["wrist_twist"])
    worse = max(side_a, key=side_a.get)
    score_a = side_a[worse]

    score_b = _table_b(s("neck"), s("trunk"), d["legs"])
    score_c = score_a + d["muscle_use"] + d["load"]
    score_d = score_b + d["muscle_use"] + d["load"]
    grand = _table_c(score_c, score_d)
    level = action_level(grand)

    # Messages: every segment that pushes the score up, both arms
    msgs = []
    if s("trunk") > 1:
        msgs.append(MESSAGES["trunk"][s("trunk")])
    if s("neck") > 1:
        msgs.append(MESSAGES["neck"][min(s("neck"), 3)])
    for side, name in (("l", "Left"), ("r", "Right")):
        ua, la = s(f"{side}_upper_arm"), s(f"{side}_lower_arm")
        if ua > 1:
            msgs.append(f"{name} {MESSAGES['upper_arm'][ua]}")
        if la > 1:
            msgs.append(f"{name} {MESSAGES['lower_arm'][la]}")

    label, meaning = LEVELS[level]
    text = f"{label}: " + (", ".join(m[0].lower() + m[1:] for m in msgs) if msgs else meaning)
    if missing:
        text += f"  (incomplete, not visible: {', '.join(missing)})"

    return {
        "grand": grand,
        "level": level,
        "label": label,
        "text": text,
        "messages": msgs,
        "segments": {k: (None if np.isnan(v) else int(v)) for k, v in seg.items()},
        "score_a": score_a,
        "score_b": score_b,
        "missing": missing,
        "complete": not missing,
    }


# ---------------------------------------------------------------------------
# Live webcam: smooth ANGLES over ~1 s, then score, so the text does not flicker
# ---------------------------------------------------------------------------
class AngleSmoother:
    """Median of each angle over the last `window` frames (NaNs ignored)."""

    def __init__(self, window=15):
        self.buf = {k: deque(maxlen=window) for k in ANGLE_NAMES}

    def update(self, angles):
        out = {}
        for k in ANGLE_NAMES:
            self.buf[k].append(angles.get(k, np.nan))
            vals = np.array(self.buf[k], dtype=float)
            vals = vals[~np.isnan(vals)]
            out[k] = float(np.median(vals)) if len(vals) else np.nan
        return out

    def reset(self):
        for b in self.buf.values():
            b.clear()


# ---------------------------------------------------------------------------
# Self-test:  python rula.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def show(title, a):
        r = rula_from_angles(a)
        print(f"{title:34s} grand {r['grand']}  level {r['level']}  | {r['text']}")
        return r

    good = {"trunk": 2, "neck": 5, "l_upper_arm": 10, "r_upper_arm": 10,
            "l_elbow": 80, "r_elbow": 80}
    print("Check 1: RULA tables")
    show("Upright, arms down, typing", good)
    show("Leaning 30°, neck 25°", {**good, "trunk": 30, "neck": 25})
    show("Right arm raised 100°", {**good, "r_upper_arm": 100})
    show("Lean 70°, neck 25°, arm 100°, elbow 20°",
         {**good, "trunk": 70, "neck": 25, "r_upper_arm": 100, "r_elbow": 20})

    print("\nCheck 2: missing joints are flagged, not hidden")
    show("Trunk not visible", {**good, "trunk": np.nan})

    print("\nCheck 3: the 2D-vs-3D cost in RULA terms (same pose, 45° lean)")
    show("GT / 3D sees 45° lean", {**good, "trunk": 45, "neck": 25})
    show("2D sees ~3° lean (foreshortened)", {**good, "trunk": 3, "neck": 4})

    print("\nCheck 4: smoother ignores a one-frame spike")
    sm = AngleSmoother(window=5)
    for t in [10, 11, 70, 12, 10]:
        out = sm.update({**good, "trunk": t})
    print(f"  trunk inputs 10, 11, 70, 12, 10 -> smoothed {out['trunk']:.0f}°")
