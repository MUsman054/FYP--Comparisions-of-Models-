"""
logger.py — per-frame CSV logging of angles for the 2D-vs-3D comparison.

One CSV per session in ./logs/. One row per frame per method:
    method = "2D"            -> angles from the extractor's 2D joints
    method = "<lifter name>" -> angles from that lifter's 3D joints

Logged RAW (no confidence gating), as decided: gate later in analysis
using the conf_* columns, so the 2D rows aren't mostly empty.

Columns
    time, frame, extractor, method, label, calibrated, latency_ms
    trunk, neck, l_upper_arm, r_upper_arm, l_elbow, r_elbow   (degrees)
    trunk_raw        trunk vs camera vertical (before calibration)
    conf_<angle>     lowest 2D confidence among the joints that angle uses
    rula_<segment>   RULA segment scores from the logged angles
    joints           the joints themselves (JSON), to recompute angles later

`label` is the test-protocol tag set in the app, e.g. "upright",
"arm_forward_90", "arm_side_90", "lean_forward". Analysis groups by it.
"""

import csv
import json
import os
import threading
import time
from datetime import datetime

import numpy as np

from angles import ANGLE_NAMES, ANGLE_JOINTS, rula_segment_scores

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

RULA_KEYS = ["trunk", "neck", "l_upper_arm", "r_upper_arm", "l_lower_arm", "r_lower_arm"]

COLUMNS = (["time", "frame", "extractor", "method", "label", "calibrated", "latency_ms"]
           + ANGLE_NAMES
           + ["trunk_raw"]
           + [f"conf_{n}" for n in ANGLE_NAMES]
           + [f"rula_{k}" for k in RULA_KEYS]
           + ["joints"])


def angle_confidences(scores):
    """Per-angle confidence = lowest score among the joints that angle uses."""
    if scores is None:
        return {n: np.nan for n in ANGLE_NAMES}
    s = np.asarray(scores, dtype=np.float64)
    return {n: float(s[idx].min()) for n, idx in ANGLE_JOINTS.items()}


def _num(x, nd=2):
    """Round for CSV; NaN/None -> empty cell (pandas reads it back as NaN)."""
    if x is None:
        return ""
    x = float(x)
    if np.isnan(x):
        return ""
    return int(round(x)) if nd == 0 else round(x, nd)


class AngleLogger:
    def __init__(self, session_name=None, log_dir=LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{name}.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._lock = threading.Lock()      # app may log from a background thread
        self.rows = 0

    def log(self, *, frame, extractor, method, angles, scores=None, joints=None,
            label="", calibrated=False, trunk_raw=None, latency_ms=None):
        conf = angle_confidences(scores)
        rula = rula_segment_scores(angles)
        row = {
            "time": round(time.time(), 3),
            "frame": frame,
            "extractor": extractor,
            "method": method,
            "label": label,
            "calibrated": int(bool(calibrated)),
            "latency_ms": _num(latency_ms, 1),
            "trunk_raw": _num(trunk_raw),
            "joints": "" if joints is None else
                      json.dumps(np.round(np.asarray(joints, dtype=np.float64), 4).tolist()),
        }
        row.update({n: _num(angles.get(n)) for n in ANGLE_NAMES})
        row.update({f"conf_{n}": _num(conf[n], 3) for n in ANGLE_NAMES})
        row.update({f"rula_{k}": _num(rula[k], 0) for k in RULA_KEYS})

        with self._lock:
            self._writer.writerow(row)
            self.rows += 1
            if self.rows % 30 == 0:        # flush about once a second at 30 fps
                self._file.flush()

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Self-test:  python logger.py   (writes logs/selftest.csv, reads it back)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from angles import compute_angles, PELVIS, THORAX, HEAD, R_SHOULDER, R_ELBOW, R_WRIST

    j = np.zeros((17, 3))
    j[THORAX], j[HEAD] = [0, -0.5, 0], [0, -0.75, 0]
    j[R_SHOULDER], j[R_ELBOW], j[R_WRIST] = [0.2, -0.5, 0], [0.2, -0.2, 0], [0.2, 0.05, 0]
    scores = np.full(17, 0.9)
    scores[PELVIS] = 0.1                    # hidden by the desk

    with AngleLogger("selftest") as log:
        for f in range(3):
            a2 = compute_angles(j[:, :2])
            log.log(frame=f, extractor="RTMPose", method="2D", angles=a2,
                    scores=scores, joints=j[:, :2], label="upright", latency_ms=12.3)
            a3 = compute_angles(j)
            log.log(frame=f, extractor="RTMPose", method="VideoPose3D", angles=a3,
                    scores=scores, joints=j, label="upright", calibrated=True,
                    trunk_raw=a3["trunk"], latency_ms=110.0)
        path, n = log.path, log.rows

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"wrote {n} rows -> {path}")
    print(f"columns ({len(COLUMNS)}):", ", ".join(c for c in COLUMNS if c != "joints"), "+ joints")
    r = rows[1]
    print(f"\nrow 2: method={r['method']} label={r['label']} trunk={r['trunk']} "
          f"r_elbow={r['r_elbow']} conf_trunk={r['conf_trunk']} rula_trunk={r['rula_trunk']}")
    print("joints round-trip:", np.allclose(np.array(json.loads(r["joints"])), j))
