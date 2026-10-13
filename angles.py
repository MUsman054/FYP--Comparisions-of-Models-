"""
angles.py — RULA-relevant joint angles from H36M-17 joints.

One code path for 2D and 3D:
    2D input (17, 2) is padded with z = 0, then goes through the SAME math as 3D.
    So the only difference between the 2D and 3D arms is depth — nothing else.

Coordinate convention (image, VideoPose3D, MotionBERT, MediaPipe world):
    x right, y DOWN, z away from camera.  Up = (0, -1, 0).

v1 scope:
    - Unsigned angles (flexion vs extension not separated yet).
    - No wrist angle: H36M-17 has no hand joints.
    - Calibration fixes camera tilt for the trunk reference only.
"""

import numpy as np

# ---------------------------------------------------------------------------
# H36M-17 joint order (same in VideoPose3D and MotionBERT)
# ---------------------------------------------------------------------------
PELVIS, R_HIP, R_KNEE, R_ANKLE = 0, 1, 2, 3
L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16

UP = np.array([0.0, -1.0, 0.0])

ANGLE_NAMES = ["trunk", "neck", "l_upper_arm", "r_upper_arm", "l_elbow", "r_elbow"]

# Joints each angle depends on (used for confidence gating)
ANGLE_JOINTS = {
    "trunk":       [PELVIS, THORAX],
    "neck":        [PELVIS, THORAX, HEAD],
    "l_upper_arm": [PELVIS, THORAX, L_SHOULDER, L_ELBOW],
    "r_upper_arm": [PELVIS, THORAX, R_SHOULDER, R_ELBOW],
    "l_elbow":     [L_SHOULDER, L_ELBOW, L_WRIST],
    "r_elbow":     [R_SHOULDER, R_ELBOW, R_WRIST],
}


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def to_3d(joints):
    """(17,2) or (17,3) -> (17,3) float64. 2D gets z = 0."""
    j = np.asarray(joints, dtype=np.float64)
    if j.shape not in [(17, 2), (17, 3)]:
        raise ValueError(f"expected (17,2) or (17,3), got {j.shape}")
    if j.shape[1] == 2:
        j = np.hstack([j, np.zeros((17, 1))])
    return j


def angle_between(a, b, eps=1e-8):
    """Unsigned angle in degrees between vectors a and b. NaN if either is ~zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def compute_angles(joints, conf=None, conf_thr=0.3, up=UP):
    """
    joints : (17,2) or (17,3) in H36M-17 order
    conf   : optional (17,) confidence; angles using a low-confidence joint -> NaN
    up     : reference 'vertical' for trunk (replace via Calibrator)
    returns: dict angle_name -> degrees
    """
    j = to_3d(joints)

    trunk_up = j[THORAX] - j[PELVIS]          # pelvis -> thorax
    trunk_down = -trunk_up                    # thorax -> pelvis
    head_vec = j[HEAD] - j[THORAX]            # thorax -> head

    out = {
        # Trunk inclination from vertical
        "trunk": angle_between(trunk_up, up),
        # Neck relative to trunk line
        "neck": angle_between(head_vec, trunk_up),
        # Upper arm relative to trunk (RULA measures arm vs trunk)
        "l_upper_arm": angle_between(j[L_ELBOW] - j[L_SHOULDER], trunk_down),
        "r_upper_arm": angle_between(j[R_ELBOW] - j[R_SHOULDER], trunk_down),
        # Elbow flexion: 0 = straight arm
        "l_elbow": angle_between(j[L_ELBOW] - j[L_SHOULDER], j[L_WRIST] - j[L_ELBOW]),
        "r_elbow": angle_between(j[R_ELBOW] - j[R_SHOULDER], j[R_WRIST] - j[R_ELBOW]),
    }

    if conf is not None:
        conf = np.asarray(conf)
        for name, idx in ANGLE_JOINTS.items():
            if np.any(conf[idx] < conf_thr):
                out[name] = np.nan

    return out


# ---------------------------------------------------------------------------
# Calibration: laptop webcams are tilted, so camera 'up' != true vertical.
# Sit upright for a few seconds -> mean trunk direction becomes the reference.
# Use ONE Calibrator per arm (2D and each lifter have their own frame).
# ---------------------------------------------------------------------------
class Calibrator:
    def __init__(self):
        self.samples = []
        self.up = UP.copy()

    def add(self, joints):
        j = to_3d(joints)
        v = j[THORAX] - j[PELVIS]
        n = np.linalg.norm(v)
        if n > 1e-8:
            self.samples.append(v / n)

    def finish(self, min_samples=10):
        if len(self.samples) < min_samples:
            return False
        m = np.mean(self.samples, axis=0)
        self.up = m / np.linalg.norm(m)
        self.samples = []
        return True

    def reset(self):
        self.samples = []
        self.up = UP.copy()


# ---------------------------------------------------------------------------
# RULA segment scores (posture-level comparison, not the full RULA table yet)
# ---------------------------------------------------------------------------
def _band(x, edges):
    """Score 1..len(edges)+1 by thresholds. NaN stays NaN."""
    if np.isnan(x):
        return np.nan
    return 1 + int(np.searchsorted(edges, x, side="right"))


def rula_segment_scores(a):
    """
    a: dict from compute_angles.
    Thresholds from RULA (McAtamney & Corlett 1993). Unsigned v1:
      upper arm 20/45/90, lower arm 60–100 = 1, neck 10/20, trunk 0/20/60.
    Trunk 'upright' uses a 5° tolerance (our choice, state it in the report).
    """
    def lower_arm(x):
        if np.isnan(x):
            return np.nan
        return 1 if 60 <= x <= 100 else 2

    return {
        "trunk": _band(a["trunk"], [5, 20, 60]),
        "neck": _band(a["neck"], [10, 20]),
        "l_upper_arm": _band(a["l_upper_arm"], [20, 45, 90]),
        "r_upper_arm": _band(a["r_upper_arm"], [20, 45, 90]),
        "l_lower_arm": lower_arm(a["l_elbow"]),
        "r_lower_arm": lower_arm(a["r_elbow"]),
    }


# ---------------------------------------------------------------------------
# Self-test: shows WHY 2D fails, with synthetic joints (no model needed)
#   python angles.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def fmt(x):
        return "undefined" if np.isnan(x) else f"{x:5.1f}°"

    def base_skeleton():
        j = np.zeros((17, 3))
        j[PELVIS] = [0, 0, 0]
        j[THORAX] = [0, -0.5, 0]
        j[HEAD] = [0, -0.75, 0]
        j[L_SHOULDER] = [-0.2, -0.5, 0]
        j[R_SHOULDER] = [0.2, -0.5, 0]
        j[L_ELBOW] = [-0.2, -0.2, 0]   # arms hanging
        j[R_ELBOW] = [0.2, -0.2, 0]
        j[L_WRIST] = [-0.2, 0.05, 0]
        j[R_WRIST] = [0.2, 0.05, 0]
        return j

    print("Test A: trunk leans TOWARD a frontal camera")
    print(f"{'true':>6} | {'3D':>9} | {'2D (z dropped)':>14}")
    for deg in [0, 15, 30, 45, 60]:
        t = np.radians(deg)
        j = base_skeleton()
        j[THORAX] = [0, -0.5 * np.cos(t), -0.5 * np.sin(t)]  # forward = -z
        a3, a2 = compute_angles(j), compute_angles(j[:, :2])
        print(f"{deg:5d}° | {fmt(a3['trunk']):>9} | {fmt(a2['trunk']):>14}")

    print("\nTest B: right upper arm at 90°, two directions")
    for label, d in [("sideways (abduction)", [0.3, 0, 0]),
                     ("forward  (flexion)  ", [0, 0, -0.3])]:
        j = base_skeleton()
        j[R_ELBOW] = j[R_SHOULDER] + d
        a3, a2 = compute_angles(j), compute_angles(j[:, :2])
        print(f"{label} | 3D {fmt(a3['r_upper_arm'])} | 2D {fmt(a2['r_upper_arm'])}")
