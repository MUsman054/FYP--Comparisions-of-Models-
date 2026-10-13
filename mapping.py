"""
mapping.py — convert 2D/3D keypoints to the H36M-17 format the lifters expect.

    RTMPose   : COCO-17       -> coco_to_h36m()
    MediaPipe : 33 landmarks  -> mediapipe_to_coco() -> coco_to_h36m()

Both extractors go through the SAME COCO -> H36M step, so the comparison is fair.
The COCO -> H36M rule is MotionBERT's own coco2h36m (lib/data/dataset_action.py).

Who uses what:
    VideoPose3D : COCO-17 directly (its in-the-wild checkpoint was trained on COCO input)
    MotionBERT  : H36M-17 from coco_to_h36m()
    angles.py   : H36M-17

Works on a single frame (17, C) or a sequence (T, 17, C), C = 2 or 3.
Left/right = the SUBJECT's left/right in all three formats.
"""

import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 order (RTMPose output)
# ---------------------------------------------------------------------------
C_NOSE, C_L_EYE, C_R_EYE, C_L_EAR, C_R_EAR = 0, 1, 2, 3, 4
C_L_SHOULDER, C_R_SHOULDER, C_L_ELBOW, C_R_ELBOW = 5, 6, 7, 8
C_L_WRIST, C_R_WRIST, C_L_HIP, C_R_HIP = 9, 10, 11, 12
C_L_KNEE, C_R_KNEE, C_L_ANKLE, C_R_ANKLE = 13, 14, 15, 16

# ---------------------------------------------------------------------------
# MediaPipe-33 indices for each COCO-17 joint, in COCO order
# ---------------------------------------------------------------------------
MP_TO_COCO = [
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
]


def mediapipe_to_coco(landmarks, visibility=None):
    """
    landmarks : (33, C) or (T, 33, C)  — 2D pixels or 3D world, C = 2 or 3
    visibility: (33,) or (T, 33)       — MediaPipe visibility, used as confidence
    returns   : (coco (…,17,C), scores (…,17) or None)
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[-2] != 33:
        raise ValueError(f"expected 33 landmarks, got shape {lm.shape}")
    coco = lm[..., MP_TO_COCO, :]
    scores = None
    if visibility is not None:
        scores = np.asarray(visibility, dtype=np.float64)[..., MP_TO_COCO]
    return coco, scores


def coco_to_h36m(kpts, scores=None):
    """
    kpts  : (17, C) or (T, 17, C) in COCO order
    scores: (17,) or (T, 17) confidences, optional
    returns: (h36m (…,17,C), h36m_scores (…,17) or None)

    Derived joints (no direct COCO equivalent):
        pelvis = mid-hips, thorax = mid-shoulders, spine = mid(pelvis, thorax)
        head   = mid-eyes (MotionBERT convention), neck/nose = nose
    A derived joint's confidence = MIN of its sources
    (e.g. pelvis is only as reliable as the less visible hip).
    """
    x = np.asarray(kpts, dtype=np.float64)
    if x.shape[-2] != 17:
        raise ValueError(f"expected 17 COCO joints, got shape {x.shape}")

    y = np.zeros_like(x)
    pelvis = 0.5 * (x[..., C_L_HIP, :] + x[..., C_R_HIP, :])
    thorax = 0.5 * (x[..., C_L_SHOULDER, :] + x[..., C_R_SHOULDER, :])

    y[..., 0, :] = pelvis
    y[..., 1, :] = x[..., C_R_HIP, :]
    y[..., 2, :] = x[..., C_R_KNEE, :]
    y[..., 3, :] = x[..., C_R_ANKLE, :]
    y[..., 4, :] = x[..., C_L_HIP, :]
    y[..., 5, :] = x[..., C_L_KNEE, :]
    y[..., 6, :] = x[..., C_L_ANKLE, :]
    y[..., 7, :] = 0.5 * (pelvis + thorax)                              # spine
    y[..., 8, :] = thorax
    y[..., 9, :] = x[..., C_NOSE, :]                                    # neck/nose
    y[..., 10, :] = 0.5 * (x[..., C_L_EYE, :] + x[..., C_R_EYE, :])     # head
    y[..., 11, :] = x[..., C_L_SHOULDER, :]
    y[..., 12, :] = x[..., C_L_ELBOW, :]
    y[..., 13, :] = x[..., C_L_WRIST, :]
    y[..., 14, :] = x[..., C_R_SHOULDER, :]
    y[..., 15, :] = x[..., C_R_ELBOW, :]
    y[..., 16, :] = x[..., C_R_WRIST, :]

    if scores is None:
        return y, None

    s = np.asarray(scores, dtype=np.float64)
    ys = np.zeros_like(s)
    s_pelvis = np.minimum(s[..., C_L_HIP], s[..., C_R_HIP])
    s_thorax = np.minimum(s[..., C_L_SHOULDER], s[..., C_R_SHOULDER])

    ys[..., 0] = s_pelvis
    ys[..., 1] = s[..., C_R_HIP]
    ys[..., 2] = s[..., C_R_KNEE]
    ys[..., 3] = s[..., C_R_ANKLE]
    ys[..., 4] = s[..., C_L_HIP]
    ys[..., 5] = s[..., C_L_KNEE]
    ys[..., 6] = s[..., C_L_ANKLE]
    ys[..., 7] = np.minimum(s_pelvis, s_thorax)
    ys[..., 8] = s_thorax
    ys[..., 9] = s[..., C_NOSE]
    ys[..., 10] = np.minimum(s[..., C_L_EYE], s[..., C_R_EYE])
    ys[..., 11] = s[..., C_L_SHOULDER]
    ys[..., 12] = s[..., C_L_ELBOW]
    ys[..., 13] = s[..., C_L_WRIST]
    ys[..., 14] = s[..., C_R_SHOULDER]
    ys[..., 15] = s[..., C_R_ELBOW]
    ys[..., 16] = s[..., C_R_WRIST]
    return y, ys


def mediapipe_to_h36m(landmarks, visibility=None):
    """Convenience: MediaPipe-33 -> COCO-17 -> H36M-17."""
    coco, sc = mediapipe_to_coco(landmarks, visibility)
    return coco_to_h36m(coco, sc)


# ---------------------------------------------------------------------------
# Self-test:  python mapping.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from angles import compute_angles, PELVIS, THORAX, SPINE

    # Upright seated person in image pixels (y down), facing the camera.
    # Subject's LEFT appears on image RIGHT (larger x).
    coco = np.zeros((17, 2))
    coco[C_NOSE] = [320, 120]
    coco[C_L_EYE], coco[C_R_EYE] = [330, 110], [310, 110]
    coco[C_L_EAR], coco[C_R_EAR] = [345, 115], [295, 115]
    coco[C_L_SHOULDER], coco[C_R_SHOULDER] = [380, 200], [260, 200]
    coco[C_L_ELBOW], coco[C_R_ELBOW] = [390, 300], [250, 300]
    coco[C_L_WRIST], coco[C_R_WRIST] = [390, 390], [250, 390]
    coco[C_L_HIP], coco[C_R_HIP] = [360, 400], [280, 400]
    coco[C_L_KNEE], coco[C_R_KNEE] = [360, 500], [280, 500]
    coco[C_L_ANKLE], coco[C_R_ANKLE] = [360, 600], [280, 600]
    scores = np.full(17, 0.9)
    scores[C_L_HIP] = 0.1  # left hip hidden by the desk

    h36m, hs = coco_to_h36m(coco, scores)

    print("Check 1: derived joints")
    print("  pelvis =", h36m[PELVIS], " (expect [320, 400])")
    print("  thorax =", h36m[THORAX], " (expect [320, 200])")
    print("  spine  =", h36m[SPINE], " (expect [320, 300])")

    print("\nCheck 2: confidence propagation (left hip hidden)")
    print(f"  pelvis score = {hs[PELVIS]:.1f}  (expect 0.1)")

    print("\nCheck 3: sequence input")
    seq, _ = coco_to_h36m(np.stack([coco] * 5))
    print("  (5,17,2) ->", seq.shape)

    print("\nCheck 4: MediaPipe path")
    mp = np.random.rand(33, 3)
    h_mp, _ = mediapipe_to_h36m(mp)
    ok = np.allclose(h_mp[PELVIS], 0.5 * (mp[23] + mp[24]))
    print("  (33,3) ->", h_mp.shape, "| pelvis = mid-hips:", ok)

    print("\nCheck 5: angles on mapped joints")
    for name, val in compute_angles(h36m, conf=hs).items():
        print(f"  {name:12s} {'NaN (gated)' if np.isnan(val) else f'{val:5.1f}°'}")
