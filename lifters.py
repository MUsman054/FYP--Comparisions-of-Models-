"""
lifters.py — pretrained 2D -> 3D lifters, used off the shelf (no training).

    VideoPose3D  : TemporalModel, 243-frame receptive field. Checkpoint
                   pretrained_h36m_detectron_coco.bin = the repo's in-the-wild model,
                   trained on COCO-format 2D input, outputs H36M-17.
    MotionBERT   : DSTformer-lite fine-tuned on H36M (FT_MB_lite_MB_ft_h36m_global_lite),
                   the checkpoint MotionBERT's own infer_wild.py uses. Input H36M-17 (x, y, conf).
    MediaPipe 3D : MediaPipe world landmarks. No extra model; only valid with the MediaPipe extractor.

Every lifter returns (17, 3): H36M order, root-relative (pelvis = origin), y down —
exactly what angles.py expects.

Live use: a lifter takes the recent history of Poses (oldest -> newest)
and returns the 3D pose of the NEWEST frame.
  - VideoPose3D is non-causal (needs 121 future frames). Live, the future is filled with
    copies of the newest frame — the same edge padding VideoPose3D uses at the end of
    every video. Some accuracy loss vs offline; state it in the report.
  - MotionBERT predicts every frame of its window; we take the last one.
  - Both use test-time flip augmentation, as in their official evaluation.

Setup (once, from the project folder):
    git clone https://github.com/facebookresearch/VideoPose3D third_party/VideoPose3D
    git clone https://github.com/Walter0807/MotionBERT third_party/MotionBERT
    python -m pip install torch
Weights auto-download to ./checkpoints/ on first use.
"""

import os
import sys
import time
import urllib.request
from functools import partial

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(ROOT, "third_party")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

VP3D_URL = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
MB_URL = ("https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
          "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")

REPO_URLS = {
    "VideoPose3D": "https://github.com/facebookresearch/VideoPose3D",
    "MotionBERT": "https://github.com/Walter0807/MotionBERT",
}

# Left/right joints, for flip augmentation
COCO_LEFT, COCO_RIGHT = [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]
H36M_LEFT, H36M_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _add_repo(name):
    path = os.path.join(THIRD_PARTY, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{path} not found. Run:  git clone {REPO_URLS[name]} third_party/{name}")
    if path not in sys.path:
        sys.path.insert(0, path)


def _download(url, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        tmp = path + ".part"   # no corrupt file if the download breaks
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _load_state(path):
    # Official checkpoints also store optimizer/RNG state, so weights_only=False.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_pos"]
    # Saved from nn.DataParallel -> strip 'module.' so it loads on CPU
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _flip(x, left, right):
    """Mirror horizontally: negate x, swap left/right joints. x: (..., J, C)."""
    y = x.clone()
    y[..., 0] *= -1
    y[..., left + right, :] = y[..., right + left, :]
    return y


def _root_relative(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x[0:1]


# ---------------------------------------------------------------------------
# VideoPose3D
# ---------------------------------------------------------------------------
class VideoPose3DLifter:
    name = "VideoPose3D"

    def __init__(self, device="cpu", flip=True):
        _add_repo("VideoPose3D")
        from common.model import TemporalModel

        self.model = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3],
                                   causal=False, dropout=0.25, channels=1024)
        weights = _download(VP3D_URL, "pretrained_h36m_detectron_coco.bin")
        self.model.load_state_dict(_load_state(weights))
        self.model.to(device).eval()

        self.device, self.flip = device, flip
        self.pad = (self.model.receptive_field() - 1) // 2   # 121
        self.window = self.pad + 1                            # past frames used, incl. current

    @torch.no_grad()
    def __call__(self, history):
        poses = [p for p in history if p.coco2d is not None][-self.window:]
        if not poses:
            return None

        w, h = poses[-1].img_wh
        seq = np.stack([p.coco2d for p in poses])           # (T, 17, 2) pixels
        seq = seq / w * 2 - [1, h / w]                      # = VideoPose3D normalize_screen_coordinates

        # Past: edge-pad to `pad` frames. Future: `pad` copies of the newest frame.
        n_past = self.window - len(seq)
        seq = np.concatenate([np.repeat(seq[:1], n_past, 0), seq,
                              np.repeat(seq[-1:], self.pad, 0)])   # (243, 17, 2)

        x = torch.from_numpy(seq).float()[None]
        if self.flip:
            x = torch.cat([x, _flip(x, COCO_LEFT, COCO_RIGHT)])
        out = self.model(x.to(self.device)).cpu()           # (B, 1, 17, 3)
        if self.flip:
            out = (out[0:1] + _flip(out[1:2], H36M_LEFT, H36M_RIGHT)) / 2
        return _root_relative(out[0, 0].numpy())

    def close(self):
        pass


# ---------------------------------------------------------------------------
# MotionBERT (lite)
# ---------------------------------------------------------------------------
class MotionBERTLifter:
    name = "MotionBERT"

    def __init__(self, device="cpu", window=243, flip=True):
        _add_repo("MotionBERT")
        from lib.model.DSTformer import DSTformer
        from lib.utils.utils_data import crop_scale

        # Values from configs/pose3d/MB_ft_h36m_global_lite.yaml
        self.model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5,
                               num_heads=8, mlp_ratio=4,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6),
                               maxlen=243, num_joints=17)
        weights = _download(MB_URL, "motionbert_lite_h36m.bin")
        self.model.load_state_dict(_load_state(weights), strict=True)
        self.model.to(device).eval()

        self.crop_scale = crop_scale
        self.device, self.flip = device, flip
        self.window = int(min(window, 243))   # shorter = faster on CPU, less context

    @torch.no_grad()
    def __call__(self, history):
        poses = list(history)[-self.window:]
        if not poses:
            return None

        xy = np.stack([p.kpts2d for p in poses])                     # (T, 17, 2)
        # RTMPose scores can exceed 1; MotionBERT saw [0, 1] confidences
        conf = np.clip(np.stack([p.scores for p in poses]), 0, 1)[..., None]
        seq = self.crop_scale(np.concatenate([xy, conf], -1), [1, 1])  # MotionBERT's own normalization

        x = torch.from_numpy(seq).float()[None]                       # (1, T, 17, 3)
        if self.flip:
            x = torch.cat([x, _flip(x, H36M_LEFT, H36M_RIGHT)])
        out = self.model(x.to(self.device)).cpu()                     # (B, T, 17, 3)
        if self.flip:
            out = (out[0:1] + _flip(out[1:2], H36M_LEFT, H36M_RIGHT)) / 2
        return _root_relative(out[0, -1].numpy())

    def close(self):
        pass


# ---------------------------------------------------------------------------
# MediaPipe world landmarks (pass-through)
# ---------------------------------------------------------------------------
class MediaPipe3DLifter:
    name = "MediaPipe 3D"

    def __init__(self, **kwargs):
        self.window = 1

    def __call__(self, history):
        if not history:
            return None
        p = history[-1]
        return None if p.world3d is None else _root_relative(p.world3d)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Registry (the app's menu reads this)
# ---------------------------------------------------------------------------
LIFTERS = {
    "VideoPose3D": VideoPose3DLifter,
    "MotionBERT": MotionBERTLifter,
    "MediaPipe 3D": MediaPipe3DLifter,
}

# Lifters that only work with specific extractors (others work with any)
ONLY_WITH = {"MediaPipe 3D": ["MediaPipe"]}

HISTORY_LEN = 243   # frames of 2D poses the app should keep


def load_lifter(name, **kwargs):
    if name not in LIFTERS:
        raise ValueError(f"unknown lifter '{name}', choose from {list(LIFTERS)}")
    return LIFTERS[name](**kwargs)


def is_compatible(lifter_name, extractor_name):
    return extractor_name in ONLY_WITH.get(lifter_name, [extractor_name])


# ---------------------------------------------------------------------------
# Self-test (no webcam): a synthetic person leans toward a frontal camera.
# The 3D pose is projected to 2D pixels, fed to each lifter, and the trunk
# angle is compared: true lean vs 2D-only vs each lifter.
#   python lifters.py
# ---------------------------------------------------------------------------
def _synthetic_pose(lean_deg, img_wh=(640, 480), f=600.0, dist=3.5):
    from extractors import Pose
    from mapping import (coco_to_h36m, C_NOSE, C_L_EYE, C_R_EYE, C_L_EAR, C_R_EAR,
                         C_L_SHOULDER, C_R_SHOULDER, C_L_ELBOW, C_R_ELBOW,
                         C_L_WRIST, C_R_WRIST, C_L_HIP, C_R_HIP,
                         C_L_KNEE, C_R_KNEE, C_L_ANKLE, C_R_ANKLE)

    # Metres, origin at pelvis, y down, camera looks along +z.
    # Facing the camera: subject's LEFT is at +x, nose points to -z.
    j = np.zeros((17, 3))
    j[C_L_HIP], j[C_R_HIP] = [0.12, 0, 0], [-0.12, 0, 0]
    j[C_L_KNEE], j[C_R_KNEE] = [0.12, 0.45, 0], [-0.12, 0.45, 0]
    j[C_L_ANKLE], j[C_R_ANKLE] = [0.12, 0.9, 0], [-0.12, 0.9, 0]
    j[C_L_SHOULDER], j[C_R_SHOULDER] = [0.2, -0.5, 0], [-0.2, -0.5, 0]
    j[C_L_ELBOW], j[C_R_ELBOW] = [0.22, -0.22, 0], [-0.22, -0.22, 0]
    j[C_L_WRIST], j[C_R_WRIST] = [0.22, 0.03, 0], [-0.22, 0.03, 0]
    j[C_NOSE] = [0, -0.68, -0.08]
    j[C_L_EYE], j[C_R_EYE] = [0.03, -0.71, -0.07], [-0.03, -0.71, -0.07]
    j[C_L_EAR], j[C_R_EAR] = [0.07, -0.69, 0], [-0.07, -0.69, 0]

    # Lean the upper body toward the camera (rotate about the x-axis at the pelvis)
    t = np.radians(lean_deg)
    upper = [C_NOSE, C_L_EYE, C_R_EYE, C_L_EAR, C_R_EAR, C_L_SHOULDER, C_R_SHOULDER,
             C_L_ELBOW, C_R_ELBOW, C_L_WRIST, C_R_WRIST]
    y, z = j[upper, 1].copy(), j[upper, 2].copy()
    j[upper, 1] = y * np.cos(t) - z * np.sin(t)
    j[upper, 2] = y * np.sin(t) + z * np.cos(t)

    # Pinhole projection, camera level with the chest
    cam = j + [0, 0.3, dist]
    uv = f * cam[:, :2] / cam[:, 2:3] + [img_wh[0] / 2, img_wh[1] / 2]
    s = np.full(17, 0.9)
    h36m, hs = coco_to_h36m(uv, s)
    return Pose(h36m, hs, img_wh, coco2d=uv, coco_scores=s)


if __name__ == "__main__":
    from angles import compute_angles, Calibrator

    lifters = {}
    for name in ["VideoPose3D", "MotionBERT"]:
        try:
            lifters[name] = load_lifter(name)
        except Exception as e:
            print(f"[skip] {name}: {e}")

    leans = [0, 15, 30, 45, 60]
    timings = {n: [] for n in lifters}
    outputs = {n: {} for n in lifters}
    for lean in leans:
        history = [_synthetic_pose(lean)] * 243      # person holds the pose
        for name, lifter in lifters.items():
            t0 = time.perf_counter()
            outputs[name][lean] = lifter(history)
            timings[name].append(time.perf_counter() - t0)

    # Calibration: the 0° pose is "sitting upright" -> its trunk direction becomes vertical.
    # Removes each lifter's built-in tilt (H36M cameras look down at the subject).
    cal = {}
    for name in lifters:
        c = Calibrator()
        for _ in range(10):
            c.add(outputs[name][0])
        c.finish()
        cal[name] = c.up

    for title, use_cal in [("RAW (vs camera vertical)", False),
                           ("CALIBRATED (vs upright pose)", True)]:
        header = f"{'true lean':>9} | {'2D only':>8}" + "".join(f" | {n:>11}" for n in lifters)
        print(f"\nTrunk angle — {title}")
        print(header)
        print("-" * len(header))
        for lean in leans:
            row = f"{lean:8d}° | {compute_angles(_synthetic_pose(lean).kpts2d)['trunk']:7.1f}°"
            for name in lifters:
                up = cal[name] if use_cal else None
                a = compute_angles(outputs[name][lean], **({"up": up} if use_cal else {}))
                row += f" | {a['trunk']:10.1f}°"
            print(row)

    print()
    for name, ts in timings.items():
        print(f"{name}: {1000 * np.mean(ts):.0f} ms per frame on CPU")
