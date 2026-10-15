"""
compare.py — run ONE image through all 7 model arms, score RULA, compare with GT.

Arms (same angle code for all, so depth is the only difference):
    2D only : RTMPose 2D, MediaPipe 2D
    3D      : RTMPose -> VideoPose3D, RTMPose -> MotionBERT,
              MediaPipe -> VideoPose3D, MediaPipe -> MotionBERT, MediaPipe 3D

Single-image mode for the video lifters: the one pose is repeated to fill the
lifter's window (state as "single-frame" in the report).

Used by:  app.py (Image tab) and dataset_eval.py (whole folder)
CLI:      python compare.py --image data/panoptic/images/frame_00000120.jpg
          (GT is picked up automatically from gt_angles.csv in the image's parent folder)
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")      # silence MediaPipe's C++ warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from angles import ANGLE_NAMES, HEAD, PELVIS, compute_angles
from extractors import EXTRACTORS, draw_pose
from lifters import load_lifter
from rula import rula_from_angles

# (arm name, extractor, lifter or None for 2D only)
ARMS = [
    ("RTMPose 2D", "RTMPose", None),
    ("MediaPipe 2D", "MediaPipe", None),
    ("RTMPose → VideoPose3D", "RTMPose", "VideoPose3D"),
    ("RTMPose → MotionBERT", "RTMPose", "MotionBERT"),
    ("MediaPipe → VideoPose3D", "MediaPipe", "VideoPose3D"),
    ("MediaPipe → MotionBERT", "MediaPipe", "MotionBERT"),
    ("MediaPipe 3D", "MediaPipe", "MediaPipe 3D"),
]
ARM_NAMES = [a[0] for a in ARMS]
COLORS = {"RTMPose": (0, 200, 0), "MediaPipe": (200, 0, 200)}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_gt(csv_path):
    """gt_angles.csv -> {image file name: {angle: degrees}}"""
    gt = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["image"]] = {a: float(row[a]) if row.get(a) not in (None, "") else np.nan
                                for a in ANGLE_NAMES}
    return gt


def find_gt_for(image_path):
    """Look for gt_angles.csv next to the image's folder (data/<set>/gt_angles.csv)."""
    p = Path(image_path)
    for folder in (p.parent, p.parent.parent):
        csv_path = folder / "gt_angles.csv"
        if csv_path.exists():
            return load_gt(csv_path).get(p.name)
    return None


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------
class Comparator:
    """
    Loads each extractor and lifter once, then compares any number of images.
    fresh_mediapipe=True rebuilds MediaPipe per image, so its video-mode tracking
    never carries over from one (unrelated) image to the next.
    """

    def __init__(self, device="cpu", motionbert_window=27, fresh_mediapipe=True):
        self.device = device
        self.mb_window = motionbert_window
        self.fresh_mediapipe = fresh_mediapipe
        self.extractors = {}
        self.lifters = {}

    def _extractor(self, name):
        if name == "MediaPipe" and self.fresh_mediapipe and name in self.extractors:
            self.extractors[name].close()
            del self.extractors[name]
        if name not in self.extractors:
            cls = EXTRACTORS[name]
            self.extractors[name] = cls(device=self.device) if name == "RTMPose" else cls()
        return self.extractors[name]

    def _lifter(self, name):
        if name not in self.lifters:
            kw = {"device": self.device}
            if name == "MotionBERT":
                kw["window"] = self.mb_window
            self.lifters[name] = load_lifter(name, **kw) if name != "MediaPipe 3D" else load_lifter(name)
        return self.lifters[name]

    def run(self, image_bgr, gt=None):
        """
        image_bgr : HxWx3 image (cv2.imread)
        gt        : optional {angle: degrees}
        returns   : dict with 'arms' (per arm results), 'gt', 'poses'
        """
        poses = {}
        for ex in ("RTMPose", "MediaPipe"):
            poses[ex] = self._extractor(ex)(image_bgr)

        gt_rula = rula_from_angles(gt) if gt else None
        arms = {}
        for arm, ex, lf in ARMS:
            pose = poses[ex]
            if pose is None:
                arms[arm] = None
                continue
            if lf is None:
                joints = pose.kpts2d
            else:
                lifter = self._lifter(lf)
                history = [pose] * max(1, getattr(lifter, "window", 1))  # single-frame mode
                joints = lifter(history)
                if joints is None:
                    arms[arm] = None
                    continue
                if lf == "MediaPipe 3D" and getattr(pose, "world_ears", None) is not None:
                    # Same head definition as GT: ear midpoint, root-relative like the lifter output
                    joints = joints.copy()
                    joints[HEAD] = pose.world_ears - pose.world3d[PELVIS]
            ang = compute_angles(joints, conf=pose.scores)
            r = rula_from_angles(ang)
            res = {"angles": ang, "rula": r, "is_3d": lf is not None}
            if gt:
                res["error"] = {a: abs(ang[a] - gt[a]) for a in ANGLE_NAMES}
                res["rula_match"] = r["level"] == gt_rula["level"]
            arms[arm] = res

        return {"arms": arms, "gt": gt, "gt_rula": gt_rula, "poses": poses}

    def close(self):
        for m in list(self.extractors.values()) + list(self.lifters.values()):
            m.close()


# ---------------------------------------------------------------------------
# Presentation helpers (used by the app and the CLI)
# ---------------------------------------------------------------------------
def _fmt(x, unit="°"):
    return "—" if x is None or np.isnan(x) else f"{x:.1f}{unit}"


def results_table(result, show_degrees=True):
    """
    Rows for a table: GT first (if any), then one row per arm.
    show_degrees=False gives the plain-language view only.
    """
    rows = []
    gt = result["gt"]
    if gt:
        r = result["gt_rula"]
        row = {"Model": "Ground truth", "RULA": r["text"]}
        if show_degrees:
            row.update({a: _fmt(gt[a]) for a in ANGLE_NAMES})
            row["Score"] = r["grand"]
        rows.append(row)

    for arm in ARM_NAMES:
        res = result["arms"][arm]
        if res is None:
            rows.append({"Model": arm, "RULA": "No person detected"})
            continue
        row = {"Model": arm, "RULA": res["rula"]["text"]}
        if gt:
            row["RULA"] += "  ✓" if res["rula_match"] else "  ✗"
        if show_degrees:
            for a in ANGLE_NAMES:
                cell = _fmt(res["angles"][a])
                if gt and not np.isnan(res["error"][a]):
                    cell += f" (err {res['error'][a]:.1f})"
                row[a] = cell
            row["Score"] = res["rula"]["grand"]
        rows.append(row)
    return rows


def draw_overlay(image_bgr, result):
    img = image_bgr.copy()
    for ex, pose in result["poses"].items():
        if pose is not None:
            draw_pose(img, pose, COLORS[ex])
    return img


def print_table(result):
    gt = result["gt"]
    head = f"{'Model':26s}" + "".join(f"{a:>13s}" for a in ANGLE_NAMES) + "  RULA"
    print(head)
    print("-" * len(head))
    if gt:
        print(f"{'Ground truth':26s}" + "".join(f"{_fmt(gt[a]):>13s}" for a in ANGLE_NAMES)
              + f"  {result['gt_rula']['text']}")
    for arm in ARM_NAMES:
        res = result["arms"][arm]
        if res is None:
            print(f"{arm:26s}  no person detected")
            continue
        cells = []
        for a in ANGLE_NAMES:
            c = _fmt(res["angles"][a])
            if gt and not np.isnan(res["error"][a]):
                c += f"±{res['error'][a]:.0f}"
            cells.append(f"{c:>13s}")
        mark = ("  ✓" if res["rula_match"] else "  ✗") if gt else ""
        print(f"{arm:26s}" + "".join(cells) + f"  {res['rula']['text']}{mark}")
    if gt:
        print("\n±N = absolute error vs ground truth (degrees);  ✓/✗ = RULA level matches GT")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", default=None, help="gt_angles.csv (default: auto-detect)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    gt = load_gt(args.gt).get(Path(args.image).name) if args.gt else find_gt_for(args.image)
    print("GT:", "found" if gt else "none (model angles only)")

    comp = Comparator(device=args.device)
    result = comp.run(img, gt)
    print_table(result)

    # Saved outside the dataset folder so it never gets mixed in with the input images
    out = Path("logs") / "compare" / (Path(args.image).stem + "_compare.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_overlay(img, result))
    print(f"\nOverlay saved: {out}  (green = RTMPose, magenta = MediaPipe)")
    comp.close()
