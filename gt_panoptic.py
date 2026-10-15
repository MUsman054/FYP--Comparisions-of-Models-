"""
gt_panoptic.py — build an image + ground-truth dataset from CMU Panoptic.

Output (inside your project):
    data/panoptic/images/frame_XXXXXXXX.jpg   input images for all 7 model arms
    data/panoptic/gt_angles.csv               GT angles per image (same names as angles.py)
    data/panoptic/gt_joints.npz               GT H36M-17 joints (3D camera frame + 2D pixels)
    data/panoptic/preview/                    a few images with the GT skeleton drawn on them

GT angles are computed with YOUR angles.py on joints mapped with YOUR mapping.py,
so GT and models use exactly the same angle definitions.

Usage (from the project folder, venv active):
    python gt_panoptic.py                    # picks the most frontal camera automatically
    python gt_panoptic.py --cam 3            # force HD camera 00_03
    python gt_panoptic.py --step 5           # denser frames (default every 10th)

Data: CMU Panoptic Studio, non-commercial research use. Cite Joo et al. in the report.
"""

import argparse
import csv
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from angles import ANGLE_NAMES, compute_angles
from mapping import coco_to_h36m

ENDPOINTS = ["http://domedb.perception.cs.cmu.edu", "http://vcl.snu.ac.kr/panoptic"]

# Panoptic joints19 -> COCO-17 (standard COCO order; left/right = subject's)
#   COCO: nose, l_eye, r_eye, l_ear, r_ear, l_sho, r_sho, l_elb, r_elb,
#         l_wri, r_wri, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank
P19_TO_COCO = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
P_NECK, P_CENTER, P_LSHO, P_RSHO = 0, 2, 3, 9

# Joints that must be visible in the image for an upper-body RULA frame
NEEDED_COCO = [0, 5, 6, 7, 8, 9, 10, 11, 12]
MIN_CONF = 0.1


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(rel_url, dest):
    """Download rel_url to dest, trying the CMU server then the SNU mirror."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for ep in ENDPOINTS:
        url = f"{ep}/{rel_url}"
        print(f"  downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
            print()
            if tmp.stat().st_size == 0:
                raise IOError("empty file")
            tmp.replace(dest)
            return
        except Exception as e:  # try the mirror
            print(f"\n    failed: {e}")
            tmp.unlink(missing_ok=True)
    sys.exit(f"Could not download {rel_url} from any server.")


# ---------------------------------------------------------------------------
# Load calibration + skeletons
# ---------------------------------------------------------------------------
def load_hd_cameras(calib_path):
    calib = json.loads(calib_path.read_text())
    cams = {}
    for c in calib["cameras"]:
        if c.get("type") != "hd":
            continue
        cams[c["node"]] = {
            "name": c["name"],
            "K": np.array(c["K"], dtype=np.float64),
            "dist": np.array(c["distCoef"], dtype=np.float64),
            "R": np.array(c["R"], dtype=np.float64),
            "t": np.array(c["t"], dtype=np.float64).reshape(3),
            "res": tuple(c["resolution"]),  # (width, height)
        }
    return cams


def load_skeletons(tar_path, extract_dir):
    """Return {hd_frame_idx: (19, 4) array [x, y, z, conf]} for the first body."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(extract_dir, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(extract_dir)
    skels = {}
    for p in sorted(extract_dir.rglob("body3DScene_*.json")):
        d = json.loads(p.read_text())
        if not d.get("bodies"):
            continue
        j = np.array(d["bodies"][0]["joints19"], dtype=np.float64).reshape(19, 4)
        skels[int(p.stem.split("_")[-1])] = j
    if not skels:
        sys.exit("No skeleton frames found in the tar.")
    return skels


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def to_cam(X_world, cam):
    """World (N,3) cm -> camera frame (N,3): x right, y down, z forward (OpenCV)."""
    return X_world @ cam["R"].T + cam["t"]


def project(X_cam, cam):
    pts, _ = cv2.projectPoints(X_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                               cam["K"], cam["dist"])
    return pts.reshape(-1, 2)


def facing_angle(j19_cam, up_c):
    """Horizontal angle (yaw) between where the subject faces and the direction to the
    camera. 0 = facing the camera. Measured in the floor plane, so leaning forward
    does not count as turning away."""
    left = j19_cam[P_LSHO, :3] - j19_cam[P_RSHO, :3]
    fwd = np.cross(left, up_c)            # body facing direction, horizontal
    to_cam = -j19_cam[P_CENTER, :3]       # camera sits at the origin
    to_cam = to_cam - (to_cam @ up_c) * up_c
    c = fwd @ to_cam / (np.linalg.norm(fwd) * np.linalg.norm(to_cam) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def world_up(skels):
    """World vertical = the world axis closest to the median neck-minus-hips direction."""
    v = np.median([s[P_NECK, :3] - s[P_CENTER, :3] for s in skels.values()], axis=0)
    axes = np.vstack([np.eye(3), -np.eye(3)])
    return axes[np.argmax(axes @ (v / np.linalg.norm(v)))]


def camera_tilt(cam, up_w):
    """Degrees between the camera's 'up' and world vertical (0 = level camera)."""
    cam_up_w = cam["R"].T @ np.array([0.0, -1.0, 0.0])
    return float(np.degrees(np.arccos(np.clip(cam_up_w @ up_w, -1, 1))))


def frame_ok(j19, cam, margin=10):
    """All needed joints confident and inside the image."""
    coco = j19[P19_TO_COCO]
    if np.any(coco[NEEDED_COCO, 3] < MIN_CONF):
        return False
    uv = project(to_cam(coco[NEEDED_COCO, :3], cam), cam)
    w, h = cam["res"]
    return bool(np.all((uv[:, 0] > margin) & (uv[:, 0] < w - margin) &
                       (uv[:, 1] > margin) & (uv[:, 1] < h - margin)))


def rank_cameras(cams, skels, up_w, n_probe=60):
    keys = sorted(skels)
    probe = keys[:: max(1, len(keys) // n_probe)]
    rows = []
    for node, cam in cams.items():
        up_c = cam["R"] @ up_w
        yaws = [facing_angle(to_cam(skels[k][:, :3], cam), up_c) for k in probe]
        inframe = np.mean([frame_ok(skels[k], cam) for k in probe])
        tilt = camera_tilt(cam, up_w)
        rows.append((node, cam["name"], float(np.median(yaws)), tilt, float(inframe)))
    # most frontal, then most level, among cameras that see the person most of the time
    rows.sort(key=lambda r: (r[4] < 0.8, r[2] + 0.5 * r[3]))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="171204_pose1_sample")
    ap.add_argument("--cam", type=int, default=None, help="HD camera node 0-30 (default: auto)")
    ap.add_argument("--step", type=int, default=10, help="keep every Nth frame")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--max-yaw", type=float, default=45.0,
                    help="skip frames where the subject faces more than this many degrees away")
    ap.add_argument("--out", default="data/panoptic")
    ap.add_argument("--keep-video", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    raw = out / "_raw"
    base = f"webdata/dataset/{args.seq}"

    print("1/5  Calibration + 3D skeletons")
    calib_path = raw / f"calibration_{args.seq}.json"
    tar_path = raw / "hdPose3d_stage1_coco19.tar"
    download(f"{base}/calibration_{args.seq}.json", calib_path)
    download(f"{base}/hdPose3d_stage1_coco19.tar", tar_path)
    cams = load_hd_cameras(calib_path)
    skels = load_skeletons(tar_path, raw / "skeletons")
    up_w = world_up(skels)
    print(f"  {len(skels)} skeleton frames, {len(cams)} HD cameras")

    print("2/5  Choosing camera")
    ranking = rank_cameras(cams, skels, up_w)
    print("  node  name   facing  tilt  in-frame")
    for node, name, yaw, tilt, inf in ranking[:5]:
        print(f"  {node:4d}  {name}  {yaw:5.1f}°  {tilt:4.1f}°  {inf:5.0%}")
    node = args.cam if args.cam is not None else ranking[0][0]
    if node not in cams:
        sys.exit(f"HD camera {node} not in calibration.")
    cam = cams[node]
    up_c = cam["R"] @ up_w
    print(f"  using camera {cam['name']}")

    print("3/5  Video")
    vid_name = f"hd_{cam['name']}.mp4"
    vid_path = raw / vid_name
    download(f"{base}/videos/hd_shared_crf20/{vid_name}", vid_path)

    print("4/5  Extracting images + GT angles")
    img_dir = out / "images"
    prev_dir = out / "preview"
    for d in (img_dir, prev_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"OpenCV could not open {vid_path}")

    rows, j3d_all, j2d_all, conf_all, names = [], [], [], [], []
    idx, kept = -1, 0
    skipped = {"no skeleton": 0, "not visible": 0, "facing away": 0}
    while kept < args.max_frames:
        if not cap.grab():
            break
        idx += 1  # Panoptic HD frames start at 0, same index as the skeleton files
        if idx % args.step:
            continue
        j19 = skels.get(idx)
        if j19 is None:
            skipped["no skeleton"] += 1
            continue
        if not frame_ok(j19, cam):
            skipped["not visible"] += 1
            continue
        j19_cam = np.hstack([to_cam(j19[:, :3], cam), j19[:, 3:]])
        yaw = facing_angle(j19_cam, up_c)
        if yaw > args.max_yaw:
            skipped["facing away"] += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        coco = j19_cam[P19_TO_COCO]
        h36m, hconf = coco_to_h36m(coco[:, :3], coco[:, 3])
        ang = compute_angles(h36m, conf=hconf)

        name = f"frame_{idx:08d}.jpg"
        cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        uv = project(h36m, cam)

        rows.append({"image": name, "frame": idx, "facing_deg": round(yaw, 1),
                     **{a: round(float(ang[a]), 2) for a in ANGLE_NAMES}})
        names.append(name)
        j3d_all.append(h36m)
        j2d_all.append(uv)
        conf_all.append(hconf)

        if kept % 10 == 0:  # every 10th kept image gets a preview
            draw_preview(frame, uv, ang, prev_dir / name)
        kept += 1
    cap.release()

    if not rows:
        sys.exit("No usable frames. Try another camera with --cam, or raise --max-yaw.")

    with open(out / "gt_angles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "gt_joints.npz", images=np.array(names),
             joints3d_cam_cm=np.array(j3d_all), joints2d_px=np.array(j2d_all),
             conf=np.array(conf_all), camera=cam["name"], sequence=args.seq)
    shutil.copy(calib_path, out / calib_path.name)

    print("5/5  Cleanup")
    if not args.keep_video:
        vid_path.unlink(missing_ok=True)
        print("  video deleted")
    if not args.keep_raw and not args.keep_video:
        shutil.rmtree(raw, ignore_errors=True)
        print("  raw downloads deleted")

    print(f"\nDone: {len(rows)} images in {img_dir}")
    print(f"      GT angles in {out / 'gt_angles.csv'}")
    print(f"      skipped: {skipped}")
    print(f"      check {prev_dir} to confirm the skeleton lines up with the person")
    print("\nGT angle summary (min / median / max):")
    for a in ANGLE_NAMES:
        v = np.array([r[a] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            print(f"  {a:12s} {v.min():6.1f}  {np.median(v):6.1f}  {v.max():6.1f}   (n={len(v)})")


# H36M-17 bones for the preview drawing
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]


def draw_preview(frame, uv, ang, path):
    img = frame.copy()
    for a, b in BONES:
        cv2.line(img, tuple(map(int, uv[a])), tuple(map(int, uv[b])), (0, 255, 0), 3)
    for p in uv:
        cv2.circle(img, tuple(map(int, p)), 5, (0, 0, 255), -1)
    y = 40
    for k in ANGLE_NAMES:
        v = ang[k]
        txt = f"{k}: {'NaN' if np.isnan(v) else f'{v:.1f} deg'}"
        cv2.putText(img, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += 40
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    main()
