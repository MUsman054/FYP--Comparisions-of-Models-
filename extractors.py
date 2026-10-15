"""
extractors.py — pretrained 2D pose extractors, used off the shelf.

    RTMPose   : rtmlib (ONNX runtime, CPU ok). YOLOX person detector + RTMPose-m, COCO-17.
    MediaPipe : PoseLandmarker (Tasks API). 33 landmarks in 2D + world 3D in one call.

Every extractor returns a Pose already converted to H36M-17 (via mapping.py),
so lifters, angles and drawing all see one format.

Install:  pip install rtmlib onnxruntime mediapipe opencv-python numpy
First run downloads weights: rtmlib -> its own cache, MediaPipe -> ./checkpoints/

Note: RTMPose scores are SimCC peak values, not probabilities (can exceed 1).
"""

import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

from mapping import coco_to_h36m, mediapipe_to_coco, mediapipe_to_h36m

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


@dataclass
class Pose:
    kpts2d: np.ndarray            # (17, 2) H36M order, pixels
    scores: np.ndarray            # (17,)  confidence per joint
    img_wh: tuple                 # (width, height), needed by lifters for normalization
    coco2d: np.ndarray = None     # (17, 2) COCO order, pixels — VideoPose3D input
    coco_scores: np.ndarray = None
    world3d: np.ndarray = None    # (17, 3) H36M order, metres — MediaPipe only
    world_ears: np.ndarray = None # (3,) ear midpoint, same frame as world3d — MediaPipe only,
                                  # used as the head point for the neck angle (matches GT)


# ---------------------------------------------------------------------------
# RTMPose (rtmlib)
# ---------------------------------------------------------------------------
class RTMPoseExtractor:
    name = "RTMPose"

    def __init__(self, mode="balanced", device="cpu"):
        from rtmlib import Body
        # mode: 'lightweight' (RTMPose-s) | 'balanced' (RTMPose-m) | 'performance' (RTMPose-x)
        self.model = Body(mode=mode, backend="onnxruntime", device=device)

    def __call__(self, frame_bgr):
        kpts, scores = self.model(frame_bgr)
        kpts, scores = np.asarray(kpts), np.asarray(scores)
        if kpts.size == 0:
            return None
        i = int(np.argmax(scores.mean(axis=1)))   # most confident person
        h36m, hs = coco_to_h36m(kpts[i], scores[i])
        h, w = frame_bgr.shape[:2]
        return Pose(h36m, hs, (w, h), coco2d=kpts[i].astype(np.float64),
                    coco_scores=scores[i].astype(np.float64))

    def close(self):
        pass


# ---------------------------------------------------------------------------
# MediaPipe PoseLandmarker (Tasks API; the old mp.solutions API is gone in 0.10.3x)
# ---------------------------------------------------------------------------
MP_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
          "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

MP_L_EAR, MP_R_EAR = 7, 8


def _ensure_mp_model(variant):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        print(f"Downloading MediaPipe '{variant}' model ...")
        tmp = path + ".part"   # avoid leaving a corrupt file if the download breaks
        urllib.request.urlretrieve(MP_URL.format(v=variant), tmp)
        os.replace(tmp, path)
    return path


class MediaPipeExtractor:
    name = "MediaPipe"

    def __init__(self, variant="full"):
        # variant: 'lite' | 'full' | 'heavy'
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=_ensure_mp_model(variant)),
            running_mode=vision.RunningMode.VIDEO,   # uses tracking between frames
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.mp = mp
        self._last_ts = -1

    def _timestamp(self):
        # VIDEO mode needs strictly increasing timestamps (ms)
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def __call__(self, frame_bgr):
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(image, self._timestamp())
        if not res.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lm = res.pose_landmarks[0]
        xy = np.array([[p.x * w, p.y * h] for p in lm])
        vis = np.array([p.visibility if p.visibility is not None else 1.0 for p in lm])
        world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])

        coco, coco_s = mediapipe_to_coco(xy, vis)
        k2, s2 = coco_to_h36m(coco, coco_s)
        k3, _ = mediapipe_to_h36m(world)
        return Pose(k2, s2, (w, h), coco2d=coco, coco_scores=coco_s, world3d=k3,
                    world_ears=0.5 * (world[MP_L_EAR] + world[MP_R_EAR]))

    def close(self):
        self.landmarker.close()


# ---------------------------------------------------------------------------
# Registry (the app's menu reads this)
# ---------------------------------------------------------------------------
EXTRACTORS = {
    "RTMPose": RTMPoseExtractor,
    "MediaPipe": MediaPipeExtractor,
}


def load_extractor(name, **kwargs):
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor '{name}', choose from {list(EXTRACTORS)}")
    return EXTRACTORS[name](**kwargs)


# ---------------------------------------------------------------------------
# Drawing (reused by app.py)
# ---------------------------------------------------------------------------
H36M_EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
              (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]


def draw_pose(img, pose, color, thr=0.3):
    """Draw H36M skeleton. Low-confidence joints/bones are skipped."""
    import cv2
    k, s = pose.kpts2d, pose.scores
    for a, b in H36M_EDGES:
        if s[a] >= thr and s[b] >= thr:
            cv2.line(img, tuple(map(int, k[a])), tuple(map(int, k[b])), color, 2)
    for i, p in enumerate(k):
        if s[i] >= thr:
            cv2.circle(img, tuple(map(int, p)), 3, color, -1)
    return img


# ---------------------------------------------------------------------------
# Webcam opening (reused by app.py). On Windows the default backend often
# fails, so try DirectShow first, then Media Foundation, then the default.
# ---------------------------------------------------------------------------
def open_camera(source="0"):
    import sys
    import cv2
    if not str(source).isdigit():
        cap = cv2.VideoCapture(source)            # video file
        return cap if cap.isOpened() else None

    backends = [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for be in backends:
        cap = cv2.VideoCapture(int(source), be)
        if cap.isOpened():
            ok, _ = cap.read()                    # some backends open but return no frames
            if ok:
                return cap
        cap.release()
    return None


# ---------------------------------------------------------------------------
# Self-test: both extractors on the webcam, side by side on one frame.
#   python extractors.py                 (webcam 0)
#   python extractors.py --source 1      (another webcam)
#   python extractors.py --source x.jpg  (image -> saves x_extractors.jpg)
# Press q to quit.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import cv2
    from angles import compute_angles

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    models = {
        "RTMPose": (RTMPoseExtractor(device=args.device), (0, 200, 0)),     # green
        "MediaPipe": (MediaPipeExtractor(), (200, 0, 200)),                 # magenta
    }

    def process(frame):
        y = 25
        for name, (model, color) in models.items():
            t0 = time.perf_counter()
            pose = model(frame)
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            if pose is None:
                text = f"{name}: no person"
            else:
                draw_pose(frame, pose, color)
                a = compute_angles(pose.kpts2d)   # raw 2D angles, no gating
                text = (f"{name} {fps:4.1f}fps | 2D trunk {a['trunk']:5.1f}  "
                        f"R arm {a['r_upper_arm']:5.1f}  R elbow {a['r_elbow']:5.1f}")
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 25
        return frame

    src = args.source
    if os.path.isfile(src) and src.lower().endswith((".jpg", ".jpeg", ".png")):
        out = process(cv2.imread(src))
        out_path = os.path.splitext(src)[0] + "_extractors.jpg"
        cv2.imwrite(out_path, out)
        print("saved", out_path)
    else:
        cap = open_camera(src)
        if cap is None:
            raise SystemExit(
                f"cannot open source {src}. Check: (1) close Zoom/Teams/browser tabs using "
                "the camera, (2) Windows Settings > Privacy > Camera > allow desktop apps, "
                "(3) try --source 1")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # No mirroring: flipping would swap the subject's left/right labels
            cv2.imshow("extractors (q to quit)", process(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    for model, _ in models.values():
        model.close()
