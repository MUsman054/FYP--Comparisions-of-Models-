"""
app.py — live 2D-vs-3D posture-angle comparison on the laptop webcam.

Run:
    python app.py                      (webcam 0, opens the browser)
    python app.py --source 1           (another webcam)
    python app.py --source clip.mp4    (a recorded video)

How to use:
    1. Pick the 2D extractor and the 3D lifter.
    2. Sit upright, press Calibrate, hold still for 3 s (sets the 0° reference).
    3. Pick a label (the test pose), press Record, hold the pose, press Stop.
    4. Each recording session is saved to logs/ as CSV (see logger.py).

Design:
    - Camera + 2D extractor run in one thread, the lifter in another, so a slow
      lifter (MotionBERT) never freezes the video. The 3D row in the CSV carries
      the frame number it belongs to, so 2D and 3D rows can be matched later.
    - Live MotionBERT uses an 81-frame window without flip (243 + flip is
      ~3-5 s per frame on a laptop CPU). Recorded clips can use the full setting.
"""

import argparse
import threading
import time
from collections import deque

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from angles import ANGLE_JOINTS, ANGLE_NAMES, Calibrator, compute_angles, rula_segment_scores
from extractors import EXTRACTORS, H36M_EDGES, draw_pose, load_extractor, open_camera
from lifters import HISTORY_LEN, LIFTERS, is_compatible, load_lifter
from logger import AngleLogger

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NO_LIFTER = "None (2D only)"
LIVE_LIFTER_KWARGS = {"MotionBERT": {"window": 81, "flip": False}}
CALIBRATION_SECONDS = 3.0

LABELS = [
    "free",
    "upright",
    "lean_forward",
    "lean_side",
    "neck_flexion",
    "arm_down_0",
    "arm_forward_90",
    "arm_side_90",
    "arm_up_180",
    "elbow_straight_0",
    "elbow_typing_90",
    "chair_rotated_90",
]

# angle name -> RULA segment key
RULA_OF = {"trunk": "trunk", "neck": "neck",
           "l_upper_arm": "l_upper_arm", "r_upper_arm": "r_upper_arm",
           "l_elbow": "l_lower_arm", "r_elbow": "r_lower_arm"}

GREEN, WHITE, GRAY, ORANGE = (0, 200, 0), (255, 255, 255), (150, 150, 150), (0, 165, 255)


# ---------------------------------------------------------------------------
# Engine: camera, models, calibration, logging (runs in background threads)
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, source, device):
        self.cap = open_camera(source)
        if self.cap is None:
            raise SystemExit(f"cannot open source {source} (see extractors.py self-test tips)")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # always the newest frame, no lag build-up
        self.device = device

        self._models = {}                            # loaded models, reused on switch-back
        self.extractor = self.lifter = None
        self.ext_name, self.lift_name = None, NO_LIFTER

        self.history = deque(maxlen=HISTORY_LEN)
        self.lock = threading.Lock()
        self.new_pose = threading.Event()
        self.frame_idx = 0

        self.view = None                             # RGB frame for the UI
        self.res2d, self.res3d = {}, {}
        self.fps2d = self.fps3d = 0.0
        self.error = ""

        self.cal2d, self.cal3d = Calibrator(), Calibrator()
        self.calibrated = False
        self.calib_until = 0.0
        self.calib_msg = "not calibrated"

        self.label = LABELS[0]
        self.logger = None
        self.ended = False

        self.running = True
        self.threads = [threading.Thread(target=self._capture_loop, daemon=True),
                        threading.Thread(target=self._lifter_loop, daemon=True)]
        for t in self.threads:
            t.start()

    # ---------------- models ----------------
    def _get(self, kind, name):
        key = (kind, name)
        if key not in self._models:
            if kind == "ext":
                kw = {"device": self.device} if name == "RTMPose" else {}
                self._models[key] = load_extractor(name, **kw)
            else:
                kw = dict(LIVE_LIFTER_KWARGS.get(name, {}), device=self.device)
                self._models[key] = load_lifter(name, **kw)
        return self._models[key]

    def set_models(self, ext_name, lift_name):
        warn = ""
        if lift_name != NO_LIFTER and not is_compatible(lift_name, ext_name):
            warn = f"⚠️ {lift_name} only works with the MediaPipe extractor. Lifter set to none. "
            lift_name = NO_LIFTER
        ext = self._get("ext", ext_name)
        lift = None if lift_name == NO_LIFTER else self._get("lift", lift_name)
        with self.lock:
            self.extractor, self.ext_name = ext, ext_name
            self.lifter, self.lift_name = lift, lift_name
            self.history.clear()
            self.res2d, self.res3d = {}, {}
            self.cal2d.reset()
            self.cal3d.reset()
            self.calibrated = False
            self.calib_msg = "not calibrated (models changed)"
        return warn + f"Loaded {ext_name} + {lift_name}. Sit upright and press Calibrate."

    # ---------------- calibration ----------------
    def start_calibration(self):
        self.cal2d.reset()
        self.cal3d.reset()
        self.calibrated = False
        self.calib_until = time.time() + CALIBRATION_SECONDS
        self.calib_msg = "calibrating — sit upright and hold still"

    def _calibrating(self):
        return time.time() < self.calib_until

    def _maybe_finish_calibration(self):
        if self.calib_until and not self._calibrating():
            self.calib_until = 0.0
            ok2 = self.cal2d.finish(min_samples=3)
            ok3 = self.cal3d.finish(min_samples=3) if self.lifter else True
            self.calibrated = ok2 and ok3
            self.calib_msg = ("calibrated ✓" if self.calibrated else
                              "calibration failed — no person seen, try again")

    # ---------------- recording ----------------
    def toggle_recording(self):
        if self.logger is None:
            name = time.strftime("session_%Y%m%d_%H%M%S")
            self.logger = AngleLogger(name)
            return True, f"Recording → {self.logger.path}"
        log, self.logger = self.logger, None
        log.close()
        return False, f"Saved {log.rows} rows → {log.path}"

    # ---------------- thread 1: camera + 2D ----------------
    def _capture_loop(self):
        t_prev = time.perf_counter()
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    self.ended = True
                    time.sleep(0.05)
                    continue

                ext = self.extractor
                if ext is None:
                    self.view = self._compose(frame)
                    continue

                t0 = time.perf_counter()
                pose = ext(frame)
                latency = (time.perf_counter() - t0) * 1000
                self.frame_idx += 1
                fidx = self.frame_idx

                if pose is None:
                    self.res2d = {}
                else:
                    pose.frame = fidx
                    with self.lock:
                        self.history.append(pose)
                    self.new_pose.set()

                    if self._calibrating():
                        self.cal2d.add(pose.kpts2d)
                    a2 = compute_angles(pose.kpts2d, up=self.cal2d.up)
                    self.res2d = {"angles": a2, "scores": pose.scores, "frame": fidx}

                    log = self.logger
                    if log is not None:
                        log.log(frame=fidx, extractor=self.ext_name, method="2D", angles=a2,
                                scores=pose.scores, joints=pose.kpts2d, label=self.label,
                                calibrated=self.calibrated,
                                trunk_raw=compute_angles(pose.kpts2d)["trunk"],
                                latency_ms=latency)
                    draw_pose(frame, pose, GREEN)

                self._maybe_finish_calibration()
                now = time.perf_counter()
                self.fps2d = 0.9 * self.fps2d + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                self.view = self._compose(frame)
            except Exception as e:                       # keep the app alive, show the error
                self.error = f"2D thread: {type(e).__name__}: {e}"
                time.sleep(0.2)

    # ---------------- thread 2: lifter ----------------
    def _lifter_loop(self):
        t_prev = time.perf_counter()
        while self.running:
            if not self.new_pose.wait(0.1):
                continue
            self.new_pose.clear()
            try:
                lift, lift_name = self.lifter, self.lift_name
                if lift is None:
                    continue
                with self.lock:
                    hist = list(self.history)
                if not hist:
                    continue

                t0 = time.perf_counter()
                p3d = lift(hist)
                latency = (time.perf_counter() - t0) * 1000
                if p3d is None or lift is not self.lifter:   # no output, or models switched meanwhile
                    continue

                newest = hist[-1]
                if self._calibrating():
                    self.cal3d.add(p3d)
                a3 = compute_angles(p3d, up=self.cal3d.up)
                self.res3d = {"angles": a3, "frame": newest.frame, "p3d": p3d}

                log = self.logger
                if log is not None:
                    log.log(frame=newest.frame, extractor=self.ext_name, method=lift_name,
                            angles=a3, scores=newest.scores, joints=p3d, label=self.label,
                            calibrated=self.calibrated, trunk_raw=compute_angles(p3d)["trunk"],
                            latency_ms=latency)

                now = time.perf_counter()
                self.fps3d = 0.8 * self.fps3d + 0.2 / max(now - t_prev, 1e-6)
                t_prev = now
            except Exception as e:
                self.error = f"3D thread: {type(e).__name__}: {e}"
                time.sleep(0.2)

    # ---------------- drawing ----------------
    def _compose(self, frame):
        """Webcam frame + trunk readout + 3D side-view inset. Returns RGB."""
        img = frame.copy()
        h, w = img.shape[:2]

        a2 = self.res2d.get("angles", {})
        a3 = self.res3d.get("angles", {})
        txt = f"trunk  2D {_fmt(a2.get('trunk'))}"
        if self.lifter is not None:
            txt += f"   3D {_fmt(a3.get('trunk'))}"
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        if self._calibrating():
            cv2.putText(img, "CALIBRATING - sit upright", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
        if self.logger is not None:
            cv2.circle(img, (w - 20, 17), 8, (0, 0, 255), -1)      # recording dot

        p3d = self.res3d.get("p3d")
        if p3d is not None:
            _draw_side_view(img, p3d, self.cal3d.up)
        return img[:, :, ::-1]


def _fmt(x):
    return "  -  " if x is None or np.isnan(x) else f"{x:5.1f}°"


def _draw_side_view(img, p3d, up, size=(170, 210)):
    """
    Inset at top-right: the 3D skeleton seen from the SIDE (depth z vs height y).
    The camera is to the LEFT of the inset, so leaning toward the camera tilts
    the body left here — exactly the motion the frontal 2D view cannot see.
    Gray line = calibrated vertical.
    """
    pw, ph = size
    h, w = img.shape[:2]
    x0, y0 = w - pw - 10, 44
    if x0 < 0 or y0 + ph > h:
        return
    roi = img[y0:y0 + ph, x0:x0 + pw]
    roi[:] = (roi * 0.3).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), WHITE, 1)
    cv2.putText(img, "3D side view", (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.putText(img, "<- camera", (x0 + 6, y0 + ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    trunk = np.linalg.norm(p3d[8] - p3d[0])
    s = 55.0 / max(trunk, 1e-6)
    cx, cy = x0 + pw // 2, y0 + int(ph * 0.62)

    def pt(v):
        return int(cx + s * v[2]), int(cy + s * v[1])     # horizontal = z (depth), vertical = y

    end = up / max(np.linalg.norm(up), 1e-6) * trunk * 1.3
    cv2.line(img, pt(np.zeros(3)), pt(end), GRAY, 1)
    for a, b in H36M_EDGES:
        cv2.line(img, pt(p3d[a]), pt(p3d[b]), ORANGE, 2)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(engine, ext_default, lift_default):
    lifter_choices = [NO_LIFTER] + list(LIFTERS)

    def tick():
        a2 = engine.res2d.get("angles", {})
        a3 = engine.res3d.get("angles", {})
        sc = engine.res2d.get("scores")
        r2 = rula_segment_scores(a2) if a2 else {}
        r3 = rula_segment_scores(a3) if a3 else {}
        has3d = engine.lifter is not None

        rows = []
        for n in ANGLE_NAMES:
            v2, v3 = a2.get(n, np.nan), a3.get(n, np.nan)
            diff = abs(v2 - v3) if has3d else np.nan
            conf = (float(np.min(sc[_joints_of(n)])) if sc is not None else np.nan)
            rows.append([n, _fmt(v2), _fmt(v3) if has3d else "", _fmt(diff) if has3d else "",
                         _score(r2.get(RULA_OF[n])), _score(r3.get(RULA_OF[n])) if has3d else "",
                         "" if np.isnan(conf) else f"{conf:.2f}"])
        table = pd.DataFrame(rows, columns=["angle", "2D", "3D", "|2D−3D|",
                                            "RULA 2D", "RULA 3D", "2D conf"])

        lag = engine.frame_idx - engine.res3d.get("frame", engine.frame_idx)
        lines = [f"**2D:** {engine.ext_name} — {engine.fps2d:4.1f} fps"]
        if has3d:
            lines.append(f"**3D:** {engine.lift_name} — {engine.fps3d:4.1f} fps "
                         f"(lags {lag} frames)")
        lines.append(f"**Calibration:** {engine.calib_msg}")
        lines.append(f"**Label:** `{engine.label}`")
        if engine.logger is not None:
            lines.append(f"**Recording:** 🔴 {engine.logger.rows} rows")
        if engine.ended:
            lines.append("**Video ended.**")
        if engine.error:
            lines.append(f"⚠️ {engine.error}")
        return engine.view, table, "  \n".join(lines)

    def on_models(ext_name, lift_name):
        msg = engine.set_models(ext_name, lift_name)
        # If the lifter was refused, show "None" in the menu; otherwise leave the menu as is
        return msg, (gr.Dropdown(value=NO_LIFTER) if msg.startswith("⚠️") else gr.Dropdown())

    def on_calibrate():
        engine.start_calibration()
        return f"Calibrating for {CALIBRATION_SECONDS:.0f} s — sit upright, hold still."

    def on_label(label):
        engine.label = label or "free"

    def on_record():
        recording, msg = engine.toggle_recording()
        return msg, gr.Button(value="⏹ Stop recording" if recording else "⏺ Start recording",
                              variant="stop" if recording else "primary")

    with gr.Blocks(title="2D vs 3D posture angles") as demo:
        gr.Markdown("## 2D vs 3D posture angles — live webcam")
        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Image(label="Webcam (green = 2D joints, inset = 3D side view)",
                                 type="numpy", format="jpeg", interactive=False)
            with gr.Column(scale=2):
                ext_dd = gr.Dropdown(list(EXTRACTORS), value=ext_default, label="2D extractor")
                lift_dd = gr.Dropdown(lifter_choices, value=lift_default, label="3D lifter")
                calib_btn = gr.Button("🎯 Calibrate (sit upright)")
                label_dd = gr.Dropdown(LABELS, value=LABELS[0], label="Test pose label",
                                       allow_custom_value=True)
                rec_btn = gr.Button("⏺ Start recording", variant="primary")
                msg = gr.Markdown()
                status = gr.Markdown()
        table = gr.Dataframe(label="Angles (degrees) and RULA segment scores", interactive=False)

        timer = gr.Timer(0.1)
        timer.tick(tick, outputs=[video, table, status], show_progress="hidden")
        ext_dd.change(on_models, [ext_dd, lift_dd], [msg, lift_dd])
        lift_dd.change(on_models, [ext_dd, lift_dd], [msg, lift_dd])
        calib_btn.click(on_calibrate, outputs=msg)
        label_dd.change(on_label, label_dd)
        rec_btn.click(on_record, outputs=[msg, rec_btn])
    return demo


def _joints_of(angle_name):
    return ANGLE_JOINTS[angle_name]


def _score(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(int(x))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--extractor", default="RTMPose", choices=list(EXTRACTORS))
    ap.add_argument("--lifter", default="VideoPose3D", choices=[NO_LIFTER] + list(LIFTERS))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    engine = Engine(args.source, args.device)
    print(engine.set_models(args.extractor, args.lifter))
    demo = build_ui(engine, args.extractor, engine.lift_name)
    try:
        demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)
    finally:
        engine.running = False
        if engine.logger is not None:
            engine.toggle_recording()      # close and save the CSV
        engine.cap.release()
