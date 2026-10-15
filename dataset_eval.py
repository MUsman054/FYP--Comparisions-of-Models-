"""
dataset_eval.py — run all 7 arms over a whole image folder and compare with GT.

Metrics (as committed in the proposal):
    Per-angle MAE in degrees, reported per angle (never averaged), with SD and P95
    Bias (mean signed error)   — shows systematic under-reading, e.g. 2D on flexion
    Coverage %                 — share of GT frames where the arm produced the angle
    RULA agreement             — per sub-score (trunk, neck, upper/lower arm)
                                 + action level (1-4) confusion matrix and Cohen's kappa
No "accuracy %": the system measures angles; RULA is scored from them.

Usage:
    python dataset_eval.py                       # data/panoptic
    python dataset_eval.py --every 3             # every 3rd image: ~1/3 of the time
    python dataset_eval.py --data data/panoptic --limit 20
Outputs: logs/eval/<dataset>/  (CSVs + two PNG figures)

Resumable: finished images are cached, so Ctrl+C and rerun continues where it stopped.
Single-image mode: MotionBERT's window is filled with copies of the one pose, so a short
window (default 27) gives the same information much faster than 81/243.
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")      # silence MediaPipe's C++ warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from angles import ANGLE_NAMES
from compare import ARM_NAMES, Comparator, load_gt

SEGMENTS = ["trunk", "neck", "l_upper_arm", "r_upper_arm", "l_lower_arm", "r_lower_arm"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def angle_metrics(pred, gt):
    """pred, gt: arrays over images (NaN = missing). Returns dict of metrics."""
    gt_ok = ~np.isnan(gt)
    both = gt_ok & ~np.isnan(pred)
    n_gt, n = int(gt_ok.sum()), int(both.sum())
    if n == 0:
        return {"n": 0, "coverage_pct": 0.0, "mae": np.nan, "sd": np.nan,
                "p95": np.nan, "bias": np.nan}
    err = pred[both] - gt[both]
    ae = np.abs(err)
    return {
        "n": n,
        "coverage_pct": 100.0 * n / max(n_gt, 1),
        "mae": float(ae.mean()),
        "sd": float(ae.std(ddof=1)) if n > 1 else 0.0,
        "p95": float(np.percentile(ae, 95)),
        "bias": float(err.mean()),
    }


def cohen_kappa(a, b, labels=(1, 2, 3, 4)):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# ---------------------------------------------------------------------------
# Figures (matplotlib, light background for the report)
# ---------------------------------------------------------------------------
# Fixed categorical order (validated palette): one color per arm, never cycled
ARM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_mae(metrics, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    n_arms = len(ARM_NAMES)
    width = 0.8 / n_arms
    x = np.arange(len(ANGLE_NAMES))
    for i, arm in enumerate(ARM_NAMES):
        vals = [metrics[arm][a]["mae"] for a in ANGLE_NAMES]
        sds = [metrics[arm][a]["sd"] for a in ANGLE_NAMES]
        ax.bar(x - 0.4 + width * (i + 0.5), vals, width * 0.9, yerr=sds,
               color=ARM_COLORS[i], label=arm, error_kw={"elinewidth": 1, "ecolor": MUTED})
    ax.set_xticks(x, ANGLE_NAMES)
    ax.set_ylabel("MAE vs ground truth (degrees)", color=INK)
    ax.set_title("Per-angle error by model arm (bars = MAE, whiskers = SD)", color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_gt_vs_pred(per_image, scatter_arms, out_png):
    """One panel per angle: GT on x, prediction on y, diagonal = perfect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=150)
    for ax, a in zip(axes.flat, ANGLE_NAMES):
        lo, hi = 0, 1
        for arm in scatter_arms:
            g = np.array([r["gt"][a] for r in per_image])
            p = np.array([r["pred"][arm][a] for r in per_image])
            ok = ~np.isnan(g) & ~np.isnan(p)
            if ok.any():
                hi = max(hi, g[ok].max(), p[ok].max())
            ax.scatter(g[ok], p[ok], s=30, color=ARM_COLORS[ARM_NAMES.index(arm)],
                       edgecolors="white", linewidths=0.6, label=arm, alpha=0.9)
        hi = hi * 1.05
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(a, color=INK, loc="left", fontsize=10)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("ground truth (°)", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted (°)", color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Ground truth vs predicted angle (dashed line = perfect)", y=0.99,
                 color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, ncol=len(scatter_arms), frameon=False, loc="upper left",
               bbox_to_anchor=(0.005, 0.955), markerscale=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _clean(x):
    """NaN -> None (JSON), recursively; numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def to_json(rec):
    return _clean(rec)


def from_json(rec):
    """Angles come back as None -> NaN; RULA dicts keep None for missing segments."""
    nan = lambda d: {a: (np.nan if d.get(a) is None else d[a]) for a in ANGLE_NAMES}
    rec["gt"] = nan(rec["gt"])
    rec["pred"] = {arm: nan(v) for arm, v in rec["pred"].items()}
    return rec


def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/panoptic", help="folder with images/ and gt_angles.csv")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--every", type=int, default=1, help="use every Nth image (3 = one third)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mb-window", type=int, default=27,
                    help="MotionBERT window in single-image mode (report this value)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results and start over")
    ap.add_argument("--scatter-arms", nargs="+",
                    default=["RTMPose 2D", "RTMPose → MotionBERT", "MediaPipe 3D"],
                    help="max 3 arms in the GT-vs-predicted figure (stays readable)")
    args = ap.parse_args()

    data = Path(args.data)
    gt_all = load_gt(data / "gt_angles.csv")
    images = sorted(p for p in (data / "images").glob("*.jpg")
                    if p.name in gt_all and not p.stem.endswith("_compare"))
    images = images[:: max(1, args.every)]
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images with GT found in {data / 'images'}")

    out = Path("logs") / "eval" / data.name
    out.mkdir(parents=True, exist_ok=True)

    # ---- cache: one JSON line per finished image, keyed by the MotionBERT window
    cache_path = out / f"cache_mb{args.mb_window}.jsonl"
    if args.fresh:
        cache_path.unlink(missing_ok=True)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = from_json(json.loads(line))
                done[rec["image"]] = rec
    todo = [p for p in images if p.name not in done]
    print(f"Images: {len(images)}  |  cached: {len(images) - len(todo)}  |  to run: {len(todo)}"
          f"  |  MotionBERT window: {args.mb_window}")

    if todo:
        comp = Comparator(device=args.device, motionbert_window=args.mb_window)
        t0 = time.perf_counter()
        with open(cache_path, "a", encoding="utf-8") as cache:
            for i, img_path in enumerate(todo, 1):
                img = cv2.imread(str(img_path))
                gt = gt_all[img_path.name]
                res = comp.run(img, gt)
                rec = {"image": img_path.name, "gt": gt, "gt_rula": res["gt_rula"],
                       "pred": {}, "rula": {}}
                for arm in ARM_NAMES:
                    r = res["arms"][arm]
                    rec["pred"][arm] = r["angles"] if r else {a: np.nan for a in ANGLE_NAMES}
                    rec["rula"][arm] = r["rula"] if r else None
                done[img_path.name] = rec
                cache.write(json.dumps(to_json(rec)) + "\n")
                cache.flush()
                el = time.perf_counter() - t0
                print(f"\r  {i}/{len(todo)}  ({el / i:.1f} s/image, ~{el / i * (len(todo) - i):.0f} s left)   ",
                      end="", flush=True)
        print()
        comp.close()

    per_image = [done[p.name] for p in images]
    long_rows = []
    for rec in per_image:
        for arm in ARM_NAMES:
            r = rec["rula"][arm]
            for a in ANGLE_NAMES:
                p, g = rec["pred"][arm][a], rec["gt"][a]
                long_rows.append({
                    "image": rec["image"], "arm": arm, "angle": a,
                    "gt": fmt(g, 2), "pred": fmt(p, 2), "abs_error": fmt(abs(p - g), 2),
                    "gt_rula_level": rec["gt_rula"]["level"],
                    "pred_rula_level": r["level"] if r else "",
                })

    # ---- angle metrics
    metrics = {arm: {} for arm in ARM_NAMES}
    for arm in ARM_NAMES:
        for a in ANGLE_NAMES:
            g = np.array([r["gt"][a] for r in per_image], dtype=float)
            p = np.array([r["pred"][arm][a] for r in per_image], dtype=float)
            metrics[arm][a] = angle_metrics(p, g)

    # ---- RULA agreement
    rula_rows, conf_rows = [], []
    for arm in ARM_NAMES:
        row = {"arm": arm}
        for seg in SEGMENTS:
            pairs = [(r["gt_rula"]["segments"][seg], r["rula"][arm]["segments"][seg])
                     for r in per_image if r["rula"][arm] is not None]
            pairs = [(g, p) for g, p in pairs if g is not None and p is not None]
            row[seg] = 100.0 * np.mean([g == p for g, p in pairs]) if pairs else np.nan
        lv = [(r["gt_rula"]["level"], r["rula"][arm]["level"])
              for r in per_image if r["rula"][arm] is not None]
        gl, pl = [g for g, _ in lv], [p for _, p in lv]
        row["action_level"] = 100.0 * np.mean(np.array(gl) == np.array(pl)) if lv else np.nan
        row["kappa"] = cohen_kappa(gl, pl)
        rula_rows.append(row)
        cm = np.zeros((4, 4), dtype=int)
        for g, p in lv:
            cm[g - 1, p - 1] += 1
        for gi in range(4):
            for pi in range(4):
                conf_rows.append({"arm": arm, "gt_level": gi + 1, "pred_level": pi + 1,
                                  "count": int(cm[gi, pi])})

    # ---- write CSVs
    def write(name, rows):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("per_image.csv", long_rows)
    write("metrics_angles.csv", [
        {"arm": arm, "angle": a, **{k: (round(v, 2) if isinstance(v, float) else v)
                                    for k, v in metrics[arm][a].items()}}
        for arm in ARM_NAMES for a in ANGLE_NAMES])
    write("rula_agreement.csv", [{k: (round(v, 2) if isinstance(v, float) else v)
                                  for k, v in r.items()} for r in rula_rows])
    write("rula_confusion.csv", conf_rows)

    # ---- figures
    try:
        plot_mae(metrics, out / "mae_by_angle.png")
        plot_gt_vs_pred(per_image, args.scatter_arms[:3], out / "gt_vs_pred.png")
        figs = "mae_by_angle.png, gt_vs_pred.png"
    except ImportError:
        figs = "skipped (pip install matplotlib)"

    # ---- print summary
    print(f"\nDataset: {data}  |  images: {len(images)}")
    print("\nMAE ± SD (degrees), per angle — lower is better")
    print(f"{'Arm':26s}" + "".join(f"{a:>14s}" for a in ANGLE_NAMES))
    for arm in ARM_NAMES:
        cells = []
        for a in ANGLE_NAMES:
            m = metrics[arm][a]
            cells.append(f"{fmt(m['mae'])}±{fmt(m['sd'])}" if m["n"] else "—")
        print(f"{arm:26s}" + "".join(f"{c:>14s}" for c in cells))

    print("\nBias (mean signed error, degrees) — negative = under-reads the angle")
    print(f"{'Arm':26s}" + "".join(f"{a:>14s}" for a in ANGLE_NAMES))
    for arm in ARM_NAMES:
        print(f"{arm:26s}" + "".join(f"{fmt(metrics[arm][a]['bias']):>14s}" for a in ANGLE_NAMES))

    print("\nCoverage % (frames where the arm produced the angle)")
    print(f"{'Arm':26s}" + "".join(f"{a:>14s}" for a in ANGLE_NAMES))
    for arm in ARM_NAMES:
        print(f"{arm:26s}" + "".join(f"{fmt(metrics[arm][a]['coverage_pct'], 0):>14s}"
                                     for a in ANGLE_NAMES))

    print("\nRULA agreement with GT (% of frames with the same score)")
    print(f"{'Arm':26s}" + "".join(f"{s:>12s}" for s in SEGMENTS) + f"{'level':>8s}{'kappa':>8s}")
    for r in rula_rows:
        print(f"{r['arm']:26s}" + "".join(f"{fmt(r[s], 0):>12s}" for s in SEGMENTS)
              + f"{fmt(r['action_level'], 0):>8s}{fmt(r['kappa'], 2):>8s}")

    print(f"\nSaved to {out}: per_image.csv, metrics_angles.csv, rula_agreement.csv, "
          f"rula_confusion.csv, figures: {figs}")


if __name__ == "__main__":
    main()
