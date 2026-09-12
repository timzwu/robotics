"""Regenerate the result figures from the per-pass CSVs in experiments/results/.

    python experiments/tools/plot_results.py            # writes progress_by_model.png, model_comparison.png, failure_geography_pi05.png into results/01_model_comparison/

Every figure reads the CSVs, so adding a pass = adding a row to RUNS below and re-running. Stages: 0 never reached
the block, 1 touched it but never closed on it, 2 grasped it then dropped / collided / ran out of time, 3 success.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES / "01_model_comparison"
BG = "#fbfbf9"
STAGE = {"no_reach": 0, "no_move": 0, "touch_no_grip": 1, "timeout": 1, "drop": 2, "collision": 2, "wrong_bowl": 2, "other": 2}
# Trials whose notes place them in a different stage than their failure bucket (read the CSV notes before adding one).
STAGE_OVERRIDES = {("smolvla_wristonly_pair", 3): 2}  # "grasps at last chunk" then timed out
STAGE_COLORS = ["#e8e8e3", "#c6d4ea", "#7fa3d6", "#2b57a5"]
STAGE_NAMES = ["never reached", "reached, no grip", "grasped, lost it", "success"]

# (csv name, label) in display order. All are the same 20 stickers on the trained pair unless the label says otherwise.
RUNS = [
    ("act100_pair", "ACT, 100 ep, 3 passes"),
    ("act100k_pair", "ACT, 100 ep, 15 passes"),
    ("smolvla_n10_pair", "SmolVLA, 10 ep"),
    ("smolvla_n25_pair", "SmolVLA, 25 ep"),
    ("smolvla_n50_pair", "SmolVLA, 50 ep"),
    ("smolvla100_pair", "SmolVLA, 100 ep"),
    ("smolvla_toponly_pair", "SmolVLA, 100 ep, overhead only"),
    ("smolvla_wristonly_pair", "SmolVLA, 100 ep, wrist only"),
    ("pi05_pair", "π0.5, 100 ep, 24 passes"),
]
MATCHED = [  # the model comparison at each library's default recipe / matched passes, same 100 episodes
    ("act100_pair", "ACT\n3 passes"),
    ("act100k_pair", "ACT\n15 passes"),
    ("smolvla100_pair", "SmolVLA\n24 passes"),
    ("pi05_pair", "π0.5\n24 passes"),
]


def load(name: str) -> list[dict]:
    """Find <name>.csv in any experiment folder under results/."""
    hits = sorted(RES.glob(f"*/{name}.csv")) + sorted(RES.glob(f"{name}.csv"))
    if not hits:
        raise SystemExit(f"no {name}.csv under {RES}")
    return list(csv.DictReader(open(hits[0])))


def stages(rows: list[dict]) -> list[int]:
    return [3 if r["success"] == "1" else STAGE_OVERRIDES.get((r["name"], int(r["position"])), STAGE[r["failure_type"]]) for r in rows]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def progress_chart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.3, 0.66 * len(RUNS) + 2.2), facecolor=BG)
    ax.set_facecolor(BG)
    for i, (name, label) in enumerate(RUNS):
        st = stages(load(name))
        left = 0.0
        for s in range(4):
            share = st.count(s) / len(st)
            ax.barh(i, share, left=left, color=STAGE_COLORS[s], edgecolor=BG, height=0.6)
            left += share
        ax.text(1.02, i, f"mean progress {sum(st) / len(st):.2f} / 3", va="center", fontsize=11)
    ax.set_yticks(range(len(RUNS)), [l for _, l in RUNS], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.set_xlabel("share of the 20 trials", fontsize=13)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS]
    ax.legend(handles, STAGE_NAMES, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12)
    ax.set_title("How far each model got: trials scored by stage reached", fontsize=15, pad=14)
    fig.text(0.01, 0.01, "Stages: 0 never reached the block, 1 touched it but never closed on it, 2 grasped it then dropped / collided / "
             "ran out of time, 3 placed it in the bowl. Same 20 trials per model.", fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def comparison_chart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=BG)
    ax.set_facecolor(BG)
    xs = np.arange(len(MATCHED))
    for x, (name, label) in zip(xs, MATCHED):
        rows = load(name)
        k = sum(r["success"] == "1" for r in rows)
        lo, hi = wilson(k, len(rows))
        p = k / len(rows)
        ax.bar(x, p, width=0.6, color="#2b57a5" if "π0.5" not in label else "#7fa3d6", zorder=2)
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#333", elinewidth=1.4, capsize=6, zorder=3)
        ax.text(x, hi + 0.03, f"{k}/20", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs, [l for _, l in MATCHED], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_ylabel("success on the trained pair (20 trials)", fontsize=12)
    ax.grid(axis="y", color="#ddd", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Same 100 demonstrations, same 20 trials: three model families", fontsize=14, pad=12)
    fig.text(0.01, 0.01, "Bars: successes out of 20; whiskers: 95% Wilson interval. ACT from scratch; SmolVLA and π0.5 fine-tuned\n"
             "(vision frozen, action expert trained) at 24 passes over the data.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def geography(panels: list[tuple[str, str, bool]], out: Path, caption: str, sigma: float = 42.0) -> None:
    """Smoothed success field per pass on the sticker layout. panels: (csv, title, star_untrained_combos)."""
    px = {int(k): v for k, v in json.load(open(RES / "sticker_map.json"))["pixels"].items()}
    xs, ys = np.array([v[0] for v in px.values()]), np.array([v[1] for v in px.values()])
    x0, x1, y0, y1 = xs.min() - 40, xs.max() + 40, ys.min() - 40, ys.max() + 40
    gx, gy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels) + 1.2, 7), facecolor=BG)
    axes = np.atleast_1d(axes)
    for ax, (name, title, star) in zip(axes, panels):
        rows = load(name)
        ax.set_facecolor(BG)
        succ = {int(r["position"]): r["success"] == "1" for r in rows}
        combo = {int(r["position"]): (r["color"], r["bowl"]) for r in rows}
        field = np.zeros_like(gx)
        wsum = np.zeros_like(gx)
        for s, (x, y) in px.items():
            w = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma**2))
            field += w * (1.0 if succ.get(s) else 0.0)
            wsum += w
        ax.imshow(field / np.maximum(wsum, 1e-9), extent=(x0, x1, y1, y0), cmap="RdBu", vmin=0, vmax=1, alpha=0.9, interpolation="bilinear")
        for s, (x, y) in px.items():
            ok = succ.get(s)
            untrained = star and combo.get(s) in (("red", "right"), ("blue", "left"))
            ax.scatter(x, y, s=260, facecolor="white" if ok else "#f4f4f4", edgecolor="#222" if not ok else "#2b57a5",
                       linewidth=1.6 if ok else 1.2, zorder=3, marker="s" if untrained else "o")
            ax.text(x, y, str(s), ha="center", va="center", fontsize=8, zorder=4)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#999", linewidth=1.2))
        for bx, lab, col in ((x0 - 45, "L bowl", "#e9dcd8"), (x1 + 45, "R bowl", "#dbe6dc")):
            ax.add_patch(matplotlib.patches.Ellipse((bx, (y0 + y1) / 2), 70, 150, color=col, zorder=1))
            ax.text(bx, (y0 + y1) / 2, lab, ha="center", va="center", fontsize=10, color="#666")
        ax.add_patch(plt.Rectangle(((x0 + x1) / 2 - 50, y1 + 20), 100, 50, color="#e5e5e0", zorder=1))
        ax.text((x0 + x1) / 2, y1 + 45, "arm base", ha="center", va="center", fontsize=10, color="#666")
        ax.set_xlim(x0 - 95, x1 + 95)
        ax.set_ylim(y1 + 85, y0 - 40)
        ax.set_aspect("equal")
        ax.axis("off")
        k = sum(succ.values())
        ax.set_title(f"{title} — {k}/20", fontsize=13, pad=10)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["failed", "succeeded"])
    cb.outline.set_visible(False)
    fig.text(0.01, 0.02, caption, fontsize=9.5, color="#555")
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    progress_chart(OUT / "progress_by_model.png")
    comparison_chart(OUT / "model_comparison.png")
    geography(
        [("smolvla100_pair", "SmolVLA, 100 ep: trained pair", False),
         ("pi05_pair", "π0.5, 100 ep: trained pair", False),
         ("pi05_all4", "π0.5, 100 ep: all four combos", True)],
        OUT / "failure_geography_pi05.png",
        "π0.5 vs SmolVLA on the same 100 episodes at matched passes; smoothed between stickers (Gaussian, σ≈42 px). Odd stickers "
        "red→left, even blue→right in the pair passes; squares = untrained combos in the four-task pass. Left/right from the operator seat.",
    )
    print("wrote progress_by_model.png, model_comparison.png, failure_geography_pi05.png")
