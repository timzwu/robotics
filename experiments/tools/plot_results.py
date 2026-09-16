"""Regenerate the result figures from the per-pass CSVs in experiments/results/.

    python experiments/tools/plot_results.py            # writes 01_model_comparison/{progress_by_model,model_comparison,failure_geography_pi05full}.png, 02_data_scaling/{scaling,failure_geography_sweep}.png, 04_camera_ablation/failure_geography_ablation.png, 06_general_llm/failure_geography_llm.png

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
STAGE_OVERRIDES = {("smolvla_wristonly_pair", 3): 2,   # "grasps at last chunk" then timed out
                   ("smolvla_simreal_pair", 2): 2,     # gripped, dropped, was recovering toward the bowl when the 30 s ran out
                   ("smolvla_simreal_75_25_pair", 3): 2}  # "barely picks up but out of motion time"
STAGE_COLORS = ["#e8e8e3", "#c6d4ea", "#7fa3d6", "#2b57a5"]
STAGE_NAMES = ["no contact with block", "contact without grip", "grip without successful placement", "success"]

# (csv name, label) in display order. All are the same 20 stickers on the trained pair unless the label says otherwise.
RUNS = [
    ("act100k_pair", "ACT, 100 demos, 15 passes"),
    ("smolvla_n10_pair", "SmolVLA, 10 demos"),
    ("smolvla_n25_pair", "SmolVLA, 25 demos"),
    ("smolvla_n50_pair", "SmolVLA, 50 demos"),
    ("smolvla100_pair", "SmolVLA, 100 demos, 24 passes"),
    ("smolvla_toponly_pair", "SmolVLA, 100 demos, overhead only"),
    ("smolvla_wristonly_pair", "SmolVLA, 100 demos, wrist only"),
    ("pi05full_pair", "π0.5, 100 demos, 18 passes"),
    ("astra_pair", "GPT-6 Astra"),
]
MATCHED = [  # the model comparison at each library's own recipe (ACT 15, SmolVLA 24, π0.5 18 passes), same 100 episodes
    ("act100k_pair", "ACT"),
    ("smolvla100_pair", "SmolVLA"),
    ("pi05full_pair", "π0.5"),
    ("astra_pair", "GPT-6 Astra, run 2\n(no task-specific fine-tuning)"),
]


def load(name: str) -> list[dict]:
    """Find <name>.csv in any experiment folder under results/."""
    hits = sorted(RES.glob(f"*/{name}.csv")) + sorted(RES.glob(f"{name}.csv"))
    if not hits:
        return []
    return list(csv.DictReader(open(hits[0])))


def stages(rows: list[dict]) -> list[int]:
    return [3 if r["success"] == "1" else STAGE_OVERRIDES.get((r["name"], int(r["position"])), STAGE[r["failure_type"]]) for r in rows]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


HEADLINE = {"act100k_pair", "smolvla100_pair", "pi05full_pair", "astra_pair"}  # the four models of the top-level comparison, bold in Figure 2


def progress_chart(out: Path) -> None:
    runs = [(n, l) for n, l in RUNS if load(n)]
    fig, ax = plt.subplots(figsize=(14.3, 0.66 * len(runs) + 2.2), facecolor=BG)
    ax.set_facecolor(BG)
    runs = [(n, l) for n, l in RUNS if load(n)]
    for i, (name, label) in enumerate(runs):
        st = stages(load(name))
        left = 0.0
        for s in range(4):
            share = st.count(s) / len(st)
            ax.barh(i, share, left=left, color=STAGE_COLORS[s], edgecolor=BG, height=0.6)
            left += share
    ax.set_yticks(range(len(runs)), [l for _, l in runs], fontsize=12)
    for lab, (name, _) in zip(ax.get_yticklabels(), runs):
        if name in HEADLINE:
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.set_xlabel("share of the 20 trials", fontsize=13)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS]
    ax.legend(handles, STAGE_NAMES, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12)
    ax.set_title("Categorical model performance on trained task pair", fontsize=15, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def comparison_chart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    matched = [(n, l) for n, l in MATCHED if load(n)]
    xs = np.arange(len(matched))
    for x, (name, label) in zip(xs, matched):
        rows = load(name)
        k = sum(r["success"] == "1" for r in rows)
        lo, hi = wilson(k, len(rows))
        p = k / len(rows)
        ax.bar(x, p, width=0.6, color="#b3672b" if "Astra" in label else "#2b57a5", zorder=2)
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#333", elinewidth=1.4, capsize=6, zorder=3)
        ax.text(x, hi + 0.03, f"{k}/20", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs, [l for _, l in matched], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_ylabel("success rate (%)", fontsize=12)
    ax.grid(axis="y", color="#ddd", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Pick-and-place task success rate across 20 trials on trained task pair", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


SWEEP = [("smolvla_n10_pair", 10, 0), ("smolvla_n25_pair", 25, 1), ("smolvla_n50_pair", 50, 4), ("smolvla100_pair", 100, 8)]  # run, demonstrations, my blind guess (see 02_data_scaling/README.md)


def scaling_chart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor=BG)
    ax.set_facecolor(BG)
    xs, ps, los, his, ks = [], [], [], [], []
    for name, n, _ in SWEEP:
        rows = load(name); k = sum(r["success"] == "1" for r in rows); lo, hi = wilson(k, len(rows))
        xs.append(n); ps.append(k / len(rows)); los.append(lo); his.append(hi); ks.append(k)
    ax.fill_between(xs, los, his, color="#2b57a5", alpha=0.12, label="95% Wilson interval (n=20)", zorder=1)
    ax.plot(xs, ps, "-o", color="#2b57a5", lw=2.2, ms=8, label="SmolVLA, measured", zorder=3)
    gx = [n for _, n, g in SWEEP if g is not None]; gy = [g / 20 for _, _, g in SWEEP if g is not None]
    ax.scatter(gx, gy, s=140, facecolors="none", edgecolors="#888", linewidths=2, label="prediction before evaluation", zorder=4)
    for x, p, k in zip(xs, ps, ks):
        ax.annotate(f"{k}/20", (x, p), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=12)
    ax.set_xscale("log"); ax.set_xticks(xs, [f"{n}\n(12 / 13 across tasks)" if n == 25 else f"{n}\n({n // 2} per task)" for n in xs], fontsize=12); ax.minorticks_off()
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_xlabel("demonstrations", fontsize=13); ax.set_ylabel("success rate", fontsize=12)
    ax.grid(axis="y", color="#ddd", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_title("Success rate by number of demonstrations on the trained task pair", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def geography(panels: list[tuple[str, str, bool]], out: Path, caption: str, sigma: float = 42.0, ncols: int | None = None) -> None:
    """Smoothed success field per pass on the sticker layout. panels: (csv, title, star_untrained_combos)."""
    px = {int(k): v for k, v in json.load(open(RES / "sticker_map.json"))["pixels"].items()}
    xs, ys = np.array([v[0] for v in px.values()]), np.array([v[1] for v in px.values()])
    x0, x1, y0, y1 = xs.min() - 40, xs.max() + 40, ys.min() - 40, ys.max() + 40
    gx, gy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols + 1.2, 4.7 * nrows + 0.7), facecolor=BG)
    axes = np.atleast_1d(axes).ravel()
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
            ax.text(x, y, str(s), ha="center", va="center", fontsize=8 if ncols >= 3 else 10, zorder=4)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#999", linewidth=1.2))
        for bx, lab, col in ((x0 - 62, "R bowl", "#e9dcd8"), (x1 + 62, "L bowl", "#dbe6dc")):  # overhead-camera frame: the operator's LEFT bowl (green) is on the image's RIGHT; pink bowl = operator's right
            ax.add_patch(matplotlib.patches.Circle((bx, (y0 + y1) / 2), 46, color=col, zorder=1))
            ax.text(bx, (y0 + y1) / 2, lab, ha="center", va="center", fontsize=9, color="#666")
        ax.add_patch(plt.Rectangle(((x0 + x1) / 2 - 50, y1 + 20), 100, 50, color="#e5e5e0", zorder=1))
        ax.text((x0 + x1) / 2, y1 + 45, "arm base", ha="center", va="center", fontsize=10, color="#666")
        ax.set_xlim(x0 - 115, x1 + 115)
        ax.set_ylim(y1 + 85, y0 - 40)
        ax.set_aspect("equal")
        ax.axis("off")
        k = sum(succ.values())
        ax.set_title(f"{title} — {k}/20", fontsize=13 if ncols >= 3 else 14, pad=8)
    fig.subplots_adjust(left=0.01, right=0.91, top=0.9, bottom=0.11 if nrows > 1 else 0.17, wspace=0.18, hspace=0.12)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.935, 0.3, 0.008, 0.45]))
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["failed", "succeeded"])
    cb.outline.set_visible(False)
    import textwrap
    fig.text(0.01, 0.012, "\n".join(textwrap.wrap(caption, 60 * ncols)), fontsize=9.5, color="#555", va="bottom")
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    progress_chart(OUT / "progress_by_model.png")
    comparison_chart(OUT / "model_comparison.png")
    scaling_chart(RES / "02_data_scaling" / "scaling.png")
    geography(
        [("smolvla100_pair", "SmolVLA, 100 ep: trained pair", False),
         ("pi05full_pair", "π0.5, 100 ep: trained pair", False),
         ("pi05full_all4", "π0.5, 100 ep: all four combos", True)],
        OUT / "failure_geography_pi05full.png",
        "π0.5 vs SmolVLA on the same 100 episodes, each at its authors' recipe (SmolVLA 24 passes, π0.5 18); smoothed between stickers (Gaussian, σ≈42 px). Odd stickers "
        "red→left, even blue→right in the pair passes; squares = untrained combos in the four-task pass. Left/right in the task names are from the operator's seat; the overhead camera (arm base at the bottom) mirrors them, so the left bowl is on the right of the map.",
    )
    if load("smolvla_wristonly_pair"):
        geography(
            [("smolvla_toponly_pair", "overhead camera only", False), ("smolvla_wristonly_pair", "wrist camera only", False), ("smolvla100_pair", "both cameras", False)],
            RES / "04_camera_ablation" / "failure_geography_ablation.png",
            "SmolVLA fine-tuned on the same 100 demonstrations with one camera stream removed; same 20 trials. Smoothed between stickers (Gaussian, σ≈42 px). Overhead-camera frame, arm base at the bottom, so the operator's left bowl is on the right.",
        )
    if load("smolvla_n50_pair"):
        geography(
            [("smolvla_n10_pair", "10 demos", False), ("smolvla_n25_pair", "25 demos", False), ("smolvla_n50_pair", "50 demos", False), ("smolvla100_pair", "100 demos", False)],
            RES / "02_data_scaling" / "failure_geography_sweep.png",
            "SmolVLA at four dataset sizes, same recipe and the same 20 trials. Smoothed between stickers (Gaussian, σ≈42 px). Overhead-camera frame, arm base at the bottom, so the operator's left bowl is on the right.",
        )
    if load("astra_pair"):
      geography(
        [("smolvla100_pair", "SmolVLA, 100 demos", False), ("pi05full_pair", "π0.5, 100 demos", False),
         ("astra_pair_noroll", "GPT-6 Astra, no demos, run 1 (no roll)", False), ("astra_pair", "GPT-6 Astra, no demos, run 2 (roll + hint)", False)],
        RES / "06_general_llm" / "failure_geography_llm.png",
        "Same 20 stickers, red→left on odd stickers and blue→right on even. GPT-6 Astra drives the arm through gripper poses, given no demonstrations "
        "(run 1 without a wrist-roll parameter, run 2 with roll, a joint tool and one prompt hint); the trained models react at 30 Hz. Smoothed between stickers (Gaussian, σ≈42 px). Overhead-camera frame, arm base at the bottom, so the operator's left bowl is on the right.",
    )
    print("wrote 01/{progress_by_model,model_comparison,failure_geography_pi05full}.png, 02/{scaling,failure_geography_sweep}.png, 04/failure_geography_ablation.png, 06/failure_geography_llm.png")
