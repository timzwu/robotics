"""Charts for the sim-to-real experiment (`sim/04_real_evals/`), reusing the post #0 chart code.
    python sim/tools/plot_sim_results.py
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "tools"))
import plot_results as pr

OUT = ROOT / "sim" / "04_real_evals"
PATHS = {"smolvla100_pair": ROOT / "experiments/results/01_model_comparison/smolvla100_pair.csv",
         "smolvla_sim_pair": OUT / "smolvla_sim_pair.csv", "smolvla_simreal_pair": OUT / "smolvla_simreal_pair.csv",
         # the same three checkpoints evaluated in the simulator on the same 20 stickers
         "smolvla_real_simeval": OUT / "smolvla_real_simeval.csv", "smolvla_sim_simeval": OUT / "smolvla_sim_simeval.csv",
         "smolvla_simreal_simeval": OUT / "smolvla_simreal_simeval.csv",
         # second round (Sept 16): the v2 sim demonstrations (05_sim_demos_v2) and the 75/25 split
         "smolvla_sim_v2_pair": OUT / "smolvla_sim_v2_pair.csv", "smolvla_simreal_v2_pair": OUT / "smolvla_simreal_v2_pair.csv",
         "smolvla_simreal_75_25_pair": OUT / "smolvla_simreal_75_25_pair.csv", "smolvla_sim_v2_simeval": OUT / "smolvla_sim_v2_simeval.csv",
         "smolvla_simreal_v2_simeval": OUT / "smolvla_simreal_v2_simeval.csv"}
SIM_OF = {"smolvla100_pair": "smolvla_real_simeval", "smolvla_sim_pair": "smolvla_sim_simeval", "smolvla_simreal_pair": "smolvla_simreal_simeval",
          "smolvla_sim_v2_pair": "smolvla_sim_v2_simeval", "smolvla_simreal_v2_pair": "smolvla_simreal_v2_simeval"}
pr.load = lambda name: list(csv.DictReader(open(PATHS[name]))) if name in PATHS and PATHS[name].exists() else []
ALL_CONDS = [("smolvla100_pair", "A · real\n100 teleop demos", "#2b57a5"), ("smolvla_sim_pair", "C · sim only\n100 scripted sim demos", "#c98a2b"),
             ("smolvla_simreal_pair", "D · sim + real\n100 sim + 100 real", "#5a9e6f"),
             ("smolvla_sim_v2_pair", "C2 · sim v2 only\n100 sim demos, 2nd recipe", "#e0b45a"),
             ("smolvla_simreal_v2_pair", "D2 · sim v2 + real\n100 sim v2 + 100 real", "#8fc79a"),
             ("smolvla_simreal_75_25_pair", "75 / 25 real : sim\n100 real + 33 sim", "#7a6fb0")]
# round 1 (A, C, D) and round 2 (the v2 demonstrations and the 75/25 split, with A and D as the reference bars) are
# charted separately: they are separate sections of the write-up. Set by `set_round()` below.
ROUND1 = ["smolvla100_pair", "smolvla_sim_pair", "smolvla_simreal_pair"]
ROUND2 = ["smolvla100_pair", "smolvla_simreal_pair", "smolvla_sim_v2_pair", "smolvla_simreal_v2_pair", "smolvla_simreal_75_25_pair"]
CONDS, COLORS = [], []


def set_round(names):
    global CONDS, COLORS
    CONDS = [(n, l) for n, l, _ in ALL_CONDS if n in names and len(pr.load(n)) >= 20]
    COLORS = [c for n, _, c in ALL_CONDS if n in names and len(pr.load(n)) >= 20]


def success_chart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.4 + 2.1 * len(CONDS), 5.2), facecolor=pr.BG); ax.set_facecolor(pr.BG)
    for i, ((name, label), col) in enumerate(zip(CONDS, COLORS)):
        rows = pr.load(name); k = sum(int(r["success"]) for r in rows); n = len(rows); lo, hi = pr.wilson(k, n)
        ax.bar(i, k / n, color=col, width=0.62)
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="#333", capsize=6, lw=1.4)
        ax.text(i, hi + 0.03, f"{k} / {n}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(CONDS)), [l for _, l in CONDS], fontsize=12)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.set_ylabel("success rate, 20 trials on the real arm", fontsize=12)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.set_title("SmolVLA fine-tuned on real, sim, or both demonstrations: success on the real arm", fontsize=14, pad=12)
    ax.text(0.5, -0.2, "Same model, recipe (20k steps × 64), task pair, 20 sticker positions and 30 s budget; bars with 95% Wilson intervals.",
            transform=ax.transAxes, ha="center", fontsize=10, color="#555")
    fig.tight_layout(); fig.savefig(out, dpi=110, facecolor=pr.BG); plt.close(fig)


def combined_chart(out: Path) -> None:
    """Per condition: success in the simulator and on the real arm, 20 trials each."""
    fig, ax = plt.subplots(figsize=(2.4 + 2.4 * len(CONDS), 5.4), facecolor=pr.BG); ax.set_facecolor(pr.BG)
    w = 0.36
    for i, ((name, label), col) in enumerate(zip(CONDS, COLORS)):
        for j, (run, hatch, lab) in enumerate(((SIM_OF.get(name, ""), "//", "in the simulator"), (name, None, "on the real arm"))):
            rows = pr.load(run)
            if not rows:
                continue
            k = sum(int(r["success"]) for r in rows); n = len(rows); lo, hi = pr.wilson(k, n); x = i + (j - 0.5) * (w + 0.04)
            ax.bar(x, k / n, width=w, color=col if j else "none", edgecolor=col, hatch=hatch, linewidth=1.6, label=lab if i == 0 else None)
            ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], color="#333", capsize=5, lw=1.3)
            ax.text(x, hi + 0.03, f"{k}/{n}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(CONDS)), [l for _, l in CONDS], fontsize=12)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.set_ylabel("success rate, 20 trials", fontsize=12)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.set_title("Where each policy works: the simulator and the real arm, same 20 sticker positions", fontsize=13.5, pad=12)
    ax.text(0.5, -0.2, "Same model and recipe; in the simulator the block sits on each sticker and the same stop-and-go loop drives the arm.\n95% Wilson intervals.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#555")
    fig.tight_layout(); fig.savefig(out, dpi=110, facecolor=pr.BG); plt.close(fig)


def stage_chart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 1.8 + 0.8 * len(CONDS)), facecolor=pr.BG); ax.set_facecolor(pr.BG)
    for i, (name, label) in enumerate(CONDS):
        st = pr.stages(pr.load(name)); left = 0.0
        for s in range(4):
            share = st.count(s) / len(st)
            ax.barh(i, share, left=left, color=pr.STAGE_COLORS[s], edgecolor=pr.BG, height=0.6); left += share
    ax.set_yticks(range(len(CONDS)), [l.replace("\n", ", ") for _, l in CONDS], fontsize=12); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.set_xlabel("share of the 20 trials", fontsize=12)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pr.STAGE_COLORS]
    ax.legend(handles, pr.STAGE_NAMES, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=11)
    ax.set_title("How far each trial got", fontsize=14, pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=110, facecolor=pr.BG); plt.close(fig)


if __name__ == "__main__":
    for names, suffix in ((ROUND1, ""), (ROUND2, "_v2")):
        set_round(names)
        success_chart(OUT / f"success_by_condition{suffix}.png")
        if all(PATHS[SIM_OF[n]].exists() for n, _ in CONDS if n in SIM_OF):
            combined_chart(OUT / f"sim_vs_real_by_condition{suffix}.png")
        stage_chart(OUT / f"stages_by_condition{suffix}.png")
        pr.geography([(n, l.replace("\n", ": "), False) for n, l in CONDS], OUT / f"failure_geography_sim2real{suffix}.png",
                     "SmolVLA on the real arm after fine-tuning on real, sim, or both demonstrations; same 20 stickers, red→left on odd stickers and blue→right on even. "
                     "Smoothed between stickers (Gaussian, σ≈42 px). Overhead-camera frame, arm base at the bottom, so the operator's left bowl is on the right.")
    print("charts written to", OUT)
