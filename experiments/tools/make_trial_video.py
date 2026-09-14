"""Stitch recorded eval trials into one fast-forward mp4, labelled with the trial's task and outcome.

    python experiments/tools/make_trial_video.py --name pi05full_pair --speed 10 --out experiments/results/01_model_comparison/pi05full_pair_10x.mp4
    python experiments/tools/make_trial_video.py --name pi05full_pair --trials 9,10,12,15,16 --speed 5   # just the successes

Reads experiments/results/<experiment>/<name>.csv for the labels and experiments/results/trials/<name>_NN.mp4 (written by
sync_rollout.py --record: overhead | wrist side by side, 1280x480, 30 s of motion per trial, think pauses cut).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="pass name, e.g. pi05_pair")
ap.add_argument("--trials", default="", help='comma list of trial positions to include (default all)')
ap.add_argument("--speed", type=int, default=10)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default="")
ap.add_argument("--thinking", action="store_true", help="llm trials: hold the frame during each call's logged thinking time (true Nx throughout, from <name>.calls.jsonl)")
args = ap.parse_args()

RES = ROOT.parent / "results"
hits = sorted(RES.glob(f"*/{args.name}.csv")) + sorted(RES.glob(f"{args.name}.csv"))
if not hits:
    sys.exit(f"[video] no {args.name}.csv under {RES}")
rows = list(csv.DictReader(open(hits[0])))
think_log = None
if args.thinking:
    import json
    think_log = [json.loads(l) for l in open(hits[0].with_name(f"{args.name}.calls.jsonl")) if l.strip()]
want = {int(t) for t in args.trials.split(",") if t.strip()} if args.trials else None
out = Path(args.out or hits[0].parent / f"{args.name}_{args.speed}x.mp4")
tmp = out.with_suffix(".raw.mp4")
writer = None
n = 0
for r in rows:
    pos = int(r["position"])
    if want is not None and pos not in want:
        continue
    src = RES / "trials" / f"{args.name}_{pos:02d}.mp4"
    if not src.exists():
        print(f"[video] missing {src.name}, skipped", flush=True)
        continue
    PLAIN = {"touch_no_grip": "touched, no grip", "no_reach": "never reached", "drop": "grasped, dropped", "wrong_bowl": "wrong bowl",
             "no_move": "no move", "timeout": "ran out of time", "collision": "collision", "other": "other"}
    outcome = "SUCCESS" if r["success"] == "1" else f"fail: {PLAIN.get(r['failure_type'], r['failure_type'])}"
    label = f"trial {pos:2d}  |  {r['task']}  |  {outcome}  |  {args.speed}x"
    cap = cv2.VideoCapture(str(src))
    holds = {}  # captured-frame index -> (seconds to hold, label) ; thinking happens BEFORE that frame's motion
    if think_log is not None:
        rec = think_log[pos - 1]
        n_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        moves = [c for c in rec["log"] if "done" not in c]
        total_move = sum(c.get("move_s", 0) for c in moves) or 1.0
        cap_fps = n_cap / total_move  # the recorder wrote one frame per loop tick (~21.5/s), not 30
        real_speed = args.speed * args.fps / cap_fps  # what "every Nth frame at 30 fps" means in real time
        acc = 0.0
        for c in rec["log"]:
            i = min(int(round(acc / total_move * n_cap)), max(n_cap - 1, 0))
            holds[i] = holds.get(i, [0.0, ""])
            holds[i][0] += c.get("think_s", 0) / real_speed
            holds[i][1] = f"thinking {c.get('think_s', 0):.1f} s" + ("  (done)" if "done" in c else "")
            acc += c.get("move_s", 0)
        label = f"trial {pos:2d}  |  {r['task']}  |  {outcome}  |  {args.speed * args.fps / 21.5:.0f}x speed including inference pauses"  # nominal: recorder ran ~21.5 fps
    k = 0
    last = None
    def stamp(frame, extra=""):
        # two-line bar: trial / task / outcome, then speed (and the thinking clock on held frames)
        col = (120, 255, 120) if r["success"] == "1" else (255, 255, 255)
        cv2.rectangle(frame, (0, frame.shape[0] - 70), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        top, bottom = label.rsplit("  |  ", 1)
        cv2.putText(frame, top, (12, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 1)
        cv2.putText(frame, bottom + ("      " + extra if extra else ""), (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        return frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if k in holds:
            hold_s, txt = holds[k]
            still = stamp((last if last is not None else frame).copy(), txt)
            for _ in range(int(round(hold_s * args.fps))):
                writer.write(still)
                n += 1
        last = frame
        if k % args.speed == 0:
            writer.write(stamp(frame))
            n += 1
        k += 1
    cap.release()
    print(f"[video] trial {pos} ({outcome}) -> {n} frames so far", flush=True)
if writer is None:
    sys.exit("[video] nothing written")
writer.release()
ffmpeg = Path(sys.executable).parent / "ffmpeg"
subprocess.run([str(ffmpeg), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                "-movflags", "+faststart", str(out)], check=True)
tmp.unlink()
print(f"[video] wrote {out}: {n} frames = {n / args.fps:.0f} s at {args.fps} fps")
