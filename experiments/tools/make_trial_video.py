"""Stitch recorded eval trials into one fast-forward mp4, labelled with the trial's task and outcome.

    python experiments/tools/make_trial_video.py --name pi05_pair --speed 10 --out experiments/results/01_model_comparison/pi05_pair_10x.mp4
    python experiments/tools/make_trial_video.py --name pi05_pair --trials 9,10,12,15,16 --speed 5   # just the successes

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
args = ap.parse_args()

RES = ROOT.parent / "results"
hits = sorted(RES.glob(f"*/{args.name}.csv")) + sorted(RES.glob(f"{args.name}.csv"))
if not hits:
    sys.exit(f"[video] no {args.name}.csv under {RES}")
rows = list(csv.DictReader(open(hits[0])))
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
    outcome = "SUCCESS" if r["success"] == "1" else f"fail: {r['failure_type']}"
    label = f"trial {pos:2d}  |  {r['task']}  |  {outcome}  |  {args.speed}x"
    cap = cv2.VideoCapture(str(src))
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % args.speed == 0:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, frame.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 255, 120) if r["success"] == "1" else (255, 255, 255), 1)
            writer.write(frame)
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
