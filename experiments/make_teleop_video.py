"""Stitch recorded teleop episodes into one fast-forward mp4 (overhead | wrist side by side).

    python experiments/make_teleop_video.py --episodes 0-49 --speed 10 --out experiments/results/teleop_50eps_10x.mp4

Reads the local LeRobot dataset, takes every `speed`-th frame of each episode, labels it, and writes H.264 via the
environment's ffmpeg. Episodes are played in the order given.
"""
import argparse, glob, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / ".cache/huggingface/lerobot/timzwu/so101_blocks"))
ap.add_argument("--episodes", default="0-49", help='"0-49" or "0-24,100-124"')
ap.add_argument("--speed", type=int, default=10)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default="experiments/results/teleop_10x.mp4")
args = ap.parse_args()
root = Path(args.root)
eps = []
for part in args.episodes.split(","):
    a, _, b = part.partition("-"); eps += list(range(int(a), int(b or a) + 1))
meta = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)))
tasks = pd.read_parquet(f"{root}/meta/tasks.parquet"); task_by_idx = {int(v): k for k, v in tasks.to_dict()["task_index"].items()}
data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)))
ep_task = data.groupby("episode_index")["task_index"].first().to_dict()
tmp = Path(args.out).with_suffix(".raw.mp4")
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 480))
caps = {}
def cap_for(cam, r):
    p = f"{root}/videos/observation.images.{cam}/chunk-{int(r[f'videos/observation.images.{cam}/chunk_index']):03d}/file-{int(r[f'videos/observation.images.{cam}/file_index']):03d}.mp4"
    if p not in caps: caps[p] = cv2.VideoCapture(p)
    return caps[p]
n = 0
for i, e in enumerate(eps):
    r = meta.iloc[e]; t0 = r["videos/observation.images.top/from_timestamp"]; t1 = r["videos/observation.images.top/to_timestamp"]
    w0 = r["videos/observation.images.wrist/from_timestamp"]
    label = f"episode {e}  |  {task_by_idx.get(int(ep_task.get(e, -1)), '')}  |  {args.speed}x"
    top, wr = cap_for("top", r), cap_for("wrist", r)
    top.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000); wr.set(cv2.CAP_PROP_POS_MSEC, w0 * 1000)
    frames = int((t1 - t0) * args.fps)
    for k in range(frames):
        ok1, f1 = top.read(); ok2, f2 = wr.read()
        if not (ok1 and ok2): break
        if k % args.speed: continue
        frame = np.hstack([f1, f2])
        cv2.rectangle(frame, (0, 450), (1280, 480), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 471), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame); n += 1
    print(f"[video] episode {e} ({i+1}/{len(eps)}) -> {n} frames so far", flush=True)
writer.release()
ffmpeg = Path(sys.executable).parent / "ffmpeg"
subprocess.run([str(ffmpeg), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", args.out], check=True)
tmp.unlink()
print(f"[video] wrote {args.out}: {n} frames = {n/args.fps:.0f} s at {args.fps} fps")
