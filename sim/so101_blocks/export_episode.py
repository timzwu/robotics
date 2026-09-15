"""Export one real episode (joint states, actions, both camera streams) for replay in the simulator.

Runs on the Mac in the `lerobot` env; the simulator never needs a Hub token this way.
    python sim/so101_blocks/export_episode.py --episode 0 --out sim/02_scene/real_episodes
Writes `episode_NNN.npz` (state, action, timestamp; LeRobot units: degrees, gripper 0-100) and
`episode_NNN_{top,wrist}.mp4` (640x480, 30 fps, as recorded).
"""
import argparse, os, json
import numpy as np, cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="timzwu/so101_blocks")
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--out", default="sim/02_scene/real_episodes")
a = ap.parse_args()
ds = LeRobotDataset(a.repo, episodes=[a.episode])
os.makedirs(a.out, exist_ok=True)
names = ds.meta.features["observation.state"]["names"]
states, actions, ts = [], [], []
writers = {}
for i in range(len(ds)):
    fr = ds[i]
    states.append(fr["observation.state"].numpy()); actions.append(fr["action"].numpy()); ts.append(float(fr["timestamp"]))
    for cam in ("top", "wrist"):
        img = fr[f"observation.images.{cam}"]  # torch float CHW in [0,1]
        img = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        if cam not in writers:
            writers[cam] = cv2.VideoWriter(f"{a.out}/episode_{a.episode:03d}_{cam}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), ds.fps, (img.shape[1], img.shape[0]))
        writers[cam].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
for w in writers.values(): w.release()
task = ds.meta.tasks.index[fr["task_index"].item()] if hasattr(ds.meta.tasks, "index") else str(fr.get("task", ""))
np.savez(f"{a.out}/episode_{a.episode:03d}.npz", state=np.stack(states), action=np.stack(actions), timestamp=np.array(ts), names=np.array(names), fps=ds.fps, task=str(task))
print(f"episode {a.episode}: {len(ds)} frames at {ds.fps} fps, task {task!r}; state min {np.stack(states).min(0).round(1)} max {np.stack(states).max(0).round(1)}")
