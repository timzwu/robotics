"""Data for the joint-offset fit (Mac, `lerobot` env). Writes to sim/02_scene/real_episodes/:
  states.npz     joint trajectories (LeRobot units) and actions of every episode, straight from the parquet files
  first_NNN.jpg  first overhead frame of the red-block episodes, and grasps.json with the block pixel found in each
  contact_sheet.jpg  the first frames with the detected block marked, for checking by eye
    python sim/so101_blocks/export_grasps.py
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import find_red_block

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="timzwu/so101_blocks")
ap.add_argument("--out", default="sim/02_scene/real_episodes")
ap.add_argument("--red-episodes", default="0-24,50-74,100-124")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{a.repo}")
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))])
states, actions = {}, {}
for e, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    states[str(int(e))] = np.stack(g["observation.state"].values); actions[str(int(e))] = np.stack(g["action"].values)
np.savez(f"{a.out}/states.npz", **{f"s{k}": v for k, v in states.items()}, **{f"a{k}": v for k, v in actions.items()})
print("episodes:", len(states))
eps = [e for part in a.red_episodes.split(",") for e in range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)]
tiles, out = [], []
for e in eps:
    ds = LeRobotDataset(a.repo, episodes=[e])
    fr0 = ds[0]
    img = (fr0["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype("uint8")
    cv2.imwrite(f"{a.out}/first_{e:03d}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    c = find_red_block(img)
    task = ds.meta.tasks.index[fr0["task_index"].item()] if hasattr(ds.meta.tasks, "index") else ""
    out.append({"episode": e, "task": str(task), "frames": len(ds), "block_px": c})
    t = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    if c: cv2.circle(t, (int(c[0]), int(c[1])), 14, (255, 255, 0), 2)
    cv2.putText(t, f"{e}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    tiles.append(cv2.resize(t, (320, 240)))
    print(e, str(task)[:34], "block px", None if c is None else [round(v, 1) for v in c])
json.dump(out, open(f"{a.out}/grasps.json", "w"), indent=1)
cols = 8; rows = -(-len(tiles) // cols)
sheet = np.zeros((rows * 240, cols * 320, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols); sheet[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = t
cv2.imwrite(f"{a.out}/contact_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", f"{a.out}/grasps.json", "and contact_sheet.jpg")
