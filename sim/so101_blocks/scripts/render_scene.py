"""Render the block-to-bowl scene from both cameras at the real rest pose, with the block on a chosen sticker.
Writes top/wrist PNGs (and side-by-side panels against real frames when given). Headless, inside the sim container:
    python /workspace/so101_blocks/scripts/render_scene.py --headless --out /out/scene --sticker 1 \
        --real-top /data/episode_000_top.mp4 --real-wrist /data/episode_000_wrist.mp4
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="So101-Blocks-v0")
parser.add_argument("--out", default="/out/scene")
parser.add_argument("--sticker", type=int, default=1, help="sticker number the red block sits on (0 = the reset event's random spot)")
parser.add_argument("--settle", type=int, default=45, help="control steps to let the scene settle before the frame is taken")
parser.add_argument("--real-top", default=None); parser.add_argument("--real-wrist", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json, os, sys, time
import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import so101_blocks.tasks  # noqa: F401  (registers the task)
from so101_blocks import rig, units
from so101_blocks.scripts.common import rgb_frame, save_png, side_by_side, read_video, apply_joint_limits



def main():
    t0 = time.time()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    rest_np = units.lerobot_to_sim([rig.REST_POSE_DEG[j] for j in units.JOINTS])
    apply_joint_limits(env, rest_np)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    print("[scene] joint names:", robot.joint_names)
    rest = torch.tensor(rest_np, dtype=torch.float32, device=env.unwrapped.device)
    if args.sticker:
        block = env.unwrapped.scene["block_red"]
        x, y = rig.STICKERS[args.sticker]
        state = block.data.default_root_state.clone()
        state[:, 0:3] = torch.tensor([x, y, rig.BLOCK_SIZE / 2 + 0.002], device=state.device) + env.unwrapped.scene.env_origins
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device); state[:, 7:] = 0.0
        block.write_root_pose_to_sim(state[:, :7]); block.write_root_velocity_to_sim(state[:, 7:])
    actions = rest.unsqueeze(0).repeat(1, 1)
    with torch.inference_mode():
        for _ in range(args.settle):
            obs, *_ = env.step(actions)
    top, wrist = rgb_frame(obs, "top"), rgb_frame(obs, "wrist")
    os.makedirs(args.out, exist_ok=True)
    save_png(f"{args.out}/sim_top.png", top); save_png(f"{args.out}/sim_wrist.png", wrist)
    info = {"task": args.task, "sticker": args.sticker, "settle_steps": args.settle, "top_shape": list(top.shape), "wrist_shape": list(wrist.shape),
            "top_std": float(top.std()), "wrist_std": float(wrist.std()),
            "joint_pos_deg_sim": [round(float(v), 2) for v in np.rad2deg(obs["policy"]["joint_pos_obs"][0].cpu().numpy())],
            "joint_pos_lerobot": [round(float(v), 2) for v in units.sim_to_lerobot(obs["policy"]["joint_pos_obs"][0].cpu().numpy())],
            "rest_pose_lerobot": [rig.REST_POSE_DEG[j] for j in units.JOINTS],
            "block_red_pos": [round(float(v), 4) for v in env.unwrapped.scene["block_red"].data.root_pos_w[0].cpu().numpy()],
            "seconds": round(time.time() - t0, 1)}
    for cam, real in (("top", args.real_top), ("wrist", args.real_wrist)):
        if real and os.path.exists(real):
            frames = read_video(real)
            if frames:
                save_png(f"{args.out}/real_{cam}.png", frames[0])
                save_png(f"{args.out}/compare_{cam}.png", side_by_side(top if cam == "top" else wrist, frames[0]))
                info[f"real_{cam}_frames"] = len(frames)
    json.dump(info, open(f"{args.out}/scene.json", "w"), indent=1)
    print("[scene]", json.dumps(info))
    return env


if __name__ == "__main__":
    import traceback
    env, rc = None, 0
    try:
        env = main()
    except BaseException:
        traceback.print_exc(); rc = 1
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            traceback.print_exc()
        app.close()   # Kit keeps the process alive otherwise; the verdict is the JSON file, not the exit code
    sys.exit(rc)
