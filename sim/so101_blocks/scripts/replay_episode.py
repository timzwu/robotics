"""Replay a real episode's joint trajectory on the simulated arm and record both sim cameras next to the real ones.
The commanded joints are the real follower's observed positions (observation.state), mapped by `units.lerobot_to_sim`.
Writes sim_top.mp4, sim_wrist.mp4, compare_top.mp4, compare_wrist.mp4, stills, and replay.npz (commanded vs achieved
joints per frame) for the tracking plot. Headless, inside the sim container:
    python /workspace/so101_blocks/scripts/replay_episode.py --headless --episode /data/episode_000.npz --out /out/replay
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="So101-Blocks-v0")
parser.add_argument("--episode", required=True, help="episode_NNN.npz from export_episode.py; the _top/_wrist.mp4 next to it are used if present")
parser.add_argument("--out", default="/out/replay")
parser.add_argument("--sticker", type=int, default=0, help="put the red block on this sticker (0 = where the real episode's block was, found in the real first frame; or the reset event's spot if no real video)")
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--gripper-from", choices=["action", "state"], default="action",
                    help="action = the leader's command (carries the squeeze; the follower's reading stalls on the block), state = the follower's reading")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json, os, sys, time
import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import so101_blocks.tasks  # noqa: F401
from so101_blocks import rig, units
from so101_blocks.scripts.common import rgb_frame, save_png, side_by_side, read_video, Mp4, apply_joint_limits, find_red_block, fingertips_w



def main():
    t0 = time.time()
    ep = np.load(args.episode, allow_pickle=True)
    state, fps = ep["state"].copy(), int(ep["fps"])
    if args.gripper_from == "action":
        state[:, 5] = ep["action"][:, 5]
    if args.max_frames:
        state = state[: args.max_frames]
    base = args.episode[: -len(".npz")]
    real = {cam: read_video(f"{base}_{cam}.mp4") if os.path.exists(f"{base}_{cam}.mp4") else [] for cam in ("top", "wrist")}
    print(f"[replay] {len(state)} frames at {fps} fps; real videos: { {k: len(v) for k, v in real.items()} }")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    cmd_rad = units.lerobot_to_sim(state)
    apply_joint_limits(env, cmd_rad[0])
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    dev = env.unwrapped.device
    block_xy = rig.STICKERS[args.sticker] if args.sticker else None
    if block_xy is None and real["top"]:
        c = find_red_block(real["top"][0])
        if c is not None:
            block_xy = rig.px_to_world(*c)
            print(f"[replay] real block found at pixel {tuple(round(v, 1) for v in c)} -> world {tuple(round(v, 4) for v in block_xy)}")
    if block_xy is not None:
        block = env.unwrapped.scene["block_red"]
        x, y = block_xy
        st = block.data.default_root_state.clone()
        st[:, 0:3] = torch.tensor([x, y, rig.BLOCK_SIZE / 2 + 0.002], device=st.device) + env.unwrapped.scene.env_origins
        st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=st.device); st[:, 7:] = 0.0
        block.write_root_pose_to_sim(st[:, :7]); block.write_root_velocity_to_sim(st[:, 7:])
    # settle at the episode's first pose
    first = torch.tensor(cmd_rad[0], dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(30):
            obs, *_ = env.step(first)
    os.makedirs(args.out, exist_ok=True)
    vids = {cam: Mp4(f"{args.out}/sim_{cam}.mp4", fps, (rig.IMG_W, rig.IMG_H)) for cam in ("top", "wrist")}
    cmp = {cam: Mp4(f"{args.out}/compare_{cam}.mp4", fps, (2 * rig.IMG_W, rig.IMG_H)) for cam in ("top", "wrist") if real[cam]}
    achieved = np.zeros_like(cmd_rad); tips = np.zeros((len(state), 2, 3))
    stills = sorted({0, len(state) // 4, len(state) // 2, 3 * len(state) // 4, len(state) - 1})
    with torch.inference_mode():
        for i in range(len(state)):
            a = torch.tensor(cmd_rad[i], dtype=torch.float32, device=dev).unsqueeze(0)
            obs, *_ = env.step(a)
            achieved[i] = obs["policy"]["joint_pos_obs"][0].cpu().numpy(); tips[i] = fingertips_w(robot)
            for cam in ("top", "wrist"):
                fr = rgb_frame(obs, cam)
                vids[cam].write(fr)
                if cam in cmp and i < len(real[cam]):
                    panel = side_by_side(fr, real[cam][i])
                    cmp[cam].write(panel)
                    if i in stills:
                        save_png(f"{args.out}/compare_{cam}_{i:04d}.png", panel)
                elif i in stills:
                    save_png(f"{args.out}/sim_{cam}_{i:04d}.png", fr)
            if i % 100 == 0:
                print(f"[replay] frame {i}/{len(state)}  {time.time() - t0:.0f}s")
    for v in list(vids.values()) + list(cmp.values()):
        v.close()
    err_deg = np.rad2deg(achieved - cmd_rad)
    # the real grasp: the first frame after the gripper opened where the command closes again (below 8 on the 0-100 scale)
    g = state[:, 5]; opened = np.where(g > 10)[0]
    grasp = int(next((i for i in range(opened[0], len(g)) if g[i] < 8), -1)) if len(opened) else -1
    mid = tips[grasp].mean(0) if grasp >= 0 else None
    info = {"episode": os.path.basename(args.episode), "frames": int(len(state)), "fps": fps, "task": str(ep["task"]), "gripper_from": args.gripper_from,
            "block_start_xy": [round(float(v), 4) for v in block_xy] if block_xy is not None else None,
            "grasp_frame": grasp, "fingertip_mid_at_grasp": [round(float(v), 4) for v in mid] if mid is not None else None,
            "fingertip_gap_at_grasp_cm": round(float(np.linalg.norm(tips[grasp, 0] - tips[grasp, 1]) * 100), 2) if grasp >= 0 else None,
            "block_minus_tip_xy": [round(float(block_xy[k] - mid[k]), 4) for k in range(2)] if (mid is not None and block_xy is not None) else None,
            "tracking_error_deg_rms_per_joint": [round(float(v), 2) for v in np.sqrt((err_deg ** 2).mean(0))],
            "tracking_error_deg_max_per_joint": [round(float(v), 2) for v in np.abs(err_deg).max(0)],
            "commanded_range_deg": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(np.rad2deg(cmd_rad).min(0), np.rad2deg(cmd_rad).max(0))],
            # frames where the real value lies outside the USD joint limits (the mapping clips them): a calibration-offset signal
            "clipped_frames_per_joint": [int(v) for v in (np.abs(units.sim_to_lerobot(cmd_rad) - state) > 1e-3).sum(0)],
            "block_red_final_pos": [round(float(v), 4) for v in env.unwrapped.scene["block_red"].data.root_pos_w[0].cpu().numpy()],
            "seconds": round(time.time() - t0, 1)}
    np.savez(f"{args.out}/replay.npz", commanded_rad=cmd_rad, achieved_rad=achieved, state_lerobot=state, joints=np.array(units.JOINTS), fingertips_w=tips)
    json.dump(info, open(f"{args.out}/replay.json", "w"), indent=1)
    print("[replay]", json.dumps(info))
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
