"""Record simulated demonstrations of the block-to-bowl task with a scripted agent, in the real dataset's format.

Each episode: the block in play is dropped at a random spot in the taped zone with a random yaw (the other block is
parked out of view); the agent reads the block's pose from the simulator, plans fingertip waypoints (hover, grasp,
lift, above the target bowl, release, rest) through inverse kinematics on the arm model (`ik.py`), and drives the arm
through smooth joint ramps at 30 Hz while both cameras are recorded. An episode is kept only if the block ends inside
the target bowl. Camera pose/focal length, lighting and robot colour are randomized on every reset (`So101-Blocks-DR-v0`).
The dataset (LeRobot v3.0: observation.state, action, observation.images.top/.wrist, task) matches the real one; the
action is the commanded joint target, the state the achieved one, both in the real dataset's units.
    python /workspace/so101_blocks/scripts/record_demos.py --headless --episodes 100 --out /out/datasets/so101_blocks_sim
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="So101-Blocks-DR-v0")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out", default="/out/datasets/so101_blocks_sim")
parser.add_argument("--repo-id", default="timzwu/so101_blocks_sim")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--reel-every", type=int, default=10, help="write the side camera's mp4 for every Nth episode (0 = never)")
parser.add_argument("--max-attempts", type=int, default=4, help="re-rolls per episode when the block does not end in the bowl")
parser.add_argument("--recovery-share", type=float, default=0.15, help="share of episodes that miss first, reopen and re-grasp")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json, os, sys, time, math, shutil
import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import so101_blocks.tasks  # noqa: F401
from so101_blocks import rig, units
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ik.py / fk.py import each other bare
import ik
from so101_blocks.scripts.common import rgb_frame, save_png, apply_joint_limits, Mp4, fingertips_w
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 30
LOG, DS, T0 = [], None, time.time()
JOINT_NAMES = [f"{j}.pos" for j in units.JOINTS]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "observation.images.top": {"dtype": "video", "shape": (rig.IMG_H, rig.IMG_W, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (rig.IMG_H, rig.IMG_W, 3), "names": ["height", "width", "channels"]},
}
TASKS = {"red": ("put the red block in the left bowl", "left"), "blue": ("put the blue block in the right bowl", "right")}
GRIP_CLOSE, GRIP_REST = 5.0, 1.0                           # LeRobot gripper units, from the real episodes
ZONE_HALF = rig.TAPE_OUTER / 2 - rig.TAPE_WIDTH - 0.018    # v2: up to the tape's inner edge minus the block's half side
RECOVERY_SHARE = 0.15                                      # v2: episodes that miss first, reopen and re-grasp


def smoothstep(t):
    return t * t * (3 - 2 * t)


def yaw_of(quat_wxyz):
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def approach_dir(rng, max_tilt_deg=25.0):
    t, ph = math.radians(rng.uniform(0.0, max_tilt_deg)), rng.uniform(0.0, 2 * math.pi)
    return np.array([math.sin(t) * math.cos(ph), math.sin(t) * math.sin(ph), -math.cos(t)])


def plan(block_xy, block_yaw, bowl_xy, q_rest, rng, recovery=False):
    """Joint-space waypoints (rad, 5 arm joints), per-segment durations (s) and the gripper command at each segment's end.
    v2 (Sept 16): varied hover height, approach tilt and speed, the jaw orientation chosen at random among the two valid
    ones, a wider and varied opening, random pauses, a small wiggle before closing, and 15% recovery episodes: the first
    grasp is deliberately 1.2-2.5 cm off, the jaws close on nothing, lift, reopen, and the second attempt (half of them
    with the jaws turned 90 deg) takes the block."""
    dur = lambda base: base * rng.uniform(0.6, 1.4)
    open_w = rng.uniform(22.0, 35.0)                             # the real operator opened to 16-37; below 22 the moving tip can land on the block's edge
    hover_z = rng.uniform(0.06, 0.14)
    ap = approach_dir(rng)
    q_hover0, _, _ = ik.solve(np.array([*block_xy, hover_z]), q_rest)
    _, _, d = ik.frame(np.concatenate([q_hover0, [0.3]]))
    cur = math.atan2(d[1], d[0])
    cands = sorted([block_yaw + k * math.pi / 2 for k in range(4)], key=lambda h: abs((h - cur + math.pi) % (2 * math.pi) - math.pi))
    heading = cands[0] if rng.uniform() < 0.6 else cands[1]      # the nearest valid orientation, or the other one
    q_hover, e1, _ = ik.solve(np.array([*block_xy, hover_z]), q_rest, heading=heading, approach=ap, down_w=0.08)
    q_grasp, e2, tilt = ik.solve(np.array([*block_xy, 0.010]), q_hover, heading=heading, approach=ap, down_w=0.08)
    wig = np.array([rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004), 0.0])
    q_wig, _, _ = ik.solve(np.array([*block_xy, 0.010]) + wig, q_grasp, heading=heading, approach=ap, down_w=0.08)
    q_bowl, e3, _ = ik.solve(np.array([*bowl_xy, rng.uniform(0.09, 0.13)]), q_hover)
    segs = [(q_hover, dur(2.4), open_w), (q_hover, rng.uniform(0.0, 0.6), open_w)]         # approach, pause
    info = {"recovery": recovery}
    if recovery:
        # the miss is perpendicular to the jaw axis of the solved grasp, far enough that neither the descent nor the
        # closing jaws touch the block (checked on the finger model: 2.3-3.0 cm clears it; a random direction did not)
        _, _, dj = ik.frame(np.concatenate([q_grasp, [0.3]]))
        perp = np.array([-dj[1], dj[0]]); perp /= np.linalg.norm(perp)
        miss = rng.uniform(0.023, 0.030) * (1.0 if rng.uniform() < 0.5 else -1.0)
        off = miss * perp
        q_miss, _, _ = ik.solve(np.array([*(np.asarray(block_xy) + off), 0.010]), q_hover, heading=heading, approach=ap, down_w=0.08)
        q_miss_up, _, _ = ik.solve(np.array([*(np.asarray(block_xy) + off), hover_z * 0.6]), q_miss, heading=heading, approach=ap, down_w=0.08)
        segs += [(q_miss, dur(1.6), open_w), (q_miss, 0.7, GRIP_CLOSE), (q_miss_up, dur(1.0), GRIP_CLOSE),   # miss, close on nothing, lift
                 (q_miss_up, rng.uniform(0.3, 0.8), open_w)]                                                   # reopen
        if rng.uniform() < 0.5:                                                                                 # turn the jaws 90 deg for the retry
            heading = cands[1] if heading == cands[0] else cands[0]
            q_grasp, e2, tilt = ik.solve(np.array([*block_xy, 0.010]), q_miss_up, heading=heading, approach=ap, down_w=0.08)
            q_wig, _, _ = ik.solve(np.array([*block_xy, 0.010]) + wig, q_grasp, heading=heading, approach=ap, down_w=0.08)
        q_hover_r, _, _ = ik.solve(np.array([*block_xy, hover_z * 0.6]), q_miss_up, heading=heading, approach=ap, down_w=0.08)
        segs += [(q_hover_r, dur(1.0), open_w)]                                                                # back above the block before the retry descends
        info["miss_cm"] = round(100 * abs(miss), 1)
    segs += [(q_grasp, dur(1.6), open_w), (q_wig, 0.4, open_w), (q_grasp, 0.7, GRIP_CLOSE + rng.uniform(-2.0, 2.0)),   # descend, wiggle, close
             (q_hover, dur(1.2), GRIP_CLOSE), (q_bowl, dur(2.4), GRIP_CLOSE), (q_bowl, rng.uniform(0.0, 0.5), GRIP_CLOSE),
             (q_bowl, 0.7, open_w), (q_rest, dur(2.4), GRIP_REST), (q_rest, 0.6, GRIP_REST)]
    info.update({"ik_err_mm": [round(1000 * e, 2) for e in (e1, e2, e3)], "grasp_tilt_deg": round(tilt, 1), "heading_deg": round(math.degrees(heading), 1),
                 "open_w": round(open_w, 1), "hover_cm": round(100 * hover_z, 1)})
    return segs, info


def main():
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    # everything from here on runs under inference mode: buffers the env creates during steps become inference tensors,
    # and a later env.reset() outside the mode fails with "Inplace update to inference tensor outside InferenceMode"
    torch.inference_mode().__enter__()
    q_rest6 = units.lerobot_to_sim([rig.REST_POSE_DEG[j] for j in units.JOINTS])
    apply_joint_limits(env, q_rest6)
    obs, _ = env.reset()
    dev = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    blocks = {"red": env.unwrapped.scene["block_red"], "blue": env.unwrapped.scene["block_blue"]}
    origins = env.unwrapped.scene.env_origins
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    ds = LeRobotDataset.create(args.repo_id, fps=FPS, features=FEATURES, root=args.out, robot_type="so_follower", use_videos=True, image_writer_threads=4)   # the real dataset says so_follower
    os.makedirs(f"{args.out}_extras", exist_ok=True)
    global DS, T0; DS, T0 = ds, t0

    def set_block(block, xy, yaw, z=rig.BLOCK_SIZE / 2 + 0.003):
        st = block.data.default_root_state.clone()
        st[:, 0:3] = torch.tensor([xy[0], xy[1], z], device=st.device) + origins
        st[:, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=st.device); st[:, 7:] = 0.0
        block.write_root_pose_to_sim(st[:, :7]); block.write_root_velocity_to_sim(st[:, 7:])

    def step(q5, grip_units):
        q = np.concatenate([q5, units.lerobot_to_sim([0, 0, 0, 0, 0, grip_units])[5:6]])
        a = torch.tensor(q, dtype=torch.float32, device=dev).unsqueeze(0)
        return env.step(a)[0], q

    if args.reel_every:
        assert "rgb_side" in obs["visual"], "the reel needs the rgb_side observation (So101-Blocks-DR-v0)"
    ep, kept = 0, 0
    while kept < args.episodes:
        colour = "red" if kept % 2 == 0 else "blue"
        task, bowl = TASKS[colour]
        done = False
        for attempt in range(1, args.max_attempts + 1):
            obs, _ = env.reset()
            if rng.uniform() < 0.3:   # v2: a third of the episodes near a sticker (the eval spots), the rest anywhere inside the tape
                sx, sy = rig.STICKERS[int(rng.integers(1, 21))]
                xy = (sx + rng.uniform(-0.012, 0.012), sy + rng.uniform(-0.012, 0.012))
            else:
                xy = (rig.TAPE_CENTER[0] + rng.uniform(-ZONE_HALF, ZONE_HALF), rig.TAPE_CENTER[1] + rng.uniform(-ZONE_HALF, ZONE_HALF))
            yaw = rng.uniform(-math.pi, math.pi)
            set_block(blocks[colour], xy, yaw)
            other = "blue" if colour == "red" else "red"
            set_block(blocks[other], (-0.25, 0.0), 0.0)
            for _ in range(15):                            # settle at rest, block lands
                obs, _ = step(q_rest6[:5], GRIP_REST)
            obs_prev = obs
            bp = blocks[colour].data.root_pos_w[0].cpu().numpy() - origins[0].cpu().numpy()
            byaw = yaw_of(blocks[colour].data.root_quat_w[0].cpu().numpy())
            bowl_xy = rig.BOWLS[bowl][:2]
            segs, info = plan(bp[:2], byaw, np.array(bowl_xy), q_rest6[:5], rng, recovery=rng.uniform() < args.recovery_share)
            reel = Mp4(f"{args.out}_extras/side_ep{ep:03d}.mp4", FPS, (rig.IMG_W, rig.IMG_H)) if (args.reel_every and ep % args.reel_every == 0 and "rgb_side" in obs["visual"]) else None
            q_from, grip_from, frames = q_rest6[:5].copy(), GRIP_REST, 0
            if True:
                for q_to, dur, grip_to in segs:
                    n = max(1, int(round(dur * FPS)))
                    for k in range(1, n + 1):
                        s = smoothstep(k / n)
                        q5 = q_from + s * (q_to - q_from)
                        grip = grip_from + s * (grip_to - grip_from)
                        # like the real record loop: the observation captured BEFORE the command, paired with that command
                        q_cmd = np.concatenate([q5, units.lerobot_to_sim([0, 0, 0, 0, 0, grip])[5:6]])
                        state = units.sim_to_lerobot(obs_prev["policy"]["joint_pos_obs"][0].cpu().numpy())
                        action = units.sim_to_lerobot(q_cmd); action[5] = grip
                        ds.add_frame({"observation.state": state.astype(np.float32), "action": action.astype(np.float32),
                                      "observation.images.top": rgb_frame(obs_prev, "top"), "observation.images.wrist": rgb_frame(obs_prev, "wrist"), "task": task})
                        if reel is not None:
                            reel.write(rgb_frame(obs_prev, "side"))
                        obs, _ = step(q5, grip)
                        obs_prev = obs
                        frames += 1
                    q_from, grip_from = q_to, grip_to
            if reel is not None:
                reel.close()
            bf = blocks[colour].data.root_pos_w[0].cpu().numpy() - origins[0].cpu().numpy()
            dist = float(np.hypot(bf[0] - bowl_xy[0], bf[1] - bowl_xy[1]))
            success = dist < rig.BOWL_INNER_R - 0.005 and bf[2] < rig.BOWL_HEIGHT
            rec = {"episode": ep, "attempt": attempt, "colour": colour, "task": task, "block_xy": [round(float(v), 4) for v in bp[:2]], "block_yaw_deg": round(math.degrees(byaw), 1),
                   "frames": frames, "final_xy": [round(float(v), 4) for v in bf[:2]], "final_z": round(float(bf[2]), 4), "dist_to_bowl_cm": round(100 * dist, 1), "success": bool(success), **info,
                   "elapsed_s": round(time.time() - t0)}
            print("[record]", json.dumps(rec), flush=True)
            LOG.append(rec)
            if success:
                ds.save_episode(parallel_encoding=False); kept += 1; done = True; break
            ds.clear_episode_buffer()
            if reel is not None:
                os.rename(f"{args.out}_extras/side_ep{ep:03d}.mp4", f"{args.out}_extras/side_ep{ep:03d}_failed{attempt}.mp4")
        if not done:
            print(f"[record] episode {ep} FAILED {args.max_attempts} times; skipping", flush=True)
        ep += 1
    return env


def finish():
    kept_eps = [r for r in LOG if r["success"]]
    summary = {"episodes_kept": len(kept_eps), "attempts": len(LOG), "red": sum(r["colour"] == "red" for r in kept_eps), "blue": sum(r["colour"] == "blue" for r in kept_eps),
               "frames_total": sum(r["frames"] for r in kept_eps), "mean_frames": round(np.mean([r["frames"] for r in kept_eps]), 1) if kept_eps else 0,
               "seconds": round(time.time() - T0), "task": args.task, "seed": args.seed, "dataset": args.out}
    json.dump({"summary": summary, "episodes": LOG}, open(f"{args.out}_extras/record_log.json", "w"), indent=1)
    print("[record] SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    import traceback
    env, rc = None, 0
    try:
        env = main()
    except BaseException:
        traceback.print_exc(); rc = 1
    finally:
        try:   # a partial dataset stays loadable: parquet footers are written by finalize
            if DS is not None:
                DS.stop_image_writer(); DS.finalize(); finish()
        except Exception:
            traceback.print_exc()
        try:
            if env is not None:
                env.close()
        except Exception:
            traceback.print_exc()
        app.close()
    sys.exit(rc)
