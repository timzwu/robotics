"""Evaluate a policy in the simulator on the 20 sticker positions, with the real eval's protocol: red->left on odd
stickers, blue->right on even, a 30 s budget of executed motion, one observation -> one chunk of 50 actions executed
with the real per-step clamp, then the next observation. The policy runs behind the HTTPS bridge
(`sim/tools/modal_policy_http.py`); this script is the client. Scoring from the physics: contact = the block moved
more than 5 mm; grip = the block rose more than 2.5 cm; success = the block ends inside the target bowl. Writes a CSV
in `experiments/tools/eval.py`'s schema (mode "sim"), a per-trial JSON, and a top|wrist mp4 per trial.
    POLICY_URL=... POLICY_TOKEN=... python /workspace/so101_blocks/scripts/eval_in_sim.py --headless \
        --name smolvla_sim_simeval --policy /outputs/<job>/checkpoints/020000/pretrained_model --out /out/simeval
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="So101-Blocks-Eval-v0")
parser.add_argument("--name", required=True)
parser.add_argument("--policy", required=True, help="checkpoint path as the server sees it")
parser.add_argument("--policy-type", default="smolvla")
parser.add_argument("--camera-rename", default="top=camera1,wrist=camera2")
parser.add_argument("--positions", default="1-20")
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--chunk", type=int, default=50)
parser.add_argument("--clamp", type=float, default=5.0, help="degrees per step, the real follower's max_relative_target")
parser.add_argument("--out", default="/out/simeval")
parser.add_argument("--block-yaw", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
import os, requests   # before Kit starts: a missing URL or module must fail cheaply, not after a GPU app is up
URL, TOKEN = os.environ.get("POLICY_URL", ""), os.environ.get("POLICY_TOKEN", "")
assert URL and TOKEN, "POLICY_URL and POLICY_TOKEN must be set"
app = AppLauncher(args).app

import base64, csv, io, json, math, sys, time
import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import so101_blocks.tasks  # noqa: F401
from so101_blocks import rig, units
from so101_blocks.scripts.common import rgb_frame, apply_joint_limits, Mp4, fingertips_w

FPS = 30
RENAME = dict(p.split("=") for p in args.camera_rename.split(",")) if args.camera_rename else {}
TASKS = {"red": ("put the red block in the left bowl", "left"), "blue": ("put the blue block in the right bowl", "right")}
FIELDS = ["name", "position", "color", "bowl", "task", "success", "failure_type", "duration_s", "notes", "timestamp", "policy", "mode"]


def jpeg_b64(rgb):
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def get_chunk(state_lerobot, frames, task):
    req = {"policy": args.policy, "policy_type": args.policy_type, "task": task, "state": [float(v) for v in state_lerobot],
           "images": {RENAME.get(k, k): jpeg_b64(v) for k, v in frames.items()}, "chunk": args.chunk, "token": TOKEN}
    for attempt in range(3):
        try:
            r = requests.post(URL, json=req, timeout=300); r.raise_for_status()
            j = r.json(); return np.asarray(j["actions"], dtype=np.float64), j.get("seconds", 0.0)
        except Exception as e:
            print(f"[simeval] request failed ({e}); retry {attempt + 1}", flush=True); time.sleep(3)
    raise RuntimeError("policy server unreachable")


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    torch.inference_mode().__enter__()
    q_rest6 = units.lerobot_to_sim([rig.REST_POSE_DEG[j] for j in units.JOINTS])
    apply_joint_limits(env, q_rest6)
    obs, _ = env.reset()
    dev = env.unwrapped.device; robot = env.unwrapped.scene["robot"]; origins = env.unwrapped.scene.env_origins
    blocks = {"red": env.unwrapped.scene["block_red"], "blue": env.unwrapped.scene["block_blue"]}
    os.makedirs(f"{args.out}/trials", exist_ok=True)
    positions = []
    for part in args.positions.split(","):
        a, _, b = part.partition("-"); positions += list(range(int(a), int(b or a) + 1))
    rows, details = [], []

    def set_block(block, xy, yaw, z=rig.BLOCK_SIZE / 2 + 0.003):
        st = block.data.default_root_state.clone()
        st[:, 0:3] = torch.tensor([xy[0], xy[1], z], device=st.device) + origins
        st[:, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=st.device); st[:, 7:] = 0.0
        block.write_root_pose_to_sim(st[:, :7]); block.write_root_velocity_to_sim(st[:, 7:])

    def block_pos(block):
        return block.data.root_pos_w[0].cpu().numpy() - origins[0].cpu().numpy()

    def step(q6_rad):
        return env.step(torch.tensor(q6_rad, dtype=torch.float32, device=dev).unsqueeze(0))[0]

    for pos in positions:
        colour = "red" if pos % 2 == 1 else "blue"
        task, bowl = TASKS[colour]; bowl_xy = rig.BOWLS[bowl][:2]
        obs, _ = env.reset()
        set_block(blocks[colour], rig.STICKERS[pos], math.radians(args.block_yaw)); set_block(blocks["blue" if colour == "red" else "red"], (-0.25, 0.0), 0.0)
        for _ in range(15):
            obs = step(q_rest6)
        b0 = block_pos(blocks[colour]).copy()
        vid = Mp4(f"{args.out}/trials/{args.name}_{pos:02d}.mp4", FPS, (2 * rig.IMG_W, rig.IMG_H))
        executed, chunks, think, moved_max, lift_max, min_tip = 0, 0, 0.0, 0.0, 0.0, 9.9
        budget = int(args.duration * FPS)
        while executed < budget:
            state = units.sim_to_lerobot(obs["policy"]["joint_pos_obs"][0].cpu().numpy())
            frames = {"top": rgb_frame(obs, "top"), "wrist": rgb_frame(obs, "wrist")}
            chunk, secs = get_chunk(state, frames, task); chunks += 1; think += secs
            cur = state.copy()
            for a in chunk:
                cur = cur + np.clip(a - cur, -args.clamp, args.clamp)      # the real follower's per-step clamp, in LeRobot units
                obs = step(units.lerobot_to_sim(cur)); executed += 1
                vid.write(np.concatenate([rgb_frame(obs, "top"), rgb_frame(obs, "wrist")], axis=1))
                bp = block_pos(blocks[colour]); moved_max = max(moved_max, float(np.linalg.norm(bp[:2] - b0[:2]))); lift_max = max(lift_max, float(bp[2] - b0[2]))
                tips = fingertips_w(robot); min_tip = min(min_tip, float(np.linalg.norm(tips.mean(0) - bp)))
                if executed >= budget:
                    break
            bp = block_pos(blocks[colour]); dist = float(np.hypot(bp[0] - bowl_xy[0], bp[1] - bowl_xy[1]))
            if dist < rig.BOWL_INNER_R - 0.005 and bp[2] < rig.BOWL_HEIGHT:
                break
        vid.close()
        bp = block_pos(blocks[colour]); dist = float(np.hypot(bp[0] - bowl_xy[0], bp[1] - bowl_xy[1]))
        success = dist < rig.BOWL_INNER_R - 0.005 and bp[2] < rig.BOWL_HEIGHT
        contact = moved_max > 0.005 or min_tip < 0.02; gripped = lift_max > 0.025
        other_xy = rig.BOWLS["right" if bowl == "left" else "left"][:2]
        in_other = float(np.hypot(bp[0] - other_xy[0], bp[1] - other_xy[1])) < rig.BOWL_INNER_R - 0.005 and bp[2] < rig.BOWL_HEIGHT
        if success: failure = ""
        elif gripped and in_other: failure = "wrong_bowl"
        elif gripped and bp[2] - b0[2] > 0.02: failure = "timeout"        # still held when the budget ran out
        elif gripped: failure = "drop"
        elif contact: failure = "touch_no_grip"
        else: failure = "no_reach"
        note = f"chunks {chunks}, think {think:.1f}s, block moved {100*moved_max:.1f} cm, lifted {100*lift_max:.1f} cm, closest fingertip {100*min_tip:.1f} cm, final {100*dist:.1f} cm from the bowl"
        row = {"name": args.name, "position": pos, "color": colour, "bowl": bowl, "task": task, "success": int(success), "failure_type": failure,
               "duration_s": round(executed / FPS, 1), "notes": note, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "policy": args.policy, "mode": "sim"}
        rows.append(row); details.append({**row, "block_start": [round(float(v), 4) for v in b0], "block_final": [round(float(v), 4) for v in bp]})
        print("[simeval]", json.dumps(row), flush=True)
        with open(f"{args.out}/{args.name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        json.dump(details, open(f"{args.out}/{args.name}.json", "w"), indent=1)
    k = sum(r["success"] for r in rows)
    print(f"[simeval] SUMMARY {args.name}: {k}/{len(rows)} success; failures", json.dumps({t: sum(r['failure_type'] == t for r in rows) for t in ('no_reach', 'touch_no_grip', 'drop')}), flush=True)
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
        app.close()
    sys.exit(rc)
