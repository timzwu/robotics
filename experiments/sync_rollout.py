"""Synchronous chunked rollout on the Mac for policies too slow to run at 30 Hz per step (SmolVLA, pi0.5).

Loop: observe -> predict one action chunk on the Mac GPU (SmolVLA: ~0.8 s on an M1 Pro) -> execute the chunk at
--fps with a per-step clamp -> observe again, until --duration seconds. The arm pauses briefly between chunks
while the Mac thinks; ACT's local rollout is open-loop in the same way (one 100-step chunk at a time), so the
two evals are comparable. Use the async policy server for real-time deployment; this is the eval path.

    python experiments/sync_rollout.py --policy experiments/checkpoints/smolvla_so101_blocks_n100_s0 \
        --task "put the red block in the left bowl" --duration 30

Ctrl-C stops and disables torque. Robot config (ports, cameras, clamp) comes from experiments/robot.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_policy(path: str, policy_type: str, device: str):
    from lerobot.policies.factory import make_pre_post_processors

    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    elif policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
    elif policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    else:
        raise SystemExit(f"unsupported policy type {policy_type}")
    pol = P.from_pretrained(path).to(device).eval()
    pre, post = make_pre_post_processors(
        pol.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return pol, pre, post


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="checkpoint directory (pretrained_model)")
    ap.add_argument("--policy-type", default="smolvla", choices=["smolvla", "pi05", "act"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of motion (thinking pauses excluded)")
    ap.add_argument("--chunk", type=int, default=50, help="actions executed per inference")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--clamp", type=float, default=None, help="max deg per step; default robot.json max_relative_target")
    ap.add_argument("--robot-config", default=str(ROOT / "robot.json"))
    ap.add_argument("--camera-rename", default="", help='e.g. "top=camera1,wrist=camera2"; usually not needed: the '
                    "checkpoint's own preprocessor renames if it was trained with --rename_map")
    args = ap.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

    r = json.load(open(args.robot_config))
    clamp = r.get("max_relative_target", 5.0) if args.clamp is None else args.clamp
    rename = dict(p.split("=") for p in args.camera_rename.split(",")) if args.camera_rename else {}
    cams = {rename.get(k, k): OpenCVCameraConfig(index_or_path=v["index_or_path"], width=v["width"],
                                                  height=v["height"], fps=v["fps"]) for k, v in r["cameras"].items()}
    pol, pre, post = load_policy(args.policy, args.policy_type, args.device)
    bot = SO101Follower(SOFollowerRobotConfig(port=r["port"], id=r["id"], cameras=cams))
    bot.connect(calibrate=False)
    moved = 0.0
    try:
        for _ in range(int(args.fps)):  # ~1 s of frames so the cameras' exposure settles
            bot.get_observation()
        c = 0
        while moved < args.duration:
            raw = bot.get_observation()
            state = np.array([raw[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
            obs = {"observation.state": torch.tensor(state)[None], "task": [args.task]}
            for k in cams:
                obs[f"observation.images.{k}"] = torch.tensor(np.ascontiguousarray(raw[k])).permute(2, 0, 1).float()[None] / 255
            t0 = time.time()
            batch = pre(obs)
            pol.reset()
            chunk = []
            with torch.no_grad():
                for _ in range(args.chunk):
                    a = post(pol.select_action(batch))
                    chunk.append((a["action"] if isinstance(a, dict) else a)[0].float().cpu().numpy())
            chunk = np.stack(chunk)
            rng = chunk.max(0) - chunk.min(0)
            print(f"[sync] chunk {c}: think {time.time() - t0:.2f}s | range {np.round(rng, 0).tolist()}", flush=True)
            cur = state.copy()
            t_exec = time.time()
            for a in chunk:
                cur = cur + np.clip(a - cur, -clamp, clamp)
                bot.send_action({f"{j}.pos": float(cur[i]) for i, j in enumerate(JOINTS)})
                time.sleep(1 / args.fps)
                if moved + (time.time() - t_exec) >= args.duration:
                    break
            moved += time.time() - t_exec
            c += 1
        print(f"[sync] done: {moved:.1f}s of motion in {c} chunks", flush=True)
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
