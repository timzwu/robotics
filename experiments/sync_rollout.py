"""Synchronous chunked rollout on the Mac for policies too slow to run at 30 Hz per step (SmolVLA, pi0.5).

Loop: observe -> predict one action chunk on the Mac GPU (SmolVLA: ~0.8 s on an M1 Pro) -> execute the chunk at
--fps with a per-step clamp -> observe again, until --duration seconds. The arm pauses briefly between chunks
while the Mac thinks; ACT's local rollout is open-loop in the same way (one 100-step chunk at a time), so the
two evals are comparable. Use the async policy server for real-time deployment; this is the eval path.

    python experiments/sync_rollout.py --policy experiments/checkpoints/smolvla_so101_blocks_n100_s0 \
        --task "put the red block in the left bowl" --duration 30

Remote variant (same loop, the chunk comes from a LeRobot policy server on Modal; for models too big for
the Mac, e.g. pi0.5). No blending, no queue: one observation out, one chunk back, execute, repeat, so the
protocol is identical to the local loop and the pause is network + GPU time instead of Mac time:

    python experiments/sync_rollout.py --server r442.modal.host:12345 --policy-type pi05 \
        --policy /outputs/<job>/checkpoints/last/pretrained_model --task "..." --duration 30

Ctrl-C stops and disables torque. Robot config (ports, cameras, clamp) comes from experiments/robot.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
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


class RemoteChunkSource:
    """One observation -> one chunk from a LeRobot policy server (lerobot.async_inference.policy_server).

    Uses the server's own protocol (Ready, SendPolicyInstructions, SendObservations, GetActions) but never
    keeps a queue: every observation is sent with must_go=True so the server always runs inference, and we
    block until the chunk for that observation comes back."""

    def __init__(self, server: str, policy_path: str, policy_type: str, device: str, chunk: int, bot) -> None:
        import pickle
        import grpc
        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        self.pickle, self.pb, self.chunk = pickle, services_pb2, chunk
        self.channel = grpc.insecure_channel(server, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.stub.Ready(services_pb2.Empty())
        cfg = RemotePolicyConfig(policy_type, policy_path, map_robot_keys_to_lerobot_features(bot), chunk, device)
        self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(cfg)))
        self.t = 0

    def get_chunk(self, raw, state):
        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks

        obs = TimedObservation(timestamp=time.time(), timestep=self.t, observation=raw, must_go=True)
        self.t += self.chunk
        self.stub.SendObservations(send_bytes_in_chunks(self.pickle.dumps(obs), self.pb.Observation, silent=True))
        deadline = time.time() + 60
        while time.time() < deadline:
            resp = self.stub.GetActions(self.pb.Empty())
            if len(resp.data):
                actions = self.pickle.loads(resp.data)  # nosec: our own server
                return np.stack([a.get_action().float().cpu().numpy() for a in actions])[: self.chunk]
            time.sleep(0.02)
        raise RuntimeError("policy server returned no chunk within 60 s")


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
    ap.add_argument("--server", default="", help="host:port of a LeRobot policy server; when set, the chunk is "
                    "fetched from there instead of computed on the Mac (--policy is then a path ON THE SERVER)")
    ap.add_argument("--policy-device", default="cuda", help="remote only: device on the server")
    ap.add_argument("--record", default="", help="write this trial's camera frames (all cameras side by side) to an mp4 "
                    "at this path, e.g. experiments/results/trials/pi05_pair_03.mp4")
    ap.add_argument("--cameras", default="", help='comma list of robot.json cameras to send, e.g. "top" for a model '
                    "trained without the wrist stream (default: all)")
    args = ap.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

    r = json.load(open(args.robot_config))
    clamp = r.get("max_relative_target", 5.0) if args.clamp is None else args.clamp
    rename = dict(p.split("=") for p in args.camera_rename.split(",")) if args.camera_rename else {}
    keep = set(args.cameras.split(",")) if args.cameras else set(r["cameras"])
    cams = {rename.get(k, k): OpenCVCameraConfig(index_or_path=v["index_or_path"], width=v["width"],
                                                  height=v["height"], fps=v["fps"]) for k, v in r["cameras"].items() if k in keep}
    bot = SO101Follower(SOFollowerRobotConfig(port=r["port"], id=r["id"], cameras=cams))
    bot.connect(calibrate=False)
    if args.server:
        remote = RemoteChunkSource(args.server, args.policy, args.policy_type, args.policy_device, args.chunk, bot)
        get_chunk = remote.get_chunk
    else:
        pol, pre, post = load_policy(args.policy, args.policy_type, args.device)

        def get_chunk(raw, state):
            obs = {"observation.state": torch.tensor(state)[None], "task": [args.task]}
            for k in cams:
                obs[f"observation.images.{k}"] = torch.tensor(np.ascontiguousarray(raw[k])).permute(2, 0, 1).float()[None] / 255
            batch = pre(obs)
            pol.reset()
            out = []
            with torch.no_grad():
                for _ in range(args.chunk):
                    a = post(pol.select_action(batch))
                    out.append((a["action"] if isinstance(a, dict) else a)[0].float().cpu().numpy())
            return np.stack(out)
    moved = 0.0
    rec = None
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        rec = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640 * len(cams), 480))
    try:
        for _ in range(int(args.fps)):  # ~1 s of frames so the cameras' exposure settles
            bot.get_observation()
        c = 0
        while moved < args.duration:
            raw = bot.get_observation()
            raw["task"] = args.task
            state = np.array([raw[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
            t0 = time.time()
            chunk = get_chunk(raw, state)
            rng = chunk.max(0) - chunk.min(0)
            print(f"[sync] chunk {c}: think {time.time() - t0:.2f}s | range {np.round(rng, 0).tolist()}", flush=True)
            cur = state.copy()
            t_exec = time.time()
            for a in chunk:
                cur = cur + np.clip(a - cur, -clamp, clamp)
                bot.send_action({f"{j}.pos": float(cur[i]) for i, j in enumerate(JOINTS)})
                if rec is not None:
                    raw = bot.get_observation()
                    rec.write(np.hstack([cv2.cvtColor(np.ascontiguousarray(raw[k]), cv2.COLOR_RGB2BGR) for k in cams]))
                else:
                    time.sleep(1 / args.fps)
                if moved + (time.time() - t_exec) >= args.duration:
                    break
            moved += time.time() - t_exec
            c += 1
        print(f"[sync] done: {moved:.1f}s of motion in {c} chunks", flush=True)
    finally:
        if rec is not None:
            rec.release()
        bot.disconnect()


if __name__ == "__main__":
    main()
