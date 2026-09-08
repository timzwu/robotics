"""Run LeRobot's async-inference policy server on a Modal GPU; the Mac is the client.

Why this exists: SmolVLA (~450M params) and pi0.5 (>3B) are too big to run on the Mac at control
rate. The model runs on a Modal GPU, the arm stays on the desk. LeRobot splits this into two
processes:
  * policy server (here, on the GPU) - receives an observation, returns a CHUNK of ~50 future actions
  * robot client (on the Mac)        - drives the arm and cameras, executes the chunk it already has
                                       while the next one is in flight. That overlap is the "async"
                                       part; without it the arm would freeze for every round trip.

The server is GENERIC: it does not take a policy. The CLIENT sends policy_type and
pretrained_name_or_path, and the server loads that model on first request. So the same server works
for smolvla_base today and your own fine-tuned checkpoint after R4.

    # 1. start the server (prints the address to use; holds a GPU until it exits)
    modal run experiments/modal_policy_server.py --minutes 30

    # 2. in a second terminal on the Mac, with the follower + both cameras plugged in:
    python experiments/eval.py --mode async --name smolvla_zeroshot \
        --policy-type smolvla --policy lerobot/smolvla_base \
        --server <ADDRESS PRINTED IN STEP 1> --duration 10

COST: the GPU is held for the whole `--minutes`, whether or not the arm is moving. A10G is about
$1.10/hour, so 30 minutes is roughly $0.55. Use `--dry-run` to print the plan and spend nothing.

STOPPING EARLY (this matters for the bill): Ctrl-C in the terminal running `modal run` stops it. But if that
process is killed any other way, THE CONTAINER KEEPS RUNNING. Always finish with:
    modal app list                      # find the ap-... id for lerobot-policy-server
    modal app stop --yes <ap-id>        # terminate it; then re-run `modal app list` to confirm

SAFETY (first run with a real arm): set `max_relative_target` on the follower in robot.json to a few
degrees so the arm can only creep, keep the power switch in reach, start from the rest pose, and run
10 seconds before anything longer. Ctrl-C on the client disconnects and torque drops by default.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import modal

APP_NAME = "lerobot-policy-server"
HF_CACHE_DIR = "/root/.cache/huggingface"
DEFAULT_GPU = "A10G"          # enough for SmolVLA. Use L40S/A100-80GB for pi0.5.
SERVER_PORT = 8080

app = modal.App(APP_NAME)

# Same base as modal_train.py, plus [async] for grpcio (the server/client wire protocol).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("lerobot[training,smolvla,pi,async]")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_LEROBOT_HOME": f"{HF_CACHE_DIR}/lerobot",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

hf_cache = modal.Volume.from_name("lerobot-hf-cache", create_if_missing=True)
# Trained checkpoints live on the outputs Volume, so the client can name one as
# /outputs/<job>/checkpoints/last/pretrained_model instead of pushing it to the Hub.
outputs = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)
OUTPUTS_DIR = "/outputs"

local_tokens = modal.Secret.from_dict(
    {k: v for k, v in {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}.items() if v}
)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
    secrets=[local_tokens],
    timeout=4 * 60 * 60,
    cpu=4,
    memory=16384,
)
def serve(minutes: int = 30, fps: int = 30, inference_latency: float = 0.033) -> dict:
    """Hold a GPU and run the policy server for `minutes`, reachable over a raw TCP tunnel."""
    # unencrypted=True gives a raw TCP socket, which is what gRPC needs.
    with modal.forward(SERVER_PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        address = f"{host}:{port}"
        print("=" * 72, flush=True)
        print(f"  POLICY SERVER ADDRESS:  {address}", flush=True)
        print(f"  Shutting down in {minutes} min. Ctrl-C here stops it early and stops the billing.", flush=True)
        print("  On the Mac:", flush=True)
        print(f"    python experiments/eval.py --mode async --name smolvla_zeroshot \\", flush=True)
        print(f"        --policy-type smolvla --policy lerobot/smolvla_base \\", flush=True)
        print(f"        --server {address} --duration 10", flush=True)
        print("=" * 72, flush=True)

        cmd = [
            sys.executable, "-m", "lerobot.async_inference.policy_server",
            "--host=0.0.0.0",
            f"--port={SERVER_PORT}",
            f"--fps={fps}",
            f"--inference_latency={inference_latency}",
            "--obs_queue_timeout=1",
        ]
        print("[policy_server] $", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd)

        deadline = time.time() + minutes * 60
        try:
            while time.time() < deadline:
                if proc.poll() is not None:
                    print(f"[policy_server] exited early, rc={proc.returncode}", flush=True)
                    break
                time.sleep(5)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
            hf_cache.commit()

    return {"address": address, "minutes": minutes, "returncode": proc.returncode}


@app.local_entrypoint()
def main(minutes: int = 30, fps: int = 30, dry_run: bool = False):
    est = round(minutes / 60 * 1.10, 2)
    print(f"Policy server on {DEFAULT_GPU} for {minutes} min. Estimated cost ~${est:.2f} "
          f"(A10G ~$1.10/h, billed for the whole window whether or not the arm moves).")
    if dry_run:
        print("--dry-run: nothing started, $0 spent.")
        return
    result = serve.remote(minutes=minutes, fps=fps)
    print(result)
