"""Run `lerobot-train` on Modal.

One Modal app, one image (lerobot + ACT/SmolVLA/pi0.5 deps), two Volumes:
  lerobot-hf-cache  -> /root/.cache/huggingface   (datasets, base models; persists across runs)
  lerobot-outputs   -> /outputs                   (checkpoints, one folder per job)

Dry run on the public SO-101 dataset:
    modal run experiments/modal_train.py --steps 2000 --batch-size 8
    # ~10 min image build the first time, then ~15 min of ACT on an A10G.

Real runs on your own dataset, e.g.:
    modal run experiments/modal_train.py --dataset $HF_USER/so101_blocks --policy act --steps 20000
    modal run experiments/modal_train.py --dataset $HF_USER/so101_blocks --policy smolvla --steps 20000 --batch-size 64 --gpu L40S
    modal run experiments/modal_train.py --dataset $HF_USER/so101_blocks --policy pi05 --steps 10000 --batch-size 32 --gpu A100-80GB

Add `--detach` after `modal run` to close the laptop while it trains:
    modal run --detach experiments/modal_train.py --steps 2000

Pull a finished checkpoint to the Mac (the run prints this exact command when it ends):
    modal run experiments/modal_train.py::pull --job-name <job_name>
    # (plain `modal volume get` on checkpoints/last fails: `last` is a symlink and the dir can hold a
    #  stray .tmp file from the save; `pull` resolves the numbered step dir and copies file by file.)

Optional: `export HF_TOKEN=...` locally before running to read private datasets / gated models
(pi0.5 needs the PaliGemma license accepted on the Hub + a token). `export WANDB_API_KEY=...` turns on W&B.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "lerobot-train"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DEFAULT_DATASET = "lerobot/svla_so101_pickplace"  # 50 eps, cameras: up + side, task: red cube -> gray bowl
DEFAULT_GPU = "A10G"

app = modal.App(APP_NAME)

# Torch from the cu128 index first (Modal's drivers are fine with it; PyPI's default cu130 wheel
# wants a newer driver floor). lerobot then sees torch already satisfied and won't swap it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("lerobot[training,smolvla,pi]")
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
outputs = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)

# Forward local HF / W&B tokens if they exist; empty otherwise. No `modal secret create` needed.
local_tokens = modal.Secret.from_dict(
    {
        k: v
        for k, v in {
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        }.items()
        if v
    }
)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
    secrets=[local_tokens],
    timeout=6 * 60 * 60,
    cpu=4,
    memory=16384,
)
def train(argv: list[str], job_name: str) -> dict:
    """Run one `lerobot-train` job. `argv` is the full flag list (output_dir/job_name included)."""
    out_dir = Path(OUTPUTS_DIR) / job_name
    if out_dir.exists() and any(out_dir.iterdir()):
        # lerobot refuses to overwrite a non-empty output_dir unless --resume=true.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = out_dir.with_name(f"{job_name}-{stamp}")
        argv = [a for a in argv if not a.startswith("--output_dir=")] + [f"--output_dir={out_dir}"]
        print(f"[modal_train] output dir existed; using {out_dir}")

    cmd = ["lerobot-train", *argv]
    print("[modal_train] $", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        tail.append(line)
        tail = tail[-60:]
    rc = proc.wait()
    elapsed = time.time() - t0

    hf_cache.commit()
    outputs.commit()

    last = out_dir / "checkpoints" / "last" / "pretrained_model"
    result = {
        "job_name": job_name,
        "returncode": rc,
        "elapsed_min": round(elapsed / 60, 1),
        "output_dir": str(out_dir),
        "checkpoint": str(last) if last.exists() else None,
        "tail": "".join(tail[-25:]) if rc != 0 else "",
    }
    print(f"[modal_train] done rc={rc} in {result['elapsed_min']} min; checkpoint={result['checkpoint']}")
    return result


def build_argv(
    dataset: str,
    policy: str,
    job_name: str,
    steps: int,
    batch_size: int,
    episodes: list[int] | None = None,
    pretrained: str | None = None,
    extra: str = "",
    rename_map: str = "",
    save_freq: int = 0,
) -> list[str]:
    """Assemble the lerobot-train flag list for act / smolvla / pi05 / diffusion.

    rename_map: JSON dict mapping the dataset's observation keys onto the pretrained policy's expected
    keys, e.g. '{"observation.images.top":"observation.images.camera1",
    "observation.images.wrist":"observation.images.camera2"}'. Needed whenever a base model (smolvla_base,
    pi05_base) was pretrained with different camera names than the dataset. Fine-tuning only."""
    argv = [
        f"--dataset.repo_id={dataset}",
        f"--output_dir={OUTPUTS_DIR}/{job_name}",
        f"--job_name={job_name}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq or steps}",
        "--log_freq=100",
        "--num_workers=4",
        f"--wandb.enable={'true' if os.environ.get('WANDB_API_KEY') else 'false'}",
    ]
    if episodes is not None:
        argv.append("--dataset.episodes=[" + ",".join(str(e) for e in episodes) + "]")
    if rename_map:
        argv.append(f"--rename_map={rename_map}")

    if policy == "act":
        argv.append(f"--policy.path={pretrained}" if pretrained else "--policy.type=act")
    elif policy == "diffusion":
        argv.append(f"--policy.path={pretrained}" if pretrained else "--policy.type=diffusion")
    elif policy == "smolvla":
        argv.append(f"--policy.path={pretrained or 'lerobot/smolvla_base'}")
    elif policy == "pi05":
        # Freeze the PaliGemma backbone, train the action expert only (fits on one GPU).
        argv += [
            "--policy.type=pi05",
            f"--policy.pretrained_path={pretrained or 'lerobot/pi05_base'}",
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
            "--policy.gradient_checkpointing=true",
            "--policy.dtype=bfloat16",
            "--policy.normalization_mapping={\"ACTION\": \"MEAN_STD\", \"STATE\": \"MEAN_STD\", \"VISUAL\": \"IDENTITY\"}",
        ]
    else:
        raise SystemExit(f"unknown policy {policy!r}; use act | diffusion | smolvla | pi05")

    if extra:
        argv += extra.split()
    return argv


def pull_command(job_name: str) -> str:
    return f"modal run experiments/modal_train.py::pull --job-name {job_name}"


@app.local_entrypoint()
def pull(job_name: str, dest: str = ""):
    """Copy <job>/checkpoints/<latest step>/pretrained_model from the outputs Volume to the Mac, file by file."""
    steps = sorted(
        e.path.rsplit("/", 1)[-1]
        for e in outputs.listdir(f"{job_name}/checkpoints")
        if e.path.rsplit("/", 1)[-1].isdigit()
    )
    if not steps:
        raise SystemExit(f"no numbered checkpoints under {job_name}/checkpoints")
    src = f"{job_name}/checkpoints/{steps[-1]}/pretrained_model"
    out = Path(dest or f"experiments/checkpoints/{job_name}")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in outputs.listdir(src):
        name = e.path.rsplit("/", 1)[-1]
        if name.startswith(".tmp"):
            continue
        with open(out / name, "wb") as fh:
            for chunk in outputs.read_file(e.path):
                fh.write(chunk)
        n += 1
    print(f"[pull] {n} files from {src} -> {out}")
    print(f"[pull] load with: ACTPolicy.from_pretrained('{out}')  (or the matching policy class)")


def parse_episodes(spec: str) -> list[int]:
    """"0-49" -> [0..49]; "0-24,50-74" -> both ranges; "3,7" -> [3, 7]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@app.local_entrypoint()
def main(
    dataset: str = DEFAULT_DATASET,
    policy: str = "act",
    steps: int = 2000,
    batch_size: int = 8,
    gpu: str = DEFAULT_GPU,
    job_name: str = "",
    pretrained: str = "",
    extra: str = "",
    episodes: str = "",
    rename_map: str = "",
    save_freq: int = 10000,
    dry_run: bool = False,
):
    """Train one policy on one dataset. See module docstring for examples.

    --save-freq N writes a checkpoint every N steps (default 10000) so a killed run keeps its progress;
    the final checkpoint is always written at --steps.

    --episodes "0-49" or "0-24,50-74" restricts training to those episode indices (inclusive ranges).
    """
    ep_list = parse_episodes(episodes) if episodes else None
    job = job_name or f"{policy}_{dataset.split('/')[-1]}_{steps}" + (f"_ep{episodes.replace(',', '_')}" if episodes else "")
    argv = build_argv(dataset, policy, job, steps, batch_size, episodes=ep_list, pretrained=pretrained or None, extra=extra, rename_map=rename_map, save_freq=save_freq)
    print(f"[modal_train] job={job} gpu={gpu}")
    print("[modal_train] lerobot-train", " ".join(argv))
    if dry_run:
        return
    fn = train if gpu == DEFAULT_GPU else train.with_options(gpu=gpu)
    # spawn + get, not .remote(): a spawned call keeps running on Modal if this terminal dies (a .remote()
    # call under `modal run --detach` did not survive the client being killed on Sept 9).
    handle = fn.spawn(argv, job)
    print(f"[modal_train] spawned; safe to close this terminal. Check with: modal app list")
    result = handle.get()
    print(result if result["returncode"] else {k: v for k, v in result.items() if k != "tail"})
    if result["returncode"] == 0 and result["checkpoint"]:
        print("\nPull the checkpoint to the Mac with:\n  " + pull_command(job))
    else:
        raise SystemExit(f"training failed (rc={result['returncode']}); last log lines above")
