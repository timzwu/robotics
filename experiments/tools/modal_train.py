"""Run `lerobot-train` on Modal.

One Modal app, one image (lerobot + ACT/SmolVLA/pi0.5 deps), two Volumes:
  lerobot-hf-cache  -> /root/.cache/huggingface   (datasets, base models; persists across runs)
  lerobot-outputs   -> /outputs                   (checkpoints, one folder per job)

Every launch prints a time + cost estimate first and then STOPS unless `--yes` is given (or `--dry-run`, which
costs nothing). Runs estimated over $40 also need `--confirm-cost`. Unknown (policy, GPU, batch) combos need a
200-step fit trial first, then their sec/step goes into MEASURED_SEC_PER_STEP below.

Dry run on the public SO-101 dataset:
    modal run experiments/tools/modal_train.py::main --steps 2000 --batch-size 8 --dry-run

Real runs on your own dataset, e.g.:
    modal run --detach experiments/tools/modal_train.py::main --dataset $HF_USER/so101_blocks --policy act --steps 20000 --yes
    modal run --detach experiments/tools/modal_train.py::main --dataset $HF_USER/so101_blocks --policy smolvla --steps 20000 --batch-size 64 --gpu L40S --yes
    modal run --detach experiments/tools/modal_train.py::main --dataset $HF_USER/so101_blocks --policy pi05 --steps 40000 --batch-size 32 --gpu H100 --gpus 2 --yes

`--detach` lets the laptop sleep; the spawned call keeps running and `modal app list` shows it.
`--gpus N` trains data-parallel on N cards with the SAME global batch (per-card batch = batch / N), so results stay
comparable; only wall time changes. It costs ~14% more than one card at the measured 0.88 scaling.

Continue an interrupted run (preemption, timeout, watchdog) from its last checkpoint, same card count:
    modal run --detach experiments/tools/modal_train.py::main --resume-from <job_name> --gpu H100 --gpus 2 --yes
A Modal preemption that re-executes the call resumes by itself (train() finds checkpoints/last in the job dir).

Pull a finished checkpoint to the Mac (the run prints this exact command when it ends):
    modal run experiments/tools/modal_train.py::pull --job-name <job_name>
    # (plain `modal volume get` on checkpoints/last fails: `last` is a symlink; `pull` takes the newest COMPLETE
    #  numbered step dir and copies file by file.)

`export HF_TOKEN=...` (or `hf auth login`) before running to read private datasets / gated models
(pi0.5 needs the PaliGemma license accepted on the Hub + a token). `export WANDB_API_KEY=...` turns on W&B.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

def _app_name_from_argv() -> str:
    """A descriptive Modal app name derived from the command line, e.g. pi05-40k-2xH100, resume-<job>,
    sweep-smolvla-20k-n10-25-50. The app list truncates after ~12 characters, so the model comes first."""
    import re as _re

    argv = sys.argv

    def opt(name: str, default: str = "") -> str:
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(name + "="):
                return a.split("=", 1)[1]
        return default

    if not any(a.endswith(("modal_train.py", "sweep.py")) or "modal_train.py::" in a or "sweep.py::" in a for a in argv):
        return "lerobot-train"  # imported from somewhere else (policy server, tests)
    sweep = any("sweep" in a for a in argv)
    policy = opt("--policy", "smolvla" if sweep else "act")
    steps = int(opt("--steps", "20000" if sweep else "2000") or 0)
    k = f"{steps // 1000}k" if steps >= 1000 else str(steps)
    if opt("--resume-from"):
        name = f"resume-{opt('--resume-from')}"
    elif sweep:
        name = f"sweep-{policy}-{k}-n{opt('--sizes', '10,25,50,100').replace(',', '-')}"
    elif "::pull" in " ".join(argv):
        name = f"pull-{opt('--job-name')}"
    elif opt("--job-name"):
        name = opt("--job-name")
    else:
        name = f"{policy}-{k}-{opt('--gpus', '1')}x{opt('--gpu', DEFAULT_GPU).upper()}"
    return _re.sub(r"[^A-Za-z0-9._-]", "-", name)[:63]


HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DEFAULT_DATASET = "lerobot/svla_so101_pickplace"  # 50 eps, cameras: up + side, task: red cube -> gray bowl
DEFAULT_GPU = "A10G"

app = modal.App(_app_name_from_argv())

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


def _hf_token() -> str:
    """HF_TOKEN from the environment, else the `hf auth login` file (so a forgotten export doesn't fail in the container)."""
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        for p in (Path.home() / ".cache" / "huggingface" / "token", Path.home() / ".huggingface" / "token"):
            if p.exists():
                tok = p.read_text().strip()
                break
    return tok


# Forward local HF / W&B tokens if they exist; empty otherwise. No `modal secret create` needed.
local_tokens = modal.Secret.from_dict(
    {k: v for k, v in {"HF_TOKEN": _hf_token(), "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}.items() if v}
)

CPU_PER_CARD = 4
MEM_GIB_PER_CARD = 24  # every rank loads pi0.5's 14.5 GB weights on the host at the same moment; 16 was too tight


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
    secrets=[local_tokens],
    timeout=24 * 60 * 60,  # Modal's hard maximum; _train_fn sets a tighter one from the estimate
    cpu=CPU_PER_CARD,
    memory=MEM_GIB_PER_CARD * 1024,
)
def train(argv: list[str], job_name: str, num_gpus: int = 1) -> dict:
    """Run one `lerobot-train` job in the container. `argv` is the full flag list (output_dir/job_name included).

    num_gpus > 1: `accelerate launch --multi_gpu` (data parallel). The launcher already divided --batch_size by
    num_gpus, so the GLOBAL batch, and therefore the recipe, is unchanged. LeRobot logs "Effective batch size: B x N".

    Resume rules (all in-place, the job dir is never renamed):
      * argv carries --resume=true              -> continue from the checkpoint named in --config_path.
      * job dir already has checkpoints/last    -> AUTO-RESUME from it. This is what a Modal preemption looks like
                                                   (the call is re-executed with the same input); before this rule
                                                   a preempted run restarted from step 0 in a renamed folder.
      * job dir non-empty but no checkpoint     -> leftovers moved to <job>-stale-<stamp>, fresh start in place.
    """
    out_dir = Path(OUTPUTS_DIR) / job_name
    last_cfg = out_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    resume = "--resume=true" in argv
    if not resume and last_cfg.exists():
        argv = [f"--config_path={last_cfg}", "--resume=true"]
        resume = True
        print(f"[modal_train] AUTO-RESUME: {out_dir} already holds a checkpoint (Modal re-execution?); continuing from it", flush=True)
    elif not resume and out_dir.exists() and any(out_dir.iterdir()):
        stale = out_dir.with_name(f"{job_name}-stale-{time.strftime('%Y%m%d-%H%M%S')}")
        out_dir.rename(stale)
        print(f"[modal_train] job dir had leftovers but no checkpoint; moved to {stale}", flush=True)

    if num_gpus > 1:
        cmd = ["accelerate", "launch", "--multi_gpu", f"--num_processes={num_gpus}",
               "-m", "lerobot.scripts.lerobot_train", *argv]
    else:
        cmd = ["lerobot-train", *argv]
    print("[modal_train] $", " ".join(cmd), flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    # start_new_session: the trainer (and torchrun's ranks, which each start their own session) must be killable
    # as a tree; errors="replace": a stray byte in the log must not raise inside this loop while the trainer runs.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace", start_new_session=True, env=env)
    wd = _arm_watchdog(proc, silence_s=45 * 60)  # a hung job must not bill until the container timeout
    tail: list[str] = []

    def reader():  # its own thread: an orphaned rank holding the pipe must never block this function's return
        for line in proc.stdout:  # type: ignore[union-attr]
            wd["t"] = time.time()
            sys.stdout.write(line)
            tail.append(line)
            del tail[:-60]

    import threading

    threading.Thread(target=reader, daemon=True).start()
    rc = proc.wait()  # returns when the trainer (or accelerate parent) is gone; the function then returns and the
    elapsed = time.time() - t0  # container is torn down, which ends anything still on the GPUs
    time.sleep(2)  # let the reader drain the last lines

    hf_cache.commit()
    outputs.commit()

    if resume:  # the saved config knows where the run actually lives
        try:
            cfg_path = next(a for a in argv if a.startswith("--config_path=")).split("=", 1)[1]
            out_dir = Path(json.loads(Path(cfg_path).read_text())["output_dir"])
        except Exception as e:  # noqa: BLE001
            print(f"[modal_train] could not read output_dir from the saved config ({e}); assuming {out_dir}")
    last = out_dir / "checkpoints" / "last" / "pretrained_model"
    result = {
        "job_name": job_name,
        "returncode": rc,
        "elapsed_min": round(elapsed / 60, 1),
        "output_dir": str(out_dir),
        "checkpoint": str(last) if last.exists() else None,
        "watchdog_fired": wd["fired"],
        "tail": "".join(tail[-25:]) if rc != 0 else "",
    }
    print(f"[modal_train] done rc={rc} in {result['elapsed_min']} min; checkpoint={result['checkpoint']}"
          + ("; KILLED BY WATCHDOG (no log output for 45 min)" if wd["fired"] else ""))
    return result


def _kill_tree(proc: subprocess.Popen) -> None:
    """End the trainer and everything under it. SIGTERM first: torchrun's handler kills every rank's process group
    (ranks run in their own sessions, so a plain kill of the parent would orphan them on the GPUs and leave the
    stdout pipe open). Then SIGKILL whatever is left, children before the parent (the tree is unreachable after)."""
    import signal

    kids = _descendants(proc.pid)  # before signalling: the tree is unreachable once the parent is gone
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=90)
    except Exception:  # noqa: BLE001
        pass
    for pid in kids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _descendants(pid: int) -> list[int]:
    """All processes under `pid`, via psutil when present (it ships with accelerate) else `ps`."""
    try:
        import psutil

        return [p.pid for p in psutil.Process(pid).children(recursive=True)]
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout.split()
        by_parent: dict[int, list[int]] = {}
        for c, par in zip(rows[0::2], rows[1::2]):
            by_parent.setdefault(int(par), []).append(int(c))
        out, todo = [], [pid]
        while todo:
            cur = todo.pop()
            for c in by_parent.get(cur, []):
                out.append(c)
                todo.append(c)
        return out
    except Exception:  # noqa: BLE001
        return []


def _arm_watchdog(proc: subprocess.Popen, silence_s: int) -> dict:
    """Kill the trainer tree if it prints nothing for `silence_s` seconds. The caller refreshes state["t"] per line.
    Healthy silences are short: a 14 GB checkpoint save or weight load is ~1-3 min, log_freq x sec/step is ~5 min."""
    import threading

    state = {"t": time.time(), "fired": False}

    def watch():
        while proc.poll() is None:
            time.sleep(max(1, min(60, silence_s // 4)))
            if proc.poll() is None and time.time() - state["t"] > silence_s:
                state["fired"] = True
                print(f"[modal_train] WATCHDOG: no output for {silence_s // 60} min, killing the trainer tree", flush=True)
                _kill_tree(proc)
                return

    threading.Thread(target=watch, daemon=True).start()
    return state


# Flags the launcher owns; `--extra` may not override them (draccus takes the LAST occurrence, silently).
MANAGED_KEYS = {"batch_size", "steps", "save_freq", "output_dir", "job_name", "dataset.repo_id", "dataset.episodes",
                "rename_map", "policy.path", "policy.pretrained_path", "policy.type", "resume", "config_path"}


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
    """Assemble the lerobot-train flag list for act / smolvla / pi05 / diffusion. batch_size here is PER PROCESS.

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
        if policy in ("act", "diffusion") and not pretrained:
            raise SystemExit("--rename-map needs a pretrained checkpoint (lerobot rejects it for a from-scratch run, after the container starts)")
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
        tokens = shlex.split(extra)  # keeps JSON values with spaces intact
        for tok in tokens:
            key = tok.lstrip("-").split("=", 1)[0]
            if key in MANAGED_KEYS:
                raise SystemExit(f"--extra may not set --{key}; use the launcher flag (the estimate would be wrong otherwise)")
        argv += tokens
    return argv


def pull_command(job_name: str) -> str:
    return f"modal run experiments/tools/modal_train.py::pull --job-name {job_name}"


# ---------------------------------------------------------------------------------------------------------------------
# Volume lookups from the Mac (no container). `last` is a symlink the SDK cannot follow, so everything goes through
# the numbered step dirs, and a step dir only counts when training_state/training_step.json exists: it is the LAST
# file lerobot writes, so its presence means the checkpoint is complete (background commits can snapshot a torn one).
# ---------------------------------------------------------------------------------------------------------------------
def _ls(path: str) -> list[str]:
    try:
        return [e.path.rsplit("/", 1)[-1] for e in outputs.listdir(path)]
    except Exception:  # noqa: BLE001  (NotFoundError and friends)
        return []


def _read_json(path: str) -> dict | None:
    try:
        return json.loads(b"".join(outputs.read_file(path)).decode())
    except Exception:  # noqa: BLE001
        return None


def _job_dirs(job: str) -> list[str]:
    """The job's folder plus any `<job>-<14-digit stamp>` copies the old launcher made on resume/preemption."""
    pat = re.compile(re.escape(job) + r"-\d{8}-\d{6}$")
    return [n for n in _ls("/") if n == job or pat.match(n)]


def latest_complete_checkpoint(job: str) -> tuple[str, str, dict] | None:
    """(job_dir, step_dir_name, training_step.json) of the newest complete checkpoint across the job's folders."""
    best = None
    for d in _job_dirs(job):
        for step in sorted((s for s in _ls(f"{d}/checkpoints") if s.isdigit()), reverse=True):
            state = _read_json(f"{d}/checkpoints/{step}/training_state/training_step.json")
            if state and (best is None or int(step) > int(best[1])):
                best = (d, step, state)
                break
    return best


@app.local_entrypoint()
def pull(job_name: str, dest: str = ""):
    """Copy the newest COMPLETE <job>/checkpoints/<step>/pretrained_model from the outputs Volume to the Mac."""
    found = latest_complete_checkpoint(job_name)
    if not found:
        raise SystemExit(f"no complete numbered checkpoint under {job_name}/checkpoints (folders seen: {_job_dirs(job_name) or 'none'})")
    d, step, state = found
    if d != job_name:
        print(f"[pull] newest checkpoint lives in {d} (old launcher suffix), step {int(step)}")
    src = f"{d}/checkpoints/{step}/pretrained_model"
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
    print(f"[pull] {n} files from {src} (step {state.get('step')}, saved with {state.get('num_processes')} card(s)) -> {out}")
    print(f"[pull] load with: ACTPolicy.from_pretrained('{out}')  (or the matching policy class)")


# ---------------------------------------------------------------------------------------------------------------------
# Money and time. Every number here is checked against experiments/results/training_runs.md.
# ---------------------------------------------------------------------------------------------------------------------
GPU_USD_PER_HOUR = {  # Modal list prices, verified Sept 12 2026
    "T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100": 2.10, "A100-40GB": 2.10, "A100-80GB": 2.50,
    "H100": 3.95, "H200": 4.54, "B200": 6.25,
}
CPU_USD_PER_CORE_HOUR = 0.047   # Modal bills CPU and RAM on top of the GPU: +8% (H100) to +29% (A10G) per card
MEM_USD_PER_GIB_HOUR = 0.008
# Single-card seconds per step measured on our runs, keyed by (policy, gpu, GLOBAL batch). Add a row after every fit
# trial on a new combo; the estimate refuses to guess. (Camera count changes it: one-camera SmolVLA ran 0.33 s/step.)
MEASURED_SEC_PER_STEP = {
    ("act", "A10G", 8): 0.166,
    ("smolvla", "L40S", 64): 0.58,   # sweep runs; the n100 run was 0.53
    ("pi05", "A100-80GB", 32): 3.07,
    ("pi05", "H100", 32): 1.19,
}
MULTI_CARD_SCALING = 0.88   # MEASURED Sept 12: pi0.5 on 2 x H100 ran 0.66 s/step vs 1.18 on one (1.77x); one H100 node, NVLink
STARTUP_H = 0.1             # container start + dataset/model pull (pi0.5 cold cache was ~10 min)
MAX_GPUS_PER_JOB = 4
MAX_RUN_HOURS = 20.0        # Modal kills a call at 24 h; a run estimated past this is refused instead of launched
CONFIRM_COST_USD = 40.0     # runs estimated above this need --confirm-cost (~4% of the credits)


def rate_per_hour(gpu: str, gpus: int) -> float:
    """$/h for N cards INCLUDING the CPU and RAM each card is given."""
    return gpus * (GPU_USD_PER_HOUR[gpu] + CPU_PER_CARD * CPU_USD_PER_CORE_HOUR + MEM_GIB_PER_CARD * MEM_USD_PER_GIB_HOUR)


def hours_for(steps: int, sec_per_step: float, gpus: int) -> float:
    compute_h = steps * sec_per_step / 3600
    return (compute_h / (gpus * MULTI_CARD_SCALING) if gpus > 1 else compute_h) + STARTUP_H


def triage(gpus: int, global_batch: int, sec_per_step: float, steps: int) -> str:
    """Printed recommendation. Cost only goes UP with more cards; the decision is wall time and the 24 h cap."""
    h1 = hours_for(steps, sec_per_step, 1)
    need = next((n for n in (1, 2, 4) if hours_for(steps, sec_per_step, n) <= MAX_RUN_HOURS), None)
    premium = f"+{100 * (1 / MULTI_CARD_SCALING - 1):.0f}%"
    if need is None:
        return f"no card count finishes under {MAX_RUN_HOURS:.0f} h; split the run (--steps then --resume-from) or use a faster GPU"
    if need > 1:
        return f"recommend {need} cards: the fewest that finish under {MAX_RUN_HOURS:.0f} h (1 card would need ~{h1:.1f} h; Modal kills at 24 h)"
    if global_batch // max(gpus, 2) < 8:
        return "recommend 1 card: per-card batch would drop below 8, the GPU is under-fed and DDP gives no speedup"
    if sec_per_step < 0.5:
        return "recommend 1 card: steps are short (small model / I/O-bound); more cards won't help. Sweeps parallelize across JOBS"
    if h1 < 4:
        return f"recommend 1 card: run is under 4 h; multi-card saves little for {premium} cost and a new failure mode"
    if h1 < 10:
        return "recommend 2 cards (after a 200-step trial on 2 cards to measure real scaling)"
    return "recommend 2 cards, 4 only if the result is needed the same day; trial first"


def estimate(policy: str, gpu: str, gpus: int, steps: int, global_batch: int, sec_per_step: float, confirm_cost: bool) -> float:
    """Print wall time and cost before any launch; refuse runs that are unknown, too long, or too expensive.
    Returns the estimated hours (used to size the container timeout)."""
    if gpus < 1 or gpus > MAX_GPUS_PER_JOB:
        raise SystemExit(f"--gpus must be 1..{MAX_GPUS_PER_JOB} (got {gpus})")
    if gpu not in GPU_USD_PER_HOUR:
        raise SystemExit(f"unknown GPU {gpu!r}; known: {', '.join(GPU_USD_PER_HOUR)} (add a price to GPU_USD_PER_HOUR for a new one)")
    rate = rate_per_hour(gpu, gpus)
    measured = MEASURED_SEC_PER_STEP.get((policy, gpu, global_batch), 0.0)
    if sec_per_step and measured and not (0.5 <= sec_per_step / measured <= 2.0) and not confirm_cost:
        raise SystemExit(f"--sec-per-step {sec_per_step} is far from the measured {measured} s/step for ({policy}, {gpu}, batch {global_batch}); "
                         f"check the number, or pass --confirm-cost to use it anyway")
    sec = sec_per_step or measured
    source = "--sec-per-step" if sec_per_step else "measured table"
    if not sec:
        if steps <= 1000:
            bound_h = 2.0  # the timeout for a trial; the bill can't exceed this
            print(f"[modal_train] ESTIMATE: fit trial ({steps} steps) on {gpus} x {gpu}, no measured sec/step. Timeout {bound_h:.0f} h "
                  f"-> worst case ~${bound_h * rate:.2f} (${rate:.2f}/h). Read updt_s + data_s from the late log lines and add "
                  f"the combo to MEASURED_SEC_PER_STEP")
            if bound_h * rate > CONFIRM_COST_USD and not confirm_cost:
                raise SystemExit(f"[modal_train] trial worst case ${bound_h * rate:.0f} exceeds ${CONFIRM_COST_USD:.0f}; re-run with --confirm-cost")
            return 0.5  # -> 2 h timeout
        raise SystemExit(f"[modal_train] no measured sec/step for ({policy}, {gpu}, batch {global_batch}); run a 200-step "
                         f"fit trial (--steps 200) and pass --sec-per-step from its late log lines (updt_s + data_s)")
    h1 = hours_for(steps, sec, 1)
    hn = hours_for(steps, sec, gpus)
    cost = hn * rate
    print(f"[modal_train] ESTIMATE: {gpus} x {gpu} for {steps} steps at {sec} s/step ({source}): ~{hn:.1f} h, ~${cost:.2f} "
          f"(${rate:.2f}/h incl. CPU+RAM)"
          + (f"; 1 card would be ~{h1:.1f} h, ~${h1 * rate_per_hour(gpu, 1):.2f} (multi-card costs "
             f"+{100 * (1 / MULTI_CARD_SCALING - 1):.0f}% at {MULTI_CARD_SCALING} scaling)" if gpus > 1 else ""))
    print(f"[modal_train] {triage(gpus, global_batch, sec, steps)}")
    if hn > MAX_RUN_HOURS:
        need = next((n for n in (2, 4) if hours_for(steps, sec, n) <= MAX_RUN_HOURS), None)
        raise SystemExit(f"[modal_train] ~{hn:.1f} h exceeds {MAX_RUN_HOURS:.0f} h (Modal kills the call at 24 h; attempt 1 of pi0.5 died this way). "
                         + (f"Use --gpus {need}." if need else "Split the run: fewer --steps now, then --resume-from."))
    if cost > CONFIRM_COST_USD and not confirm_cost:
        raise SystemExit(f"[modal_train] estimated ${cost:.0f} exceeds ${CONFIRM_COST_USD:.0f}; re-run with --confirm-cost")
    return hn


def _train_fn(gpu: str, gpus: int = 1, timeout_s: int | None = None):
    """The train function sized for the request: N cards of `gpu`, CPU and RAM scaled to match, and the container
    timeout set from the estimate (2 x expected + 1 h, capped at 24 h) so a hung job cannot bill for a day."""
    if gpus < 1 or gpus > MAX_GPUS_PER_JOB:
        raise SystemExit(f"--gpus must be 1..{MAX_GPUS_PER_JOB} (got {gpus})")
    opts = {}
    if timeout_s:
        opts["timeout"] = int(min(timeout_s, 24 * 60 * 60))
    if gpus > 1:
        opts.update(gpu=f"{gpu}:{gpus}", cpu=CPU_PER_CARD * gpus, memory=MEM_GIB_PER_CARD * 1024 * gpus)
    elif gpu != DEFAULT_GPU:
        opts["gpu"] = gpu
    return train.with_options(**opts) if opts else train


def launch(gpu: str, gpus: int, argv: list[str], job: str, hours: float, yes: bool) -> dict:
    """The one place a container is started. Requires --yes (the estimate was printed first) and reports truthfully."""
    if not yes:
        raise SystemExit("[modal_train] estimate printed above; nothing launched. Re-run with --yes to start it (or --dry-run)")
    if not any(a in ("--detach", "-d") for a in sys.argv):
        print("[modal_train] WARNING: without `modal run --detach` the app stops if this terminal closes or the laptop sleeps", flush=True)
    fn = _train_fn(gpu, gpus, timeout_s=int(hours * 2 * 3600 + 3600))
    # spawn + get, not .remote(): a spawned call keeps running on Modal if this client dies under --detach
    # (a .remote() call under `modal run --detach` did not survive the client being killed on Sept 9).
    handle = fn.spawn(argv, job, gpus)
    print(f"[modal_train] spawned {job} on {gpus} x {gpu}, timeout {min(hours * 2 + 1, 24):.1f} h. Check with: modal app list", flush=True)
    try:
        result = handle.get()
    except Exception as e:  # noqa: BLE001  (FunctionTimeoutError, InternalFailure, ...)
        raise SystemExit(f"[modal_train] the Modal call ended abnormally: {type(e).__name__}: {e}\n"
                         f"If a checkpoint was saved, continue with: modal run --detach experiments/tools/modal_train.py::main "
                         f"--resume-from {job} --gpu {gpu} --gpus {gpus} --yes")
    print(result if result["returncode"] else {k: v for k, v in result.items() if k != "tail"})
    if result["returncode"] == 0 and result["checkpoint"]:
        print("\nPull the checkpoint to the Mac with:\n  " + pull_command(Path(result["output_dir"]).name))
    else:
        why = "killed by the watchdog (no log output for 45 min)" if result.get("watchdog_fired") else f"rc={result['returncode']}"
        raise SystemExit(f"[modal_train] training failed ({why}); last log lines above. Resume with --resume-from {job} --gpus {gpus}")
    return result


def parse_episodes(spec: str) -> list[int]:
    """"0-49" -> [0..49]; "0-24,50-74" -> both ranges; "3,7" -> [3, 7]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise SystemExit(f"--episodes: bad item {part!r}; use e.g. 0-49 or 0-24,50-74")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if b < a:
            raise SystemExit(f"--episodes: range {part!r} is backwards")
        out.extend(range(a, b + 1))
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
    resume_from: str = "",
    gpus: int = 1,
    sec_per_step: float = 0.0,
    confirm_cost: bool = False,
    yes: bool = False,
    dry_run: bool = False,
):
    """Train one policy on one dataset. See the module docstring for examples.

    Flow: build the command -> print the estimate and a card-count recommendation -> refuse if unknown / over 20 h /
    over $40 without --confirm-cost -> stop unless --yes -> spawn on Modal -> print the pull command.

    --gpus N: data-parallel on N cards of --gpu. --batch-size stays the GLOBAL batch (per-card = batch / N), so the
    recipe is unchanged and results stay comparable; only wall time changes.

    --resume-from <job>: continue an interrupted run from its newest complete checkpoint (weights, optimizer,
    scheduler, step). Must use the same card count it was saved with (the checkpoint stores the per-card batch).

    --save-freq N writes a checkpoint every N steps (default 10000); the final one is always written at --steps.

    --episodes "0-49" or "0-24,50-74" restricts training to those episode indices (inclusive ranges).
    """
    gpu = gpu.upper()
    if resume_from:
        found = latest_complete_checkpoint(resume_from)
        if not found:
            raise SystemExit(f"--resume-from {resume_from}: no complete checkpoint on the Volume (folders seen: {_job_dirs(resume_from) or 'none'})")
        job, step_dir, state = found
        cfg = _read_json(f"{job}/checkpoints/{step_dir}/pretrained_model/train_config.json") or {}
        saved_gpus, per_card, step = int(state.get("num_processes", 1)), int(state["batch_size"]), int(state["step"])
        total_steps = int(cfg.get("steps", 0))
        pol = (cfg.get("policy") or {}).get("type", "?")
        remaining = total_steps - step
        if remaining <= 0:
            raise SystemExit(f"{job} already reached step {step} of {total_steps}; nothing to resume")
        if gpus != saved_gpus:
            raise SystemExit(f"{job} was saved with {saved_gpus} card(s) at per-card batch {per_card} (global {per_card * saved_gpus}); "
                             f"pass --gpus {saved_gpus}. A different count would silently change the global batch")
        argv = [f"--config_path={OUTPUTS_DIR}/{job}/checkpoints/{step_dir}/pretrained_model/train_config.json", "--resume=true"]
        print(f"[modal_train] RESUME {job} from step {step} of {total_steps} ({remaining} to go), {pol}, {saved_gpus} x {gpu}, "
              f"global batch {per_card * saved_gpus}")
        print("[modal_train] lerobot-train", " ".join(argv))
        hours = estimate(pol, gpu, saved_gpus, remaining, per_card * saved_gpus, sec_per_step, confirm_cost)
        if dry_run:
            return
        launch(gpu, saved_gpus, argv, job, hours, yes)
        return

    ep_list = parse_episodes(episodes) if episodes else None
    if gpus < 1 or gpus > MAX_GPUS_PER_JOB:
        raise SystemExit(f"--gpus must be 1..{MAX_GPUS_PER_JOB} (got {gpus})")
    if batch_size % gpus:
        raise SystemExit(f"--batch-size {batch_size} must divide by --gpus {gpus} (the global batch is kept)")
    global_batch = batch_size
    per_card = batch_size // gpus
    job = job_name or f"{policy}_{dataset.split('/')[-1]}_{steps}" + (f"_ep{episodes.replace(',', '_')}" if episodes else "")
    if _job_dirs(job):
        raise SystemExit(f"{job} already exists on the Volume ({', '.join(_job_dirs(job))}). Use --resume-from {job} to continue it, "
                         f"or --job-name for a new run")
    argv = build_argv(dataset, policy, job, steps, per_card, episodes=ep_list, pretrained=pretrained or None, extra=extra,
                      rename_map=rename_map, save_freq=save_freq)
    print(f"[modal_train] job={job} gpu={gpu}" + (f" x{gpus} (per-card batch {per_card}, global {global_batch})" if gpus > 1 else ""))
    print("[modal_train] lerobot-train", " ".join(argv))
    hours = estimate(policy, gpu, gpus, steps, global_batch, sec_per_step, confirm_cost)
    if dry_run:
        return
    launch(gpu, gpus, argv, job, hours, yes)
