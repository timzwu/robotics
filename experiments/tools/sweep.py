"""Data-scaling sweep: train the same policy on nested, seeded episode subsets, in parallel on Modal.

Subsets are nested (the 10 are inside the 25, inside the 50, inside the 100) from one seeded shuffle,
so the curve measures "more of the same data", not different data.

Dry run on the public dataset (50 episodes), tiny steps, just to prove parallel launch:
    modal run experiments/tools/sweep.py::sweep --sizes 5,10 --steps 200

Real sweep on your own dataset:
    modal run experiments/tools/sweep.py::sweep --dataset $HF_USER/so101_blocks --policy smolvla \
        --sizes 10,25,50,100 --steps 20000 --batch-size 64 --gpu L40S

Writes experiments/results/02_data_scaling/sweep_<policy>_<seed>.json (the subsets + job names + results) so eval.py and
the plot know exactly which episodes each checkpoint saw. Add `--dry-run` to print commands only.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal  # noqa: E402

from modal_train import (  # noqa: E402
    CONFIRM_COST_USD, DEFAULT_DATASET, DEFAULT_GPU, MAX_GPUS_PER_JOB, _job_dirs, _train_fn, app, build_argv, estimate,
    parse_episodes, pull_command, rate_per_hour,
)

MAX_CONCURRENT_GPUS = 4  # jobs x cards launched at once; --allow-many-gpus overrides

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "02_data_scaling"


def dataset_num_episodes(repo_id: str) -> int:
    """Read total_episodes from meta/info.json on the Hub without downloading the dataset.

    Sends the Hugging Face token (HF_TOKEN, or the `hf auth login` file) so private datasets work too.
    """
    from urllib.request import Request, urlopen

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        token = token_file.read_text().strip() if token_file.exists() else ""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json"
    req = Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urlopen(req, timeout=30) as r:  # noqa: S310
        return int(json.load(r)["total_episodes"])


def nested_subsets(total: int, sizes: list[int], seed: int, strata: int = 0, pool: list[int] | None = None) -> dict[int, list[int]]:
    """Nested seeded subsets. With strata=K, episodes are grouped in contiguous blocks of K (one task
    per block, the way record_r2.sh lays them out) and every subset takes an equal share from each
    block, so a 10-episode subset of a four-task dataset holds 3/3/2/2 rather than whatever a plain
    shuffle happens to give (seed 0 gave 6/1/1/2)."""
    rng = random.Random(seed)
    ids = pool if pool is not None else list(range(total))
    if not strata:
        order = list(ids)
        rng.shuffle(order)
        return {n: sorted(order[:n]) for n in sizes}
    blocks = [ids[i : i + strata] for i in range(0, len(ids), strata)]
    for b in blocks:
        rng.shuffle(b)
    out = {}
    for n in sizes:
        picked, i = [], 0
        while len(picked) < n:  # round-robin over the blocks so shares differ by at most one
            b = blocks[i % len(blocks)]
            k = i // len(blocks)
            if k < len(b):
                picked.append(b[k])
            i += 1
        out[n] = sorted(picked)
    return out


@app.local_entrypoint()
def sweep(
    dataset: str = DEFAULT_DATASET,
    policy: str = "smolvla",
    sizes: str = "10,25,50,100",
    steps: int = 20000,
    batch_size: int = 64,
    gpu: str = DEFAULT_GPU,
    seed: int = 0,
    strata: int = 0,
    pool: str = "",
    rename_map: str = "",
    save_freq: int = 10000,
    gpus: int = 1,
    sec_per_step: float = 0.0,
    confirm_cost: bool = False,
    allow_many_gpus: bool = False,
    yes: bool = False,
    tag: str = "",
    dry_run: bool = False,
):
    wanted = [int(s) for s in sizes.split(",") if s.strip()]
    total = dataset_num_episodes(dataset)
    pool_ids = parse_episodes(pool) if pool else None
    limit = len(pool_ids) if pool_ids else total
    usable = [n for n in wanted if n <= limit]
    skipped = [n for n in wanted if n > limit]
    if skipped:
        print(f"[sweep] {'pool' if pool_ids else 'dataset'} has {limit} episodes; skipping sizes {skipped}")
    if not usable:
        raise SystemExit("[sweep] nothing to run")

    subsets = nested_subsets(total, usable, seed, strata, pool_ids)
    name = dataset.split("/")[-1]
    jobs = {n: f"{policy}_{name}_n{n}_s{seed}{('_' + tag) if tag else ''}" for n in usable}

    manifest = {
        "dataset": dataset,
        "policy": policy,
        "seed": seed,
        "strata": strata,
        "pool": pool,
        "gpus": gpus,
        "steps": steps,
        "batch_size": batch_size,
        "gpu": gpu,
        "subsets": {str(n): subsets[n] for n in usable},
        "jobs": {str(n): jobs[n] for n in usable},
        "results": {},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / f"sweep_{policy}_{name}_s{seed}{('_' + tag) if tag else ''}.json"

    gpu = gpu.upper()
    if gpus < 1 or gpus > MAX_GPUS_PER_JOB:
        raise SystemExit(f"--gpus must be 1..{MAX_GPUS_PER_JOB} (got {gpus})")
    if batch_size % gpus:
        raise SystemExit(f"--batch-size {batch_size} must divide by --gpus {gpus} (the global batch is kept)")
    existing = [jobs[n] for n in usable if _job_dirs(jobs[n])]
    if existing:
        raise SystemExit(f"[sweep] already on the Volume: {', '.join(existing)}; use --tag for a new sweep or --resume-from per job")
    argvs = {n: build_argv(dataset, policy, jobs[n], steps, batch_size // gpus, episodes=subsets[n], rename_map=rename_map, save_freq=save_freq) for n in usable}
    for n in usable:
        print(f"[sweep] n={n:>3} job={jobs[n]} episodes={subsets[n]}")
        print("        lerobot-train", " ".join(argvs[n]))
    total = len(usable) * gpus
    if total > MAX_CONCURRENT_GPUS and not allow_many_gpus:
        raise SystemExit(f"[sweep] {len(usable)} jobs x {gpus} cards = {total} {gpu}s at once "
                         f"(~${rate_per_hour(gpu, total):.0f}/h); pass --allow-many-gpus to override")
    hours = estimate(policy, gpu, gpus, steps, batch_size, sec_per_step, confirm_cost)  # per job (its own gates apply)
    total_cost = hours * rate_per_hour(gpu, gpus) * len(usable)
    print(f"[sweep] {len(usable)} jobs in parallel: total ~${total_cost:.0f}")
    if total_cost > CONFIRM_COST_USD and not confirm_cost:
        raise SystemExit(f"[sweep] total ~${total_cost:.0f} exceeds ${CONFIRM_COST_USD:.0f}; re-run with --confirm-cost")
    if dry_run:
        print("[sweep] dry run; nothing launched, manifest not written")
        return
    if not yes:
        raise SystemExit("[sweep] estimate printed above; nothing launched. Re-run with --yes to start (or --dry-run)")
    if not any(a in ("--detach", "-d") for a in sys.argv):
        print("[sweep] WARNING: without `modal run --detach` the app stops if this terminal closes or the laptop sleeps")

    fn = _train_fn(gpu, gpus, timeout_s=int(hours * 2 * 3600 + 3600))
    handles = {n: fn.spawn(argvs[n], jobs[n], gpus) for n in usable}  # all sizes train at once
    print(f"[sweep] launched {len(handles)} jobs on {gpu}; waiting...")

    for n, h in handles.items():
        try:
            res = h.get()
        except Exception as e:  # one failure shouldn't hide the others
            res = {"job_name": jobs[n], "returncode": -1, "error": repr(e)}
        manifest["results"][str(n)] = {k: v for k, v in res.items() if k != "tail"}
        if res.get("output_dir"):
            jobs[n] = Path(res["output_dir"]).name  # the folder the run actually wrote (pull must use it)
        status = "ok" if res.get("returncode") == 0 else f"FAILED rc={res.get('returncode')}"
        print(f"[sweep] n={n:>3} {status} {res.get('elapsed_min', '?')} min")
        if res.get("tail"):
            print(res["tail"])
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[sweep] manifest: {manifest_path}")
    print("[sweep] pull checkpoints with:")
    for n in usable:
        if manifest["results"].get(str(n), {}).get("returncode") == 0:
            print("  " + pull_command(jobs[n]))
