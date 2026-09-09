"""20-position evaluation protocol for "put the [red|blue] block in the [left|right] bowl".

Runs on the Mac. Walks the numbered dot stickers in order, tells you which block and bowl, runs the
policy for a fixed duration, then asks for the outcome. Every trial is appended to the CSV immediately,
so a crash or a break loses nothing; re-running the same command resumes at the first unfinished position.

Modes (--mode):
  local   ACT on the Mac:      runs `lerobot-rollout --strategy.type=base --policy.path=<ckpt>`
  async   SmolVLA / pi0.5:     runs `python -m lerobot.async_inference.robot_client` against a policy
                               server (Modal, or local) for --duration seconds, then stops it
  sync    SmolVLA / pi0.5 on the Mac: runs `experiments/sync_rollout.py` (predict a chunk, execute it,
                               repeat; ~0.8 s think pause per chunk). The eval path used for R3/R4.
  manual  no robot command:    you run the policy yourself; this just drives the protocol + CSV.
                               Use it to test the script without hardware.

Examples:
    python experiments/eval.py --mode manual --name test --positions 3 --out /tmp/eval_test.csv
    python experiments/eval.py --mode local --name act_50 --policy experiments/checkpoints/act_so101_blocks_20000
    python experiments/eval.py --mode async --name smolvla_50 --policy-type smolvla \
        --policy $HF_USER/smolvla_blocks_50 --server 127.0.0.1:8080
    python experiments/eval.py --summary experiments/results/act_50.csv     # just recompute the summary

Robot config comes from experiments/robot.json (ports, arm ids, cameras for this rig).
Output: experiments/results/<name>.csv + a printed summary with a 95% Wilson interval and failure breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
ROBOT_JSON = HERE / "robot.json"

FIELDS = [
    "name", "position", "color", "bowl", "task", "success", "failure_type", "duration_s",
    "notes", "timestamp", "policy", "mode",
]
FAILURE_TYPES = {
    "1": "no_reach",        # never got near the block
    "2": "touch_no_grip",   # reached / touched the block but never closed on it
    "3": "drop",            # grasped, dropped before the bowl
    "4": "wrong_bowl",      # placed in the other bowl
    "5": "no_move",         # policy froze / barely moved
    "6": "timeout",         # still going when time ran out
    "7": "collision",       # hit bowl/table/arm hard enough to stop (incl. grasped-then-hit-bowl)
    "8": "other",
}
# Four block/bowl combos, cycled so every 4 positions cover all of them.
COMBOS = [("red", "left"), ("blue", "right"), ("red", "right"), ("blue", "left")]


def combo_for(position: int, combos: list[tuple[str, str]] = COMBOS) -> tuple[str, str]:
    return combos[(position - 1) % len(combos)]


def parse_combos(spec: str) -> list[tuple[str, str]]:
    """"red:left,blue:right" -> [("red","left"),("blue","right")]. Empty -> all four."""
    if not spec:
        return COMBOS
    out = []
    for part in spec.split(","):
        color, bowl = part.strip().split(":")
        if (color, bowl) not in COMBOS:
            raise SystemExit(f"unknown combo {part!r}; choose from {COMBOS}")
        out.append((color, bowl))
    return out


def task_for(color: str, bowl: str) -> str:
    return f"put the {color} block in the {bowl} bowl"


def load_robot() -> dict:
    if not ROBOT_JSON.exists():
        sys.exit(f"missing {ROBOT_JSON}. Create it with port, id, and cameras for this rig.")
    return json.loads(ROBOT_JSON.read_text())


def rename_cameras(cams: dict, spec: str) -> dict:
    """"top=camera1,wrist=camera2" -> the same camera dict under the policy's names. For checkpoints
    trained with --rename_map (smolvla_base / pi05_base fine-tunes expect camera1/camera2)."""
    if not spec:
        return cams
    mapping = dict(part.split("=") for part in spec.split(","))
    return {mapping.get(k, k): v for k, v in cams.items()}


def cameras_arg(cams: dict) -> str:
    inner = ", ".join(
        f"{k}: {{type: {v['type']}, index_or_path: {v['index_or_path']}, width: {v['width']}, "
        f"height: {v['height']}, fps: {v['fps']}}}"
        for k, v in cams.items()
    )
    return "{ " + inner + " }"


def build_command(args, task: str) -> list[str] | None:
    if args.mode == "manual":
        return None
    robot = load_robot()
    common = [
        f"--robot.type={robot.get('robot_type', 'so101_follower')}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--robot.cameras={cameras_arg(rename_cameras(robot['cameras'], args.camera_rename))}",
    ]
    # Safety clamp: max degrees the follower may move per control step. Keeps a bad policy from
    # lunging into the table on a first zero-shot run. Omitted entirely when absent/null.
    clamp = robot.get("max_relative_target") if args.clamp is None else (args.clamp or None)
    if clamp is not None:
        common.append(f"--robot.max_relative_target={clamp}")
    if args.mode == "local":
        if not args.policy:
            sys.exit("--policy (checkpoint path or Hub id) is required for --mode local")
        return [
            "lerobot-rollout", "--strategy.type=base", f"--policy.path={args.policy}",
            f"--policy.device={args.local_device}", *common,
            f"--task={task}", f"--duration={args.duration}",
        ]
    if args.mode == "sync":
        if not (args.policy and args.policy_type):
            sys.exit("--policy and --policy-type are required for --mode sync")
        cmd = [
            sys.executable, str(HERE / "sync_rollout.py"), f"--policy={args.policy}",
            f"--policy-type={args.policy_type}", f"--task={task}", f"--duration={args.duration}",
            f"--robot-config={ROBOT_JSON}",
        ]
        if args.camera_rename:
            cmd.append(f"--camera-rename={args.camera_rename}")
        if args.clamp is not None:
            cmd.append(f"--clamp={args.clamp}")
        return cmd
    if args.mode == "async":
        if not (args.policy and args.policy_type):
            sys.exit("--policy and --policy-type are required for --mode async")
        server = args.server or robot.get("async_server", "127.0.0.1:8080")
        return [
            sys.executable, "-m", "lerobot.async_inference.robot_client",
            f"--server_address={server}", *common,
            f"--task={task}", f"--policy_type={args.policy_type}",
            f"--pretrained_name_or_path={args.policy}", f"--policy_device={args.policy_device}",
            f"--actions_per_chunk={args.actions_per_chunk}",
            f"--chunk_size_threshold={args.chunk_size_threshold}",
        ]
    sys.exit(f"unknown mode {args.mode}")


def run_trial(cmd: list[str] | None, duration: float, start_marker: str | None = None) -> float:
    """Run the policy command for up to `duration` seconds. Returns wall time actually used."""
    t0 = time.time()
    if cmd is None:
        input(f"  Run the policy now (~{duration:.0f}s). Press Enter when the trial is over... ")
        return time.time() - t0
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    if start_marker is None:
        proc = subprocess.Popen(cmd)
        try:
            proc.wait(timeout=duration + 60)  # rollout stops itself at --duration; +60 s for model load and think pauses
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
            raise
        return time.time() - t0

    # Async client: the clock starts only when the control loop starts (after the server has loaded the
    # model, which can take 20-30 s on the first trial), so every trial gets the full `duration` of motion.
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # else the client's log lines arrive in blocks, late
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    started = None
    try:
        for line in proc.stdout:
            print("   ", line.rstrip()[:160])
            if started is None and start_marker in line:
                started = time.time()
                print(f"  [eval] control loop started; running {duration:.0f}s")
            if started is not None and time.time() - started >= duration:
                break
            if started is None and time.time() - t0 > duration + 120:
                print("  [eval] client never started its control loop; stopping")
                break
        proc.send_signal(signal.SIGINT)   # robot_client runs forever; stop it cleanly
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        raise
    return (time.time() - started) if started else (time.time() - t0)


def ask_outcome() -> tuple[int, str, str]:
    while True:
        s = input("  Success? [y/n] ").strip().lower()
        if s in ("y", "n"):
            break
    if s == "y":
        notes = input("  Notes (optional): ").strip()
        return 1, "", notes
    menu = "  ".join(f"{k}={v}" for k, v in FAILURE_TYPES.items())
    while True:
        f = input(f"  Failure type [{menu}]: ").strip()
        if f in FAILURE_TYPES:
            break
        if f in FAILURE_TYPES.values():
            break
    ftype = FAILURE_TYPES.get(f, f)
    notes = input("  Notes (optional): ").strip()
    return 0, ftype, notes


def done_positions(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {int(r["position"]) for r in csv.DictReader(fh)}


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(path: Path) -> None:
    if not path.exists():
        print(f"no results at {path}")
        return
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    k = sum(int(r["success"]) for r in rows)
    lo, hi = wilson(k, n)
    print(f"\n== {path.name}: {k}/{n} = {100 * k / max(n, 1):.0f}% success  (95% CI {100 * lo:.0f}–{100 * hi:.0f}%)")
    fails = Counter(r["failure_type"] for r in rows if r["success"] == "0")
    if fails:
        print("   failures:", ", ".join(f"{t} ×{c}" for t, c in fails.most_common()))
    by_combo = Counter()
    tot_combo = Counter()
    for r in rows:
        key = f"{r['color']}→{r['bowl']}"
        tot_combo[key] += 1
        by_combo[key] += int(r["success"])
    print("   by combo:", ", ".join(f"{c} {by_combo[c]}/{tot_combo[c]}" for c in sorted(tot_combo)))
    failed_pos = sorted(int(r["position"]) for r in rows if r["success"] == "0")
    if failed_pos:
        print("   failed positions:", failed_pos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="condition name, e.g. act_50 (used for the CSV filename)")
    ap.add_argument("--mode", choices=["local", "async", "sync", "manual"], default="manual")
    ap.add_argument("--policy", default="", help="checkpoint dir / Hub id (local), or Hub id on the server (async)")
    ap.add_argument("--policy-type", default="", help="async only: act | smolvla | pi05")
    ap.add_argument("--policy-device", default="cuda", help="async only: device on the policy server")
    ap.add_argument("--local-device", default="mps", help="local only: device on the Mac (mps or cpu)")
    ap.add_argument("--camera-rename", default="", help='send cameras under other names, e.g. '
                    '"top=camera1,wrist=camera2" for a checkpoint fine-tuned from smolvla_base')
    ap.add_argument("--clamp", type=float, default=None, help="override robot.json max_relative_target "
                    "(degrees per step); 0 disables the clamp. Default: use robot.json")
    ap.add_argument("--server", default="", help="async only: host:port of the policy server")
    ap.add_argument("--actions-per-chunk", type=int, default=50)
    ap.add_argument("--chunk-size-threshold", type=float, default=0.5)
    ap.add_argument("--positions", type=int, default=20, help="number of dot positions (default 20)")
    ap.add_argument("--combos", default="", help='restrict tasks, e.g. "red:left,blue:right" for a policy '
                    "that was only trained on those (default: cycle all four)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds per trial")
    ap.add_argument("--out", default="", help="CSV path (default experiments/results/<name>.csv)")
    ap.add_argument("--summary", metavar="CSV", help="print the summary for an existing CSV and exit")
    args = ap.parse_args()

    if args.summary:
        summarize(Path(args.summary))
        return
    if not args.name:
        ap.error("--name is required")

    combos = parse_combos(args.combos)
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.name}.csv"
    done = done_positions(out)
    todo = [p for p in range(1, args.positions + 1) if p not in done]
    print(f"[eval] {args.name}: {len(done)} done, {len(todo)} to go -> {out}")
    if not todo:
        summarize(out)
        return

    try:
        for pos in todo:
            color, bowl = combo_for(pos, combos)
            task = task_for(color, bowl)
            print(f"\n=== Position {pos}/{args.positions}: {color.upper()} block on dot {pos}, target {bowl.upper()} bowl")
            input("  Place the block, clear the mat, then press Enter to start... ")
            cmd = build_command(args, task)
            used = run_trial(cmd, args.duration, start_marker="Control loop thread starting" if args.mode == "async" else None)
            success, ftype, notes = ask_outcome()
            append_row(out, {
                "name": args.name, "position": pos, "color": color, "bowl": bowl, "task": task,
                "success": success, "failure_type": ftype, "duration_s": f"{used:.1f}", "notes": notes,
                "timestamp": datetime.now().isoformat(timespec="seconds"), "policy": args.policy,
                "mode": args.mode,
            })
    except KeyboardInterrupt:
        print("\n[eval] stopped; progress is saved. Re-run the same command to resume.")
    summarize(out)


if __name__ == "__main__":
    main()
