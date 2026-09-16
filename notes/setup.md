# Setup notes

Hardware notes describe this particular rig. The checklist below records the initial setup; current experiments and results are in the [repository overview](../README.md).

## Environment

- Machine: macOS (Apple Silicon)
- System Python: 3.13 (Homebrew) — **caveat below**

### Python version

Per the [official install guide](https://huggingface.co/docs/lerobot/installation),
lerobot needs **Python >= 3.12** (and PyTorch >= 2.10). System Python here is 3.13,
but the guide recommends an isolated **Miniforge (conda)** env so you don't touch
Homebrew's Python:

```bash
# Install Miniforge (conda-forge based, open source) — one-time, machine-wide
# NOTE: macOS has no `wget` by default (`brew install wget`, or use `curl -fL -O`).
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh"   # accept license, default location, yes to conda init

# Create and activate the env
conda create -y -n lerobot python=3.12
conda activate lerobot   # run this each new shell

# ffmpeg — required for video decoding (TorchCodec). Installed INTO the env so the
# version is pinned to this environment rather than system-wide.
conda install -y -c conda-forge ffmpeg      # got ffmpeg 8.1.2, incl. libsvtav1 encoder

# lerobot + the Feetech motor SDK (SO-101 uses Feetech servos — required for the arms)
pip install 'lerobot[core_scripts,feetech]==0.6.1' modal
```

Verify:

```bash
python -c "import importlib.metadata, torch; print(importlib.metadata.version('lerobot'), torch.__version__)"  # experiment environment: 0.6.1, 2.11.0
python -c "import scservo_sdk; print('feetech SDK OK')"
ffmpeg -hide_banner -encoders | grep svtav1    # confirm the AV1 encoder is present
```

## Hardware: SO-101 (leader + follower)

Bought **pre-assembled** (PartaBot Full Kit) → skip assembly, and skip
`lerobot-setup-motors`. Setting motor IDs requires connecting one motor at a time
*before* they're daisy-chained, so a pre-assembled arm necessarily has its IDs set
already. (If calibration can't find the motors, that assumption was wrong — revisit.)

### ⚡ Power supplies — the one step that can destroy hardware

| Supply | Arm | Why |
|---|---|---|
| **12V 5A** | **Follower** (gripper) | 12V STS3215 motors; holds its own weight + grips |
| **5V 4A** | **Leader** (handle) | Lower-voltage motors; back-driven by hand |

**Swapping these can burn the motors.** Label the bricks with tape. Each arm needs
*both* its power supply (motor power) *and* a USB cable to the computer (data only).

### Wiring

The 3-pin servo cable (black/red/white) plugs into the board's **D/V/G** port
(Data / VCC / Ground). Boards have two D/V/G ports wired in parallel — either works.
The connector is keyed, so it can't be inserted backwards.

### Find ports → calibrate

```bash
lerobot-find-port     # run once per arm; unplug the arm when prompted to identify it
```

Port names (`/dev/tty.usbmodem…`) are **not permanent** — they can change on replug,
especially into a different physical USB port. Calibration does NOT depend on them.

```bash
# Follower uses --robot.*  |  Leader uses --teleop.*   <- easy to mix up
lerobot-calibrate --robot.type=so101_follower  --robot.port=<PORT> --robot.id=follower_arm
lerobot-calibrate --teleop.type=so101_leader   --teleop.port=<PORT> --teleop.id=leader_arm
```

Calibration = move all joints to mid-range → Enter → sweep each joint through its full
range. Saved permanently to `~/.cache/huggingface/lerobot/calibration/` under the `id`
(so it survives unplugs/reboots — you don't recalibrate each session). Output is just
per-joint `homing_offset` (zero point) + `range_min`/`range_max` (travel) in raw 12-bit
encoder ticks. These values define this arm’s joint coordinates. Consistent calibration helps compare recordings; it does not by itself make a dataset portable across robot bodies.

**Use the same `id`s (`follower_arm`, `leader_arm`) for teleop, recording, and eval.**

### Teleoperate

```bash
bash scripts/teleop.sh     # leader moves by hand, follower mirrors. Ctrl+C to stop.
```

The follower **snaps to the leader's current pose on start** — clear the workspace and
pose the leader near the follower first. The follower goes **limp when unpowered**, so
support it when powering down.

### Gotchas seen

- **Long multi-line commands break in the terminal** — a stray newline (or a space after
  a `\`) splits the command and the second half runs as a bogus command. Put long
  invocations in a script file (`scripts/`) instead of pasting them.
- **`objc[...] Class AVFFrameReceiver is implemented in both …libavdevice…`** — harmless
  noise on macOS (`cv2` and `av` each bundle libavdevice). Not an error. May become a
  real suspect if cameras crash later.

## Secrets — don't leak keys

Layered protection so API keys / tokens never land in a commit:

1. **Never hardcode tokens.** Read from env (`os.environ["HF_TOKEN"]`) or a
   gitignored `.env`. Never paste a literal key into code or a notebook.
2. **Authenticate via login commands** — each stores its secret *outside* this repo:
   - `gh auth login` → macOS Keychain
   - `huggingface-cli login` → `~/.cache/huggingface/`
   - `wandb login` → `~/.netrc`
3. **Watch notebooks** — cell outputs are saved JSON. Never `print()` a token; clear
   outputs before committing.
4. **gitleaks pre-commit hook** (`.git/hooks/pre-commit`) scans staged changes and
   blocks any commit containing a detected secret. Review any finding and remove the secret or document a narrowly scoped false-positive exception.
   - Note: this hook is local to this clone (`.git/hooks/` is untracked) — it won't
     follow a fresh `git clone`. Reinstall it, or move to a tracked pre-commit config.

## Initial setup checklist (historical)

- [x] GitHub auth (gh, HTTPS) + gitleaks pre-commit hook
- [x] Miniforge + `lerobot` conda env (Python 3.12.13)
- [x] ffmpeg 8.1.2 + lerobot 0.5.1 (torch 2.10.0) + Feetech SDK
- [x] SO-101 arrived; ports found; both arms calibrated
- [x] Teleoperation working (leader → follower mirroring)
- [ ] Workspace: clear table, **clamp the follower base**, fixed camera spot
- [ ] Camera setup → teleop with cameras (check framing)
- [ ] Record first dataset (start with ONE well-placed camera)
- [ ] Train a policy (ACT) → evaluate → iterate
