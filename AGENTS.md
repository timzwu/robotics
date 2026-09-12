# AGENTS.md

Canonical, shared instructions for any coding agent (Claude Code, Codex, others) working in this
repository. `CLAUDE.md` imports this file. Keep this file public-safe: it is committed to a public repo.

## What this repo is

A personal learning workspace for imitation learning on a low-cost robot arm with
[HuggingFace LeRobot](https://github.com/huggingface/lerobot). It is **not** a fork of
`huggingface/lerobot`; the `lerobot` library is a dependency. The content is experiment scripts,
setup notes, and results for one rig: an SO-101 leader/follower arm pair, two USB cameras, a mat.

The experiment plan (one task, three policies, a data-scaling sweep, a camera ablation, an honest
20-trial eval protocol) is described in `experiments/README.md`. Read that first.

## Layout

| Path | What |
|---|---|
| `experiments/` | The real content: Modal trainer, data-scaling sweep, 20-position eval protocol, rig config, results. See `experiments/README.md`. |
| `experiments/results/` | Committed CSVs, comparison tables, plots (empty until evals run). |
| `experiments/checkpoints/` | Pulled checkpoints. **Gitignored.** |
| `notes/setup.md` | Environment and hardware setup walkthrough with rationale. Human-facing. Its TODO list is historical. |
| `scripts/` | Reusable shell wrappers, e.g. `scripts/teleop.sh`. Long CLI invocations go here, never pasted. |
| `scratch/`, `notebooks/` | Placeholders, currently empty. |
| `outputs/`, `data/`, `wandb/`, `*.pt`, `*.ckpt` | Gitignored artifacts. |

## Environment

- macOS, Apple Silicon. Conda (Miniforge) env named `lerobot`, Python 3.12. Activate before anything:
  `conda activate lerobot` (or prefix `~/miniforge3/envs/lerobot/bin/`).
- Installed: **lerobot 0.6.1, torch 2.11.0**, ffmpeg 8.x (inside the env, via conda-forge), Feetech SDK
  (`pip install 'lerobot[core_scripts,feetech]'`), `modal` CLI.
- Training runs on **Modal** (`experiments/tools/modal_train.py`); the Mac does teleop, recording, and eval.
- Harmless startup noise on macOS: `objc[...] Class AVFFrameReceiver is implemented in both ... cv2 ... av`.
  Ignore it. Do **not** swap in conda-forge opencv/av to silence it; that adds a second OpenMP runtime next
  to torch's and causes real crashes (`OMP: Error #15`).

## Hardware (SO-101)

- Arm ids used everywhere: `follower_arm`, `leader_arm`. Both arms are calibrated; calibration lives in
  `~/.cache/huggingface/lerobot/calibration/` and persists across unplugs. Do not recalibrate by default.
- Ports are machine-specific and can change on replug: re-run `lerobot-find-port`. Current values are in
  `scripts/teleop.sh` and `experiments/tools/robot.json`.
- **Power: 12V 5A to the follower, 5V 4A to the leader. Swapping them burns the motors.**
- CLI flag gotcha: the follower uses `--robot.*`, the leader uses `--teleop.*`.
- Cameras: `overhead` and `wrist`, 640x480 at 30 fps, indices in `experiments/tools/robot.json`. Re-enumerate with
  `lerobot-find-cameras opencv` (must run in a terminal that has macOS camera permission).

## Commands

```bash
conda activate lerobot
bash scripts/teleop.sh                                   # leader moves by hand, follower mirrors
lerobot-find-port                                        # once per arm, when a port changes
lerobot-find-cameras opencv                              # camera indices + a test frame each

# Modal (costs money; see experiments/README.md for the full recipe)
modal run experiments/tools/modal_train.py::main --dry-run           # prints the lerobot-train command, no spend
modal run --detach experiments/tools/modal_train.py::main --steps 2000 --batch-size 8 --yes
modal run experiments/tools/modal_train.py::pull --job-name <job>   # copy a checkpoint to experiments/checkpoints/
modal run experiments/tools/sweep.py::sweep --sizes 5,10 --steps 200 --dry-run

# Eval protocol (no robot needed in manual mode)
python experiments/tools/eval.py --mode manual --name test --positions 2 --out /tmp/eval_test.csv
python experiments/tools/eval.py --summary experiments/results/<name>.csv
```

## Validation

There is no test suite. Before committing a script change, run the cheapest thing that exercises it:
`--dry-run` for the Modal scripts, `--mode manual` with a temp CSV for `eval.py`, and
`python -c "import lerobot"` after any environment change. Say in the commit or handoff what was run.

## Conventions

- Scripts are plain Python with argparse or Modal local entrypoints; no framework, no packaging.
- Camera keys in configs must match the dataset's `observation.images.*` names.
- Results go in `experiments/results/` and are committed; checkpoints and outputs never are.
- Put long multi-line commands in `scripts/`; pasted multi-line commands break in the owner's terminal.
- Notebooks: clear outputs before committing.

## Secrets

Never hardcode tokens. Read from env (`HF_TOKEN`, `WANDB_API_KEY`) or a gitignored `.env`. Auth via login
commands, which store secrets outside the repo (`gh auth login`, `huggingface-cli login`, `wandb login`,
`modal token`). A gitleaks pre-commit hook (`.git/hooks/pre-commit`, local to this clone, not tracked) blocks
commits containing detected secrets; reinstall it after a fresh clone (`brew install gitleaks`).

## Git

- Default branch `main`. Do not commit or push unless the user asks; when asked, review the staged diff for
  keys and tokens first.
- Commit messages: one line, imperative, saying what changed and why.

## Local Private Agent Context

This repository may optionally have local agent context under `.agent-private/` (gitignored).

If `.agent-private/` exists, read the relevant files before substantial work, especially:

- `.agent-private/MEMORY.md`
- `.agent-private/CURRENT.md`
- `.agent-private/HANDOFF.md`

These files are local/private state and MUST NOT be committed, staged, quoted into public documentation,
copied into issues/PRs, or otherwise published. Private context may inform implementation work, but it must
not leak into public artifacts unless the user explicitly decides that the information is safe to publish.

When recording new persistent information:

- Public, repository-appropriate facts may be added to normal project documentation (this file,
  `experiments/README.md`, `notes/`).
- Personal, internal, uncertain, unpublished, or sensitive context goes to `.agent-private/`.
- When uncertain, default to `.agent-private/`.

Conversation memory is not project memory. At the end of every working session (not only on request), update
`.agent-private/CURRENT.md` and `.agent-private/HANDOFF.md`, and record durable facts, decisions, and
learnings in the matching `.agent-private/` file. Before committing, check that no private-context file or
private information has entered the staged diff (`git status`, `git diff --cached --stat`).
