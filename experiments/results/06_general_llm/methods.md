# Astra control loop and run instructions

## Control loop

Each turn, the model receives the overhead and wrist images, the fingertip pose, and the six joint readings. It
chooses one tool:

| Tool | Action |
|---|---|
| `move_to(x, y, z, pitch_deg, roll_deg, gripper)` | Request an absolute fingertip pose; inverse kinematics converts it to joint targets. |
| `move_joints(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)` | Set joint targets directly. |
| `done` | End the trial. Success is scored separately by the operator. |

The controller moves toward the target, then returns the achieved pose, joint readings, motion time remaining,
blocked/unreachable flags, a `holding` estimate, and fresh images. `holding` is inferred from the jaw opening;
it is not a force-sensor measurement.

The [rollout script](../../tools/llm_rollout.py) uses LeRobot's placo inverse-kinematics solver and the SO-101 URDF.
Motion runs at 30 Hz, limited to 3° per tick, with targets no more than 15° ahead of the measured joints. A move
ends on arrival, after 0.5 s without movement, or after 8 s. Poses outside the workspace are clipped; solutions
with excessive position error or below the floor check are refused.

## Prompt and comparison

The [recorded prompt](prompt_plain.txt) describes coordinates, reach limits, image orientation, gripper readings,
the motion budget, and the roll hint. It includes calibration derived from the rig and recorded demonstrations:
fingertip height at the mat and jaw readings with and without a block. No demonstration trajectories or bowl
coordinates are supplied to the model. `--prompt coached` adds bowl coordinates and a grasp recipe for debugging;
it was not used for these results.

The task uses the operator's viewpoint: left is negative y in the arm's frame and appears on the right of the
overhead image. Position units are metres. The close command is floored at 6; jaws holding the block read about
11–13. The runtime floor setting is `--z-floor -0.005`; the prompt rounds it to −0.01.

The same 20 starting positions, alternating red→left / blue→right instructions, 30 s motion budget, and scoring
apply as in the policy tests. Astra differs in its calibrated interface, pose-by-pose control, and inference time:
run 2 took a median 103 s of thinking per trial, excluded from the budget. It used high reasoning effort, two
640×480 images at high detail per call, and no call limit. The four-model table compares these systems as configured.

## Run it

Run from the repository root in the `lerobot` environment, with the rig configured in
[`robot.json`](../../tools/robot.json). Supply a key through `OPENAI_API_KEY` (the script also accepts
`~/.openai_key`) and access to the configured model, `gpt-6-astra`. The
[evaluation script](../../tools/eval.py) accepts `--llm-model` to select another model; that would be a new condition.

1. Optional pose check: `python experiments/tools/llm_rollout.py --calibrate`. **Hold the arm: this disables torque.**
   Place the closed gripper around the block by hand; check fingertip z near 0.01 and jaw opening around 11–13.
   This checks the pose convention, not motor calibration.
2. Run one shakedown trial:

```bash
python experiments/tools/llm_rollout.py --task "put the red block in the left bowl" --record experiments/results/trials/astra_shakedown.mp4
```

3. Run the 20-trial pass with a new name to keep the published results intact:

```bash
python experiments/tools/eval.py --mode llm --llm-effort high --name astra_repeat --combos red:left,blue:right --out experiments/results/06_general_llm/astra_repeat.csv --record-dir experiments/results/trials
```

The pass writes a CSV and an adjacent `.calls.jsonl` containing the prompt, targets, achieved poses, timing, and
tokens. `llm_rollout.py --dry-run` substitutes a simulated arm but still calls the model API.

