# 06 · A general language model driving the arm (GPT-6 Astra)

**Results (2026-09-13): pass 1 (no roll) 12/20; pass 2 (roll + joint tool + one hint) 17/20.** Write-up `astra.md`; per-call log `astra_pair.calls.jsonl`; prompt `prompt_plain.txt`; map `failure_geography_llm.png`; reel `astra_pair_10x.mp4`; justification feed video `astra_pair_feed_highlights_6x.mp4` (five successes, made with `tools/make_llm_feed_video.py`); pass 1 record `astra_pass1_recovered.md` + `astra_pair_noroll.csv`.

**Pass 1 (Sept 12 evening, no wrist roll in the interface): 12/20**, blue→right 8/10, red→left 4/10, all 8 failures touch_no_grip; the
model reached the block on every trial and called done every time (median 18 calls, 127 s of thinking, 22 s of motion, ~210k input
tokens per trial). Its per-call log was deleted when the interface was found to lack roll; the trial-level record was reconstructed as `astra_pair_noroll.csv` (`astra_pass1_recovered.md`).
**Pass 2 (rerun): roll and a joint tool added, plus one hint sentence** (match the roll to the block): **17/20**. Two things
changed at once, so the difference between the passes cannot be attributed to the capability alone; a roll-without-hint pass would separate them.

**Question.** The three trained models in `01_model_comparison/` each saw 100 demonstrations. What does a general
language model given no demonstrations do on the same task, given the cameras and a way to move the gripper?

**Why this design.** It copies the interface of the independent GPT-6 Astra arm test published by Robocurve on Sept 4 2026
(red block into a bowl, 19/20 on a pair of industrial YAM arms): the model receives the camera frames and the current
gripper pose each turn and answers with an absolute pose; the arm's inverse kinematics turns that into joint angles; the
model gets the result plus fresh frames.

**What the model can command (every joint).** Two motion tools. `move_to(x, y, z, pitch_deg, roll_deg, gripper)`: an
absolute fingertip pose; on this five-axis arm position plus pitch plus roll are the five free parameters (yaw is fixed by
the base pointing at the target), so every joint is reachable through a pose. `move_joints(shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper)`: the six joints directly, for fine single-joint adjustments. The observation
each turn carries the fingertip pose AND the six joint angles. (The first pass had no roll and no joint tool.)

**One deliberate hint in the prompt (Tim's decision, Sept 13):** after defining roll, the prompt says "Match the roll to how the
block is turned in the image." It is the only sentence that goes beyond describing the interface, kept on purpose because
aligning the jaws to the block is the key strategy for a grip on this arm, and the write-up states it.

**What is held constant with the other passes.** Same 20 stickers, same instructions alternating red/left and
blue/right, same 30 s of arm motion with thinking time excluded, same failure scoring.
**What is different, and why this is an extension rather than a row in the model table:** no demonstrations, pose or
joint targets instead of 30 Hz joint commands, minutes of thinking between moves, 3°/tick motion (other passes 5), no
call limit and high reasoning effort (Robocurve: 20 calls, medium), one 5-DoF hobby arm with two cameras versus their two
industrial arms with three.

**Tools.** `tools/llm_rollout.py` (the loop; `--calibrate` to measure the table height and a grasp pose by hand;
`--dry-run` for a simulated arm), `tools/eval.py --mode llm` (the 20-trial protocol). Kinematics: LeRobot's placo
solver on `so101_new_calib.urdf` from TheRobotStudio/SO-ARM100. Key in `OPENAI_API_KEY` or `~/.openai_key`.

**Run order.**
1. Optional live check before the pass: `python experiments/tools/llm_rollout.py --calibrate` (torque drops: hold the arm), set the closed
   gripper on the block by hand and confirm the printed fingertip z is near 0.01 and the gripper reading 11–13.
2. One shakedown trial: `python experiments/tools/llm_rollout.py --task "put the red block in the left bowl" --record experiments/results/trials/astra_shakedown.mp4`
3. The pass: `python experiments/tools/eval.py --mode llm --name astra_pair --combos red:left,blue:right --out experiments/results/06_general_llm/astra_pair.csv --record-dir experiments/results/trials`
   (writes `astra_pair.calls.jsonl` next to it with every call's target, achieved pose, think time and tokens).

**Frame facts from the demonstrations** (forward kinematics on the recorded joints, `so101_new_calib.urdf`; the URDF's `gripper_frame_link` sits at the fingertips): rest pose (0.18, 0, 0.01); at the moment the jaws close on the block the fingertip point is at z 0.001–0.017 (median 0.009), pitch 76–89° below horizontal; carrying at z 0.07–0.11; release over the operator's left bowl at about (0.24, −0.15, 0.06) and the right bowl at (0.24, +0.15, 0.05). Empty-closed gripper reads ~1; holding the block it reads 11–13 while commanded ~3–11. In the runs the close command is floored at 6 (`GRIPPER_MIN_CMD`, so an empty close cannot crank the servo), which is why the prompt says the jaws stop "near 6 when they meet nothing". The prompt rounds the floor to z = −0.01; the run refused poses below `--z-floor -0.005`. The task's left/right are from the operator's seat, so "left" is negative y in the arm's frame.

**What the model is told** (`--prompt plain`, the default; the exact text is logged per trial): the base frame and units, that z = 0 is the table at the fingertips, reach and floor limits, the gripper range and what the opening reading means after a close, the operator-seat left/right convention, the image orientation, and the 30 s motion budget. Also told, as calibration: the jaw reading with and without the block, and the grasp height at the fingertips; and each result reports `holding`. Not told: where the block or bowls are, or how to grasp. `--prompt coached` adds those and exists for debugging only.

**API and motion.** OpenAI Responses API (`responses.create` with function tools, `previous_response_id` chaining, reasoning effort high, images at high detail); motion `--clamp 3` degrees per tick at 30 Hz, command never more than 15° ahead of the arm.

**Cost.** Robocurve reported $0.94 per Astra trial. Ours: two 640×480 frames per call at high detail, no call limit; median 14 calls per trial in pass 2 (18 in pass 1).
