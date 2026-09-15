# Methods · the scene

## What is built
`sim/so101_blocks/` is an Isaac Lab environment (`So101-Blocks-v0`) of the real rig: the SO-101 on an A2 cutting mat,
the taped 20 cm zone with the 20 numbered stickers at their measured pixel positions, the two bowls, a red and a blue
block, an overhead camera and a wrist camera at 640×480, 30 Hz control (120 Hz physics, decimation 4). It reuses NVIDIA's
workshop package for the robot asset (actuator gains, joint limits) and the observation/recording helpers, so the
workshop's LeRobot recorder writes `observation.images.top` / `.wrist` and `observation.state` in the real dataset's
format.

Assets are generated once on the laptop by `so101_blocks/make_assets.py` from `rig.py` and committed: `mat.png`
(the grid, the tape square and the stickers drawn at 4 px/mm), `mat.usda` (a 3 mm slab carrying the texture),
`bowl_left.usda` / `bowl_right.usda` (solids of revolution with triangle-mesh colliders, so a block can land inside).

## Where the numbers come from
Robot frame: origin at the base rotation axis on the mat, +X toward the zone, +Y the robot's left. The overhead camera
sees +X as image-up and +Y as image-left. "Left / right bowl" in the task strings are the operator's, facing the arm.

| quantity | value | source |
|---|---|---|
| overhead scale and base axis | 11.67 px/cm, axis at pixel (334, 459) | MEASURED: tape from the base to stickers 1, 3, 8, 9 (4½/3¼, 11¼/3⅛, 11¼/3⅓, 5/2¾ in forward/sideways), fitted to their pixels, residuals < 6 mm; the 49 real grasps agree to 3 mm |
| sticker positions | `experiments/results/sticker_map.json` pixels → metres | measured on the real overhead frame, through the map above |
| tape square | 24.3 cm outer (284 px), centre 20.4 cm ahead of the axis | pixels × measured scale |
| bowls | 11.1 cm across, 5.1 cm tall, centres from their pixel centres (±19 cm sideways) | MEASURED (4⅜ × 2 in); centres from pixels |
| block | 2.54 cm cube (1 in), 12 g | MEASURED side; the gripper reading when closed on it maps to 2.6 cm in the model; mass ESTIMATE |
| overhead camera | 41.3 cm above the mat, straight down, 67° horizontal field of view | MEASURED height (16¼ in); focal length derived from the scale |
| wrist camera | the workshop's mount position, rolled 180° (the jaws hang from the top of the real frame), tilted 70° down | chosen by eye from a tilt sweep against the real wrist frame |
| rest pose | shoulder_lift −100°, elbow 96.4°, wrist_flex 55.6°, gripper 1 | first frame of the real dataset |

Replace the estimates with tape-measure values in `rig.py`, rerun `make_assets.py`, re-render.

## Joint units
The real dataset is in degrees (LeRobot `use_degrees=True`, zero at the calibration mid-point); the workshop USD's
joints are also zeroed at the calibration mid-point with limits in degrees, so the arm joints map 1:1. This arm's
calibrated sweep is wider than the model's limits on four joints (the elbow rests at 96.4° against a 90° limit), so the
scripts widen the simulated limits to the calibrated ranges before the first reset; nothing in the real data is clipped.

The gripper is calibrated from the model's geometry: the fingertips touch at Jaw = −10° and open 0.14 cm per degree;
value 1 is "closed" and the calibration sweep is 129.8° per 100 units, so `Jaw = −10 + 1.298 (value − 1)`. Cross-check
from the real data: the follower reads 11.4 when closed on the 2.5 cm block, which maps to 3.5° = a 2.6 cm gap in the
model; the approach opening 16.3 maps to 9.9° = 3.5 cm. In the replay the gripper follows the leader's command, not
the follower's reading: the reading stalls on the block, the command (5.1) carries the squeeze.

## Commands (session 2, Sept 14–15, Lambda A6000: ~2 h 10 min ≈ $2.40 including three failed starts, plus a 25 min verification run ≈ $0.45)
```bash
python sim/so101_blocks/export_episode.py --episode 0            # Mac: joints + both camera videos of one real episode
python sim/so101_blocks/export_grasps.py                          # Mac: every episode's joints, first frames, block pixels
LAMBDA_YES=1 python3 sim/tools/lambda_vm.py launch --type gpu_1x_a6000 --region us-south-2
scp -i ~/.ssh/lambda_ed25519 sim/tools/vm_session2.sh ubuntu@<ip>:~/ ; scp -r sim/so101_blocks sim/02_scene/real_episodes ubuntu@<ip>:~/
ssh -i ~/.ssh/lambda_ed25519 ubuntu@<ip> 'NGC_KEY=$(cat) bash ~/vm_session2.sh all' < <NGC key file>   # setup, image build, scene, replay
ssh ... 'bash ~/vm_session2.sh wrist'                            # wrist-camera tilt/offset sweep
python3 sim/tools/lambda_vm.py terminate
<usd env> python sim/so101_blocks/fit_axis.py                     # Mac: the base-axis fit from the real grasps
```
`vm_session2.sh` builds the workshop's sim image from its Dockerfile with one patch: the pinned ffmpeg download (a
GitHub "latest" release asset) returns 404, so Ubuntu's ffmpeg is installed instead. The container run mounts the
workshop's `docker/env` file (its entrypoint `source`s it under `set -e`), is capped at 25 minutes, and every script
shuts the simulator down in a `finally` (Kit otherwise keeps the process alive after a crash).

## Results
**Scene.** Both cameras render at 640×480 in 17 s per scene; the overhead frame matches the real one to a few pixels on
the tape square and the stickers (`compare_top.png`); a block placed on sticker 1 renders 5 px from the sticker's
measured pixel (the block's top face is 2.5 cm nearer the camera). Bowls appear larger and further out than the real
ones: the real camera is higher than the 45 cm estimate.

**Replay of real episode 0** (777 frames, 30 Hz, `replay.json`): the simulated arm tracks the real joint trajectory
with an RMS error of 0.05–1.2° per joint (max 5°, wrist roll and gripper), 68 s for the episode. No frame is clipped.

**Reach.** With the block placed where the real one was (found in the real first frame), the first replay closed the
gripper 4.6 cm beyond it. Across all 75 red-block episodes, taking the grasp as the closed-gripper frame with the lowest
fingertips, 49 grasps have the model's fingertips at the mat (6 ± 5 mm), and the reach error was a constant +42 mm
forward (spread 13 mm) with no dependence on the block's position. Joint-zero offsets cannot produce that (they move
the fingertip height, which is right), so the fault was the pixel-to-world map. Four tape measurements settled it: the
scale is 11.67 px/cm, not 14 (the tape square is 24 cm, not the 20 cm I had assumed); with the measured map the grasps'
median error is 3 mm.

**Verification with the measured map** (episodes 0, 5, 7 replayed, `replay*.json`, `replay_ep*_grasp_wrist.png`): the
overhead frames now coincide with the real ones (tape, stickers, bowls), and at each real grasp the simulated fingertips
arrive within 0.4–1.6 cm of the block in x and y, which is about the operator's own slack. None of the three replays
lifts the block: in episode 0 the fingertips come down on its top face, in episode 5 they close around it but it is
pushed rather than carried, in episode 7 they close beside it. That is the expected limit of a blind replay at 1 cm,
and it is not what the experiment needs (the scripted agent plans its grasps in sim). The block is a measured 1 inch
cube, so the residual is the placement: the block's position comes from its pixel in the first frame, and 1 cm is the
combined error of that detection, the map and where the operator actually closed on it.

Files: `scene.json`, `sim_top.png`, `sim_wrist.png`, `compare_top.png`, `compare_wrist.png` (measured map),
`replay.json` / `replay.npz` (episode 0: commanded vs achieved joints, fingertip positions), `replay_ep05.json`,
`replay_ep07.json`, `replay_ep0_compare_{top,wrist}.mp4`, `replay_ep*_grasp_*.png` (the grasp moment, sim vs real).
