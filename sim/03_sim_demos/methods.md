# Methods · simulated demonstrations

## What runs
`so101_blocks/scripts/record_demos.py` in the workshop's sim container (Isaac Lab 2.3.2, LeRobot 0.4.3) on a Lambda
GPU VM, environment `So101-Blocks-DR-v0`. One episode: the block in play (red for the left bowl, blue for the right,
alternating) is dropped at a random spot inside the taped zone with a random yaw; the other block is parked out of
view; the arm starts at the real rest pose. The scripted agent reads the block's pose from the simulator and plans
fingertip waypoints on the arm model (`ik.py`, damped least squares on `fk.py`, sub-millimetre): hover 10 cm above
the block, descend to 1 cm, close, lift, move above the target bowl at 11 cm, open, return to rest. Segments are
smooth joint ramps at 30 Hz with ±20% random durations; the gripper opens to 20 on the real 0–100 scale for the
approach, closes to 5 (a squeeze, as the leader arm commanded in the real episodes), rests at 1. The episode is kept
only if the block ends within 4.6 cm of the bowl's centre and below the rim; otherwise it is re-rolled (up to 4 tries).

Randomized on every reset (the workshop's recipe): overhead camera pose ±2 cm / ±3°, both cameras' focal length ±8–10%,
robot colour (white, white, orange, teal, black), dome and lamp intensity, lamp colour tint and direction.

## Dataset
LeRobot v3.0, `observation.state` and `action` as the real dataset (degrees, gripper 0–100, the same six names),
`observation.images.top` / `.wrist` at 640×480 30 fps (AV1), task strings identical to the real ones, robot type
`so_follower`. The action is the commanded joint target and the state the achieved one, each frame pairing the
observation captured before the command with that command, as LeRobot's real recorder does. A third camera
(`camera_side`, a three-quarter view) is rendered for every tenth episode into `_extras/side_epNNN.mp4` for the reel;
it is not in the dataset. Verified: a 2-episode smoke dataset written by 0.4.3 loads in LeRobot 0.6.1 on the laptop.

## Commands

These commands describe sim v1 at commit `1026340`. The current recorder generates sim v2; setting its recovery share to zero does not recreate v1.

```bash
LAMBDA_YES=1 python3 sim/tools/lambda_vm.py launch --type gpu_1x_a10 --region us-east-1     # the A6000 was sold out
scp -i ~/.ssh/lambda_ed25519 sim/tools/vm_session2.sh ubuntu@<ip>:~/ ; scp -r sim/so101_blocks sim/02_scene/real_episodes ubuntu@<ip>:~/
ssh ... 'NGC_KEY=$(cat) bash ~/vm_session2.sh setup && bash ~/vm_session2.sh build' < <NGC key file>
ssh ... 'EPISODES=2 TAG=_smoke bash ~/vm_session2.sh record'                                # smoke, then look
ssh ... 'EPISODES=100 SEED=1 bash ~/vm_session2.sh record'
scp -r ubuntu@<ip>:~/isaac-out/datasets/so101_blocks_sim* sim/03_sim_demos/                # then terminate
python experiments/tools/make_teleop_video.py --root sim/03_sim_demos/so101_blocks_sim --episodes 0-9 --speed 10 --out sim/03_sim_demos/sim_demos_10x.mp4
```

## What broke
- On the A10 host, installing the driver's graphics package pulled the whole driver stack to a newer version than the
  loaded kernel module, so NVML, CUDA and the container device spec all failed until a reboot; the setup script now
  detects the mismatch and says so.
- Isaac Lab buffers created during stepping under `torch.inference_mode()` cannot be updated by a later `env.reset()`
  outside it: the whole recording loop now runs under inference mode.
- `configclass` fields are not class attributes, so the randomized scene redefines its cameras instead of copying them.

## Results
100 episodes kept from 107 attempts in 59 minutes on an A10 (35 s per attempt including AV1 encoding); the seven
discarded attempts were drops in transit or missed grasps, all recovered on the next roll. Kept episodes: 334–412 frames
(11.1–13.7 s), grasp tilt from vertical median 3.6°, max 8.6°; the block ends a median 1.2 cm (max 2.4) from the bowl's
centre. Block positions cover x 13–29 cm ahead of the base and y ±8 cm, the taped zone minus a margin.

Joint ranges against the real dataset (observation.state, degrees; gripper 0–100):

| joint | sim | real |
|---|---|---|
| shoulder pan | −45 to 43 | −39 to 42 |
| shoulder lift | −100 to 52 | −104 to 62 |
| elbow | −59 to 97 | −94 to 97 |
| wrist flex | 56 to 102 | 11 to 99 |
| wrist roll | −54 to 51 | −139 to 45 |
| gripper | 1 to 20 | 1 to 37 |

The arm joints span a subset of the real ranges; the wrist roll and the gripper opening are narrower in sim (the agent rolls only
to align with the block and opens the jaws to 20), and the real operator bent the wrist further at times.
Verified on the laptop with LeRobot 0.6.1: 100 episodes, 36,771 frames, two tasks 50/50, 480×640 images decode.
