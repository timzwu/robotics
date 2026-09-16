# Methods · simulated demonstrations, second recipe

Same pipeline as `../03_sim_demos/methods.md` (workshop sim container, Isaac Lab 2.3.2, LeRobot 0.4.3, environment
`So101-Blocks-DR-v0`, `record_demos.py` on a Lambda A10). What changed, by axis:

## 1. Recovery attempts (`--recovery-share 0.15`)
15% of episodes are planned as a miss-then-retry: the first descent lands 2.3–3.0 cm to one side of the block,
perpendicular to the jaw axis so the fingers pass beside the block instead of knocking it; the jaws close on nothing,
the arm lifts, reopens, hops to a fresh hover and grasps again. Half the retries also turn the jaw heading 90°. The
episode is kept only if the second grasp puts the block in the bowl. In the real dataset a few teleop episodes contain
accidental retries; v1 had none.

## 2. Varied grasps
- Approach direction tilted up to 25° from vertical, random per episode (v1: straight down).
- Jaw heading: the nearest of the four grasp headings 60% of the time, the second nearest 40% (v1: always nearest).
  With the wrist roll limit widened to ±180° this uses the whole roll range.
- Jaw opening on approach 22–35 on the 0–100 scale (v1: 20); hover height 6–14 cm (v1: 10); segment durations
  ×0.6–1.4 (v1: ±20%); a ±4 mm wiggle on the grasp point; close command ±2 around the calibrated squeeze.

## 3. Appearance
- The mat is textured from the real overhead frames: the median of the first frames of the real episodes, its tape
  square, stickers and bowl shadows pasted at their measured positions (`make_assets.py mat_texture_photo`;
  `--drawn` gives the v1 synthetic mat).
- Per reset: block and bowl diffuse colour jittered around their measured colours, mat tint ±, overhead and wrist camera
  poses ±3 cm / ±5° (v1: ±2 cm / ±3°), focal ±8–10%, robot colour, dome and lamp as before.

## 4. Block positions
30% of episodes place the block within 1.2 cm of a random sticker (the twenty evaluation spots); the rest uniform inside
the tape's inner edge minus half a block (v1 kept a 5 cm margin, so the edges were never seen). The logged position is
the settled pose after the drop; five of 269 attempts bounced outside the tape and two of those were kept.

## Commands
```bash
LAMBDA_YES=1 python3 sim/tools/lambda_vm.py launch --type gpu_1x_a10 --region us-east-1
scp -i ~/.ssh/lambda_ed25519 sim/tools/vm_session2.sh ubuntu@<ip>:~/ ; scp -r sim/so101_blocks sim/02_scene/real_episodes ubuntu@<ip>:~/
ssh ... 'NGC_KEY=$(cat) bash ~/vm_session2.sh setup && bash ~/vm_session2.sh build' < <NGC key file>
ssh ... 'EPISODES=2 TAG=_v2smoke SEED=2 bash ~/vm_session2.sh record'          # smoke, looked at before the batch
ssh ... 'EPISODES=200 SEED=2 TAG=_v2 RECOVERY=0.15 bash ~/vm_session2.sh record'
scp -r ubuntu@<ip>:~/isaac-out/datasets/so101_blocks_sim_v2* sim/05_sim_demos_v2/ ; python3 sim/tools/lambda_vm.py terminate
python experiments/tools/make_teleop_video.py --root sim/05_sim_demos_v2/so101_blocks_sim_v2 --episodes 0-9 --speed 10 --out sim/05_sim_demos_v2/sim_demos_v2_10x.mp4
python sim/tools/make_side_reel.py sim/05_sim_demos_v2/so101_blocks_sim_v2_extras --out sim/05_sim_demos_v2/sim_demos_v2_side_10x.mp4
```
The dataset's video metadata was aligned to LeRobot 0.6.1 before the push (as for v1); the merged
`so101_blocks_simreal_v2` is real episodes 0–149 followed by sim v2 as 150–349, built with `aggregate_datasets`.

## Results
200 kept from 269 attempts (26% re-rolled) in 152 minutes on an A10, 34 s per attempt. Kept episodes: 304–605 frames
(10.1–20.2 s, median 13.2), grasp tilt median 7.1°, 90th percentile 14°, max 23°; the block ends a median 1.2 cm (max 2.4)
from the bowl's centre. Block x 11–29 cm ahead of the base, y ±9 cm (99th percentile), the full taped zone.

Why the agent fails more often than in v1 (7%): of the 69 re-rolled attempts 53 never lifted the block (the jaws closed
beside or across it) and 16 dropped it in transit. Recovery attempts failed 33% of the time against 24% for plain ones
(the second grasp starts from a hover chosen after the miss). Tilt is the other driver: attempts with a jaw tilt
under 15° failed 18–24% of the time, attempts at 15–23° failed 49%. The heading soft term in the IK lets the jaw land
a few degrees off the planned heading, which matters most when the approach is already tilted. None of this affects the
kept episodes: a re-roll discards the whole attempt, and the kept set is 200 clean or recovered placements.

Joint ranges against the real dataset (observation.state, degrees; gripper 0–100):

| joint | sim v2 | real |
|---|---|---|
| shoulder pan | −60 to 44 | −39 to 42 |
| shoulder lift | −100 to 65 | −104 to 62 |
| elbow | −74 to 97 | −94 to 97 |
| wrist flex | 37 to 102 | 11 to 99 |
| wrist roll | −180 to 180 | −139 to 45 |
| gripper | 1 to 35 | 1 to 37 |

The wrist roll now covers more than the real range (the real operator never rolled past 45°) and the gripper opening
matches. Verified on the laptop with LeRobot 0.6.1: 200 episodes, 82,126 frames, two tasks 100/100, 480×640 images decode.

## Fine-tunes on this data (Sept 16)
Same recipe as C and D (`lerobot/smolvla_base`, 20,000 × 64, cameras renamed top→camera1 wrist→camera2, L40S):
- C2 `smolvla_so101_blocks_sim_v2_20000_ep0-99`: sim v2 episodes 0–99 only (matched to C's 100).
- D2 `smolvla_so101_blocks_simreal_v2_20000_ep0-49_100-149_150-249`: real pair 100 + sim v2 0–99 (matched to D's 200).
- 75 / 25: `smolvla_so101_blocks_simreal_20000_ep0-49_100-149_150-182`: real pair 100 + the first 33 sim episodes
  (from the v1 merged dataset, since it started before v2 was recorded), 3 h 14 min, loss → 0.025.
