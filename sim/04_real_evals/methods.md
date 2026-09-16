# Methods · sim demonstrations on the real arm

## First-round fine-tunes (sim v1)
All from `lerobot/smolvla_base`, LeRobot's default SmolVLA recipe, 20,000 steps at batch 64 on one L40S, cameras
renamed top→camera1, wrist→camera2, image input shape patched to 480×640 after training (as for every smolvla_base
fine-tune here). The recipe is held fixed, so the passes over the data differ with its size: this was the decision, the
formula rather than matched passes.

| condition | job | data | frames | passes | wall | cost | final loss |
|---|---|---|---|---|---|---|---|
| A · real | `smolvla_so101_blocks_n100_s0` (post #0) | real episodes 0–49, 100–149 | 54,000 | 24 | 177 min | ≈$5.75 | 0.027 |
| C · sim | `smolvla_so101_blocks_sim_20000` | `timzwu/so101_blocks_sim`, all 100 | 36,771 | 35 | 192 min | ≈$7.50 | 0.003 |
| D · sim + real | `smolvla_so101_blocks_simreal_20000_ep0-49_100-149_150-249` | the merged dataset, real 0–49 and 100–149 + sim 150–249 | 90,771 | 14 | 195 min | ≈$7.55 | 0.024 |

The merged dataset `timzwu/so101_blocks_simreal` (private) is the real 150 episodes followed by the 100 sim episodes,
built with LeRobot's `aggregate_datasets`; the 50 real crossed-task episodes are excluded by the episode list, as in A.
The sim dataset's video metadata had to be aligned with the real one's before merging (LeRobot 0.4.3 writes extra
encoder fields).

## Evaluation
`experiments/tools/eval.py --mode sync`, the stop-and-go loop on the laptop (50-step chunks, one inference per chunk),
20 trials per condition on the fixed stickers, red→left on odd stickers and blue→right on even, 30 s per trial, the
failure scored by the same categories as post #0. Commands:
```bash
python experiments/tools/eval.py --mode sync --name smolvla_sim_pair --policy-type smolvla --combos red:left,blue:right \
  --policy experiments/checkpoints/smolvla_so101_blocks_sim_20000 --camera-rename top=camera1,wrist=camera2 \
  --record-dir sim/04_real_evals/trials --out sim/04_real_evals/smolvla_sim_pair.csv
# same with smolvla_simreal_pair and the simreal checkpoint
python sim/tools/plot_sim_results.py
python experiments/tools/make_trial_video.py --name smolvla_sim_pair --speed 10 --results-dir sim/04_real_evals --out sim/04_real_evals/smolvla_sim_pair_10x.mp4
```
Stage scoring follows post #0 (`experiments/tools/plot_results.py`: no contact / contact without grip / grip without
placement / success from the failure category), with one override recorded there: D's trial 2 is scored "grip without
placement", not "contact", because it gripped and dropped the block and was recovering toward the bowl when the 30 s ran
out (the category `timeout` maps to "contact" by default). The counts in the README were computed with the same rule.

Before the evals, one prediction per checkpoint on a training frame through the eval's own loader: C reproduces a sim
frame within 1.5° and is up to 15° off on a real frame; D within ~2° on both.

## Evaluation in the simulator
The same three checkpoints on the same 20 stickers in the nominal (non-randomized) scene, `so101_blocks/scripts/
eval_in_sim.py`: the block on the sticker, square to the arm, the arm at rest; then the real eval's loop, one
observation → one chunk of 50 actions executed with the 5°/step clamp at 30 Hz → the next observation, until 30 s of
executed motion or the block is in the bowl. The policy runs on Modal behind a small HTTPS bridge
(`sim/tools/modal_policy_http.py`), the same LeRobot version the checkpoints were trained with, producing the chunk
exactly as the laptop eval does; the bridge was checked against a recorded real action (within 2°) before use. Scoring
from the physics: contact = the block moved more than 5 mm or the midpoint between the fingertips came within 2 cm of its centre; closed on
the block = it rose more than 2.5 cm; success = inside the target bowl. One sticker was run first and its video
reviewed. Cost: ~2 h of A10 ≈ $2.85 and ~2 h of the Modal A10G endpoint ≈ $2.20 for 61 trials.
```bash
POLICY_TOKEN=<token> modal deploy sim/tools/modal_policy_http.py     # prints the URL; stop with `modal app stop --yes policy-http`
ssh ... 'POLICY_URL=<url> POLICY_TOKEN=<token> NAME=smolvla_sim_simeval POLICY=/outputs/<job>/checkpoints/020000/pretrained_model bash ~/vm_session2.sh simeval'
python experiments/tools/make_trial_video.py --name smolvla_sim_simeval --speed 10 --results-dir sim/04_real_evals --out sim/04_real_evals/smolvla_sim_simeval_10x.mp4
```

### C2 (Sept 16)
Same protocol and bridge for `smolvla_so101_blocks_sim_v2_20000_ep0-99`. One difference in the scene: since Sept 16
the mat asset is the photographed one (`make_assets.py`, default), so C2 was evaluated on the mat it trained on, as
C and D were on theirs (the drawn mat); A trained on neither. Result 2 / 20 (stickers 2 and 3): contact 9, lifted 2,
no_reach 11, touch_no_grip 7. Cost ~35 min of A10 ≈ $0.75 and ~20 min of the endpoint ≈ $0.40. Reel
`smolvla_sim_v2_simeval_10x.mp4` from the 20 trial videos. D2 (`smolvla_so101_blocks_simreal_v2_20000_ep0-49_100-149_150-249`),
same protocol, same photo mat, ≈$1.20: 4 / 20 (stickers 2, 3, 13, 20), contact 18, lifted 4, touch_no_grip 14, no_reach 2;
reel `smolvla_simreal_v2_simeval_10x.mp4`.

## Round 2 on the arm (Sept 16)
Same protocol for the three round-2 checkpoints (recipes in `../../experiments/results/training_runs.md`; the data in
`../05_sim_demos_v2/`): C2 0 / 20 (no_reach 18, touch_no_grip 2), D2 7 / 20 (touch_no_grip 8, no_reach 5), 75 / 25
3 / 20 (touch_no_grip 11, no_reach 4, timeout 1, no_move 1; trial 3 gripped and ran out of time, scored as a grip in the
stage chart via `STAGE_OVERRIDES`). Charts for this round are the `_v2` files, drawn by the same script with A and D as
reference bars. Trial videos kept only for D2 (`trials/`, gitignored) and its reel; the C2 and 75 / 25 videos were deleted.

## Reading the failure
- **Sim v1 did not learn a reliable grasp.** The sim-only policy scored 0/20 on the arm and 0/20 in sim, despite reaching the simulator's contact/proximity threshold in 13/20 trials. Successful scripted demonstrations did not produce a successful learned policy, even in its training environment.
- **Co-training v1 reduced grasp conversion.** On the arm, real-only and co-trained v1 both reached contact in 17/20 trials, but gripped in 10 and 6 respectively. They completed 9 and 4 placements. The co-trained policy's misses included small offsets and wrong wrist rolls; these observations do not isolate which data-generation choice caused the difference.
- **Sim v2 improved co-training.** Real-world success rose from 4/20 to 7/20 and in-sim success from 1/20 to 4/20. Sim-only v2 remained at 0/20 on the arm and reached 2/20 in sim. The 75/25 real:sim v1 condition scored 3/20 on the arm.
- **Next tests.** Hold the training budget and data ratio fixed while changing recovery examples, approach variation, timing, and appearance one at a time. V2 changed these together, so this experiment establishes the combined result, not the contribution of each change. Compare earlier checkpoints to test whether training length helps.
