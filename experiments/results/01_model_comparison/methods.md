# Pick-and-place: setup, protocol, and commands

Shared methods for the six experiments in
[Teaching a Robot to Pick Up a Block](https://timzwu.substack.com/p/teaching-a-robot-to-pick-up-a-block).

**Setup:** SO-101 follower + leader arm, overhead and wrist cameras at 640×480, a cutting mat with a taped
randomization zone, 20 numbered dot stickers as fixed eval positions, two bowls. "Left" and "right" are from the
operator's seat; the overhead camera is mirrored, so the operator's left bowl appears on the right of every map.

**Task and data:** put one block in the instructed bowl. The trained pair is red→left and blue→right: 50
demonstrations each, about 10% with a deliberate missed grasp and retry. The full recording contains 150
demonstrations; only episodes `0–49,100–149` belong to this pair. The 50 swapped-pair demonstrations
(red→right, blue→left) are excluded from every fine-tune reported here. Camera keys in the data and rig config
are `top` (overhead) and `wrist`.

## Evaluation protocol

Twenty fixed starting positions, one trial each, with the same block colors, orientations, and order across compared
conditions. The main passes alternate red→left and blue→right. The instruction test cycles all four combinations:
five trials each, giving ten trained-pair and ten swapped-pair trials.

Each trial allows 30 s of arm motion, excluding inference pauses. Success means the block ends in the instructed
bowl within that budget; a missed grasp or drop can still end in success after recovery. Failures are scored by the
furthest stage reached: no contact, contact without grip, or grip without placement.

Success tables report 95% Wilson intervals. At 10/20, the interval is about 30–70%. These intervals describe trial
uncertainty, not variation across independent training runs. Each condition has one evaluation pass.

ACT uses LeRobot's local rollout with 100-step action chunks. SmolVLA and π0.5 use the stop-and-go client with
50-step chunks; RTC is tested separately. Astra requests poses or joint targets through a custom controller.
Training settings, pretraining, and control interfaces differ, so the model comparison does not isolate architecture.

## Commands

Everything below assumes the `lerobot` conda env (see [setup notes](../../../notes/setup.md)) and a Modal account. Training runs on
Modal; teleop, recording and evaluation run on the Mac connected to the arm. Training launches print a time and cost
estimate and require `--yes`. Run commands from the repository root; replace placeholder paths and set `HF_USER`
to the dataset owner. The recorded dataset is private, so reproducing training requires access or your own data.

```bash
# recording: see experiments/tools/record_r2.sh for the six 25-demonstration batches
# adapt its account, ports, and camera indices before use; upload separately
# provide HF_TOKEN in the environment for private data and π0.5's gated tokenizer

# train, each at its authors' recipe
modal run --detach experiments/tools/modal_train.py::main --dataset $HF_USER/so101_blocks --policy act --episodes 0-49,100-149 --steps 100000 --batch-size 8 --yes
modal run --detach experiments/tools/modal_train.py::main --dataset $HF_USER/so101_blocks --policy smolvla --episodes 0-49,100-149 --steps 20000 --batch-size 64 --gpu L40S --rename-map '{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --yes
modal run --detach experiments/tools/modal_train.py::main --dataset $HF_USER/so101_blocks --policy pi05full --episodes 0-49,100-149 --steps 30000 --batch-size 32 --gpu H100 --gpus 4 --confirm-cost --yes
# data-scaling sweep (nested subsets, one launch)
modal run --detach experiments/tools/sweep.py::sweep --dataset $HF_USER/so101_blocks --policy smolvla --sizes 10,25,50 --pool 0-49,100-149 --strata 25 --seed 0 --steps 20000 --batch-size 64 --gpu L40S --rename-map '{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --yes
# pull a checkpoint (plain `modal volume get` on checkpoints/last fails: `last` is a symlink)
modal run experiments/tools/modal_train.py::pull --job-name <job> --dest experiments/checkpoints/<job>

# evaluate: ACT locally; VLAs through the stop-and-go client, either on the Mac or against a policy server on a GPU
python experiments/tools/eval.py --mode local --name act100k_pair --combos red:left,blue:right --policy experiments/checkpoints/<act-job>
python experiments/tools/eval.py --mode sync --name smolvla100_pair --policy-type smolvla --combos red:left,blue:right --policy experiments/checkpoints/<smolvla-job> --camera-rename top=camera1,wrist=camera2
modal run experiments/tools/modal_policy_server.py --minutes 60 --gpu L40S      # prints host:port; billed for the whole window
python experiments/tools/eval.py --mode sync  --name pi05full_pair --combos red:left,blue:right --policy-type pi05 --policy /outputs/<job>/checkpoints/last/pretrained_model --server <host:port> --record-dir experiments/results/trials
python experiments/tools/eval.py --summary experiments/results/01_model_comparison/pi05full_pair.csv

# the language-model loop (needs OPENAI_API_KEY): see experiments/results/06_general_llm/README.md
# charts and reels
python experiments/tools/plot_results.py
python experiments/tools/make_trial_video.py --name pi05full_pair --speed 10 --out experiments/results/01_model_comparison/pi05full_pair_10x.mp4
```

Notes

- SmolVLA fine-tunes of `lerobot/smolvla_base` need the camera rename (`top→camera1`, `wrist→camera2`) at training and eval time; π0.5 checkpoints do not, because their config is built from this dataset.
- Modal Volumes: `lerobot-hf-cache` (datasets, base models), `lerobot-outputs` (checkpoints). `modal volume ls lerobot-outputs`.
- Camera ablation used dataset copies that omit one camera stream ([camera ablation](../04_camera_ablation/)).
- `results/` is committed; `checkpoints/`, `results/trials/` and the two superseded Astra motion-only cuts are not.

## Pipeline check on public data

This runs a short paid training job, then checks that its checkpoint loads on the Mac. Add `--dry-run` to the
training command to print the plan without starting training.

```bash
modal run --detach experiments/tools/modal_train.py::main --steps 2000 --batch-size 8 --yes        # ACT on lerobot/svla_so101_pickplace
modal run experiments/tools/modal_train.py::pull --job-name act_svla_so101_pickplace_2000
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; p=ACTPolicy.from_pretrained('experiments/checkpoints/act_svla_so101_pickplace_2000'); print(p.config)"
```
