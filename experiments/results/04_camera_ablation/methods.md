# Camera ablation: overhead, wrist, or both — methods

I removed one camera stream from each dataset copy with `lerobot-edit-dataset remove_feature`:
`so101_blocks_nowrist` and `so101_blocks_notop`. Both used episodes `0–49,100–149`, 20k steps × batch 64 on an L40S.
The jobs were `smolvla_nowrist_n100` and `smolvla_notop_n100`, with rename maps `top=camera1` and `wrist=camera2`.
Evaluation used `experiments/tools/eval.py --mode sync` with `--cameras top` or `--cameras wrist`.
See the [training log](../training_runs.md) and [evaluation commands](../01_model_comparison/methods.md#commands).
