# Data scaling: SmolVLA on 10–100 demonstrations — methods

The [manifest](sweep_smolvla_so101_blocks_s0.json) records the subsets and jobs. Sampling used seed 0, restricted to
episodes `0–49,100–149`, and balanced across the recording batches. The 100-demonstration baseline is
[`smolvla100_pair.csv`](../01_model_comparison/smolvla100_pair.csv).

From the repository root, with `HF_USER` set to the dataset owner:

```bash
modal run --detach experiments/tools/sweep.py::sweep --dataset $HF_USER/so101_blocks --policy smolvla --sizes 10,25,50 --pool 0-49,100-149 --strata 25 --seed 0 --steps 20000 --batch-size 64 --gpu L40S --rename-map top=camera1,wrist=camera2 --yes
```

Fixed steps mean different amounts of data reuse: roughly 240, 95, 48, and 24 passes through the data. That does
not by itself establish overfitting. A sweep with matched dataset passes would test a different training budget.
Each checkpoint's declared image input shape was corrected to 480×640 after training; the base config declared
256×256. This is the input shape, not the model's internal image resolution.

Evaluate through `experiments/tools/eval.py --mode sync`, using `--combos red:left,blue:right` and
`--camera-rename top=camera1,wrist=camera2`; see the [evaluation commands](../01_model_comparison/methods.md#commands).
Checkpoints are named `smolvla_so101_blocks_n{N}_s0`, where `N` is 10, 25, or 50.

[Scaling chart](scaling.png) · [Outcomes by position](failure_geography_sweep.png) ·
[10-demo trials](smolvla_n10_pair.csv) · [25-demo trials](smolvla_n25_pair.csv) · [50-demo trials](smolvla_n50_pair.csv)
