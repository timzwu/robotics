# Instruction following

Can models follow block–bowl instructions absent from their fine-tuning data? SmolVLA and π0.5 trained on
red→left and blue→right. This test cycles all four combinations: ten trials on the trained pair and ten
on the swapped pair, red→right and blue→left.

| Model | Trained pair | Swapped pair | Total | 95% interval for total |
|---|---|---|---|---|
| [SmolVLA](smolvla100_all4.csv) | 7/10 | 3/10 | 10/20 | 30–70% |
| [π0.5, whole model fine-tuned](pi05full_all4.csv) | 2/10 | 3/10 | 5/20 | 11–47% |
| [π0.5, frozen-backbone control](pi05_all4.csv) | 1/10 | 1/10 | 2/20 | 3–30% |

Both main models completed three trials on unseen pairings. No wrong-bowl failures were recorded, but most
failures happened before placement. With ten trials per pair at different positions, this gives limited evidence
about instruction following.

To repeat the test, use the [evaluation command](../01_model_comparison/methods.md#commands) with `--combos`
omitted and a new run name and CSV. ACT is omitted because it does not take a language instruction.

[SmolVLA notes](smolvla100_all4.md) · [π0.5 notes](../01_model_comparison/pi05full.md) ·
[Control notes](../01_model_comparison/pi05.md) · [π0.5 reel](pi05full_all4_10x.mp4)
