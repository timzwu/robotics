# Data scaling

How much demonstration data does SmolVLA need for this task? I trained it on nested subsets with the same
20k steps and batch size 64, then evaluated each checkpoint at the same 20 positions.

| Demonstrations | Per task | Predicted successes | Measured | 95% interval |
|---|---|---|---|---|
| 10 | 5 + 5 | 0 | 0/20 | 0–16% |
| 25 | 13 + 12 | 1 | 1/20 | 1–24% |
| 50 | 25 + 25 | 4 | 8/20 | 22–61% |
| 100 | 50 + 50 | 8 | 9/20 | 26–66% |

The largest gain came between 25 and 50 demonstrations, earlier than I expected. Doubling from 50 to 100
added one success. One checkpoint and 20 trials per size cannot establish a minimum data requirement or a plateau.
Smaller datasets also received more passes through the data because training steps were held fixed.

[Methods and commands](methods.md) · [Scaling chart](scaling.png) ·
[Outcomes by position](failure_geography_sweep.png) · [Subset manifest](sweep_smolvla_so101_blocks_s0.json)

Trial data: [10 demonstrations](smolvla_n10_pair.csv), [25](smolvla_n25_pair.csv),
[50](smolvla_n50_pair.csv), [100](../01_model_comparison/smolvla100_pair.csv).
