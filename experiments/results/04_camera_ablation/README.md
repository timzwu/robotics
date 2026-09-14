# Camera ablation

Does SmolVLA need the overhead camera, the wrist camera, or both? Each configuration used the same
100 demonstrations, training recipe, and 20-trial evaluation.

| Cameras | Success | 95% interval |
|---|---|---|
| [Overhead + wrist](../01_model_comparison/smolvla100_pair.csv) | 9/20 | 26–66% |
| [Overhead only](smolvla_toponly_pair.csv) | 4/20 | 8–42% |
| [Wrist only](smolvla_wristonly_pair.csv) | 12/20 | 39–78% |

Wrist-only scored highest. With overhead only, I saw the arm approach the block but struggle to align the grasp.
Wrist-only sometimes kept moving after placement, which could reflect difficulty recognizing completion.

The 12/20 versus 9/20 result does not establish that adding the overhead view hurts. Repeated runs and tasks
requiring more awareness of the surrounding scene would help test when it is useful.

[Methods](methods.md) · [Training log](../training_runs.md) · [Outcomes by position](failure_geography_ablation.png)
