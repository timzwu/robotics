# Model comparison

How do ACT, SmolVLA, π0.5, and Astra perform on the same pick-and-place task?
The three robot policies trained on 100 demonstrations. All systems used the same 20 starting positions
and a 30 s motion budget, excluding inference pauses.

| Model | Training | Success | 95% interval |
|---|---|---|---|
| [ACT](act100k_pair.csv) | Task policy from scratch; 100k steps × batch 8 | 4/20 | 8–42% |
| [SmolVLA](smolvla100_pair.csv) | Fine-tuned; 20k × 64, backbone frozen | 9/20 | 26–66% |
| [π0.5](pi05full_pair.csv) | Whole model fine-tuned; 30k × 32 | 7/20 | 18–57% |
| [Astra, run 2](../06_general_llm/astra_pair.csv) | No task demonstrations or fine-tuning | 17/20 | 64–95% |

Grasping was the common failure point. π0.5 touched the block in 19 trials but gripped it in seven;
SmolVLA gripped it in ten of 17 contacts. These runs do not establish a reliable ranking between the two.

Astra used a different controller, a calibrated prompt with a grasp hint, and a median 103 s of thinking per trial.
The table compares the systems as configured, including these differences in control and compute.

Two controls are retained: ACT with shorter training (20k steps) scored 1/20;
π0.5 with a frozen backbone scored 5/20. The latter also used a different training length,
so it does not isolate the effect of unfreezing.

[Methods and commands](methods.md) ·
[Comparison chart](model_comparison.png) · [Stages reached](progress_by_model.png) · [π0.5 reel](pi05full_pair_10x-publication-copy.mp4)
