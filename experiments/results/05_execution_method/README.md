# Execution method

Does Real-Time Chunking (RTC) improve on stop-and-go execution? I tested both on the same SmolVLA checkpoint,
trained on 100 demonstrations, using the same 20 positions and 30 s motion budget.

| Execution | Success | 95% interval |
|---|---|---|
| [Stop-and-go](../01_model_comparison/smolvla100_pair.csv) | 9/20 | 26–66% |
| [Real-Time Chunking](smolvla100_rtc_pair.csv) | 3/20 | 5–36% |

RTC computes the next action chunk during the current one, conditioning it on actions already committed to
execution. Motion was continuous, but I saw repeated back-and-forth corrections near the block. Twice the arm
grasped it and then let go.

This RTC configuration scored lower than baseline. I have not isolated whether timing, RTC settings, implementation details, or policy errors caused the difference.

[Settings and command](methods.md) · [Shared evaluation protocol](../01_model_comparison/methods.md#evaluation-protocol)
