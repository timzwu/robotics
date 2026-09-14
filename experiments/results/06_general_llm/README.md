# Language-model control

Can Astra control the arm from camera images and motion tools, without demonstrations or fine-tuning on the task?
Each turn, it receives two images and the arm's pose, requests a move, then gets fresh images and feedback.

| Run | Interface | Success | 95% interval |
|---|---|---|---|
| [Run 1](astra_pair_noroll.csv) | Gripper position and pitch | 12/20 | 39–78% |
| [Run 2](astra_pair.csv) | Added wrist roll, direct joint control, and a roll hint | 17/20 | 64–95% |

Astra retried missed grasps, changed approach, and moved the arm aside for a clearer view. Several things changed
between runs, so the improvement cannot be attributed to wrist roll alone. 

The same 30 s motion budget applies as in the policy tests, but run 2 used a median 103 s of thinking per trial,
excluded from that budget. Its calibrated interface and prompt also differ from the trained policies.

[Results and observations](astra.md) · [Control loop and run instructions](methods.md) ·
[Prompt](prompt_plain.txt) · [Run 2 call log](astra_pair.calls.jsonl) · [Position map](failure_geography_llm.png)

[Full reel](astra_pair_10x_with_thinking.mp4) · [Annotated highlights](astra_pair_feed_highlights_10x_true_time.mp4)
(both at 10× speed, including inference pauses).
