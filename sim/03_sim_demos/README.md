# 03 · 100 simulated demonstrations

Step 3 of the sim-to-real plan: the demonstrations for conditions C (sim only) and D (sim + real), recorded by a
scripted agent in the domain-randomized scene. Sept 15, 2026.

**Result.** 100 episodes, 50 "put the red block in the left bowl" and 50 "put the blue block in the right bowl", in
the real dataset's format and units, from 107 attempts (7 discarded: the block missed the bowl). One A10 hour.
Dataset: `timzwu/so101_blocks_sim` on the Hub (private, like the real one).

| | sim demos | real demos (post #0) |
|---|---|---|
| episodes, trained pair | 100 | 100 (of 150) |
| length per episode | 11–14 s (median 12.2) | 18–30 s (median ~18) |
| frames | 36,771 | 54,000 |
| operator | scripted agent, 93% first-try success | Tim, 100 of 101 kept |
| cost | ~$1.30 of GPU time, 59 min | ~2 h at the leader arm |
| wrist roll used | ±54° | −139° to +45° |
| gripper opening | up to 20 | up to 37 |

The sim demos are shorter and more uniform than the teleop ones: the agent always approaches from the same hover
height with the jaws barely wider than the block, and rolls the wrist only to align with the block. That is a
property of the data, not a bug; whether it matters is part of what the experiment measures.

Reels, 10x: `sim_demos_10x.mp4` (overhead | wrist, the dataset's own cameras, first ten episodes) and
`sim_demos_side_10x.mp4` (a third-person view of the same ten episodes, not in the dataset).

Watch: `sim_demos_side_10x.mp4`.

`methods.md` has the agent, the randomization, the dataset format check against LeRobot 0.6.1, the commands and what
broke. `so101_blocks_sim_extras/record_log.json` has every attempt: block pose, grasp tilt, final distance to the bowl.
