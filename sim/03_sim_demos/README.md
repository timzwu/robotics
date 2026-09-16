# Sim v1 demonstrations

A scripted controller generated 100 successful pick-and-place demonstrations: 50 red blocks into the
left bowl and 50 blue blocks into the right. It used known object poses and inverse kinematics to
move through hover, grasp, lift, placement, and return.

| Measure | Sim v1 |
|---|---|
| Accepted demonstrations | 100 from 107 attempts |
| Median episode length | 12.2 seconds |
| Recorded frames | 36,771 |
| Recording time | 59 minutes on an A10 |
| Recording compute cost | About $1.30 |

Camera images, observed joints, commanded joints, and task instructions were saved in LeRobot format.
Camera pose, lighting, and robot color varied between episodes. Approaches and gripper openings were
uniform, and there were no deliberate recovery attempts.

These demonstrations trained the sim-only and co-trained v1 policies. Their shorter, more uniform
movements motivated the [sim v2 changes](../05_sim_demos_v2/).

[Methods](methods.md) · [Policy results](../04_real_evals/) ·
[First ten demos, overhead and wrist, 10×](sim_demos_10x.mp4) ·
[Side view, outside the training data](sim_demos_side_10x.mp4)
