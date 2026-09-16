# Isaac Sim pipeline check

Can the SO-101 run headlessly in Isaac Sim on a cloud GPU? The smoke test loaded the arm,
simulated a falling block, and rendered two camera views on a Lambda A6000.

| Check | Result |
|---|---|
| Physics | Block settled from 10 cm to 1.5 cm above the ground |
| Rendering | Both cameras returned frames; overhead resolution 640×480 |
| Runtime | 33 seconds with the container image cached |
| Session cost | About $0.55, including failed setup attempts |

![SO-101 overhead view](so101_frame.png)

Isaac Sim 5.1 ran inside NVIDIA's Isaac Lab 2.3.2 container. This verified the physics and
rendering pipeline; task setup and robot control followed in [scene calibration](../02_scene/).

[Methods and setup](methods.md) · [Test results](smoke.json) · [Side view](so101_frame_side.png)
