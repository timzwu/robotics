"""LeRobot units <-> simulator joint angles for the SO-101.

The real dataset (`timzwu/so101_blocks`, LeRobot 0.6.1, `use_degrees=True`) stores the five arm joints in DEGREES with
0 at the calibration mid-point, and the gripper as 0-100 (closed-open). The workshop's USD ("so101_new_calib") has
its joint zeros at the same calibration mid-point and limits in degrees, so the arm joints map 1:1 (clipped to the USD
limits). The gripper is calibrated from geometry: the model's fingertips touch at Jaw = -10 deg and open 0.14 cm per
degree; the real value 1.0 is "closed" and the calibration sweep is 129.8 deg per 100 units, so
Jaw_deg = -10 + 1.298 * (value - 1). Check: the real reading 11.4 when closed on the 2.5 cm block gives 3.5 deg = a
2.6 cm gap in the model; the approach opening 16.3 gives 9.9 deg = 3.5 cm.
Whether the 1:1 mapping (signs, offsets) is right is checked by replaying a real episode (`scripts/replay_episode.py`).
Joint order everywhere: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
                     =  Rotation,     Pitch,         Elbow,      Wrist_Pitch, Wrist_Roll, Jaw.
"""
import math
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
USD_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
USD_LIMITS_DEG = np.array([[-110, 110], [-100, 100], [-100, 90], [-95, 95], [-160, 160], [-10, 100]], dtype=np.float64)
# This arm's calibrated sweep (follower_arm.json half-spans: 110, 104.7, 97.3, 101.5, 180) is wider than the USD's on four
# joints, and the real elbow rests at 96.4 deg; the scripts widen the simulated joint limits to these before the first
# reset (Articulation.write_joint_position_limit_to_sim), so nothing in the real data gets clipped.
SIM_LIMITS_DEG = np.array([[-110, 110], [-104.7, 104.7], [-100, 97.3], [-101.5, 101.5], [-180, 180], [-10, 100]], dtype=np.float64)
GRIPPER_CLOSED_VALUE, GRIPPER_CLOSED_DEG, GRIPPER_DEG_PER_UNIT = 1.0, -10.0, 1.298
SIGN = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)   # flipped per joint if the replay shows a mirrored joint
OFFSET_DEG = np.zeros(6)                                  # per-joint zero offset, if the replay shows one

def lerobot_to_sim(values) -> np.ndarray:
    """(…, 6) LeRobot values -> USD joint angles in radians."""
    v = np.asarray(values, dtype=np.float64)
    deg = v.copy()
    deg[..., 5] = GRIPPER_CLOSED_DEG + GRIPPER_DEG_PER_UNIT * (v[..., 5] - GRIPPER_CLOSED_VALUE)
    deg = SIGN * deg + OFFSET_DEG
    deg = np.clip(deg, SIM_LIMITS_DEG[:, 0], SIM_LIMITS_DEG[:, 1])
    return np.deg2rad(deg)

def sim_to_lerobot(rad) -> np.ndarray:
    """USD joint angles in radians -> LeRobot values (degrees; gripper 0-100)."""
    deg = (np.rad2deg(np.asarray(rad, dtype=np.float64)) - OFFSET_DEG) / SIGN
    out = deg.copy()
    out[..., 5] = GRIPPER_CLOSED_VALUE + (deg[..., 5] - GRIPPER_CLOSED_DEG) / GRIPPER_DEG_PER_UNIT
    return out

if __name__ == "__main__":
    rest = [0.4, -100.0, 96.4, 55.6, 0.4, 1.0]
    r = lerobot_to_sim(rest); print("rest ->", np.round(np.rad2deg(r), 1), "-> back", np.round(sim_to_lerobot(r), 1))
