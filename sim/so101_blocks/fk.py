"""Forward kinematics of the workshop's SO-101 model, offline (numpy only), from the joint frames in
`assets/so101_chain.json` (extracted from the USD). Used to check joint-zero offsets between the real arm's calibration
and the model without a simulator: fingertip positions for a joint trajectory, validated against the simulator's own
body poses recorded by `scripts/replay_episode.py` (replay.npz: `fingertips_w`).
Frame: the base body pose as spawned (`rig.ROBOT_POS`, yaw `rig.ROBOT_YAW_DEG`); PhysX joint convention: child pose =
parent ∘ (localPos0, localRot0) ∘ Rz(theta) ∘ (localPos1, localRot1)^-1.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = json.load(open(os.path.join(HERE, "assets", "so101_chain.json")))["joints"]
ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T(p, R):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_poses(q_rad, base_pos, base_yaw_deg):
    """Dict body -> 4x4 world pose for joint angles (radians, ORDER)."""
    poses = {"base": T(np.asarray(base_pos, float), Rz(np.deg2rad(base_yaw_deg)))}
    for name, a in zip(ORDER, q_rad):
        j = CHAIN[name]
        parent, child = poses[j["body0"]], j["body1"]
        J0 = T(j["localPos0"], quat_to_R(j["localRot0"]))
        J1 = T(j["localPos1"], quat_to_R(j["localRot1"]))
        poses[child] = parent @ J0 @ T(np.zeros(3), Rz(a)) @ np.linalg.inv(J1)
    return poses


def fingertips(q_rad, base_pos, base_yaw_deg, tip_fixed, tip_jaw):
    """(2, 3): fixed fingertip and moving-jaw tip in the world."""
    P = body_poses(q_rad, base_pos, base_yaw_deg)
    f = P["gripper"] @ np.append(tip_fixed, 1.0)
    m = P["jaw"] @ np.append(tip_jaw, 1.0)
    return np.stack([f[:3], m[:3]])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    import rig
    r = np.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "02_scene", "replay.npz"))
    q, tips_sim = r["achieved_rad"], r["fingertips_w"]
    errs = []
    for i in range(0, len(q), 25):
        t = fingertips(q[i], rig.ROBOT_POS, rig.ROBOT_YAW_DEG, rig.FIXED_TIP_IN_GRIPPER_FRAME, rig.JAW_TIP_IN_JAW_FRAME)
        errs.append(np.linalg.norm(t - tips_sim[i], axis=1))
    errs = np.array(errs)
    print(f"FK vs simulator fingertips over {len(errs)} frames: mean {errs.mean()*1000:.2f} mm, max {errs.max()*1000:.2f} mm")
