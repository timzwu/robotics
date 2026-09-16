"""Inverse kinematics for the scripted agent, on the same arm model as `fk.py` (numpy only, so it can be tested on the
laptop). Solves the five arm joints for a fingertip-midpoint target with a soft "fingers pointing down" and "jaws
across the block" preference; the gripper joint is set separately. Damped least squares with a numeric Jacobian.
"""
import numpy as np
import fk, rig, units

ARM = slice(0, 5)
LIM = np.deg2rad(units.SIM_LIMITS_DEG[:5])


def frame(q_rad):
    """Fingertip midpoint, finger direction (unit, from the gripper origin toward the tips) and opening direction
    (unit, from the fixed tip to the moving tip), all in the world."""
    P = fk.body_poses(q_rad, rig.ROBOT_POS, rig.ROBOT_YAW_DEG)
    f = (P["gripper"] @ np.append(rig.FIXED_TIP_IN_GRIPPER_FRAME, 1.0))[:3]
    m = (P["jaw"] @ np.append(rig.JAW_TIP_IN_JAW_FRAME, 1.0))[:3]
    mid = (f + m) / 2
    a = mid - P["gripper"][:3, 3]; a /= np.linalg.norm(a)
    d = m - f; d -= a * (d @ a); d /= np.linalg.norm(d)
    return mid, a, d


def residual(q5, target, down_w, heading, heading_w, q_prev, reg_w, approach=None):
    q = np.concatenate([q5, [np.deg2rad(20.0)]])          # jaws open for the geometry
    mid, a, d = frame(q)
    want = np.array([0.0, 0.0, -1.0]) if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
    r = [mid - target, down_w * (a - want)]
    if heading is not None:
        h = np.array([np.cos(heading), np.sin(heading), 0.0])
        r.append(heading_w * (d - h * np.sign(d @ h if abs(d @ h) > 1e-6 else 1.0)))   # either jaw side across the face
    r.append(reg_w * (q5 - q_prev))
    return np.concatenate(r)


def solve(target, q_init, down_w=0.05, heading=None, heading_w=0.03, reg_w=0.01, iters=60, damping=1e-3, approach=None):
    """Returns (q5, position error in metres, tilt from vertical in degrees). `approach` = desired finger direction
    (unit vector, default straight down); joint limits enforced by clipping."""
    q = np.clip(np.asarray(q_init, float)[:5].copy(), LIM[:, 0], LIM[:, 1])
    q_prev = q.copy()
    for _ in range(iters):
        r = residual(q, target, down_w, heading, heading_w, q_prev, reg_w, approach)
        J = np.zeros((len(r), 5)); eps = 1e-5
        for j in range(5):
            dq = np.zeros(5); dq[j] = eps
            J[:, j] = (residual(q + dq, target, down_w, heading, heading_w, q_prev, reg_w, approach) - r) / eps
        step = np.linalg.solve(J.T @ J + damping * np.eye(5), -J.T @ r)
        q = np.clip(q + np.clip(step, -0.3, 0.3), LIM[:, 0], LIM[:, 1])
        if np.linalg.norm(step) < 1e-6:
            break
    # second stage: position only, from the posture found above, so the fingertips land within a millimetre
    q_prev = q.copy()
    for _ in range(30):
        r = residual(q, target, 0.0, None, 0.0, q_prev, 1e-3)
        J = np.zeros((len(r), 5)); eps = 1e-5
        for j in range(5):
            dq = np.zeros(5); dq[j] = eps
            J[:, j] = (residual(q + dq, target, 0.0, None, 0.0, q_prev, 1e-3) - r) / eps
        step = np.linalg.solve(J.T @ J + damping * np.eye(5), -J.T @ r)
        q = np.clip(q + np.clip(step, -0.2, 0.2), LIM[:, 0], LIM[:, 1])
        if np.linalg.norm(step) < 1e-7:
            break
    mid, a, d = frame(np.concatenate([q, [np.deg2rad(20.0)]]))
    return q, float(np.linalg.norm(mid - target)), float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))


if __name__ == "__main__":
    rest = units.lerobot_to_sim([rig.REST_POSE_DEG[j] for j in units.JOINTS])
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, zone in (("grasp z=1cm", 0.01), ("hover z=10cm", 0.10)):
        errs, tilts = [], []
        for _ in range(200):
            x = rig.TAPE_CENTER[0] + rng.uniform(-0.09, 0.09); y = rig.TAPE_CENTER[1] + rng.uniform(-0.09, 0.09)
            q, e, tilt = solve(np.array([x, y, zone]), rest, heading=rng.uniform(0, np.pi))
            errs.append(e); tilts.append(tilt)
        errs, tilts = np.array(errs) * 1000, np.array(tilts)
        print(f"{name:14s} position error: median {np.median(errs):.1f} mm, 95th {np.percentile(errs, 95):.1f}, max {errs.max():.1f} | finger tilt from vertical: median {np.median(tilts):.0f} deg, max {tilts.max():.0f}")
    for bowl, (bx, by, _) in rig.BOWLS.items():
        q, e, tilt = solve(np.array([bx, by, 0.10]), rest)
        print(f"above the {bowl} bowl ({bx:.3f}, {by:.3f}, 0.10): error {1000*e:.1f} mm, tilt {tilt:.0f} deg, joints deg {np.round(np.degrees(q), 1)}")
