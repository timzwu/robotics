"""Where is the arm's base axis in the overhead frame? Fitted from the real grasps: at the moment the real gripper closed on
the block, the model's fingertips (forward kinematics of the replayed joints) must be at the block's pixel position. Runs
on the data exported by `export_grasps.py`; the grasp frame is the closed-phase frame where the fingertips are lowest.
    <usd env python> sim/so101_blocks/fit_axis.py
Reports the reach error under the current `rig.AXIS_PX`, the axis shift that removes it, and the fingertip heights.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig, fk, units

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_scene", "real_episodes")
TIP = lambda q: fk.fingertips(q, rig.ROBOT_POS, rig.ROBOT_YAW_DEG, rig.FIXED_TIP_IN_GRIPPER_FRAME, rig.JAW_TIP_IN_JAW_FRAME).mean(0)

def grasp_points():
    G = [g for g in json.load(open(f"{D}/grasps.json")) if g["block_px"]]
    Z = np.load(f"{D}/states.npz")
    rows = []
    for g in G:
        e = g["episode"]; s, a = Z[f"s{e}"], Z[f"a{e}"]; gr = a[:, 5]
        opened = np.where(gr > 10)[0]
        closed = [i for i in range(opened[0], len(gr)) if gr[i] < 8] if len(opened) else []
        if not closed:
            continue
        win = [i for i in closed if i <= closed[0] + 60]
        z = [TIP(units.lerobot_to_sim(s[i]))[2] for i in win]
        i = win[int(np.argmin(z))]
        rows.append((e, i, s[i], g["block_px"], min(z)))
    return rows

if __name__ == "__main__":
    rows = grasp_points()
    S = np.array([r[2] for r in rows]); PX = np.array([r[3] for r in rows])
    t = np.array([TIP(units.lerobot_to_sim(s)) for s in S])
    clean = (t[:, 2] > -0.006) & (t[:, 2] < 0.02)          # the fingertips at the mat: a real grasp, not a close in the air
    b = np.array([rig.px_to_world(u, v) for u, v in PX])
    r = (t[:, :2] - b)[clean] * 1000
    print(f"{len(rows)} grasps, {clean.sum()} with the fingertips at the mat (height {1000*t[clean,2].mean():.1f} +- {1000*t[clean,2].std():.1f} mm)")
    print(f"reach error under AXIS_PX={rig.AXIS_PX}: median dx {np.median(r[:,0]):+.1f} mm (MAD {np.median(np.abs(r[:,0]-np.median(r[:,0]))):.1f}), "
          f"median dy {np.median(r[:,1]):+.1f} mm (MAD {np.median(np.abs(r[:,1]-np.median(r[:,1]))):.1f}); rms {np.sqrt((r**2).mean()):.1f} mm")
    dv, du = np.median(r[:, 0]) / 10 * rig.PX_PER_CM, np.median(r[:, 1]) / 10 * rig.PX_PER_CM
    print(f"axis shift that removes the median error: v {dv:+.0f} px, u {du:+.0f} px -> AXIS_PX = ({rig.AXIS_PX[0]+du:.0f}, {rig.AXIS_PX[1]+dv:.0f})")
