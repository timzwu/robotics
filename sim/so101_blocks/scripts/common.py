"""Shared helpers for the headless scripts (run inside the workshop's sim container with Isaac Sim's python)."""
import os
import numpy as np


def apply_joint_limits(env, rest_rad) -> None:
    """Widen the simulated joint limits to the calibrated ranges (units.SIM_LIMITS_DEG) and set the rest pose as the
    default joint position, unclipped. Call after gym.make and before the first env.reset."""
    import torch
    from so101_blocks import units
    robot = env.unwrapped.scene["robot"]
    assert list(robot.joint_names) == units.USD_JOINTS, robot.joint_names
    lim = torch.tensor(np.deg2rad(units.SIM_LIMITS_DEG), dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0).repeat(robot.num_instances, 1, 1)
    robot.write_joint_position_limit_to_sim(lim)
    robot.data.default_joint_pos[:] = torch.tensor(np.asarray(rest_rad), dtype=torch.float32, device=env.unwrapped.device)
    print("[limits] sim joint limits (deg):", np.rad2deg(robot.data.joint_pos_limits[0].cpu().numpy()).round(1).tolist())


def fingertips_w(robot):
    """World positions (2, 3) of the fixed fingertip and the moving-jaw tip, from the body poses and the USD tip offsets."""
    import torch
    from so101_blocks import rig
    names = list(robot.body_names)
    out = []
    for body, off in (("gripper", rig.FIXED_TIP_IN_GRIPPER_FRAME), ("jaw", rig.JAW_TIP_IN_JAW_FRAME)):
        i = names.index(body)
        p = robot.data.body_pos_w[0, i]; q = robot.data.body_quat_w[0, i]   # (w, x, y, z)
        w, x, y, z = (float(v) for v in q)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        out.append(p.cpu().numpy() + R @ np.asarray(off))
    return np.stack(out)


def rgb_frame(obs: dict, name: str) -> np.ndarray:
    """(H, W, 3) uint8 from the env's visual observation group."""
    img = obs["visual"][f"rgb_{name}"][0]
    img = img.detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def save_png(path: str, rgb: np.ndarray) -> None:
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


class Mp4:
    def __init__(self, path: str, fps: int, size_wh: tuple[int, int]):
        import cv2
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size_wh)
        self.cv2 = cv2

    def write(self, rgb: np.ndarray) -> None:
        self.w.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.w.release()


def side_by_side(left: np.ndarray, right: np.ndarray, label_l: str = "sim", label_r: str = "real") -> np.ndarray:
    import cv2
    h = min(left.shape[0], right.shape[0])
    out = np.concatenate([left[:h], right[:h]], axis=1).copy()
    for x, t in ((8, label_l), (left.shape[1] + 8, label_r)):
        cv2.putText(out, t, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def find_red_block(rgb: np.ndarray, zone=(170, 490, 60, 370), min_px: int = 80):
    """Centroid (u, v) of the red block in a real overhead frame, or None. Hue-based (the block looks pink under the lamp),
    inside the taped zone only (u0, u1, v0, v1), largest connected blob."""
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = (((h <= 12) | (h >= 165)) & (s > 70) & (v > 90)).astype(np.uint8)
    u0, u1, v0, v1 = zone
    mask[:v0] = 0; mask[v1:] = 0; mask[:, :u0] = 0; mask[:, u1:] = 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_px:
        return None
    return float(cent[i][0]), float(cent[i][1])


def read_video(path: str) -> list[np.ndarray]:
    import cv2
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
