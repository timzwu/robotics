"""Geometry of the real rig, in the robot frame, used by the scene and by the asset generator.

Frame: origin = the arm's base rotation axis at the mat surface; +X forward (from the base toward the taped zone);
+Y = the robot's left; Z up. In the overhead camera image the base is at the bottom, so +X is image-up and +Y is
image-LEFT. "Left bowl" and "right bowl" in the task strings are from the OPERATOR's seat, facing the arm:
the operator's left bowl (green) is at -Y (the robot's right, image right); the operator's right bowl (pink) at +Y.

Sources: the overhead frame `rig/top-2026-09-07.jpg` (640x480) with the sticker pixel map
`experiments/results/sticker_map.json`; the scale and the axis pixel are estimates from the tape square and the base
plate (ESTIMATE = to be replaced by a tape-measure value; see `sim/02_scene/methods.md`).
"""
import math

# --- overhead image -> world (MEASURED 2026-09-15) ------------------------------------------------------------
# Tape measure from the base to stickers 1, 3, 8, 9 (forward / sideways: 4.5/3.25, 11.25/3.125, 11.25/3.33, 5/2.75 in)
# fitted to their pixels: one scale, residuals < 6 mm. The 49 real grasps agree with this map to 3 mm (fit_axis.py).
# The first guess (14 px/cm from an assumed 20 cm tape square) was 20% off: the square is ~24 cm.
PX_PER_CM = 11.67
AXIS_PX = (334.1, 458.8)         # pixel under the base rotation axis (image x, y)
IMG_W, IMG_H = 640, 480

def px_to_world(u: float, v: float) -> tuple[float, float]:
    """Overhead pixel -> (x forward, y left) in metres, on the mat plane."""
    return (AXIS_PX[1] - v) / PX_PER_CM / 100.0, (AXIS_PX[0] - u) / PX_PER_CM / 100.0

# --- mat, tape, stickers ------------------------------------------------------------------------------------------
MAT_SIZE = (0.60, 0.90)          # x (depth) by y (width): the frame alone is 55 x 41 cm, so the mat is larger than A2 (ESTIMATE)
MAT_CENTER = (0.22, 0.0)         # the arm sits ~8 cm from the mat's back edge (ESTIMATE)
MAT_THICKNESS = 0.003
MAT_RGB = (17, 92, 85)           # sampled from the real frame
MAT_GRID_RGB = (150, 165, 160)
TAPE_OUTER = 284 / PX_PER_CM / 100                                   # 24.3 cm outer square (measured scale x 284 px)
TAPE_WIDTH = 0.018
TAPE_CENTER = ((AXIS_PX[1] - (80 + 352) / 2) / PX_PER_CM / 100, (AXIS_PX[0] - (188 + 472) / 2) / PX_PER_CM / 100)   # from the tape's pixel edges
STICKER_R = 0.015                # ~3 cm dot stickers (35 px at the measured scale)
STICKER_RGB = (235, 205, 40)
STICKER_WHITE = {7}              # sticker 7 is a white dot
STICKER_PX = {1: (434, 319), 2: (399, 260), 3: (425, 124), 4: (295, 212), 5: (279, 141), 6: (376, 184), 7: (237, 261),
              8: (233, 129), 9: (253, 315), 10: (370, 137), 11: (419, 212), 12: (357, 230), 13: (325, 172), 14: (370, 302),
              15: (320, 256), 16: (247, 182), 17: (312, 312), 18: (416, 168), 19: (324, 133), 20: (278, 267)}
STICKERS = {k: px_to_world(*v) for k, v in STICKER_PX.items()}   # metres, robot frame

# --- bowls and blocks -----------------------------------------------------------------------------------------------
BOWL_OUTER_R = 0.0556            # MEASURED 2026-09-15: 4 3/8 in across the rim, 2 in tall (the rim looks 13.5 cm in the frame
BOWL_INNER_R = 0.051             # because it is 5 cm nearer the camera than the mat)
BOWL_BASE_R = 0.042
BOWL_HEIGHT = 0.051
_b = lambda u, v: px_to_world(u, v)
BOWLS = {                        # name -> (centre x, centre y, rgb) from the bowls' pixel centres; the OPERATOR's left/right as in the task strings
    "right": (*_b(105, 215), (205, 185, 178)),   # pink, +Y, image-left
    "left": (*_b(555, 205), (140, 165, 150)),    # green, -Y, image-right
}
BLOCK_SIZE = 0.0254              # MEASURED 2026-09-15: a 1 inch cube
BLOCK_MASS = 0.012               # kg, ESTIMATE
BLOCK_RGB = {"red": (185, 40, 55), "blue": (35, 70, 190)}

# --- cameras --------------------------------------------------------------------------------------------------------
TOP_CAM_HEIGHT = 0.413           # MEASURED 2026-09-15: 16 1/4 in from the mat to the lens
TOP_CAM_XY = px_to_world(IMG_W / 2, IMG_H / 2)   # the pixel under the image centre, assuming the camera looks straight down
TOP_CAM_APERTURE = 20.955        # mm, Isaac Sim's default horizontal aperture; focal length is derived so the mat scale matches
TOP_CAM_FOCAL = TOP_CAM_APERTURE * TOP_CAM_HEIGHT / (IMG_W / PX_PER_CM / 100.0)   # mm; 20.6 at 0.45 m
TOP_CAM_HFOV_DEG = 2 * math.degrees(math.atan(IMG_W / PX_PER_CM / 100.0 / 2 / TOP_CAM_HEIGHT))

# wrist camera: the workshop's mount position (gripper frame, OpenGL camera convention), but the real camera is mounted
# upside-down relative to the workshop's (the jaws hang from the top of the real frame) and looks more steeply along
# the fingers: roll 180 deg about the optical axis, tilt -60 deg (ESTIMATE, from the real wrist frames)
import os as _os
WRIST_CAM_POS = (float(_os.environ.get("WRIST_X", -0.005)), 0.06, -0.062)
WRIST_CAM_TILT_DEG = float(_os.environ.get("WRIST_TILT", -70.0))   # env overrides let one VM session render a sweep
WRIST_CAM_ROLL_DEG = float(_os.environ.get("WRIST_ROLL", 180.0))
WRIST_CAM_FOCAL = 13.5

# fingertips in their body frames (from the USD collision meshes; the two tips touch at Jaw = -10 deg)
JAW_TIP_IN_JAW_FRAME = (-0.0115, -0.0819, 0.0195)
FIXED_TIP_IN_GRIPPER_FRAME = (-0.0119, -0.0008, -0.1041)

# --- robot ------------------------------------------------------------------------------------------------------------
# The workshop USD carries a +90 deg root rotation that Isaac Lab's spawn replaces, so the asset is spawned with yaw 90
# (as the workshop does); in that frame the base rotation axis sits at (0.023, 0.021) and the base's underside at
# z = 0.030. Place the asset so the axis is at the origin, the underside on the mat, and zero joints point along +X.
ROBOT_POS = (-0.0231, -0.0208, -0.0301)
ROBOT_YAW_DEG = 90.0
# real rest pose from the dataset (LeRobot units: degrees, gripper 0-100)
REST_POSE_DEG = {"shoulder_pan": 0.4, "shoulder_lift": -100.0, "elbow_flex": 96.4, "wrist_flex": 55.6, "wrist_roll": 0.4, "gripper": 1.0}

if __name__ == "__main__":
    print(f"top camera at {TOP_CAM_XY}, height {TOP_CAM_HEIGHT} m, focal {TOP_CAM_FOCAL:.1f} mm, HFOV {TOP_CAM_HFOV_DEG:.1f} deg")
    for k in sorted(STICKERS): print(f"sticker {k:2d}: x {STICKERS[k][0]*100:5.1f} cm  y {STICKERS[k][1]*100:5.1f} cm")
