"""Isaac Lab environment: the real block-to-bowl rig. One SO-101, the cutting mat with the taped zone and the 20
numbered stickers, two bowls, a red and a blue block, an overhead camera and a wrist camera, at the real placements
(`so101_blocks/rig.py`). Cameras are named `camera_top` / `camera_wrist` so the workshop's recorder writes
`observation.images.top` / `.wrist`, the keys of the real dataset.

Built on NVIDIA's Sim-to-Real SO-101 workshop package (`sim_to_real_so101`: the robot asset with its actuator gains,
the observation/event helpers); its lightbox, mat and vial task are not used.
"""
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101.assets.so101 import SO101_CFG
from sim_to_real_so101.mdp import (JointPositionActionCfg, image, image_raw, joint_pos, randomize_robot_color,
                                   reset_joints_by_offset, reset_root_state_uniform)

from so101_blocks import rig, units

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
# Isaac Lab refuses default joint positions outside the USD limits at spawn (the real elbow rests at 96.4 deg, the USD's
# limit is 90), so the spawn default is clipped to the USD limits; the scripts then widen the limits and set the true
# rest pose (scripts/common.apply_joint_limits) before the first reset.
REST_RAD = np.clip(units.lerobot_to_sim([rig.REST_POSE_DEG[j] for j in units.JOINTS]),
                   np.deg2rad(units.USD_LIMITS_DEG[:, 0]), np.deg2rad(units.USD_LIMITS_DEG[:, 1]))
PARK = (-0.25, 0.0, rig.BLOCK_SIZE / 2)     # the block not in play waits behind the base, out of both cameras' view


def _quat(rpy_deg):
    return tuple(float(v) for v in euler_angles_to_quat(np.array(rpy_deg, dtype=np.float64), degrees=True))


def _quat_from_matrix(R):
    """(w, x, y, z) from a 3x3 rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2 * np.sign(R[2, 1] - R[1, 2] or 1)
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2 * np.sign(R[0, 2] - R[2, 0] or 1)
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2 * np.sign(R[1, 0] - R[0, 1] or 1)
    return tuple(float(v) for v in (w, x, y, z))


def _wrist_quat():
    """Tilt about the mount's X axis, then roll about the camera's own optical axis (local Z): R = Rx(tilt) @ Rz(roll)."""
    t, r = np.deg2rad(rig.WRIST_CAM_TILT_DEG), np.deg2rad(rig.WRIST_CAM_ROLL_DEG)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return _quat_from_matrix(Rx @ Rz)


camera_base = TiledCameraCfg(
    prim_path="",
    update_period=0.0,
    height=rig.IMG_H,
    width=rig.IMG_W,
    data_types=["rgb", "depth", "instance_id_segmentation_fast"],
    colorize_instance_segmentation=True,
    spawn=sim_utils.PinholeCameraCfg(projection_type="pinhole", focal_length=rig.TOP_CAM_FOCAL,
                                     horizontal_aperture=rig.TOP_CAM_APERTURE, clipping_range=(0.01, 20.0)),
)


@configclass
class BlocksSceneCfg(InteractiveSceneCfg):
    env_spacing = 4.0
    num_envs = 1

    # the desk: a static slab under the mat (Isaac Lab's GroundPlaneCfg needs a Nucleus USD; this needs nothing)
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.CuboidCfg(size=(3.0, 3.0, 0.02), collision_props=sim_utils.CollisionPropertiesCfg(),
                                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35), roughness=0.9)),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -rig.MAT_THICKNESS - 0.01)))
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=100.0, color=(1.0, 1.0, 1.0)))
    lamp = AssetBaseCfg(prim_path="/World/lamp", spawn=sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 0.98, 0.95), angle=2.0),
                        init_state=AssetBaseCfg.InitialStateCfg(rot=_quat((25.0, 35.0, 0.0))))

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=rig.ROBOT_POS, rot=_quat((0.0, 0.0, rig.ROBOT_YAW_DEG)),
            joint_pos={n: float(v) for n, v in zip(units.USD_JOINTS, REST_RAD)}),
    )

    mat = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Mat", spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSETS}/mat.usda"),
                       init_state=AssetBaseCfg.InitialStateCfg(pos=(rig.MAT_CENTER[0], rig.MAT_CENTER[1], 0.0)))
    bowl_left = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/BowlLeft", spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSETS}/bowl_left.usda"),
                             init_state=AssetBaseCfg.InitialStateCfg(pos=(rig.BOWLS["left"][0], rig.BOWLS["left"][1], 0.0)))
    bowl_right = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/BowlRight", spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSETS}/bowl_right.usda"),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(rig.BOWLS["right"][0], rig.BOWLS["right"][1], 0.0)))

    block_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlockRed",
        spawn=sim_utils.CuboidCfg(size=(rig.BLOCK_SIZE,) * 3,
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=rig.BLOCK_MASS),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(v / 255 for v in rig.BLOCK_RGB["red"]), roughness=0.7)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(rig.STICKERS[1][0], rig.STICKERS[1][1], rig.BLOCK_SIZE / 2)),
    )
    block_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlockBlue",
        spawn=sim_utils.CuboidCfg(size=(rig.BLOCK_SIZE,) * 3,
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=rig.BLOCK_MASS),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(v / 255 for v in rig.BLOCK_RGB["blue"]), roughness=0.7)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PARK),
    )

    # overhead camera: straight down, image-up = +X (toward the zone), so the base is at the bottom of the frame
    camera_top = camera_base.replace(prim_path="{ENV_REGEX_NS}/camera_top")
    camera_top.offset = TiledCameraCfg.OffsetCfg(pos=(rig.TOP_CAM_XY[0], rig.TOP_CAM_XY[1], rig.TOP_CAM_HEIGHT),
                                                 rot=_quat((0.0, 0.0, -90.0)), convention="opengl")
    # wrist camera on the gripper's camera mount (workshop placement)
    camera_wrist = camera_base.replace(prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam")
    camera_wrist.spawn = sim_utils.PinholeCameraCfg(projection_type="pinhole", focal_length=rig.WRIST_CAM_FOCAL,
                                                    horizontal_aperture=rig.TOP_CAM_APERTURE, clipping_range=(0.01, 20.0))
    camera_wrist.offset = TiledCameraCfg.OffsetCfg(pos=rig.WRIST_CAM_POS, rot=_wrist_quat(), convention="opengl")


@configclass
class ActionsCfg:
    joint_positions = JointPositionActionCfg(asset_name="robot", joint_names=units.USD_JOINTS, scale=1.0, use_default_offset=False)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_obs = ObsTerm(func=joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class VisualCfg(ObsGroup):
        rgb_top = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "rgb", "normalize": False})
        depth_top = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "depth"})
        instance_id_seg_top = ObsTerm(func=image_raw, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "instance_id_segmentation_fast"})
        rgb_wrist = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False})
        depth_wrist = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "depth"})
        instance_id_seg_wrist = ObsTerm(func=image_raw, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "instance_id_segmentation_fast"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(func=reset_joints_by_offset, mode="reset",
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=units.USD_JOINTS), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})
    robot_colour = EventTerm(func=randomize_robot_color, mode="reset", params={"color_names": ["white"]})
    # the block in play: anywhere in the taped zone (inside the tape, with a margin), random yaw
    reset_block_red = EventTerm(func=reset_root_state_uniform, mode="reset",
                                params={"asset_cfg": SceneEntityCfg("block_red"),
                                        "pose_range": {"x": (-0.07, 0.07), "y": (-0.07, 0.07), "yaw": (-3.14159, 3.14159)},   # stays inside the tape (inner half-width 8.65 cm, block half-side 1.25)
                                        "velocity_range": {}})


@configclass
class BlocksEnvCfg(ManagerBasedRLEnvCfg):
    scene: BlocksSceneCfg = BlocksSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        self.decimation = 4                    # 120 Hz physics, 30 Hz control = the real recording rate
        self.episode_length_s = 30.0           # the real trial budget
        self.scene.num_envs = 1
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.rendering_mode = "quality"
        self.viewer.eye = (-0.35, -0.45, 0.35)
        self.viewer.lookat = (0.15, 0.0, 0.05)
        # the red block's reset range is relative to its default position: centre it on the zone
        self.scene.block_red.init_state.pos = (rig.TAPE_CENTER[0], rig.TAPE_CENTER[1], rig.BLOCK_SIZE / 2 + 0.002)


# ---------------------------------------------------------------------------------------------------------------------
# Domain-randomized variant for recording sim demonstrations (the workshop's recipe: camera pose and focal length, robot
# colour, lighting), plus a third "side" camera for the reel of the data being generated. `camera_side` is never part
# of the dataset: the recorder picks `top` and `wrist` explicitly.
# ---------------------------------------------------------------------------------------------------------------------
from sim_to_real_so101.mdp import randomize_camera_focal_length, randomize_camera_pose
import isaaclab.utils.math as math_utils
from isaaclab.sim import get_current_stage
from pxr import Gf, Sdf


def randomize_lights(env, env_ids, dome_range=(60.0, 300.0), sun_range=(400.0, 1800.0), tint=0.12,
                     sun_pitch_range=(15.0, 50.0), sun_yaw_range=(-60.0, 60.0)):
    """Dome and distant light intensity, a warm/cool tint, and the distant light's direction, on every reset."""
    stage = get_current_stage()
    u = lambda lo, hi: float(math_utils.sample_uniform(lo, hi, (1,), device="cpu").item())
    with Sdf.ChangeBlock():
        dome = stage.GetPrimAtPath("/World/dome"); sun = stage.GetPrimAtPath("/World/lamp")
        if dome.IsValid():
            dome.GetAttribute("inputs:intensity").Set(u(*dome_range))
        if sun.IsValid():
            sun.GetAttribute("inputs:intensity").Set(u(*sun_range))
            t = u(-tint, tint)
            sun.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0 + t, 1.0, 1.0 - t))
            q = _quat((u(*sun_pitch_range), u(*sun_yaw_range), 0.0))
            attr = sun.GetAttribute("xformOp:orient")
            if attr.IsValid():
                attr.Set(Gf.Quatd(q[0], q[1], q[2], q[3]) if attr.GetTypeName() == "quatd" else Gf.Quatf(q[0], q[1], q[2], q[3]))


def _look_at(eye, target):
    """OpenGL camera rotation (columns = camera X right, Y up, Z backward) looking from eye at target, Z-up world."""
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


@configclass
class BlocksDRSceneCfg(BlocksSceneCfg):
    # the dataset cameras, as in BlocksSceneCfg but rgb only (depth / segmentation are never recorded)
    camera_top = camera_base.replace(prim_path="{ENV_REGEX_NS}/camera_top", data_types=["rgb"])
    camera_top.offset = TiledCameraCfg.OffsetCfg(pos=(rig.TOP_CAM_XY[0], rig.TOP_CAM_XY[1], rig.TOP_CAM_HEIGHT),
                                                 rot=_quat((0.0, 0.0, -90.0)), convention="opengl")
    camera_wrist = camera_base.replace(prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam", data_types=["rgb"])
    camera_wrist.spawn = sim_utils.PinholeCameraCfg(projection_type="pinhole", focal_length=rig.WRIST_CAM_FOCAL,
                                                    horizontal_aperture=rig.TOP_CAM_APERTURE, clipping_range=(0.01, 20.0))
    camera_wrist.offset = TiledCameraCfg.OffsetCfg(pos=rig.WRIST_CAM_POS, rot=_wrist_quat(), convention="opengl")
    camera_side = camera_base.replace(prim_path="{ENV_REGEX_NS}/camera_side")
    camera_side.data_types = ["rgb"]
    camera_side.spawn = sim_utils.PinholeCameraCfg(projection_type="pinhole", focal_length=14.0, horizontal_aperture=rig.TOP_CAM_APERTURE, clipping_range=(0.01, 20.0))
    # a three-quarter view from the operator's right, looking at the zone
    camera_side.offset = TiledCameraCfg.OffsetCfg(pos=(0.55, 0.45, 0.40), rot=_quat_from_matrix(_look_at(np.array([0.55, 0.45, 0.40]), np.array([0.18, 0.0, 0.03]))), convention="opengl")


@configclass
class BlocksDRObservationsCfg(ObservationsCfg):
    @configclass
    class VisualCfg(ObsGroup):
        rgb_top = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "rgb", "normalize": False})
        rgb_wrist = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False})
        rgb_side = ObsTerm(func=image, params={"sensor_cfg": SceneEntityCfg("camera_side"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


@configclass
class BlocksDREventCfg(EventCfg):
    camera_top_pose = EventTerm(func=randomize_camera_pose, mode="reset",
                                params={"prim_path_pattern": "{ENV_REGEX_NS}/camera_top",
                                        "pos_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
                                        "rot_range": {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}})
    camera_top_fov = EventTerm(func=randomize_camera_focal_length, mode="reset",
                               params={"focal_length_range": (rig.TOP_CAM_FOCAL * 0.92, rig.TOP_CAM_FOCAL * 1.08), "asset_cfg": SceneEntityCfg("camera_top")})
    camera_wrist_fov = EventTerm(func=randomize_camera_focal_length, mode="reset",
                                 params={"focal_length_range": (rig.WRIST_CAM_FOCAL * 0.9, rig.WRIST_CAM_FOCAL * 1.1), "asset_cfg": SceneEntityCfg("camera_wrist")})
    robot_colour = EventTerm(func=randomize_robot_color, mode="reset", params={"color_names": ["white", "white", "orange", "teal", "black"]})
    lights = EventTerm(func=randomize_lights, mode="reset", params={})


@configclass
class BlocksDREnvCfg(BlocksEnvCfg):
    scene: BlocksDRSceneCfg = BlocksDRSceneCfg()
    observations: BlocksDRObservationsCfg = BlocksDRObservationsCfg()
    events: BlocksDREventCfg = BlocksDREventCfg()
