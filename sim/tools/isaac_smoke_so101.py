"""Runs inside the Isaac Sim 5.1 container: <python.sh> /smoke.py. Headless: ground plane + light, the workshop's SO-101,
a red cube, a top-down camera; step physics; write /out/so101_frame.png and /out/smoke.json. Exit 0 = pass."""
import json, os, sys, time, traceback
t0 = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 480})
ROBOT_USD = "/workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd"   # the file the workshop's assets/so101.py loads
try:
    import numpy as np
    from pxr import Usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    assert os.path.getsize(ROBOT_USD) > 1_000_000, f"{ROBOT_USD} is a git-lfs pointer, not the asset"
    world = World(stage_units_in_meters=1.0)
    world.scene.add(GroundPlane(prim_path="/World/ground", size=2.0, color=np.array([0.6, 0.6, 0.6])))  # local; no asset-server fetch
    create_prim("/World/light", "DomeLight", attributes={"inputs:intensity": 30.0})  # 1000 washed the frame out; a distant light gives the shading
    create_prim("/World/sun", "DistantLight", attributes={"inputs:intensity": 3000.0}, orientation=euler_angles_to_quat(np.array([0.0, 40.0, 0.0]), degrees=True))
    add_reference_to_stage(usd_path=ROBOT_USD, prim_path="/World/so101")
    n_prims = sum(1 for _ in Usd.PrimRange(get_current_stage().GetPrimAtPath("/World/so101")))
    cube = world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.25, 0.0, 0.1]), size=0.03, color=np.array([0.9, 0.1, 0.1])))
    cam = Camera(prim_path="/World/cam", position=np.array([0.15, 0.0, 1.5]), resolution=(640, 480))
    world.reset(); cam.initialize()
    cam.set_clipping_range(0.01, 100.0)  # the default near plane is 1 m: a camera closer than that sees only the dome light (uniform frame)
    OVERHEAD = (np.array([0.15, 0.0, 1.5]), euler_angles_to_quat(np.array([0.0, 90.0, 0.0]), degrees=True))  # +X forward pitched 90 deg = looking down; 5 mm focal ~ 24 deg FOV covers ~0.6 x 0.45 m from 1.5 m
    cam.set_world_pose(*OVERHEAD, camera_axes="world")
    z0 = float(cube.get_world_pose()[0][2])
    for _ in range(120):
        world.step(render=True)
    z1 = float(cube.get_world_pose()[0][2])
    # the RTX renderer returns an empty array until its first frame is ready (shader compile on a cold cache): poll, up to ~2 min
    rgba = np.asarray(cam.get_rgba()); waited = 0
    while rgba.ndim != 3 and waited < 1200:
        world.step(render=True); waited += 1
        rgba = np.asarray(cam.get_rgba())
    # the first frames can be the bare dome light while the USD/materials are still loading: keep rendering until the frame has content
    while (rgba.ndim != 3 or float(rgba[..., :3].std()) < 5) and waited < 1800:
        for _ in range(30):
            world.step(render=True)
        waited += 30; rgba = np.asarray(cam.get_rgba())
    render_steps = 120 + waited
    side = None
    if rgba.ndim == 3:  # second frame, three-quarter view, for the record
        cam.set_world_pose(np.array([0.8, 0.5, 0.5]), euler_angles_to_quat(np.array([0.0, 25.0, 210.0]), degrees=True), camera_axes="world")
        for _ in range(60):
            world.step(render=True)
        side = np.asarray(cam.get_rgba())
        if side.ndim == 3:
            cv2_import = __import__("cv2"); cv2_import.imwrite("/out/so101_frame_side.png", cv2_import.cvtColor(np.ascontiguousarray(side[..., :3]).astype("uint8"), cv2_import.COLOR_RGB2BGR))
    os.makedirs("/out", exist_ok=True)
    got_frame = rgba.ndim == 3 and rgba.shape[:2] == (480, 640)
    if got_frame:
        import cv2  # opencv-python-headless is pinned in Isaac Sim 5.1's bundled python
        cv2.imwrite("/out/so101_frame.png", cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]).astype("uint8"), cv2.COLOR_RGB2BGR))
    info = {"robot_usd": ROBOT_USD, "robot_prims": n_prims, "cube_z_before": z0, "cube_z_after": z1,
            "frame_shape": list(rgba.shape), "frame_std": float(rgba[..., :3].std()) if got_frame else None,
            "render_steps": render_steps, "seconds": round(time.time() - t0, 1)}
    info["ok"] = bool(n_prims > 10 and z1 < z0 and got_frame and info["frame_std"] > 5)
    json.dump(info, open("/out/smoke.json", "w"), indent=1); print("SMOKE", json.dumps(info))
    app.close(); sys.exit(0 if info["ok"] else 2)
except Exception as e:  # app.close() may end the process with code 0, so the verdict is written first
    traceback.print_exc()
    os.makedirs("/out", exist_ok=True)
    json.dump({"ok": False, "error": repr(e), "traceback": traceback.format_exc()}, open("/out/smoke.json", "w"), indent=1)
    app.close(); sys.exit(1)
