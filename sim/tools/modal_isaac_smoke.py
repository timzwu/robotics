"""Isaac Sim headless smoke test on a Modal L40S: build the image (pip isaacsim 5.1), start Isaac Sim headless, drop a
cube on a ground plane, step the physics, render one camera frame to the `lerobot-outputs` Volume (/outputs/sim_smoke/).
    modal run sim/tools/modal_isaac_smoke.py
Cost: image build (CPU) plus a few minutes of L40S (~$2/h). The first run also fills an extension cache Volume."""
import modal

app = modal.App("isaac-sim-smoke")
cache = modal.Volume.from_name("isaac-sim-cache", create_if_missing=True)
outputs = modal.Volume.from_name("lerobot-outputs")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglu1-mesa", "libxt6", "libxrandr2", "libxinerama1", "libxcursor1", "libxi6", "libvulkan1",
                 "vulkan-tools", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libxkbcommon0", "libatomic1", "git")
    .pip_install("isaacsim[all,extscache]==5.1.0", extra_index_url="https://pypi.nvidia.com")
    .pip_install("numpy<2", "pillow")
    .env({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_USER": "user"})
)


@app.function(image=image, gpu="L40S", timeout=45 * 60, volumes={"/root/.cache": cache, "/outputs": outputs})
def smoke() -> str:
    import os, subprocess, time, traceback
    log = []
    log.append(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    vk = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
    log.append("vulkan devices: " + ", ".join(l.split("=")[-1].strip() for l in (vk.stdout or "").splitlines() if "deviceName" in l))
    t0 = time.time()
    try:
        from isaacsim import SimulationApp
        sim_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 640, "height": 480})
        log.append(f"SimulationApp up in {time.time() - t0:.0f}s")
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.sensors.camera import Camera
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        cube = world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, 0.5]), size=0.05, color=np.array([0.9, 0.1, 0.1])))
        cam = Camera(prim_path="/World/cam", position=np.array([0.6, 0.0, 0.5]), frequency=20, resolution=(640, 480), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        world.reset(); cam.initialize()
        try:
            from isaacsim.core.utils.rotations import euler_angles_to_quat
            cam.set_world_pose(position=np.array([0.6, 0.0, 0.4]), orientation=euler_angles_to_quat(np.array([0, 25, 180]), degrees=True))
        except Exception as e:
            log.append(f"camera pose set skipped: {e}")
        z0 = float(cube.get_world_pose()[0][2])
        for i in range(120):
            world.step(render=True)
        z1 = float(cube.get_world_pose()[0][2])
        log.append(f"cube z: {z0:.3f} -> {z1:.3f} after 120 physics steps (should fall to ~0.025)")
        rgba = cam.get_rgba()
        os.makedirs("/outputs/sim_smoke", exist_ok=True)
        from PIL import Image
        arr = np.asarray(rgba)
        log.append(f"frame shape {arr.shape}, dtype {arr.dtype}, mean {float(arr[..., :3].mean()):.1f}")
        Image.fromarray(arr[..., :3].astype("uint8")).save("/outputs/sim_smoke/frame.png")
        outputs.commit()
        log.append("saved /outputs/sim_smoke/frame.png")
        sim_app.close()
    except Exception:
        log.append("FAILED:\n" + traceback.format_exc()[-3000:])
    log.append(f"total {time.time() - t0:.0f}s")
    return "\n".join(log)


@app.local_entrypoint()
def main():
    print(smoke.remote())
