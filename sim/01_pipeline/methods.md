# Methods · pipeline check

## Commands (from the laptop)
```bash
LAMBDA_YES=1 python3 sim/tools/lambda_vm.py launch --type gpu_1x_a6000 --region us-south-2   # waits for ssh, prints the ip
scp -i ~/.ssh/lambda_ed25519 sim/tools/vm_setup.sh sim/tools/isaac_smoke_so101.py ubuntu@<ip>:~/
ssh -i ~/.ssh/lambda_ed25519 ubuntu@<ip> 'NGC_KEY=$(cat) bash ~/vm_setup.sh' < <file holding the NGC key>
scp -i ~/.ssh/lambda_ed25519 'ubuntu@<ip>:~/isaac-out/*' sim/01_pipeline/
python3 sim/tools/lambda_vm.py terminate
```
The NGC key never appears on a command line: it arrives on stdin, is used for `docker login nvcr.io`, and the login is
removed at the end of the script. Lambda has no stop state, only terminate; `terminate` polls until the instance list is
empty and refuses to report success otherwise.

## What `vm_setup.sh` does
Installs the driver's graphics libraries (see below), regenerates the container device spec, logs into NGC, clones the
[workshop repo](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) and pulls only the SO-101 USD through git-lfs
(23.5 MB; a 1 MB size check catches an unpulled pointer file), pulls `nvcr.io/nvidia/isaac-lab:2.3.2` (Isaac Sim 5.1.0 +
Isaac Lab 2.3.2, ~20 GB, cached on the VM's disk for the session), then runs the smoke script with Isaac Sim's own python,
with the shader and asset caches on host directories so a second run starts in 27 s instead of 127 s.

## What `isaac_smoke_so101.py` checks
Headless `SimulationApp`; ground plane, dome + distant light; the SO-101 USD referenced into the stage; a 3 cm dynamic
cube dropped from 10 cm; a 640×480 camera 1.5 m above the base looking down. 120 physics steps with rendering, then the
frame is read (polling until it has content), and a second frame from a three-quarter view. Verdict `ok` = more than 10
robot prims, the cube lower than it started, a 480×640 frame with pixel std above 5. Written to `smoke.json`; the
verdict comes from that file, not the exit code, because Isaac Sim's python exits 0 even after a traceback.

## What broke, in order (four runs, ~25 min)
1. **Vulkan could not start inside the container** (`ERROR_INCOMPATIBLE_DRIVER`, "no suitable CUDA GPU"). Lambda's image
   ships the headless `-server` NVIDIA driver without the graphics libraries; the container toolkit's Vulkan ICD pointed at
   a `libGLX_nvidia.so.0` that did not exist on the host. Fix: `apt-get install libnvidia-gl-580-server` (matched to the
   driver version) on the host.
2. Still failing: the container device spec (`/var/run/cdi/nvidia.yaml`) is generated at boot, before those libraries
   existed, and the legacy `--gpus all` path did not expose the library under the name the ICD wanted. Fix: regenerate the
   spec (`nvidia-ctk cdi generate`) and run the container with `--device nvidia.com/gpu=all`.
3. **Empty frame.** The renderer returns an empty array until its first frame is ready. Fix: poll.
4. **Uniform grey frame** (std 0.2) with the renderer at 55 frames/s: the camera's default near clipping plane is 1 m, and
   the camera sat 0.9 m above the ground, so everything was clipped and only the dome light remained. Found by rendering the
   same scene from 3 m (content appeared) and from a Replicator look-at camera (the near half of the ground came out black).
   Fix: `set_clipping_range(0.01, 100)` and the camera at 1.5 m. A 1000-intensity dome light with no sun also washed the
   image out; 30 + a 3000 distant light gives shading.

## Notes for step 2
- The A6000 was sold out earlier in the day and available an hour later; the launcher falls back to nothing, so pick the
  type from `lambda_vm.py types` each time. An A10 (24 GB, $1.29/h) would also do for rendering.
- Everything installed on the VM is lost at terminate: the ~20 GB image pull (~3 min) and the apt step repeat each session.
  A Lambda persistent filesystem in the region would keep the Docker image and caches; worth it from step 2 on.
- The robot mesh is yellow in the workshop USD; the real arm is white. Colour randomization covers this in step 3.

## Modal, retested (Sept 15, `sim/tools/modal_isaac_smoke2.py`)
Modal's own Isaac Lab example renders a video on an L40S from NVIDIA's `isaac-lab` container, so the same smoke test was
run inside that container (`isaac-lab:2.3.2`, pulled from NGC, entrypoint cleared) rather than a pip install. Result:
Vulkan now creates the GPU device (Isaac Sim starts in 28 s and reports "Graphics API: Vulkan"), and the first render
submission fails with `ERROR_DEVICE_LOST`, after which no frame is produced. So the sandbox exposes the GPU to Vulkan
for device creation but not for graphics work; CUDA is unaffected. A `vulkaninfo` from a slim image is not a valid
test (it also fails on a host where the container renders). Not tried: the `3.0.0-beta2-post1` image the example uses.
Two runs, ≈ $1.
