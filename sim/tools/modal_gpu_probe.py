"""Ten-second probe: what GPU, driver and CUDA does a Modal L40S host expose? (Isaac Sim needs RT cores and a recent driver.)
    modal run sim/tools/modal_gpu_probe.py
Cost: seconds of L40S time (about $0.01)."""
import modal

app = modal.App("isaac-gpu-probe")
image = modal.Image.debian_slim(python_version="3.11").apt_install("pciutils", "vulkan-tools", "libvulkan1", "mesa-vulkan-drivers")


@app.function(image=image, gpu="L40S", timeout=120)
def probe() -> str:
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv"], capture_output=True, text=True).stdout
    glibc = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    os_rel = open("/etc/os-release").read().split("\n")[0]
    import glob, os
    vk = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
    vk_out = (vk.stdout or vk.stderr)
    gpus = [l.strip() for l in vk_out.splitlines() if "deviceName" in l or "driverName" in l or "deviceType" in l]; vk_err = vk.stderr[-400:]
    nv_libs = sorted(os.path.basename(f) for f in glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-*") + glob.glob("/usr/lib/x86_64-linux-gnu/libGLX_nvidia*"))[:20]
    icd = glob.glob("/etc/vulkan/icd.d/*") + glob.glob("/usr/share/vulkan/icd.d/*")
    caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES")
    return f"{out}\n{glibc}\n{os_rel}\nNVIDIA_DRIVER_CAPABILITIES={caps}\nvulkan: {gpus}\nvulkan stderr: {vk_err}\nnvidia libs: {nv_libs}\nicd files: {icd}"


@app.local_entrypoint()
def main():
    print(probe.remote())
