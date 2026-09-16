"""Second Isaac Sim smoke test on Modal, this time inside NVIDIA's own `isaac-lab` container (the way Modal's
isaac_lab_rl example does it) instead of a pip install into a slim image. Same check as `sim/01_pipeline`: load the
SO-101 USD, drop a cube, step the physics, render one frame headless.
    NGC_KEY_FILE=~/robotics/claude-nvidia-ngc.txt modal run sim/tools/modal_isaac_smoke2.py [--tag 2.3.2]
Cost: the image pull on Modal's builder plus a few minutes of L40S (~$2/h).
"""
import os, pathlib
import modal

TAG = os.environ.get("ISAAC_LAB_TAG", "2.3.2")
# this module is imported inside the container as well (to hydrate the function): keep the laptop-only paths guarded
if modal.is_local():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    USD = pathlib.Path(os.environ.get("SO101_USD", "/private/tmp/claude-501/-Users-twu-robotics-physical-ai/03ab074a-df55-4acb-a71b-ba9346f5ca97/scratchpad/workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd"))
    key_file = os.environ.get("NGC_KEY_FILE", "")
    ngc = modal.Secret.from_dict({"REGISTRY_USERNAME": "$oauthtoken", "REGISTRY_PASSWORD": open(os.path.expanduser(key_file)).read().strip()}) if key_file else None
else:
    ROOT, USD, ngc = pathlib.Path("/"), pathlib.Path("/"), None

app = modal.App(f"isaac-sim-smoke2-{TAG.replace('.', '-')}")
outputs = modal.Volume.from_name("lerobot-outputs")
image = (
    modal.Image.from_registry(f"nvcr.io/nvidia/isaac-lab:{TAG}", add_python="3.11", secret=ngc)
    .entrypoint([])
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"})
    .add_local_file(str(ROOT / "sim/tools/isaac_smoke_so101.py"), "/smoke.py", copy=True)
    .add_local_file(str(USD), "/workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd", copy=True)
)


@app.function(image=image, gpu="L40S", timeout=30 * 60, volumes={"/outputs": outputs})
def smoke() -> str:
    import json, os, subprocess
    os.makedirs("/outputs/sim_smoke2", exist_ok=True)
    if not os.path.exists("/out"):
        os.symlink("/outputs/sim_smoke2", "/out")
    import time
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    log = ["gpu: " + r.stdout.strip()]; print(log[0], flush=True)
    # stream Kit's output as it comes, and give up after 8 minutes: a hung renderer must not burn the 30-minute cap
    p = subprocess.Popen(["/workspace/isaaclab/_isaac_sim/python.sh", "/smoke.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0, kept = time.time(), []
    for line in p.stdout:
        if any(k in line for k in ("RTX", "Vulkan", "GPU devices", "INCOMPATIBLE", "Startup Complete", "Traceback", "Error]", "SMOKE", "Shutting Down", "[ext: omni.gpu", "Loading user config", "carb.graphics")):
            print(line.rstrip()[:200], flush=True); kept.append(line.rstrip())
        if time.time() - t0 > 480:
            p.kill(); kept.append("KILLED after 480 s"); print("KILLED after 480 s", flush=True); break
    p.wait()
    log += [f"exit {p.returncode}"] + kept[-25:]
    try:
        log.append("smoke.json: " + json.dumps(json.load(open("/outputs/sim_smoke2/smoke.json"))))
    except Exception as e:
        log.append(f"no smoke.json ({e})")
    outputs.commit()
    return "\n".join(log)


@app.local_entrypoint()
def main():
    print(smoke.remote())
