#!/usr/bin/env bash
# Runs ON the Lambda VM (Ubuntu 22.04; Docker + NVIDIA container toolkit preinstalled). From the Mac:
#   scp -i ~/.ssh/lambda_ed25519 -o StrictHostKeyChecking=accept-new sim/tools/vm_setup.sh sim/tools/isaac_smoke_so101.py ubuntu@<ip>:~/
#   ssh -i ~/.ssh/lambda_ed25519 ubuntu@<ip> 'NGC_KEY=$(cat) bash ~/vm_setup.sh' < ~/robotics/claude-nvidia-ngc.txt
# The NGC key arrives on stdin, is used for the registry login, and the login is removed at the end.
set -euo pipefail
IMG="nvcr.io/nvidia/isaac-lab:2.3.2"      # Isaac Sim 5.1.0 + Isaac Lab 2.3.2, runs as root; the base of the workshop's own sim image
PY=/workspace/isaaclab/_isaac_sim/python.sh
D="sudo docker"                            # ubuntu is not in the docker group on a fresh Lambda VM
WS=~/Sim-to-Real-SO-101-Workshop
USD=source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd
echo "== GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "== NVIDIA graphics libs (Lambda ships the headless -server driver; Vulkan inside the container needs the host's libGLX_nvidia)"
if [ ! -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
  V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
  sudo apt-get update -qq >/dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "libnvidia-gl-${V}-server" >/dev/null
  [ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ] && echo "installed libnvidia-gl-${V}-server" || { echo "libGLX_nvidia still missing"; exit 5; }
else echo "present"; fi
# the container device spec (CDI) is generated at boot, before those libs existed: regenerate so the container gets libGLX_nvidia
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml >/dev/null 2>&1 && echo "cdi spec: $(grep -c GLX_nvidia /var/run/cdi/nvidia.yaml) GLX entries"
echo "== docker login nvcr.io"; printf '%s' "$NGC_KEY" | $D login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null && echo ok || { echo "NGC login failed"; exit 4; }
echo "== git-lfs"; command -v git-lfs >/dev/null || { sudo apt-get update -qq >/dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs >/dev/null; }
git lfs install --skip-repo >/dev/null
echo "== workshop code"; [ -d "$WS" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git "$WS"
git -C "$WS" lfs pull --include="$USD"
[ "$(stat -c %s "$WS/$USD")" -gt 1000000 ] && echo "SO-101 USD ok ($(stat -c %s "$WS/$USD") bytes)" || { echo "USD is still an LFS pointer"; exit 3; }
echo "== pull $IMG (first time: many GB)"; $D image inspect "$IMG" >/dev/null 2>&1 && echo "already present" || $D pull "$IMG"
mkdir -p ~/isaac-cache/{kit,ov,pip,glcache,computecache} ~/isaac-out
echo "== headless smoke: SO-101 in an empty scene, one frame (first run also compiles shaders: minutes of RTX log lines)"
rc=0
$D run --rm --device nvidia.com/gpu=all --network=host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v ~/isaac-cache/kit:/isaac-sim/kit/cache:rw -v ~/isaac-cache/ov:/root/.cache/ov:rw -v ~/isaac-cache/pip:/root/.cache/pip:rw \
  -v ~/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw -v ~/isaac-cache/computecache:/root/.nv/ComputeCache:rw \
  -v "$WS":/workshop:ro -v ~/isaac-out:/out:rw -v ~/isaac_smoke_so101.py:/smoke.py:ro \
  --entrypoint "$PY" "$IMG" /smoke.py || rc=$?
$D logout nvcr.io >/dev/null 2>&1 || true
# Kit's python exits 0 even after a traceback, so the verdict comes from smoke.json, not the exit code
grep -q '"ok": true' ~/isaac-out/smoke.json 2>/dev/null || rc=2
echo "== smoke exit $rc"; ls -la ~/isaac-out; cat ~/isaac-out/smoke.json 2>/dev/null || true
exit $rc
