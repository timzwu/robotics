#!/usr/bin/env bash
# Session 2, runs ON the Lambda VM. Builds the workshop's sim image and runs our scene scripts headless. From the Mac:
#   scp -i ~/.ssh/lambda_ed25519 -o StrictHostKeyChecking=accept-new sim/tools/vm_session2.sh ubuntu@<ip>:~/
#   scp -i ~/.ssh/lambda_ed25519 -r sim/so101_blocks sim/02_scene/real_episodes ubuntu@<ip>:~/
#   ssh -i ~/.ssh/lambda_ed25519 ubuntu@<ip> 'NGC_KEY=$(cat) bash ~/vm_session2.sh' < ~/robotics/claude-nvidia-ngc.txt
# Usage on the VM: bash vm_session2.sh [setup|build|scene|replay|all]   (default all)
set -euo pipefail
STEP="${1:-all}"
BASE="nvcr.io/nvidia/isaac-lab:2.3.2"
IMG="teleop-docker:latest"                   # the workshop's sim image (Isaac Lab 2.3.2 + LeRobot 0.4.3), built here
D="sudo docker"
WS=~/Sim-to-Real-SO-101-Workshop
USD=source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd
PKG=~/so101_blocks
DATA=~/real_episodes
OUT=~/isaac-out
mkdir -p ~/isaac-cache/{kit,ov,pip,glcache,computecache} "$OUT"

setup() {
  echo "== GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "== NVIDIA graphics libs"
  if [ ! -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
    V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
    sudo apt-get update -qq >/dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "libnvidia-gl-${V}-server" >/dev/null
    [ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ] && echo "installed libnvidia-gl-${V}-server" || { echo "libGLX_nvidia still missing"; exit 5; }
  else echo "present"; fi
  # installing the GL package can pull the whole driver stack to a newer version than the loaded kernel module (A10 host, Sept 15:
  # 570.148 loaded, 570.195 installed); then NVML, CUDA and the CDI spec all fail until a reboot loads the new module
  nvidia-smi >/dev/null 2>&1 || { echo "driver/library mismatch after the install: reboot the VM (sudo reboot), then rerun setup"; exit 11; }
  sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml >/dev/null 2>&1 && echo "cdi spec: $(grep -c GLX_nvidia /var/run/cdi/nvidia.yaml) GLX entries" || { echo "cdi spec generation failed"; exit 12; }
  echo "== docker login nvcr.io"; printf '%s' "$NGC_KEY" | $D login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null && echo ok || { echo "NGC login failed"; exit 4; }
  echo "== git-lfs"; command -v git-lfs >/dev/null || { sudo apt-get update -qq >/dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs >/dev/null; }
  git lfs install --skip-repo >/dev/null
  echo "== workshop code"; [ -d "$WS" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git "$WS"
  git -C "$WS" lfs pull --include="$USD"
  [ "$(stat -c %s "$WS/$USD")" -gt 1000000 ] && echo "SO-101 USD ok ($(stat -c %s "$WS/$USD") bytes)" || { echo "USD is still an LFS pointer"; exit 3; }
  echo "== base image $BASE"
  if $D image inspect "$BASE" >/dev/null 2>&1; then echo "already present"; else
    for i in 1 2 3 4 5 6; do $D pull "$BASE" && break; echo "pull attempt $i failed (NGC token errors are transient); retrying in 20 s"; sleep 20; done
    $D image inspect "$BASE" >/dev/null 2>&1 || { echo "pull failed"; exit 9; }
  fi
}

build() {
  echo "== build $IMG from the workshop Dockerfile (pip installs: several minutes)"
  # the Dockerfile's ffmpeg download (BtbN "latest" release asset) returns 404 as of Sept 2026: use Ubuntu's ffmpeg instead
  python3 - "$WS/docker/sim/Dockerfile" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s2 = re.sub(r'RUN curl --proto "=https".*?rm /tmp/ffmpeg\.tar\.xz\n',
            'RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*\n', s, flags=re.S)
open(p, "w").write(s2); print("Dockerfile ffmpeg step:", "patched" if s2 != s else "already patched or not found")
PY
  $D image inspect "$IMG" >/dev/null 2>&1 && echo "already built" || { $D build -t "$IMG" -f "$WS/docker/sim/Dockerfile" "$WS" > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; echo "build failed"; exit 6; }; tail -3 "$OUT/build.log"; }
  $D image inspect "$IMG" >/dev/null 2>&1 && echo "image ok" || { echo "build failed"; exit 6; }
}

run_in_sim() {  # run_in_sim <script args...>; the workshop's entrypoint pip-installs its package from the mounted source, then execs our command
  # the entrypoint `source`s /root/env under set -e, so the file must exist; PYTHONPATH covers both packages even if the editable install misbehaves
  # a hard cap: Kit can outlive a crashed script; on timeout the container is killed (it is --rm, so also removed)
  $D rm -f simjob >/dev/null 2>&1 || true
  timeout -k 30 "${JOB_TIMEOUT:-1500}" $D run --rm --name simjob --device nvidia.com/gpu=all --network=host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e WRIST_TILT="${WRIST_TILT:--70}" -e WRIST_ROLL="${WRIST_ROLL:-180}" -e WRIST_X="${WRIST_X:--0.005}" -e POLICY_URL="${POLICY_URL:-}" -e POLICY_TOKEN="${POLICY_TOKEN:-}" \
    -e PYTHONPATH=/workspace:/workspace/Sim-to-Real-SO-101-Workshop/source -v "$WS/docker/env":/root/env:ro \
    -v ~/isaac-cache/kit:/isaac-sim/kit/cache:rw -v ~/isaac-cache/ov:/root/.cache/ov:rw -v ~/isaac-cache/pip:/root/.cache/pip:rw \
    -v ~/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw -v ~/isaac-cache/computecache:/root/.nv/ComputeCache:rw \
    -v "$WS/source":/workspace/Sim-to-Real-SO-101-Workshop/source \
    -v "$PKG":/workspace/so101_blocks:ro -v "$DATA":/data:ro -v "$OUT":/out:rw \
    "$IMG" python "$@"
  rc=$?; $D kill simjob >/dev/null 2>&1 || true; return $rc
}

scene() {
  echo "== scene render (headless)"
  run_in_sim /workspace/so101_blocks/scripts/render_scene.py --headless --out /out/scene --sticker 1 \
    --real-top /data/episode_000_top.mp4 --real-wrist /data/episode_000_wrist.mp4 2>&1 | tee "$OUT/scene.log" | grep -E "^\[scene\]|^\[limits\]|Error|error|Traceback|Startup Complete" | grep -v "carb.windowing\|GLFW\|platforminfo" || true
  [ -f "$OUT/scene/scene.json" ] || { echo "no scene.json; last lines of scene.log:"; grep -v "Warning" "$OUT/scene.log" | tail -40; exit 7; }
  ls -la "$OUT/scene"; cat "$OUT/scene/scene.json"
}

wrist_sweep() {  # wrist-camera tilt candidates, one frame each, to pick the mount angle by eye against the real wrist frame
  for combo in ${WRIST_SWEEP:-"-70:-0.005" "-70:0.015" "-70:-0.025"}; do
    t=${combo%%:*}; x=${combo##*:}
    echo "== wrist sweep tilt $t x $x"
    WRIST_TILT=$t WRIST_X=$x run_in_sim /workspace/so101_blocks/scripts/render_scene.py --headless --out "/out/wrist_t${t}_x${x}" --sticker 12 --settle 30 > "$OUT/wrist_t${t}_x${x}.log" 2>&1 || true
    ls "$OUT/wrist_t${t}_x${x}" 2>/dev/null | tr '\n' ' '; echo
  done
}

replay() {  # EP=000 (default) selects /data/episode_EP.npz; output in /out/replay_EP
  EP="${EP:-000}"
  echo "== replay real episode $EP (headless)"
  run_in_sim /workspace/so101_blocks/scripts/replay_episode.py --headless --episode "/data/episode_$EP.npz" --out "/out/replay_$EP" --sticker 0 2>&1 | tee "$OUT/replay_$EP.log" | grep -E "^\[replay\]|^\[limits\]|Error|error|Traceback" | grep -v "carb.windowing\|GLFW\|platforminfo\|OgnSd" || true
  [ -f "$OUT/replay_$EP/replay.json" ] || { echo "no replay.json; last lines of replay_$EP.log:"; grep -v "Warning" "$OUT/replay_$EP.log" | tail -40; exit 8; }
  ls "$OUT/replay_$EP" | tr '\n' ' '; echo; grep -E "block_start_xy|block_red_final_pos|fingertip_mid_at_grasp|block_minus_tip" -A3 "$OUT/replay_$EP/replay.json" | tr -d '\n '; echo
}

record() {  # EPISODES=100 (default); a smoke run: EPISODES=2 bash vm_session2.sh record
  N="${EPISODES:-100}"; OUTD="/out/datasets/so101_blocks_sim${TAG:-}"
  echo "== record $N sim demonstrations -> $OUTD (headless, DR)"
  JOB_TIMEOUT=10800 run_in_sim /workspace/so101_blocks/scripts/record_demos.py --headless --episodes "$N" --out "$OUTD" --seed "${SEED:-0}" --repo-id "timzwu/so101_blocks_sim${TAG:-}" --recovery-share "${RECOVERY:-0.15}" 2>&1 | tee "$OUT/record${TAG:-}.log" | grep -E "^\[record\]|^\[limits\]|Error|error|Traceback" | grep -v "carb.windowing\|GLFW\|platforminfo\|OgnSd" || true
  [ -f "$OUT/${OUTD#/out/}_extras/record_log.json" ] || { echo "no record_log.json; last lines of record.log:"; grep -v "Warning" "$OUT/record${TAG:-}.log" | tail -40; exit 10; }
  du -sh "$OUT/${OUTD#/out/}"; ls "$OUT/${OUTD#/out/}"; ls "$OUT/${OUTD#/out/}_extras" | head
}

simeval() {  # NAME, POLICY (path on the Modal volume), POLICY_URL, POLICY_TOKEN; POSITIONS=1-20
  : "${POLICY_URL:?set POLICY_URL}" "${POLICY_TOKEN:?set POLICY_TOKEN}" "${NAME:?set NAME}" "${POLICY:?set POLICY}"
  echo "== sim eval $NAME on $POLICY, positions ${POSITIONS:-1-20}"
  JOB_TIMEOUT=7200 run_in_sim /workspace/so101_blocks/scripts/eval_in_sim.py --headless --name "$NAME" --policy "$POLICY" --positions "${POSITIONS:-1-20}" --out /out/simeval 2>&1 | tee "$OUT/simeval_$NAME.log" | grep -E "^\[simeval\]|^\[limits\]|Error|error|Traceback" | grep -v "carb.windowing\|GLFW\|platforminfo\|OgnSd" || true
  [ -f "$OUT/simeval/$NAME.csv" ] || { echo "no csv; last lines:"; grep -v "Warning" "$OUT/simeval_$NAME.log" | tail -30; exit 13; }
}

case "$STEP" in
  setup) setup ;;
  build) build ;;
  scene) scene ;;
  wrist) wrist_sweep ;;
  record) record ;;
  simeval) simeval ;;
  replay) replay ;;
  all) setup; build; scene; replay ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
$D logout nvcr.io >/dev/null 2>&1 || true
echo "== done: $STEP"
