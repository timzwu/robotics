#!/usr/bin/env bash
# R2: record the dataset in runs of 25, one task string per run, into ONE local dataset.
# Runs 1-4 cover the four tasks (ep 0-99); runs 5-6 add 25 more of the base pair (ep 100-149) so the
# two-task pool is 100 episodes, 50 per task.
# Usage: bash experiments/tools/record_r2.sh <run 1-6> [dataset name]   (run 1 creates, later runs resume)
# Pushed to the Hub privately afterwards from Python (push_to_hub stays false here so a crash mid-run
# never leaves a half-uploaded repo). Ports and camera indices match experiments/tools/robot.json.
# "left"/"right" are from the OPERATOR seat facing the arm, which is mirrored in the overhead camera
# (operator-left = image-right). The eval protocol uses the same convention.
set -euo pipefail
RUN="${1:?run number 1-4}"; NAME="${2:-so101_blocks}"
case "$RUN" in
  1) TASK="put the red block in the left bowl";  RESUME=false ;;
  2) TASK="put the blue block in the right bowl"; RESUME=true ;;
  3) TASK="put the red block in the right bowl"; RESUME=true ;;
  4) TASK="put the blue block in the left bowl";  RESUME=true ;;
  5) TASK="put the red block in the left bowl";  RESUME=true ;;   # +25 of the base pair (ep 100-124)
  6) TASK="put the blue block in the right bowl"; RESUME=true ;;   # +25 of the base pair (ep 125-149)
  *) echo "run must be 1-6"; exit 1 ;;
esac
echo "run $RUN / 6 : \"$TASK\"  (25 episodes, 30 s each, 10 s reset; resume=$RESUME)"
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B610339821 --robot.id=follower_arm \
  '--robot.cameras={ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5B3D0414741 --teleop.id=leader_arm \
  --dataset.repo_id="timzwu/$NAME" --dataset.root="$HOME/.cache/huggingface/lerobot/timzwu/$NAME" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=25 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false --dataset.no_stamp=true --resume="$RESUME" --display_data=true
