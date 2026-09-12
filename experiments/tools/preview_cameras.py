"""Live preview of every camera in experiments/tools/robot.json at the EXACT resolution the dataset will record.

Why: `lerobot-find-cameras` grabs the sensor's full field of view (e.g. 1920x1080), which is NOT what gets
recorded. Recording uses the width/height in robot.json (640x480), and on this webcam that is a 4:3 CROP of
the 16:9 sensor. This tool shows the cropped view so you can frame the rig against what the policy will see.

    python experiments/tools/preview_cameras.py            # all cameras, one window each; press q to quit
    python experiments/tools/preview_cameras.py --save     # also write a frame per camera to outputs/preview/

Run it in your own Terminal (needs macOS camera permission for that app).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

ROBOT_JSON = Path(__file__).with_name("robot.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write one frame per camera to outputs/preview/")
    args = ap.parse_args()

    cams = json.loads(ROBOT_JSON.read_text())["cameras"]
    caps: dict[str, cv2.VideoCapture] = {}
    for name, c in cams.items():
        cap = cv2.VideoCapture(c["index_or_path"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        cap.set(cv2.CAP_PROP_FPS, c["fps"])
        caps[name] = cap
        print(f"{name}: index {c['index_or_path']} requested {c['width']}x{c['height']} "
              f"-> got {int(cap.get(3))}x{int(cap.get(4))}")
    time.sleep(1.0)  # warm-up

    print("Press q in any window to quit.")
    saved = False
    try:
        while True:
            for name, cap in caps.items():
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                cv2.putText(frame, f"{name} {w}x{h}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(name, frame)
                if args.save and not saved:
                    out = Path("outputs/preview"); out.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if args.save and not saved:
                saved = True
                print("saved one frame per camera to outputs/preview/")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
