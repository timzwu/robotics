"""A general language model drives the arm through gripper poses (no demonstrations, no robot training).

Replicates the interface of Robocurve's GPT-6 Astra arm test on our rig: each turn the model gets the camera frames
and the current gripper pose, and answers with one `move_to` tool call (absolute gripper pose in the arm's base frame
plus a gripper opening). Inverse kinematics (LeRobot's placo solver on the SO-101 URDF) turns the pose into joint
angles; the arm moves there at a bounded speed with every command held within `clamp` degrees of the arm's ACTUAL
position; the model gets the achieved pose, a stall flag and a holding flag, plus fresh frames. Trial rules match every
other pass (30 s of arm motion, thinking excluded) plus the Robocurve budget of 20 model calls.

    python experiments/tools/llm_rollout.py --task "put the red block in the left bowl" --record trial.mp4 --frames-dir frames/
    python experiments/tools/llm_rollout.py --calibrate      # torque off; prints the gripper pose live (hold the arm!)
    python experiments/tools/llm_rollout.py --dry-run --task "..."   # no robot: simulated arm + drawn scene; tests the loop

--prompt plain (default) tells the model only the interface and the physical facts (frame, units, reach, gripper
semantics, image orientation, budget). --prompt coached adds the bowl coordinates and a grasp recipe; not a fair
replication, kept for debugging. The exact prompt is written into the --log JSONL.

Key: OPENAI_API_KEY, else ~/.openai_key.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = JOINTS[:5]
DEFAULT_URDF = Path.home() / "robotics/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
GRIPPER_OPEN, GRIPPER_CLOSED = 37.0, 0.0  # our gripper.pos range in the demonstrations (LeRobot 0–100 units)
GRIPPER_MIN_CMD = 6.0  # never command tighter than this: the demos held the block at a command of ~8 (it reads 11–13); 0 cranks the servo
REST_Q = np.array([0.0, -100.0, 96.0, 56.0, 0.0, 1.0])  # the arm's resting joints in the demonstrations
# Joint box solved INSIDE the IK (placo enforces it): the demonstrations' ranges plus margin. wrist_roll is held near
# zero because the target orientation keeps the jaws horizontal; a wide roll box admits flipped-arm solutions.
JOINT_BOX = {"shoulder_pan": (-70, 70), "shoulder_lift": (-104, 70), "elbow_flex": (-97, 97), "wrist_flex": (0, 100), "wrist_roll": (-150, 90)}
# Workspace box for requested poses (metres): the mat plus the bowls, nothing behind or beside the base.
X_RANGE, Y_RANGE, Z_MAX = (0.12, 0.35), (-0.25, 0.25), 0.25
LEAD_DEG = 15.0      # how far the commanded position may run ahead of the REAL arm (servos sag several degrees under load)
MOVE_CAP_S = 8.0     # a single move never takes longer than this

PROMPT_PLAIN = """You are controlling a small 5-axis robot arm (SO-101) with a parallel-jaw gripper, on a table.
Task: {task}

Coordinates: metres in the arm's base frame. +x points forward across the table away from the base, +y points to the
arm's own left, +z is up. The pose you command is the point between the fingertips. z = 0 is the table surface: with
the fingers pointing straight down (pitch 90), fingertips on the mat means z is about 0.00 to 0.01. The arm cannot go
below z = {z_floor:.2f}. Reachable: x from {x0:.2f} to {x1:.2f}, y from {y0:.2f} to {y1:.2f}, z up to {zmax:.2f}; with the fingers
straight down the arm reaches up to about z 0.13. The arm starts at its rest pose, x 0.18, y 0.00, z 0.01, fingers
pointing forward and down.

Orientation: `pitch_deg` is the finger direction below horizontal (0 = forward, 90 = straight down). `roll_deg` turns
the jaws about the finger axis: at 0 the jaws open along the line from the base to the point, so they close on the
near and far faces of a block; at 90 or -90 they open sideways and close on its left and right faces. Match the roll
to how the block is turned in the image. `gripper` is the jaw opening: 0 = closed, {open:.0f} = fully open. There is no force sensor. After a close command the
reported opening tells you what happened: the jaws stop near 6 when they meet nothing and near 12 when the block is
between them (the result also reports `holding`).

Left and right in the task are from the OPERATOR's seat, facing the arm from the far side of the table: the
operator's left is the arm's right (negative y), and the operator's right is the arm's left (positive y).

Two cameras each turn. The overhead camera sees the whole mat with the arm's base at the bottom edge of the image, so
+x is toward the top of the image and +y (the arm's left) is toward the left of the image; the operator's "left"
therefore appears on the right side of the image. The wrist camera looks out from between the fingers.

Each turn, call one tool: `move_to` with an absolute target pose (position, pitch, roll, gripper), or `move_joints`
with the six joint angles directly (every joint of the arm is yours: base rotation, shoulder, elbow, wrist bend, wrist
roll, gripper; the current angles are reported each turn), or `done` when the block is in the bowl or the task cannot
be completed. The arm gets {duration:.0f} seconds of motion in total{calls_clause}; thinking time is not counted. The result of each
move reports where the gripper actually ended up; if it differs from the target, the pose was out of reach or the arm
was blocked, so adjust."""

PROMPT_COACHED = PROMPT_PLAIN + """

Hints: the block sits on the mat between x 0.15 and 0.32, y -0.10 and 0.10. The operator's left bowl is at about
x 0.24, y -0.15 and the right bowl at x 0.24, y +0.15; release from about z 0.06. To grasp: open the gripper, move
above the block at z 0.10 with pitch 90, descend to z {z_grasp:.2f} with the jaws straddling the block, close to about
{grasp:.0f}, lift to z 0.10, move over the bowl, open."""

TOOLS = [
    {"type": "function", "name": "move_to",
     "description": "Move the fingertip point to an absolute pose in the arm base frame and set the gripper opening. Executed at a bounded speed; returns the achieved pose.",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "number", "description": "metres, forward from the base"},
         "y": {"type": "number", "description": "metres, positive to the arm's own left"},
         "z": {"type": "number", "description": "metres above the table surface"},
         "pitch_deg": {"type": "number", "description": "finger direction below horizontal: 0 = pointing forward, 90 = pointing straight down"},
         "roll_deg": {"type": "number", "description": "rotation of the jaws about the finger axis: 0 = jaws open along the line from the base to the point (they close on the near and far faces of a block), 90 or -90 = jaws open sideways (they close on the left and right faces)"},
         "gripper": {"type": "number", "description": "jaw opening, 0 (closed) to 37 (fully open)"},
         "why": {"type": "string", "description": "one short phrase: what this move is for"}},
         "required": ["x", "y", "z", "pitch_deg", "roll_deg", "gripper", "why"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "move_joints",
     "description": "Command the six joints directly, in degrees (gripper in opening units). Use this for fine adjustments of a single joint, e.g. bend the wrist or roll the gripper; every joint is reachable this way.",
     "parameters": {"type": "object", "properties": {
         "shoulder_pan": {"type": "number", "description": "base rotation, degrees; positive turns the arm to its own left"},
         "shoulder_lift": {"type": "number", "description": "shoulder, degrees; -100 is folded back at rest, larger values reach forward and down"},
         "elbow_flex": {"type": "number", "description": "elbow, degrees; 96 is folded at rest, smaller values straighten the arm"},
         "wrist_flex": {"type": "number", "description": "wrist bend, degrees; about 55 at rest; larger values point the fingers further down"},
         "wrist_roll": {"type": "number", "description": "wrist roll, degrees; 0 at rest"},
         "gripper": {"type": "number", "description": "jaw opening, 0 (closed) to 37 (fully open)"},
         "why": {"type": "string", "description": "one short phrase: what this move is for"}},
         "required": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper", "why"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "done", "description": "Stop: the block is in the bowl, or the task cannot be completed.",
     "parameters": {"type": "object", "properties": {"outcome": {"type": "string"}}, "required": ["outcome"], "additionalProperties": False}, "strict": True},
]


def _finger_axes(x: float, y: float, pitch_deg: float):
    """Finger direction d (the frame's local z; verified on the demonstrations: local z points straight down at every
    grasp), yawed toward (x, y) and pitched below horizontal, plus the roll-zero reference axes: yv0 horizontal (the
    jaw hinge axis) and xv0 = yv0 x d (the direction the jaws open along; on this URDF the jaws open along local x)."""
    yaw, p = math.atan2(y, x), math.radians(pitch_deg)
    d = np.array([math.cos(yaw) * math.cos(p), math.sin(yaw) * math.cos(p), -math.sin(p)])
    yv0 = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return d, yv0, np.cross(yv0, d)


def pose_matrix(x: float, y: float, z: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """4x4 pose of the gripper frame (the fingertip point). pitch = finger direction below horizontal (90 = straight
    down). roll = rotation of the jaws about the finger axis: 0 = the jaws open along the line from the base to the
    point (they close on the block's front and back faces), 90 = the jaws open sideways (left and right faces)."""
    d, yv0, xv0 = _finger_axes(x, y, pitch_deg)
    r = math.radians(roll_deg)
    xv = math.cos(r) * xv0 + math.sin(r) * yv0
    yv = -math.sin(r) * xv0 + math.cos(r) * yv0
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xv, yv, d, [x, y, z]
    return T


def pose_to_xyzp(T: np.ndarray) -> dict:
    x, y, z = (float(v) for v in T[:3, 3])
    pitch = math.degrees(math.asin(float(np.clip(-T[2, 2], -1, 1))))
    _, yv0, xv0 = _finger_axes(x, y, pitch)
    yl = T[:3, 1]
    roll = math.degrees(math.atan2(-float(np.dot(yl, xv0)), float(np.dot(yl, yv0))))
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "pitch_deg": round(pitch, 0), "roll_deg": round(roll, 0)}


def make_kinematics(urdf: str):
    from lerobot.model.kinematics import RobotKinematics

    kin = RobotKinematics(urdf, target_frame_name="gripper_frame_link", joint_names=ARM)
    for name, (lo, hi) in JOINT_BOX.items():  # solve inside the box instead of clamping after (clamping changes the pose)
        kin.robot.set_joint_limits(name, math.radians(lo), math.radians(hi))
    return kin


def solve_ik(kin, q0: np.ndarray, T: np.ndarray, iters: int = 60, tol: float = 0.003) -> tuple[np.ndarray, float]:
    """LeRobot's solver takes one QP step per call; iterate it for a large jump and retry from other seeds (the roll
    joint has two ways round, so a seed with the roll pre-set helps)."""
    best = None
    seeds = [np.array(q0[:5], float), np.array([0.0, -60.0, 60.0, 60.0, 0.0]), np.array([0.0, -60.0, 60.0, 60.0, -90.0]), np.array([0.0, -60.0, 60.0, 60.0, 60.0])]
    for seed in seeds:
        q = seed.copy()
        err = float("inf")
        for _ in range(iters):
            q = kin.inverse_kinematics(q, T, orientation_weight=0.2)
            if not np.all(np.isfinite(q)):
                break
            err = float(np.linalg.norm(kin.forward_kinematics(q)[:3, 3] - T[:3, 3]))
            if err < tol:
                return q, err
        if np.all(np.isfinite(q)) and (best is None or err < best[1]):
            best = (q, err)
    return best if best is not None else (np.array(q0[:5], float), float("inf"))


def jpeg_bytes(rgb: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


class SimArm:
    """Stand-in for --dry-run: goes exactly where commanded, draws a scene, no hardware."""

    def __init__(self, cams, kin=None):
        self.q = REST_Q.copy()
        self.cams = cams
        self.bus = self
        self.kin = kin
        self.block = (0.23, 0.0)  # where the drawn block sits, in the base frame
        self.held = False

    def sync_read(self, _name):
        return {j: float(v) for j, v in zip(JOINTS, self.q)}

    def get_observation(self):
        obs = {f"{j}.pos": float(v) for j, v in zip(JOINTS, self.q)}
        for k in self.cams:
            img = np.full((480, 640, 3), 225, dtype=np.uint8)
            cv2.rectangle(img, (150, 60), (490, 420), (200, 200, 200), -1)
            cv2.circle(img, (80, 240), 55, (120, 170, 120), -1)
            cv2.circle(img, (560, 240), 55, (170, 130, 130), -1)
            if k == "top":
                cv2.rectangle(img, (300, 250), (330, 280), (220, 30, 30), -1)
            else:
                cv2.rectangle(img, (280, 300), (360, 380), (220, 30, 30), -1)
            obs[k] = img
        return obs

    def send_action(self, a):
        q = np.array([a[f"{j}.pos"] for j in JOINTS])
        if self.kin is not None and q[5] < 12:  # closing: if the fingertips are at the block, the jaws stop at 12 (else at the command)
            xyz = self.kin.forward_kinematics(q[:5])[:3, 3]
            near = np.hypot(xyz[0] - self.block[0], xyz[1] - self.block[1]) < 0.03 and xyz[2] < 0.03
            if near or self.held:
                q[5] = max(q[5], 12.0)
                self.held = True
        if q[5] > 15:
            self.held = False
        self.q = q

    def disconnect(self):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="put the red block in the left bowl")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of arm motion (model thinking excluded)")
    ap.add_argument("--max-calls", type=int, default=0, help="model calls per trial, 0 = no limit: the 30 s of motion is the only limit (Robocurve used 20 calls)")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="reasoning effort (Robocurve: medium)")
    ap.add_argument("--prompt", default="plain", choices=["plain", "coached"])
    ap.add_argument("--detail", default="high", choices=["low", "high", "auto"], help="image detail sent to the model")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clamp", type=float, default=3.0, help="max joint degrees per tick at 30 Hz (3 = about 90 deg/s; the other passes used robot.json's 5)")
    ap.add_argument("--z-grasp", type=float, default=0.01, help="coached prompt only: fingertip height for a grasp (demos: 0.001–0.017)")
    ap.add_argument("--z-floor", type=float, default=-0.005, help="lowest allowed fingertip z (demos bottomed at about -0.005 with a 25 mm block)")
    ap.add_argument("--grasp", type=float, default=3.0, help="coached prompt only: gripper command that holds the block (demos commanded ~3–11)")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--robot-config", default=str(ROOT / "robot.json"))
    ap.add_argument("--record", default="", help="write the camera frames during motion to this mp4 (motion only, think pauses cut)")
    ap.add_argument("--frames-dir", default="", help="save the two JPEGs the model saw at each call here")
    ap.add_argument("--log", default="", help="append one JSON line for the trial (header, prompt, every call) to this file")
    ap.add_argument("--calibrate", action="store_true", help="torque off, print the gripper pose live (Ctrl-C to stop). HOLD THE ARM when torque drops.")
    ap.add_argument("--dry-run", action="store_true", help="no robot: simulated arm and a drawn scene; exercises the model loop")
    args = ap.parse_args()

    kin = make_kinematics(args.urdf)
    r = json.load(open(args.robot_config))
    clamp = float(args.clamp)
    if clamp <= 0:
        sys.exit("--clamp must be > 0 (it is the per-step speed bound; 0 would never move)")
    cam_names = list(r["cameras"])

    if args.dry_run:
        bot = SimArm(cam_names, kin)
    else:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

        cams = {k: OpenCVCameraConfig(index_or_path=v["index_or_path"], width=v["width"], height=v["height"], fps=v["fps"]) for k, v in r["cameras"].items()}
        # max_relative_target: the robot itself refuses any goal farther than `clamp` from the PRESENT position.
        bot = SO101Follower(SOFollowerRobotConfig(port=r["port"], id=r["id"], cameras=cams, max_relative_target=LEAD_DEG))
        bot.connect(calibrate=False)

    def present() -> np.ndarray:
        d = bot.bus.sync_read("Present_Position")
        return np.array([d[j] for j in JOINTS], dtype=float)

    def observation(retries: int = 3):
        for i in range(retries):
            try:
                return bot.get_observation()
            except (TimeoutError, RuntimeError) as e:
                print(f"[llm] camera read failed ({e}); retry {i + 1}", flush=True)
                time.sleep(0.2)
        raise RuntimeError("cameras not delivering frames")

    if args.calibrate:
        print("Torque goes OFF now: hold the arm. Printing the fingertip pose 2x per second; Ctrl-C to stop.", flush=True)
        bot.bus.disable_torque()
        try:
            while True:
                q = present()
                print(f"joints {np.round(q, 1).tolist()}  ->  fingertips {pose_to_xyzp(kin.forward_kinematics(q[:5]))}  gripper {q[5]:.1f}", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            bot.disconnect()
        return

    import openai

    key = os.environ.get("OPENAI_API_KEY") or (Path.home() / ".openai_key").read_text().strip()
    client = openai.OpenAI(api_key=key, timeout=180, max_retries=3)
    tmpl = PROMPT_COACHED if args.prompt == "coached" else PROMPT_PLAIN
    system = tmpl.format(task=args.task, z_floor=args.z_floor, x0=X_RANGE[0], x1=X_RANGE[1], y0=Y_RANGE[0], y1=Y_RANGE[1], zmax=Z_MAX,
                         open=GRIPPER_OPEN, duration=args.duration, z_grasp=args.z_grasp, grasp=args.grasp,
                         calls_clause=f" and at most {args.max_calls} calls" if args.max_calls else "")

    rec = None
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        rec = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640 * len(cam_names), 480))
    if args.frames_dir:
        Path(args.frames_dir).mkdir(parents=True, exist_ok=True)
    cam_label = {"top": "overhead camera", "wrist": "wrist camera"}

    def observe(call_no: int):
        obs = observation()
        q = present()
        pose = pose_to_xyzp(kin.forward_kinematics(q[:5]))
        pose["gripper"] = round(float(q[5]), 1)
        joints_now = {j: round(float(v), 1) for j, v in zip(JOINTS, q)}
        content = [{"type": "input_text", "text": f"Current fingertip pose: {json.dumps(pose)}. Current joints (deg): {json.dumps(joints_now)}."}]
        for k in cam_names:
            jpg = jpeg_bytes(obs[k])
            if args.frames_dir:
                Path(args.frames_dir, f"call{call_no:02d}_{k}.jpg").write_bytes(jpg)
            content.append({"type": "input_text", "text": f"{cam_label.get(k, k)}:"})
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "detail": args.detail})
        return q, pose, content

    def execute(target: np.ndarray, t_budget: float) -> tuple[float, bool]:
        """Move toward `target`. The commanded position is integrated toward the target at most `clamp` degrees per
        tick (eased near the goal) but is never allowed more than LEAD_DEG ahead of the arm's REAL position, so the
        servos can push through their load sag without the command ever running away (the Sept 12 lurch). The move
        ends when the arm is at the target, when it has settled (no movement for 0.5 s: reached within sag, or the
        gripper stopped on an object), when it is blocked (settled while the command is still short of the target),
        or at min(budget, MOVE_CAP_S). Returns (seconds used, blocked)."""
        t0 = time.time()
        history: list[np.ndarray] = []
        blocked = False
        cur = present()
        while True:
            t_step = time.time()
            p = present()
            step = np.minimum(clamp, np.maximum(1.0, 0.25 * np.abs(target - cur)))
            cur = cur + np.clip(target - cur, -step, step)
            cur = np.clip(cur, p - LEAD_DEG, p + LEAD_DEG)
            bot.send_action({f"{j}.pos": float(cur[i]) for i, j in enumerate(JOINTS)})
            if rec is not None:
                try:
                    o = bot.get_observation()
                    rec.write(np.hstack([cv2.cvtColor(np.ascontiguousarray(o[k]), cv2.COLOR_RGB2BGR) for k in cam_names]))
                except (TimeoutError, RuntimeError):
                    pass
            time.sleep(max(0.0, 1 / args.fps - (time.time() - t_step)))
            history.append(p.copy())
            gap = target - p
            cmd_at_target = np.all(np.abs(target - cur) < 0.5)
            settled = len(history) > 15 and np.all(np.abs(history[-1] - history[-16]) < 0.5)
            if np.all(np.abs(gap[:5]) < 2.0) and (abs(gap[5]) < 1.5 or settled):
                break
            if settled:
                blocked = not cmd_at_target and bool(np.any(np.abs(gap[:5]) > 3.0))
                break
            if time.time() - t0 >= min(t_budget, MOVE_CAP_S):
                break
        time.sleep(0.25)  # let the servos settle before the achieved pose is read (counted as motion)
        return time.time() - t0, bool(blocked)

    header = {"task": args.task, "model": args.model, "effort": args.effort, "prompt": args.prompt, "detail": args.detail, "clamp": clamp,
              "max_calls": args.max_calls, "duration": args.duration, "z_floor": args.z_floor, "z_grasp": args.z_grasp, "grasp": args.grasp,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "system_prompt": system}
    moved, calls, prev_id, log = 0.0, 0, None, []
    usage = {"in": 0, "out": 0, "cached": 0, "reasoning": 0}
    outcome, exit_code = "motion time used up", 0
    q, pose, content = observe(0)
    inp = [{"role": "user", "content": [{"type": "input_text", "text": "Begin. " + content[0]["text"]}, *content[1:]]}]
    nudged = False
    try:
        while (not args.max_calls or calls < args.max_calls) and moved < args.duration:
            t_think = time.time()
            resp = None
            for attempt in range(3):
                try:
                    resp = client.responses.create(model=args.model, instructions=system, tools=TOOLS, reasoning={"effort": args.effort},
                                                   input=inp, previous_response_id=prev_id, tool_choice="required",
                                                   parallel_tool_calls=False, max_output_tokens=16000)
                    break
                except openai.APIError as e:  # the arm is stationary while we wait, so retrying is safe
                    print(f"[llm] API error ({type(e).__name__}: {str(e)[:120]}); retry {attempt + 1}", flush=True)
                    time.sleep(3 * (attempt + 1))
            if resp is None:
                outcome, exit_code = "api error", 2
                break
            think = time.time() - t_think
            calls += 1
            prev_id = resp.id
            u = resp.usage
            usage["in"] += u.input_tokens
            usage["out"] += u.output_tokens
            usage["cached"] += getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            usage["reasoning"] += getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
            fcs = [o for o in resp.output if o.type == "function_call"]
            if not fcs:
                print(f"[llm] call {calls}: no tool call (status {resp.status}; {resp.output_text[:80]!r})", flush=True)
                log.append({"call": calls, "no_tool_call": True, "status": resp.status, "text": resp.output_text[:500], "think_s": round(think, 2)})
                if nudged:
                    outcome = "no tool call"
                    break
                nudged = True
                inp = [{"role": "user", "content": [{"type": "input_text", "text": "Reply with a tool call: move_to or done."}]}]
                continue
            fc = fcs[0]
            a = json.loads(fc.arguments)
            outputs = [{"type": "function_call_output", "call_id": extra.call_id, "output": json.dumps({"error": "one move per turn; this call was ignored"})}
                       for extra in fcs[1:]]  # every call_id must be answered or the next turn is rejected
            if fc.name == "done":
                print(f"[llm] call {calls}: done -> {a.get('outcome')!r} (think {think:.1f}s)", flush=True)
                log.append({"call": calls, "done": a.get("outcome"), "think_s": round(think, 2), "in_tokens": u.input_tokens, "out_tokens": u.output_tokens, "response_id": resp.id})
                outcome = "model said done"
                break
            if fc.name == "move_joints":
                jt = np.array([float(a[j]) for j in ARM])
                jc = np.array([np.clip(v, *JOINT_BOX[j]) for v, j in zip(jt, ARM)])
                g = float(np.clip(a["gripper"], GRIPPER_MIN_CMD, GRIPPER_OPEN))
                T_sol = kin.forward_kinematics(jc)
                req = {"joints": {j: float(v) for j, v in zip(ARM, jt)}, "gripper": float(a["gripper"])}
                x, y, z = (float(v) for v in T_sol[:3, 3])
                clipped = bool(np.any(jc != jt) or g != float(a["gripper"]))
                q5, ik_err = jc, 0.0
                unreachable = bool(T_sol[2, 3] < args.z_floor - 0.005)  # a joint target that would drive the fingertips into the mat
                req["pitch_deg"], req["roll_deg"] = pose_to_xyzp(T_sol)["pitch_deg"], pose_to_xyzp(T_sol)["roll_deg"]
            else:
                req = {"x": float(a["x"]), "y": float(a["y"]), "z": float(a["z"]), "pitch_deg": float(a["pitch_deg"]), "roll_deg": float(a.get("roll_deg", 0.0)), "gripper": float(a["gripper"])}
                x = float(np.clip(req["x"], *X_RANGE))
                y = float(np.clip(req["y"], *Y_RANGE))
                z = float(np.clip(req["z"], args.z_floor, Z_MAX))
                g = float(np.clip(req["gripper"], GRIPPER_MIN_CMD, GRIPPER_OPEN))
                clipped = bool((x, y, z, g) != (req["x"], req["y"], req["z"], req["gripper"]))
                q5, ik_err = solve_ik(kin, q[:5], pose_matrix(x, y, z, req["pitch_deg"], req["roll_deg"]))
                T_sol = kin.forward_kinematics(q5) if np.all(np.isfinite(q5)) else None
                unreachable = bool(T_sol is None or ik_err > 0.015 or T_sol[2, 3] < args.z_floor - 0.005)
            if unreachable:
                result = {"error": "pose unreachable (or below the table); the arm did not move", "requested": req, "ik_error_m": round(ik_err, 3) if math.isfinite(ik_err) else None,
                          "achieved": pose, "motion_seconds_left": round(max(0.0, args.duration - moved), 1), "calls_left": (args.max_calls - calls) if args.max_calls else None}
                t_used, stalled = 0.0, False
            else:
                target_q = np.append(q5, g)
                t_used, stalled = execute(target_q, args.duration - moved)
                moved += t_used
                q, pose, content = observe(calls)
                holding = bool(g < 15 and pose["gripper"] - g > 4)  # jaws stopped well above the commanded closure: something is between them
                result = {"achieved": pose, "achieved_joints": {j: round(float(v), 1) for j, v in zip(JOINTS, q)}, "requested": ({"x": x, "y": y, "z": z, "pitch_deg": req["pitch_deg"], "roll_deg": req["roll_deg"], "gripper": g} if fc.name == "move_to" else req), "clipped": clipped,
                          "blocked": stalled, "holding": holding, "motion_seconds_used": round(moved, 1),
                          "motion_seconds_left": round(max(0.0, args.duration - moved), 1), "calls_left": (args.max_calls - calls) if args.max_calls else None}
            print(f"[llm] call {calls}: {fc.name} {a.get('why', '')!r} -> ({x:.3f},{y:.3f},{z:.3f}) p{req['pitch_deg']:.0f} r{req['roll_deg']:.0f} g{g:.0f}"
                  + (f" | UNREACHABLE (ik err {ik_err * 100:.1f} cm)" if unreachable else f" | got {pose}{' BLOCKED' if stalled else ''}{' holding' if result.get('holding') else ''}")
                  + f" | think {think:.1f}s move {t_used:.1f}s | tokens {u.input_tokens}/{u.output_tokens}", flush=True)
            log.append({"call": calls, "tool": fc.name, "why": a.get("why"), "requested": req, "executed": None if unreachable else result["requested"], "clipped": clipped,
                        "unreachable": unreachable, "ik_error_m": round(ik_err, 4) if math.isfinite(ik_err) else None, "achieved": pose,
                        "blocked": stalled, "holding": result.get("holding"), "move_s": round(t_used, 2), "think_s": round(think, 2),
                        "in_tokens": u.input_tokens, "out_tokens": u.output_tokens, "response_id": resp.id})
            inp = [{"type": "function_call_output", "call_id": fc.call_id, "output": json.dumps(result)}, *outputs,
                   {"role": "user", "content": content if not unreachable else [{"type": "input_text", "text": "The arm did not move. Choose another pose."}]}]
    except KeyboardInterrupt:
        outcome, exit_code = "interrupted", 130
    except Exception as e:  # noqa: BLE001
        outcome, exit_code = f"crashed: {type(e).__name__}: {str(e)[:200]}", 3
        print(f"[llm] {outcome}", flush=True)
    finally:
        if rec is not None:
            rec.release()
        try:  # back to rest before torque drops: lift straight up first so the jaws clear the bowl rim, then fold
            if calls:
                qn = present()
                T = kin.forward_kinematics(qn[:5])
                T[2, 3] = min(T[2, 3] + 0.06, 0.16)
                q_up, err = solve_ik(kin, qn[:5], T)
                if err < 0.02:
                    execute(np.append(q_up, max(qn[5], GRIPPER_MIN_CMD)), 4.0)
                execute(REST_Q.copy(), 6.0)
                print("[llm] returned to rest", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[llm] return-to-rest failed ({e}); releasing torque where it is", flush=True)
        bot.disconnect()
        print(f"[llm] end: {outcome}; {calls} calls, {moved:.1f}s of motion, tokens {usage['in']} in ({usage['cached']} cached) / {usage['out']} out ({usage['reasoning']} reasoning)", flush=True)
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(json.dumps({**header, "outcome": outcome, "exit_code": exit_code, "calls": calls, "moved_s": round(moved, 1), "usage": usage, "log": log}) + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
