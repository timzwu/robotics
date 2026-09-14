"""Stitch the language-model trials into one mp4 with the model's own justification feed beside the cameras.

    python experiments/tools/make_llm_feed_video.py --name astra_pair
    python experiments/tools/make_llm_feed_video.py --name astra_pair --trials 5,11,13,18 --out .../astra_highlights_feed.mp4

Inputs: experiments/results/<experiment>/<name>.calls.jsonl (one line per trial, written by llm_rollout.py --log; every
call with its `why`, requested and achieved pose, think and move seconds), <name>.csv (my verdict and notes) and
experiments/results/trials/<name>_NN.mp4 (overhead | wrist, motion only, think pauses cut).

Layout: overhead camera over wrist camera on the left; on the right a feed of the model's calls. Each call is shown in two
phases: a still hold with "thought N s" and the justification, then its motion at `--speed` times real time (the mp4s were
captured at ~21.5 frames/s, so playback is timed from the logged move seconds, not the frame count). Frames are assigned to
calls in proportion to each call's logged move time; the recorder is released before the return-to-rest, so the mapping
covers the whole file.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--trials", default="", help="comma list of positions to include (default all)")
ap.add_argument("--speed", type=float, default=6.0, help="motion playback, times real time")
ap.add_argument("--hold", type=float, default=1.0, help="seconds to hold on each justification before its motion")
ap.add_argument("--done-hold", type=float, default=2.0)
ap.add_argument("--true-time", action="store_true", help="hold each call for its logged thinking time divided by --speed (true Nx throughout) instead of a fixed --hold")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default="")
args = ap.parse_args()

hits = sorted(RES.glob(f"*/{args.name}.calls.jsonl"))
if not hits:
    sys.exit(f"[feed] no {args.name}.calls.jsonl under {RES}")
records = [json.loads(l) for l in open(hits[0]) if l.strip()]
rows = {int(r["position"]): r for r in csv.DictReader(open(hits[0].with_name(f"{args.name}.csv")))}
want = {int(t) for t in args.trials.split(",") if t.strip()} if args.trials else None
out = Path(args.out or hits[0].parent / f"{args.name}_feed_{args.speed:g}x.mp4")
tmp = out.with_suffix(".raw.mp4")

W, H = 1920, 960
CAM_W, CAM_H = 640, 480
PANEL_X = CAM_W + 32
PANEL_W = W - PANEL_X - 32
BG, PANEL, INK, DIM, ACCENT, GOOD, BAD, WARN = (18, 18, 20), (26, 26, 30), (235, 232, 226), (128, 126, 120), (214, 132, 58), (96, 190, 120), (222, 90, 80), (222, 180, 70)


def font(size, mono=False, bold=False):
    path = "/System/Library/Fonts/Menlo.ttc" if mono else "/System/Library/Fonts/Helvetica.ttc"
    try:
        return ImageFont.truetype(path, size, index=1 if bold and not mono else 0)
    except OSError:
        return ImageFont.load_default()


F_H1, F_BODY, F_SMALL, F_MONO = font(30, bold=True), font(27), font(22), font(21, mono=True)


def wrap(draw, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def pose_str(p):
    if not p:
        return ""
    if "x" in p:
        return f"({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f}) pitch {p['pitch_deg']:.0f} roll {p['roll_deg']:.0f} grip {p['gripper']:.0f}"
    if "joints" in p:  # move_joints: six joint angles, then the gripper
        j = p["joints"]
        short = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wflex", "wrist_roll": "roll"}
        return "joints " + " ".join(f"{short.get(k, k)} {v:.0f}" for k, v in j.items()) + f"  grip {p.get('gripper', 0):.0f}"
    return " ".join(f"{k[:5]} {v:.0f}" for k, v in p.items() if isinstance(v, (int, float)))


def result_line(c):
    if c.get("unreachable"):
        return f"UNREACHABLE (ik error {c.get('ik_error_m', 0) * 100:.1f} cm), no motion", BAD
    flags = []
    if c.get("holding"):
        flags.append("HOLDING")
    if c.get("blocked"):
        flags.append("blocked")
    return f"got {pose_str(c.get('achieved'))}" + (f"   {' '.join(flags)}" if flags else ""), (GOOD if c.get("holding") else DIM)


def render(cams, header, entries, current, phase, clocks, verdict=None, phase_t=None):
    """cams: 1280x480 BGR frame or None. entries: list of call dicts up to and including current."""
    img = Image.new("RGB", (W, H), BG)
    if cams is not None:
        top, wrist = cams[:, :CAM_W], cams[:, CAM_W:]
        img.paste(Image.fromarray(cv2.cvtColor(top, cv2.COLOR_BGR2RGB)), (0, 0))
        img.paste(Image.fromarray(cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)), (0, CAM_H))
    d = ImageDraw.Draw(img)
    d.text((12, 8), "overhead camera", font=F_SMALL, fill=INK)
    d.text((12, CAM_H + 8), "wrist camera", font=F_SMALL, fill=INK)
    d.rectangle([PANEL_X - 16, 0, W, H], fill=PANEL)
    y = 22
    d.text((PANEL_X, y), header, font=F_H1, fill=INK)
    y += 44
    d.text((PANEL_X, y), clocks, font=F_MONO, fill=DIM)
    y += 40
    d.line([PANEL_X, y, W - 32, y], fill=(50, 50, 56), width=2)
    y += 16
    # Lay the feed out bottom-up so the current entry is always visible: build blocks, then draw the ones that fit.
    blocks = []
    for e in entries:
        is_cur = e is current
        lines = []
        if "done" in e:
            lines.append(("h", f"call {e['call']} · done · thought {e.get('think_s', 0):.1f} s", ACCENT if is_cur else DIM))
            for l in wrap(d, e["done"], F_BODY, PANEL_W):
                lines.append(("b", l, INK if is_cur else DIM))
            if verdict and is_cur:
                lines.append(("v", verdict[0], verdict[1]))
                for l in wrap(d, verdict[2], F_SMALL, PANEL_W):
                    lines.append(("s", l, DIM))
        else:
            head = f"call {e['call']} · {e['tool']} · thought {e.get('think_s', 0):.1f} s"
            if is_cur:
                head += "   ·   " + ((f"thinking… {phase_t:.1f} s" if phase_t is not None else "thinking done, about to move") if phase == "hold" else f"moving  {e.get('move_s', 0):.1f} s")
            lines.append(("h", head, ACCENT if is_cur else DIM))
            for l in wrap(d, e.get("why", ""), F_BODY, PANEL_W):
                lines.append(("b", l, INK if is_cur else (200, 198, 192)))
            lines.append(("m", "→ " + pose_str(e.get("executed") or e.get("requested")), (170, 168, 160) if is_cur else DIM))
            if not is_cur or phase == "after":
                txt, col = result_line(e)
                lines.append(("m", txt, col))
        blocks.append((is_cur, lines))
    hgt = {"h": 32, "b": 34, "m": 28, "v": 34, "s": 28}
    heights = [sum(hgt[k] for k, _, _ in ls) + 18 for _, ls in blocks]
    avail = H - y - 16
    keep = []
    used = 0
    for b, h in zip(reversed(blocks), reversed(heights)):
        if used + h > avail:
            break
        keep.append(b)
        used += h
    for is_cur, lines in reversed(keep):
        if is_cur:
            d.rectangle([PANEL_X - 10, y - 6, W - 28, y + sum(hgt[k] for k, _, _ in lines) + 6], fill=(38, 34, 30))
        for kind, text, col in lines:
            f = {"h": F_SMALL, "b": F_BODY, "m": F_MONO, "v": font(27, bold=True), "s": F_SMALL}[kind]
            d.text((PANEL_X, y), text, font=f, fill=col)
            y += hgt[kind]
        y += 18
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
n_out = 0


def emit(frame, seconds):
    global n_out
    for _ in range(max(1, int(round(seconds * args.fps)))):
        writer.write(frame)
        n_out += 1


for idx, rec in enumerate(records, 1):
    if want is not None and idx not in want:
        continue
    row = rows.get(idx, {})
    cap = cv2.VideoCapture(str(RES / "trials" / f"{args.name}_{idx:02d}.mp4"))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log = rec["log"]
    moves = [c for c in log if "done" not in c]
    total_move = sum(c.get("move_s", 0) for c in moves) or 1.0
    # frame range per call, proportional to logged move seconds
    ranges, acc = {}, 0.0
    for c in moves:
        a = int(round(acc / total_move * n_frames))
        acc += c.get("move_s", 0)
        b = int(round(acc / total_move * n_frames))
        ranges[c["call"]] = (a, max(b, a))
    needed = {}
    for c in moves:
        a, b = ranges[c["call"]]
        k = max(1, int(round(c.get("move_s", 0) * args.fps / args.speed))) if b > a else 0
        for j in np.linspace(a, b - 1, k).round().astype(int) if k else []:
            needed.setdefault(int(j), None)
    frames, i = {}, 0
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in needed:
            frames[i] = fr
        last = fr
        i += 1
    cap.release()
    first = frames[min(frames)] if frames else last
    ok = row.get("success", "").strip().lower() in ("1", "true", "y", "yes")
    header = f"Trial {idx}/20 · {rec['task']}"
    think_total, motion_total = 0.0, 0.0
    entries = []
    cur_frame = first
    emit(render(cur_frame, header, [], None, "hold", "motion 0.0 / 30 s · thinking 0 s"), 0.8)
    for c in log:
        entries.append(c)
        think_total += c.get("think_s", 0)
        clocks = f"motion {motion_total:4.1f} / 30 s · thinking {think_total:5.1f} s · calls {c['call']}"
        if args.true_time:
            clocks += f" · {args.speed:g}x speed including inference pauses"
        if "done" in c:
            verdict = ("✓ SUCCESS (judged by the operator)", GOOD) if ok else (f"✗ FAILURE: {row.get('failure_type', '')}", BAD)
            emit(render(cur_frame, header, entries, c, "hold", clocks, (verdict[0], verdict[1], "Note: " + row.get("notes", ""))), max(args.done_hold, c.get("think_s", 0) / args.speed) if args.true_time else args.done_hold)
            break
        if args.true_time:
            th = c.get("think_s", 0)
            nf = max(1, int(round(th / args.speed * args.fps)))
            for q in range(nf):
                writer.write(render(cur_frame, header, entries, c, "hold", clocks, phase_t=th * (q + 1) / nf))
                n_out += 1
        else:
            emit(render(cur_frame, header, entries, c, "hold", clocks), args.hold)
        a, b = ranges[c["call"]]
        k = max(1, int(round(c.get("move_s", 0) * args.fps / args.speed))) if b > a else 0
        for step, j in enumerate(np.linspace(a, b - 1, k).round().astype(int) if k else []):
            fr = frames.get(int(j), cur_frame)
            cur_frame = fr
            t = motion_total + c.get("move_s", 0) * (step + 1) / k
            writer.write(render(fr, header, entries, c, "move", f"motion {t:4.1f} / 30 s · thinking {think_total:5.1f} s · calls {c['call']}" + (f" · {args.speed:g}x speed including inference pauses" if args.true_time else "")))
            n_out += 1
        motion_total += c.get("move_s", 0)
        emit(render(cur_frame, header, entries, c, "after", f"motion {motion_total:4.1f} / 30 s · thinking {think_total:5.1f} s · calls {c['call']}" + (f" · {args.speed:g}x speed including inference pauses" if args.true_time else "")), 0.35)
    print(f"[feed] trial {idx}: {len(log)} calls, {n_frames} frames -> {n_out / args.fps:.0f} s so far", flush=True)

writer.release()
FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(out)])
if r.returncode == 0:
    tmp.unlink()
    print(f"[feed] wrote {out} ({n_out / args.fps:.0f} s, {out.stat().st_size / 1e6:.1f} MB)")
else:
    tmp.rename(out)
    print(f"[feed] ffmpeg missing or failed; wrote raw mp4v {out}")
