"""Stitch the side-camera clips of the recorded sim episodes into one fast-forward mp4 (the data being generated,
seen from a third-person view). Mac, `lerobot` env (needs ffmpeg on PATH).
    python sim/tools/make_side_reel.py --clips 'sim/03_sim_demos/so101_blocks_sim_extras/side_ep*.mp4' --speed 10 --out sim/03_sim_demos/sim_demos_side_10x.mp4
"""
import argparse, glob, os, subprocess, tempfile
ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True); ap.add_argument("--speed", type=int, default=10); ap.add_argument("--out", required=True)
a = ap.parse_args()
clips = sorted(c for c in glob.glob(a.clips) if "failed" not in c)
assert clips, "no clips"
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    for c in clips:
        f.write(f"file '{os.path.abspath(c)}'\n")
    lst = f.name
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                "-vf", f"setpts=PTS/{a.speed},drawtext=text='sim demonstrations, {a.speed}x':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5",
                "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out], check=True)
os.unlink(lst)
print(a.out, len(clips), "clips")
