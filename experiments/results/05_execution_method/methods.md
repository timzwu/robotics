# Execution method: stop-and-go vs Real-Time Chunking — methods

From the repository root:

```bash
python experiments/tools/eval.py --mode rtc --name smolvla100_rtc_pair --combos red:left,blue:right --rtc-horizon 10 --rtc-guidance 10 --camera-rename top=camera1,wrist=camera2 --policy experiments/checkpoints/smolvla_so101_blocks_n100_s0
```
The baseline uses `--mode sync` on the same checkpoint.
