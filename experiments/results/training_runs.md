# Training runs

Every fine-tune behind a reported result: the data it saw, the recipe, the GPU, the time and cost, and the final
training loss. Passes = steps × batch ÷ frames in the dataset. Costs are approximate; pre-Sept 12 runs and the frozen-backbone π0.5 control are GPU-only. Other rows include CPU and RAM. Smoke tests, fit trials and interrupted attempts are omitted.

| date (UTC) | job | policy | data | steps × batch | passes | GPU | wall | cost | final loss |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | act_so101_blocks_20000_ep0-49 | ACT from scratch (ResNet-18 ImageNet backbone) | `so101_blocks` ep 0–49: red→left, blue→right | 20,000 × 8 | 5.7 | A10G | 55.3 min | ≈$1.01 | 0.18 |
| 2026-09-08 | act_so101_blocks_20000_ep0-49_100-149 | ACT from scratch (ResNet-18 ImageNet backbone) | `so101_blocks` ep 0–49 + 100–149: red→left, blue→right | 20,000 × 8 | ~2.9 | A10G | 55.3 min | ≈$1.01 | 0.19 |
| 2026-09-09 | smolvla_so101_blocks_n100_s0 | SmolVLA fine-tune from `lerobot/smolvla_base` | `so101_blocks` ep 0–49 + 100–149: red→left, blue→right | 20,000 × 64 | 24 | L40S | 176.8 min | ≈$5.75 | 0.027 |
| 2026-09-09 | act_so101_blocks_100000_ep0-49_100-149 | ACT continued from the 20k checkpoint (LeRobot default length) | same 100 episodes | +80,000 × 8 (100k total) | 15 total | A10G | ~3.75 h | ≈$4.10 | 0.118 |
| 2026-09-10 | smolvla_so101_blocks_n10_s0 | SmolVLA fine-tune (sweep) | 10 episodes of the pair (5 per task), balanced | 20,000 × 64 | ~240 | L40S | 194 min | ≈$6.30 | ≈0.01 |
| 2026-09-10 | smolvla_so101_blocks_n25_s0 | SmolVLA fine-tune (sweep) | 25 episodes of the pair (12/13 per task) | 20,000 × 64 | ~95 | L40S | 196 min | ≈$6.40 | 0.013 |
| 2026-09-10 | smolvla_so101_blocks_n50_s0 | SmolVLA fine-tune (sweep) | 50 episodes of the pair (25 per task) | 20,000 × 64 | ~48 | L40S | ~195 min | ≈$6.30 | not captured |
| 2026-09-10 | smolvla_nowrist_n100 | SmolVLA fine-tune, OVERHEAD camera only (camera ablation) | `so101_blocks_nowrist` ep 0–49 + 100–149 | 20,000 × 64 | ~24 | L40S | 111 min | ≈$3.60 | 0.032 |
| 2026-09-10 | smolvla_notop_n100 | SmolVLA fine-tune, WRIST camera only (camera ablation) | `so101_blocks_notop` ep 0–49 + 100–149 | 20,000 × 64 | ~24 | L40S | 110 min | ≈$3.60 | 0.028 |
| 2026-09-12 | pi05_so101_blocks_40000_ep0-49_100-149 | π0.5 frozen-backbone control, matched to SmolVLA (24 passes) | same 100 pair episodes | 40,000 × 32 | 24 | H100 | ~13.2 h total, including restart | ≈$65 total (GPU-only) | 0.049 |
| 2026-09-13 | pi05full_so101_blocks_30000_ep0-49_100-149 | π0.5 fine-tune, PI's recipe: whole model trains (no freeze flags), LR 2.5e-5 cosine to 30k, bf16, grad checkpointing | same 100 pair episodes | 30,000 × 32 (global; 8 per card) = 18 passes | 18 | 4 × H100 | 362 min | ≈$100 | 0.012 |
| 2026-09-15 | smolvla_so101_blocks_sim_20000 | SmolVLA fine-tune from `lerobot/smolvla_base`, condition C of the sim experiment | `so101_blocks_sim`: 100 scripted sim episodes, red→left, blue→right (36,771 frames) | 20,000 × 64 | 35 | L40S | 192 min | ≈$7.50 | 0.003 |
| 2026-09-15 | smolvla_so101_blocks_simreal_20000_ep0-49_100-149_150-249 | SmolVLA fine-tune from `lerobot/smolvla_base`, condition D | `so101_blocks_simreal`: the 100 real pair episodes + the 100 sim episodes (90,771 frames) | 20,000 × 64 | 14 | L40S | 195 min | ≈$7.55 | 0.024 |
| 2026-09-16 | smolvla_so101_blocks_simreal_20000_ep0-49_100-149_150-182 | SmolVLA fine-tune from `lerobot/smolvla_base`, 75 / 25 real : sim split | `so101_blocks_simreal`: the 100 real pair episodes + the first 33 sim (v1) episodes (~66,000 frames) | 20,000 × 64 | 19 | L40S | 194 min | ≈$7.50 | 0.025 |
| 2026-09-16 | smolvla_so101_blocks_simreal_v2_20000_ep0-49_100-149_150-249 | SmolVLA fine-tune from `lerobot/smolvla_base`, condition D2: real + sim v2 | `so101_blocks_simreal_v2`: the 100 real pair episodes + sim v2 episodes 0–99 (95,184 frames) | 20,000 × 64 | 14 | L40S | 194 min | ≈$7.50 | 0.023 |
| 2026-09-16 | smolvla_so101_blocks_sim_v2_20000_ep0-99 | SmolVLA fine-tune from `lerobot/smolvla_base`, condition C2: sim v2 only | `so101_blocks_sim_v2` episodes 0–99: 100 scripted sim v2 episodes (41,184 frames) | 20,000 × 64 | 31 | L40S | 193 min | ≈$7.50 | 0.006 |
