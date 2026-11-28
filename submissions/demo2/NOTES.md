# Notes: demo2

Goal: match the top of the leaderboard (BuzinD, 96.6%) with at most a fifth of its compute:
5.2 TFLOP of training (it uses 26.2) and 24 MFLOP to label one image (it uses 119).

How I work: one question per experiment, 3 seeds per setting, chosen on the `seed=1` split
(4,320 check images). Accuracy is the mean of the 3 runs; ± is the standard error of that
mean. The details and plots are in `2-climb.ipynb`; every run is a line in `runs/`.

| # | Question | Expected | Result | Decision |
|---|---|---|---|---|
| 0 | Does the harness reproduce the leader (BuzinD's settings)? | about 96.5% | 96.86 ± 0.16; 1 view 96.16, 2 views 96.61, 4 views 96.84, 8 views 96.86 | trust the harness; 4 views give all of the benefit of 8 |
| 1 | Same compute: wider network or more epochs? (widths x 0.5-1.5, 4-16 epochs) | the leader's width is about right | at ~4 TFLOP: width 1 x 4 epochs 86.3, width 0.75 x 8 90.1, width 0.5 x 16 93.4; each doubling of epochs ≈ +4 points | spend compute on epochs, not width |
| 2 | Training tricks, one at a time at the probe (width 0.5, 16 epochs, 93.4) | small gains each | batch 64 +1.0 (real), batch 256 −0.7; lr 0.003 −0.8 (real); weight decay, weight averaging, label smoothing, zero-init, no augmentation: all within luck; half the epochs at 32x32 −1.0 for 37% less compute | keep batch 64; test lr with batch; test low-res at equal compute |
| 3 | Other shapes at the probe | 4x4 squares lose little | 4x4 squares −2.5 (real); no stage 1 −1.1 for −35% compute (a tie); wider late / deeper +0.2 / +0.7 for +37% / +70% compute | keep the leader's shape |
| 4 | How far does "narrower and longer" go? (widths x 0.25-0.5, up to 128 epochs) | best width about 1/3 | at ~4.5 TFLOP: 0.25 x 64 epochs 93.9, 0.33 x 40 94.3, 0.5 x 16 93.4; at ~9 TFLOP: 0.25 95.3, 0.33 95.9, 0.5 95.8. Small networks gain +1.2 from 4 views | width 1/3 to 1/2 |
| 5 | Batch size x learning rate, at the probe | smaller batch wants a smaller rate | higher rate better at every batch; batch 32 ≈ batch 64 at lr 0.024 (94.8); batch 128 at most 93.7 | batch 64 (twice as fast as 32 on a CPU), lr 0.024 |
| 6 | Training on 32x32 images first, at equal compute (~4 TFLOP) | a tie | 16 epochs at 64x64 93.5; 26 epochs, half at 32x32 94.3 (+0.75, real); 36 epochs, 3/4 at 32x32 94.4 (+0.9, real) | train most epochs on 32x32 images |
| 7 | Everything together, each setting filling 5 TFLOP (batch 64, lr 0.024; width 1/3 or 1/2; 1/2, 3/4 or 7/8 of the epochs at 32x32) | about 96.5 | 95.8 to 96.0 for 1/2 and 3/4 at either width (all within luck); 7/8 worse (95.1-95.3); lr 0.048 a tie. Seeds now differ by up to 1.5 points | candidate: width 1/2, 45 epochs, 3/4 at 32x32; still 0.9 below the leader on this split |
| 8 | Do regularizers help long training? (at the candidate, 6 seeds) | +0.2 to +0.5 each | label smoothing +0.19, weight averaging −0.14, weight decay 0.05 −0.07 (all luck); mixup −0.53 (real). Training accuracy 95.0 vs check 95.7: the network does not memorize, it underfits | no regularizers: they fight a problem we do not have |
| 9 | AdamW or SGD (Nesterov momentum 0.9, weight decay 5e-4)? (at the candidate) | a tie | SGD lr 0.1: 96.44 vs AdamW 95.87 (+0.57, real); SGD lr 0.05 / 0.2 / 0.4: 96.07 / 96.05 / 95.73 | SGD, lr 0.1 |
| 10 | The AdamW candidate at half, 1x and 2x the compute | +1 point per doubling | 22 epochs (2.5 TFLOP) 94.31, 45 (4.9) 95.89, 90 (9.7) 96.91: the leader's accuracy at 37% of its compute | the curve is still rising: compute buys accuracy |
| 11 | How much compute does SGD need to match the leader (96.86 here)? | about 1/4 | 45 epochs (4.9 TFLOP, 6 seeds) 96.30; 60 (6.5) 96.70; 75 (8.2) 97.00; 90 (9.7) 97.08 | **choice: 60 epochs** (1/4 of the leader's training compute), 4 views (1/7 of its answering compute). At 1/5 we are 0.5 short (real) |
| 12 | Confirm on fresh splits (7 and 3, 3 runs each): leader vs ours | a tie | leader 97.10; ours (60 epochs) 96.86 (−0.24 ± 0.16, luck); 45 epochs 96.45 (−0.65, real); 75 epochs 97.26 (+0.16, luck) | keep 60 epochs, as decided before these runs |
| 13 | Did each part of the leader earn its place? (one removed at a time) | all of them | turning and mirroring −1.32 (real; without them training accuracy 99.9: memorizing); max pool −1.03 (real); 20% warm-up −0.65 (real); weight decay −0.18 (luck); residual shortcuts −0.08 (luck); views 5-8 +0.02 (exp 0) | the shortcuts, the weight decay and views 5-8 did not earn their place |
| 14 | What did each of our choices buy? (one undone at a time, same compute) | — | the leader's width −2.03; no turning −1.13; all epochs at 64x64 −0.69 (all real); batch 128 −0.55 (luck, noisy); AdamW −0.16 (luck: SGD's +0.57 at 45 epochs did not hold at 60) | width, augmentation and small images first carry the result |

GPU: about 70 minutes of one RTX 4090 (Runpod), about $1.06 in all, for 330 training runs.
