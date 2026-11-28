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
