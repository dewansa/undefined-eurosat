# Notes: demo2

Goal: match the top of the leaderboard (BuzinD, 96.6%) with at most a fifth of its compute:
5.2 TFLOP of training (it uses 26.2) and 24 MFLOP to label one image (it uses 119).

How I work: one question per experiment, 3 seeds per setting, chosen on the `seed=1` split
(4,320 check images). Accuracy is the mean of the 3 runs; ± is the standard error of that
mean. The details and plots are in `2-climb.ipynb`; every run is a line in `runs/`.

| # | Question | Expected | Result | Decision |
|---|---|---|---|---|
