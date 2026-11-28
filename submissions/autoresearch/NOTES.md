# Notes: autoresearch

Goal: beat the leader before us, BuzinD (96.57% test, 26.2 TFLOP to train, 119 MFLOP per
image), on all three numbers at once, by climbing from a naive CNN with generic
techniques. Nothing is taken from BuzinD's code or from the demos: every choice is made by
an experiment.

How I work: one question per experiment, written with a prediction before the runs;
3 seeds per setting; chosen on split 11 (4,320 held-back images), confirmed on splits 7
and 3. Accuracy is the mean of the seeds, ± its standard error. The bar on split 11 is
BuzinD's own model.py scored here: **96.67%** (97.06% on split 7, 96.81% on split 3).
Every run is a line in `runs/`; the plots are in `2-climb.ipynb`.

| # | Question | Expected | Result | Decision |
|---|---|---|---|---|
| 0 | Where does the naive CNN stand (`c16 p c32 p c64 p fl`, Adam 1e-3, 10 epochs)? | about 80% | 78.7 ± 1.9 at 10 epochs; 84.3 at 30, but training accuracy 94 by then | it starts to memorize after ~15 epochs |
| 1 | Is Adam's default rate right? | maybe higher | 3e-4 74.9, 1e-3 79.9, 3e-3 81.6, 1e-2 72.6 | small gain; revisit later |
| 2 | Standardize the input, or batch norm in every layer? | both help | standardize 87.4 ± 0.4 (+8.7, real); batch norm learns as fast (train 93) but check accuracy swings 57-87 from epoch to epoch | batch norm, but the end point is a lottery |
| 3 | Flatten or global average pool? | pooling saves parameters | too noisy to read (every batch-norm run swings) | ask again once the swings are gone |
| 4 | Does warming up and annealing the rate to zero (cosine) settle batch norm? | yes | batch norm + cosine 3e-3: 91.5 ± 0.2 (standardize + cosine 89.2); flatten 91.5 (train 97), pool 91.0, **one more conv at 8x8, then pool: 94.2 ± 0.0** | batch norm, cosine, Adam 3e-3; head = conv at 8x8 then pool |
| 5 | At 30 epochs, do flipped and turned copies stop the memorizing? | +2 | none 92.2 (train 99.9!), mirror 94.3, all flips 95.3, all 8 symmetries 95.6, plus shifts of up to 4 pixels **96.0 ± 0.1** (train 96.5) | all 8 symmetries and shifts |
| 6 | The plain CNN spends 9.4 MFLOP in each stage: which earn their keep? (30 epochs) | the full-size layer least | reference (32 MF) 96.45; stem with stride 2 (29 MF) 96.62; start at 16x16 (15 MF) 95.65 (−1.0); half the width (8.9 MF) 94.48 (−2.0). Training accuracy = check accuracy everywhere | the 64x64 layer is wasted; the 32x32 stage earns its FLOPs; the networks **underfit** |
| 7 | At ~10 MFLOP, which block buys the most accuracy? | separable | plain 94.5, plain on smaller images 94.3, residual 95.2, separable 96.2, inverted residual 96.1 — but the leaderboard's counter charges depthwise convolutions' backward pass as dense ones (65x the forward pass, not 2x): separable training is *counted* as 54 TFLOP, not 18 | residual blocks: the best of the blocks the counter charges fairly |
