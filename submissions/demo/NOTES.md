# Notes: demo

One line per experiment: what I changed, what I expected, what happened. The details are in
`2-build-model.ipynb`. Accuracy is on the `seed=1` check images, averaged over 2 seeds;
differences under 1.4 points are luck.

| Change | Expected | Result | Keep? |
|---|---|---|---|
| Starting point: average over the image instead of the big last layer, twice the channels, batch norm, turning | the parameters do more useful work | 85.0% (only 5 epochs fit in 90 s) | yes |
| Learning rate from 0.001 to 1 | a sweet spot in between | 0.003 to 0.1 tie | 0.01, the middle |
| 5 shapes, each with the epochs that fit in 90 s | stride 2: less detail, more epochs | 4 blocks with stride 2: 92.3% (12 epochs); at full size 88.4% (5 epochs) | 4 blocks, stride 2 |
| Learning rate again, for the new shape | might move | 0.003 to 0.032 tie | 0.01 |
| Batch size from 32 to 256 | not sure | 32 to 128 tie; 256 worse (90.3%) | 64 |
| No batch norm (with its own best learning rate) | worse | 80.9% (−12); up to 40% of channels dead | keep batch norm |
| Turning and mirroring | might hurt: the light comes from one side | tie (+0.2); won by 1.7 in a run with more epochs | on |
| Weight decay 0.01 and 0.1 | less memorizing | tie; training and validation only 2 points apart | off |
| Check on other images: `uv run score` | about the same | 92.1% (the baseline: 78.1%) | submit |

Next ideas: train longer and grow the network (10 minutes allowed); answer 8 times (+0.5).
