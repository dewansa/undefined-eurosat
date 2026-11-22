# EuroSAT class leaderboard

Train a convolutional network that looks at a satellite photo of Europe and says what
kind of land it shows: forest, river, highway, a town, a crop field, and so on. Then
see how you rank against your classmates.

**Leaderboard:** https://dewansa.github.io/undefined-eurosat/

The photos come from [EuroSAT](https://zenodo.org/records/7711810) (MIT license): 27,000
color images, 64x64 pixels, taken by the Sentinel-2 satellite, in 10 classes:

| | | | | |
|---|---|---|---|---|
| 0 Annual crop | 1 Forest | 2 Herbaceous vegetation | 3 Highway | 4 Industrial |
| 5 Pasture | 6 Permanent crop | 7 Residential | 8 River | 9 Sea or lake |

We split them once: **21,600 training images** you get, and **5,400 test images** whose
labels are secret. GitHub trains your model, asks it about the test images, and a
separate job that never runs your code compares its answers with the secret labels.

The leaderboard shows these numbers for each submission:

- **Test accuracy:** how many test images the model labels correctly. Higher is better.
  The **±** next to it is how far luck alone can move a score on 5,400 images (about 1
  point). Two scores closer than that may not really differ; the rank shows a **≈** then.
  Point at an accuracy to see it class by class.
- **Parameters:** how many numbers the model learns by gradient descent.
- **Size:** all the memory the trained model holds on to, in bytes: parameters, buffers,
  and any other tensor it keeps.
- **Compute:** how much arithmetic your `train()` did, in FLOPs (one multiply or add is
  one FLOP). It counts the convolutions and matrix multiplies of every forward and
  backward pass. Unlike time, it is the same on every computer. A count marked **\*** may
  be too low: your code ran operations the counter has no formula for.
- **Compute per image:** the same count for labelling one test image with the trained model.
- **Training time:** how long your `train()` ran on GitHub's computer.
- **Training curves:** whatever your `train()` logged, epoch by epoch (see below).

The chart draws a line through the **best trade-offs**: submissions that no other
submission beats on both accuracy and cost.

## What you need

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python and everything else for you.
- A GitHub account.
- About 250 MB of disk for the images, and a few GB of memory.

## How to submit

1. **Fork** this repo on GitHub, then clone your fork and set it up:

   ```sh
   git clone https://github.com/<your-github-name>/undefined-eurosat
   cd undefined-eurosat
   uv sync
   ```

2. **Make your folder** by copying the baseline. Name it after your GitHub name, exactly:
   `submissions/<your-github-name>`. For a second idea, use `<your-github-name>-<anything>`,
   for example `submissions/alice-wide`.

   ```sh
   cp -r submissions/baseline submissions/<your-github-name>
   ```

3. **Improve the model.** Edit `submissions/<your-github-name>/model.py`. You own the
   whole training loop: the network, the optimizer, the epochs, augmentation, everything.

4. **Score it on your computer:**

   ```sh
   uv run score submissions/<your-github-name>
   ```

   The first run downloads the images (about 240 MB) into `data/`. Because the test labels
   are secret, this holds back 20% of the training images, trains on the rest, and checks
   your model on the held-back ones. It prints the accuracy for each class and the most
   common mix-ups. The leaderboard's number will be close, not equal.

   More options:

   ```sh
   uv run score submissions/<you> --quick          # train on a quarter of the images: a fast try
   uv run score submissions/<you> --report report  # save pictures of your first layer's
                                                   # filters and of the images it got wrong
   uv run score submissions/<you> --flops          # also count the compute (trains twice)
   ```

5. **Open a pull request** from your fork. GitHub checks your code, then trains your model
   and scores it. Before you push, you can run the same checks yourself:

   ```sh
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```

6. **See your score.** A few minutes later, a bot comments on the pull request: your test
   accuracy, how it compares with the baseline and your best merged score, the accuracy
   for each class, and the most common mix-ups. Each new push is scored again.

7. **It merges itself** when everything passed and the pull request only changes your own
   folder(s). Then GitHub scores it once more on main, counts its compute, and your row
   becomes *reviewed*. Every merged submission keeps its own row, so your earlier scores
   stay. Want to try again? Open a new pull request. (The first time you open a pull
   request, the teacher may need to approve the run before it starts.)

## The rules

Your `model.py` must define this function:

```python
def train(train_set, log) -> nn.Module:
    ...
    return model
```

- `train_set.images` holds the 21,600 training images as one `uint8` tensor of shape
  `(21600, 3, 64, 64)`, values 0 to 255. `train_set.labels` holds their labels, 0 to 9.
  Convert images to floats **one batch at a time** (`x.float() / 255`): all of them as
  floats would take about 1 GB. (`train_set[i]` also works and gives one float image.)
- `log(...)` draws your training curves on the leaderboard. It is optional. **Watching your
  model is free:** give it your validation images once, then log the model each epoch:

  ```python
  log.validate_on(valid_images, valid_labels)  # once: uint8 images like train_set.images
  for epoch in range(EPOCHS):
      ...
      log(model, train_loss=train_loss)  # saves a copy of the weights: milliseconds
  ```

  After `train()` returns and the clock has stopped, the scorer checks every saved copy on
  those images and fills in `valid_loss` and `valid_accuracy`. That time does not count.
  You can also pass numbers you measured yourself, with any of `train_loss`, `valid_loss`,
  `train_accuracy`, `valid_accuracy`. That measuring is training time. So is anything your
  training *uses*, such as validation accuracy for stopping early: the free check only
  happens afterwards.

  Log as often as you like. Up to 50 copies are kept (and at most 1 GB); past that, the
  curve keeps every 2nd, 4th... call, plus the last one. At most 5,400 validation images
  are used.
- Return your trained model. We test it on the CPU with float images of shape
  `(batch, 3, 64, 64)` and values 0 to 1. For each batch it must return one row of 10 class
  scores per image, shape `(batch, 10)`. The class with the highest score is its answer.
- **Train only on `train_set`.** No other data, no pretrained weights, and no downloads.
- **Training must finish in 10 minutes** on GitHub's computer, which has 4 CPU cores and no
  GPU. The baseline takes under a minute there. The time GitHub reports is the one that counts.
- If you change the input (for example, normalize it), do it inside your model, because the
  test images arrive unchanged.
- Put all your code in `model.py` and only change files inside your own folder.

## Ideas to try

- Train for more epochs, and watch your curves: stop before the dashed (validation) line
  turns up.
- More filters per layer, or more layers. Add `nn.BatchNorm2d` after each convolution.
- Normalize each color channel inside the model.
- Augment: satellite photos have no "up", so flips and 90° rotations make new, valid images.
- A learning-rate schedule, such as `OneCycleLR` or cosine annealing.
- Look at `--report`: which classes does your model confuse, and why might it?

## Project layout

```
src/eurosat/data.py           downloads and loads the images
src/eurosat/score.py          trains a submission and measures it (the `score` command)
src/eurosat/leaderboard.py    grades answers with the secret labels; keeps the rows
submissions/<name>/model.py   one folder per submission
results/                      one small JSON file per merged (reviewed) submission
results/pending/              one file per open pull request (not reviewed)
site/index.html               the leaderboard page
tools/make_data.py            teacher only: builds the train and test files
tests/                        checks for the scoring code and all submissions
.github/workflows/check.yml          checks every pull request, trains and answers (read-only)
.github/workflows/pr-leaderboard.yml grades, comments, updates the leaderboard, auto-merges
.github/workflows/publish.yml        scores merged submissions and publishes them
```

## For the teacher

- **Secret labels:** the repo secret `TEST_LABELS` holds the 5,400 test labels as one line of
  digits. Only the grading steps read it; the jobs that run student code never get it. Keep
  a copy of `private/test-labels.txt` from `tools/make_data.py` somewhere safe.
- **Data:** `tools/make_data.py` builds `dist/eurosat-train.npz` and `dist/eurosat-test.npz`
  (test images, no labels), uploaded to the GitHub release `data-v1`. `data.py` checks their
  SHA-256. The original EuroSAT is public, so a determined student could look test images up
  in it; the secret labels stop casual tuning on the test set, not deliberate cheating.
- **Auto-merge:** a pull request merges itself when all checks pass, every changed folder was
  scored, and every changed file is inside `submissions/<author>` or `submissions/<author>-*`.
  Anything else waits for you. To turn it off, set the repo variable `AUTO_MERGE` to `false`
  (Settings > Secrets and variables > Actions > Variables).
- **Pages:** Settings > Pages, Source: "GitHub Actions".
- **Score again:** Actions > "Score and publish" > Run workflow, with folder names or `all`.
