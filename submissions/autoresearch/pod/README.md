# Running the experiments on a rented GPU

Every run in `../runs/` was trained on one rented GPU: an RTX 4090 on Runpod's Secure Cloud,
$0.89 an hour, 10 runs at a time. You do not need it: the notebooks read the saved runs.
It is here so you can see how, and do the same for your own experiments.

## How it works

The GPU machine (a *pod*) needs no login and no API key once it starts:

1. `worker.sh` is its start command. It clones the branch, uses the PyTorch that comes
   with the image (`runpod/pytorch`), and checks that the GPU really works (`gpu_check.py`
   trains one step on the GPU and on the CPU and compares the numbers).
2. Every 30 seconds it pulls the branch and runs the experiments named in `queue.txt`
   (names from `../experiments.py`). Finished runs are skipped, so the queue only grows.
3. It serves `status.txt`, `log.txt`, `gpu.txt` and `runs/*.jsonl` on port 8000, at
   `https://<pod id>-8000.proxy.runpod.net/`.
4. To give it work: add an experiment to `experiments.py`, add its name to `queue.txt`,
   push. To collect results: `python pod/fetch.py <pod id>`.

Pod settings: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, port `8000/http`,
environment variables `BRANCH=<your branch>` and `WORKERS=10`, start command:

```sh
bash -c "curl -sL https://raw.githubusercontent.com/<you>/undefined-eurosat/<branch>/submissions/autoresearch/pod/worker.sh | bash"
```

## Python 3.12 on the pod, 3.14 on your computer

The repo asks for Python 3.14; the image comes with 3.12 and a CUDA build of PyTorch 2.8.
Installing 3.14 with a CUDA PyTorch would download several GB, so the pod uses what it has,
and three small things make the same code run on both:

- `pull.sh` adds `from __future__ import annotations` to the pod's copy of
  `src/eurosat/data.py` (it names its own class in a type hint, which 3.14 reads lazily).
- `search.py` uses no syntax newer than 3.12.
- The repo's FLOP counter checks for uncounted operations with a PyTorch feature that 2.8
  lacks; `search.Counted` then counts with the same counter and formulas without that
  check. The totals are identical (`tests/test_autoresearch.py` checks the submission's).

The GPU does float32 arithmetic like a CPU (`search.py` turns off the GPU's TF32
shortcut), so a setting scores the same on the GPU and on your CPU: experiment 8's best
probe scored 96.88% on split 11 with `uv run score` on a CPU, and 96.89% on average over
3 seeds on the GPU.

## What it cost

| | |
|---|---|
| A Community Cloud 4090 ($0.34/h) | its GPU was broken ("CUDA unknown error", found by `gpu_check.py` in a minute); deleted: about $0.02 |
| The Secure Cloud 4090 ($0.89/h) | 2 h 05 min for all 231 runs of `../runs/`: $1.86 |
| **Total** | **about $1.87** |

## Lessons

- **Check the GPU before you trust it.** A cheap host can show a GPU that PyTorch cannot
  use. `worker.sh` stops with "GPU BROKEN" in `status.txt` if the check fails.
- **Many small trainings at once.** A network this small leaves most of a GPU idle, so 10
  runs share it.
- **Stop the pod when you are done.** It bills every hour it exists.
