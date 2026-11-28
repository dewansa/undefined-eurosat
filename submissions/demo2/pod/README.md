# Running the experiments on a rented GPU

Every run in `../runs/` was trained on one rented GPU (an RTX 4090 on Runpod, $0.89 an
hour), 8 runs at a time. All 318 runs took about 70 minutes and cost about a dollar. You
do not need this: the notebooks read the saved runs. It is here so that you can see
how, and do the same for your own experiments.

## How it works

The GPU machine (a *pod*) needs no login from you once it starts:

1. `worker.sh` is its start command. It clones this branch, uses the PyTorch that comes
   with the image (`runpod/pytorch`), and serves its results on port 8000.
2. Every 30 seconds it pulls the branch and runs `step.sh`, which runs every experiment
   named in `queue.txt` (names from `../plans.py`). Finished runs are skipped, so the
   same queue can be run again safely.
3. To give it work, add a plan to `plans.py`, add its name to `queue.txt`, and push.
4. To collect results, download `runs/*.jsonl` from the pod's port 8000 (Runpod gives
   each port an address like `https://<pod id>-8000.proxy.runpod.net/runs/`).

Settings for the pod: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, port
`8000/http`, environment variables `BRANCH=<your branch>` and `WORKERS=8`, start command:

```sh
bash -c "curl -sL https://raw.githubusercontent.com/<you>/undefined-eurosat/<branch>/submissions/demo2/pod/worker.sh | bash"
```

## Lessons that cost us time (and a few cents)

- **Stop or delete the pod when you are done.** It bills every hour it exists.
- **A cheap machine can be broken.** Our first two pods (community cloud) could not use
  their GPU at all: `nvidia-smi` worked but PyTorch reported "CUDA unknown error", with
  every version of PyTorch. It was the host, not our code. Check the GPU first
  (`worker.sh` writes `gpu.txt`) and move to another machine if it fails.
- **Use the PyTorch in the image** instead of installing your own: installing the CUDA
  build downloads several GB and took us minutes. The repo asks for Python 3.14 and the
  image has 3.12; `step.sh` adds one line to the pod's copy of `data.py` to make it work.
- **Many small trainings at once.** A network this small leaves most of a GPU idle, so
  8 runs share it. Each run still uses exactly the same code and float32 arithmetic as
  on a CPU (`lab.py` turns off the GPU's TF32 shortcut), so the results carry over.
