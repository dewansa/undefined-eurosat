#!/bin/bash
# Run once on a fresh Runpod pod with the image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# (from the web terminal, or with Runpod's exec): bash setup.sh <branch>
# It uses the PyTorch that comes with the image (Python 3.12), installs nothing,
# checks that the GPU really works, and downloads the images.
set -e
BRANCH=${1:-main}
cd /root
[ -d repo ] || git clone -q --branch "$BRANCH" https://github.com/dewansa/undefined-eurosat repo
bash /root/repo/submissions/autoresearch/pod/pull.sh
cd /root/repo/submissions/autoresearch
python3 pod/gpu_check.py
python3 -c 'import sys; sys.path.insert(0, "../../src"); from eurosat import data; print(len(data.train_set()), "training images")'
