#!/bin/bash
# One round of the worker: run every experiment named in queue.txt (finished runs are skipped).
cd /root/repo/submissions/demo2
# The repo asks for Python 3.14; the image has an older one. This line is all it needs
# (data.py names its own class in a type hint). It changes the pod's copy only.
sed -i '1i from __future__ import annotations' /root/repo/src/eurosat/data.py
export PYTHONPATH=/root/repo/src
if ! python3 -c 'import torch; torch.zeros(1).cuda()' 2> $WWW/diag.txt; then
  echo "no CUDA $(date -u +%T): see diag.txt" > $WWW/status.txt; exit
fi
queue=$(grep -v '^#' pod/queue.txt | xargs)
if [ -z "$queue" ]; then echo "idle $(date -u +%T)" > $WWW/status.txt; exit; fi
echo "running $queue since $(date -u +%T) at $(git rev-parse --short HEAD)" > $WWW/status.txt
python3 lab.py $queue --device cuda --workers ${WORKERS:-8}
echo "idle $(date -u +%T), done: $queue" > $WWW/status.txt
