#!/bin/bash
# One round of the worker: run every experiment named in queue.txt (finished runs are skipped).
cd /root/repo/submissions/demo2
{
  date -u; nvidia-smi; ls -la /dev/nvidia*
  python -c 'import torch; print("venv torch", torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name())' 2>&1 | tail -2
  /usr/bin/python3 -c 'import torch; print("image torch", torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name())' 2>&1 | tail -2
} > $WWW/diag.txt 2>&1
if ! python -c 'import torch; torch.zeros(1).cuda()' 2>/dev/null; then
  echo "no CUDA in the venv $(date -u +%T): see diag.txt" > $WWW/status.txt; exit
fi
queue=$(grep -v '^#' pod/queue.txt | xargs)
if [ -z "$queue" ]; then echo "idle $(date -u +%T)" > $WWW/status.txt; exit; fi
echo "running $queue since $(date -u +%T) at $(git rev-parse --short HEAD)" > $WWW/status.txt
python lab.py $queue --device cuda --workers ${WORKERS:-8}
echo "idle $(date -u +%T), done: $queue" > $WWW/status.txt
