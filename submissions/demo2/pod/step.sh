#!/bin/bash
# One round of the worker: run every experiment named in queue.txt (finished runs are skipped).
cd /root/repo/submissions/demo2
queue=$(grep -v '^#' pod/queue.txt | xargs)
if [ -z "$queue" ]; then echo "idle $(date -u +%T)" > $WWW/status.txt; exit; fi
echo "running $queue since $(date -u +%T) at $(git rev-parse --short HEAD)" > $WWW/status.txt
python lab.py $queue --device cuda --workers ${WORKERS:-8}
echo "idle $(date -u +%T), done: $queue" > $WWW/status.txt
