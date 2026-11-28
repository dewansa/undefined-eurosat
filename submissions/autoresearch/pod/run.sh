#!/bin/bash
# Pull the branch, then run experiments in the background: bash run.sh 0-naive 1-lr
# Progress goes to /root/log.txt; results to submissions/autoresearch/runs/<name>.jsonl.
bash /root/repo/submissions/autoresearch/pod/pull.sh
cd /root/repo/submissions/autoresearch
nohup python3 search.py "$@" --device cuda --workers "${WORKERS:-8}" >> /root/log.txt 2>&1 &
echo "started $* (pid $!)"
