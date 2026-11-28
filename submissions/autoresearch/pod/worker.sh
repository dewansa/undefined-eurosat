#!/bin/bash
# Turn a fresh Runpod PyTorch pod into an experiment worker that needs no login.
# Use it as the pod's start command (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
# port 8000/http, environment variable BRANCH=<your branch>):
#   bash -c "curl -sL <raw URL of this file> | bash"
# Then:
#   - it checks the GPU (gpu.txt) and downloads the images,
#   - every 30 seconds it pulls the branch and runs the experiments named in pod/queue.txt
#     (finished runs are skipped, so the same queue can stay there),
#   - it serves status.txt, log.txt, gpu.txt and runs/*.jsonl on port 8000, at
#     https://<pod id>-8000.proxy.runpod.net/
BRANCH=${BRANCH:-main}
WWW=/root/www
mkdir -p $WWW/runs
(cd $WWW && nohup python3 -m http.server 8000 > /dev/null 2>&1 &)
echo "starting $(date -u +%T)" > $WWW/status.txt
git clone -q --branch "$BRANCH" https://github.com/dewansa/undefined-eurosat /root/repo
cd /root/repo/submissions/autoresearch
bash pod/pull.sh > $WWW/log.txt 2>&1
if ! python3 pod/gpu_check.py > $WWW/gpu.txt 2>&1; then
  echo "GPU BROKEN $(date -u +%T): see gpu.txt. Delete this pod and try another." > $WWW/status.txt
  sleep infinity
fi
python3 -c 'import sys; sys.path.insert(0, "../../src"); from eurosat import data; data.train_set()' >> $WWW/log.txt 2>&1
# Start from the runs already committed, so a new pod skips them.
cp -n runs/*.jsonl $WWW/runs/ 2>/dev/null
export AUTORESEARCH_RUNS=$WWW/runs
while true; do
  bash pod/pull.sh >> $WWW/log.txt 2>&1
  queue=$(grep -v '^#' pod/queue.txt | xargs)
  echo "running $queue since $(date -u +%T) at $(git rev-parse --short HEAD)" > $WWW/status.txt
  python3 search.py $queue --device cuda --workers "${WORKERS:-8}" >> $WWW/log.txt 2>&1
  echo "idle $(date -u +%T) at $(git rev-parse --short HEAD), done: $queue" > $WWW/status.txt
  sleep 30
done
