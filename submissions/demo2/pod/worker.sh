#!/bin/bash
# Turn a fresh GPU pod into an experiment worker. It needs no SSH and no API key:
#   - it installs this repo's branch and the CUDA build of PyTorch,
#   - every 30 seconds it pulls the branch and runs the experiments in pod/queue.txt,
#   - it serves its results on port 8000: runs/*.jsonl, log.txt and status.txt.
# Start it as the pod's command:
#   bash -c "curl -sL <raw URL of this file> | BRANCH=<branch> bash"
set -x
BRANCH=${BRANCH:-main}
WWW=/root/www
mkdir -p $WWW/runs
cd $WWW && nohup python3 -m http.server 8000 > /dev/null 2>&1 &
echo "installing $(date -u)" > $WWW/status.txt
pip install -q uv
git clone -q --branch "$BRANCH" https://github.com/dewansa/undefined-eurosat /root/repo
uv venv -q -p 3.14 /root/venv
. /root/venv/bin/activate
uv pip install -q torch numpy pandas
uv pip install -q --no-deps -e /root/repo
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())' > $WWW/gpu.txt 2>&1
while true; do
  git -C /root/repo fetch -q origin "$BRANCH" && git -C /root/repo reset -q --hard FETCH_HEAD
  LAB_RUNS=$WWW/runs WWW=$WWW bash /root/repo/submissions/demo2/pod/step.sh >> $WWW/log.txt 2>&1
  sleep 30
done
