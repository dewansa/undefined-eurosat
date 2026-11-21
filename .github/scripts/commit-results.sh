#!/usr/bin/env bash
# Commit any changes in results/ and push them to main.
# Tries again if someone else pushed to main in the meantime.
# Usage: commit-results.sh "commit message"
set -euo pipefail

git add results/
if git diff --cached --quiet; then
  echo "Nothing changed."
  exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git commit -m "$1"
for attempt in 1 2 3 4 5; do
  if git pull --rebase && git push; then
    echo "changed=true" >> "${GITHUB_OUTPUT:-/dev/null}"
    exit 0
  fi
  echo "Push failed (try $attempt). Trying again."
  sleep $((attempt * 5))
done
exit 1
