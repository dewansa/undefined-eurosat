#!/usr/bin/env bash
# Print the submission folders to score, one name per line.
#
#   changed-submissions.sh all          every submission
#   changed-submissions.sh A...B        submissions changed in that git range
#
# The name rule matches leaderboard.py's NAME_PATTERN, so a folder that trains
# is never dropped later. At most 10 folders, so one run cannot take forever.
set -euo pipefail

if [ "$1" = "all" ]; then
  names=$(ls submissions)
else
  names=$(git diff --name-only "$1" -- submissions/ | cut -d/ -f2 | sort -u)
fi

count=0
for name in $names; do
  # Keep valid folder names that still have a model.py (a deleted folder is skipped).
  if [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,38}$ ]] && [ -f "submissions/$name/model.py" ]; then
    count=$((count + 1))
    if [ "$count" -gt 10 ]; then
      echo "::error::More than 10 submission folders changed. Split them into smaller pull requests." >&2
      exit 1
    fi
    echo "$name"
  elif [ -d "submissions/$name" ]; then
    echo "::warning::Skipping submissions/$name: names use letters, digits, . _ - (up to 39), and need a model.py." >&2
  fi
done
