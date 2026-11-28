#!/bin/bash
# Bring the pod's copy of the branch up to date.
# The repo asks for Python 3.14; the image has 3.12. One line makes data.py work on 3.12:
# it names its own class in a type hint, which 3.14 reads lazily and 3.12 does not.
set -e
cd /root/repo
git checkout -q -- src/eurosat/data.py
git pull -q --ff-only
sed -i '1i from __future__ import annotations' src/eurosat/data.py
git log -1 --format='at %h %s'
