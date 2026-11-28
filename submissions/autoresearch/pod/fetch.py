"""Download the pod's finished runs into runs/:  python pod/fetch.py <pod id>"""

import re
import sys
import urllib.request
from pathlib import Path


def get(url: str) -> str:
    # Runpod's proxy refuses Python's default User-Agent, so send a plain one.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(request, timeout=60).read().decode()


base = f"https://{sys.argv[1]}-8000.proxy.runpod.net"
runs = Path(__file__).resolve().parents[1] / "runs"
runs.mkdir(exist_ok=True)
listing = get(f"{base}/runs/")
for name in sorted(set(re.findall(r'href="([\w.-]+\.jsonl)"', listing))):
    text = get(f"{base}/runs/{name}")
    (runs / name).write_text(text)
    print(f"{name}: {text.count(chr(10))} runs")
print(get(f"{base}/status.txt").strip())
