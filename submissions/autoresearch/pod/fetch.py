"""Download the pod's finished runs into runs/:  python pod/fetch.py <pod id>"""

import re
import sys
import urllib.request
from pathlib import Path

base = f"https://{sys.argv[1]}-8000.proxy.runpod.net"
runs = Path(__file__).resolve().parents[1] / "runs"
runs.mkdir(exist_ok=True)
listing = urllib.request.urlopen(f"{base}/runs/", timeout=30).read().decode()
for name in sorted(set(re.findall(r'href="([\w.-]+\.jsonl)"', listing))):
    text = urllib.request.urlopen(f"{base}/runs/{name}", timeout=60).read().decode()
    (runs / name).write_text(text)
    print(f"{name}: {text.count(chr(10))} runs")
print(urllib.request.urlopen(f"{base}/status.txt", timeout=30).read().decode().strip())
