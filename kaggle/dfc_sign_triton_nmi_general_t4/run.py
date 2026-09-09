#!/usr/bin/env python3
from __future__ import annotations
import pathlib
import subprocess
import sys
import urllib.request

url = 'https://raw.githubusercontent.com/YellowJune/T3st/dfc-nmi-sign-gpu-20260909/kaggle/dfc_sign_triton_nmi/run.py'
work = pathlib.Path('/kaggle/working')
out = work / 'frozen_dfc_sign_gate.py'
with urllib.request.urlopen(url, timeout=30) as r:
    out.write_bytes(r.read())
# Execute as a real Python process with /kaggle/working as cwd/script directory.
# The frozen benchmark copies triton_dfc_adamw.py there before importing it.
subprocess.run([sys.executable, str(out)], cwd=str(work), check=True)
