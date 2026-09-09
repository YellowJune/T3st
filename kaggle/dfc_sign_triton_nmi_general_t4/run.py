#!/usr/bin/env python3
from __future__ import annotations
import pathlib, runpy, urllib.request

url = 'https://raw.githubusercontent.com/YellowJune/T3st/dfc-nmi-sign-gpu-20260909/kaggle/dfc_sign_triton_nmi/run.py'
out = pathlib.Path('/kaggle/working/frozen_dfc_sign_gate.py')
with urllib.request.urlopen(url, timeout=30) as r:
    out.write_bytes(r.read())
runpy.run_path(str(out), run_name='__main__')
