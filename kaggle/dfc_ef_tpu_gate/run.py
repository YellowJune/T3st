from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO="https://github.com/YellowJune/T3st.git"
BRANCH="dfc-h200-final"
WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
CHECKOUT=WORK/'T3st_dfc_ef_tpu'
OUT=WORK/'dfc_ef_tpu_results'; OUT.mkdir(parents=True,exist_ok=True)


def run(name,cmd,cwd=None,required=True):
    t=time.time(); p=subprocess.run(cmd,cwd=str(cwd) if cwd else None,text=True,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    row={'name':name,'command':cmd,'returncode':p.returncode,'success':p.returncode==0,
         'wall_seconds':time.time()-t,'stdout_tail':p.stdout[-12000:],'stderr_tail':p.stderr[-12000:]}
    (OUT/f'{name}.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
    print(f"===== {name} rc={p.returncode} {row['wall_seconds']:.1f}s =====")
    print(row['stdout_tail']); print(row['stderr_tail'],file=sys.stderr)
    if required and p.returncode: raise SystemExit(f'{name} failed')
    return row

if CHECKOUT.exists(): shutil.rmtree(CHECKOUT)
run('clone',['git','clone','--depth','1','--branch',BRANCH,REPO,str(CHECKOUT)])
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
(OUT/'source_commit.txt').write_text(commit+'\n')

# Kaggle TPU images ship the TPU-compatible JAX stack. Do not pip-replace JAX,
# because doing so can break libtpu compatibility. Record the environment first.
run('jax_environment',[sys.executable,'-c',
    "import jax,sys; print(sys.version); print(jax.__version__); print(jax.default_backend()); "
    "print(jax.device_count()); [print(d) for d in jax.devices()]"])
run('tpu_cross_substrate',[sys.executable,'tpu_dfc_ef_jax.py',
    '--exact-coordinates','1048579','--exact-steps','24',
    '--timing-sizes','1048576,4194304,16777216,67108864',
    '--timing-steps','20','--stride','8',
    '--output',str(OUT/'tpu_dfc_ef_jax.json')],cwd=CHECKOUT/'remote')
manifest={'schema_version':1,'protocol':'dfc-ef-kaggle-tpu-gate-v1','source_commit':commit,
          'files':sorted(p.name for p in OUT.iterdir())}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
shutil.make_archive(str(WORK/'dfc_ef_tpu_results'),'zip',OUT)
print(json.dumps(manifest,indent=2))
