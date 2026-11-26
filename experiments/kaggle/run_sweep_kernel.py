# Kaggle notebook template (filled in by launch.py): runs one sweep config of
# the CS224W structural-optimizer project; results, logs and checkpoints go to
# /kaggle/working, the notebook's output.
import os
import subprocess
import sys
from pathlib import Path

CONFIG = '__CONFIG__'
PROJECT = ('https://github.com/HARSHALK2598/cs224w-structural-optimizer',
           '__PROJECT_BRANCH__')
PYG = ('https://github.com/HARSHALK2598/pytorch_geometric',
       '__PYG_BRANCH__')
OUT = Path('/kaggle/working/results')
ROOT = Path('/tmp/proj')
LOG = open('/kaggle/working/run.log', 'a')
SKIP = ('Embedding raw data', 'Warning')


def sh(cmd, cwd=None):
    print('$', cmd, flush=True)
    LOG.write(f'$ {cmd}\n')
    LOG.flush()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    for line in proc.stdout:
        if any(s in line for s in SKIP):
            continue
        print(line, end='', flush=True)
        LOG.write(line)
        LOG.flush()
    if proc.wait() != 0:
        raise RuntimeError(f'failed ({proc.returncode}): {cmd}')


gpu = subprocess.run('nvidia-smi', shell=True,
                     capture_output=True).returncode == 0
build = 'cu126' if gpu else 'cpu'
pip = f'{sys.executable} -m pip install -q'
sh(f'git clone -q --branch {PROJECT[1]} {PROJECT[0]} {ROOT}')
sh(f'git clone -q --branch {PYG[1]} {PYG[0]} {ROOT}/pytorch_geometric')
sh(f'{sys.executable} -m pip uninstall -q -y torchvision torchaudio || true')
sh(f'{pip} torch==2.12.0 --index-url https://download.pytorch.org/whl/{build}')
sh(f'{pip} pyg_lib==0.9.0 '
   f'-f https://data.pyg.org/whl/torch-2.12.0+{build}.html')
sh(f'{pip} -e {ROOT}/pytorch_geometric')
sh(f'{pip} relbench==3.1.0 pytorch-frame==0.3.0 '
   f'sentence-transformers==6.3.0 pyyaml pandas')
sh('git log --oneline -1', cwd=ROOT)
sh('git log --oneline -1', cwd=ROOT / 'pytorch_geometric')
OUT.mkdir(parents=True, exist_ok=True)
sh(f'{sys.executable} experiments/sweep.py {CONFIG} --out {OUT} '
   f'--save-checkpoints', cwd=ROOT)
print('DONE', flush=True)
