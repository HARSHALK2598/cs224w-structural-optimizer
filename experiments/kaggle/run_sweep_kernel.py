# Kaggle script: runs one sweep config of the CS224W structural-optimizer project
# and writes results, logs and checkpoints to /kaggle/working (the notebook's output).
import os, subprocess, sys, shutil
from pathlib import Path

CONFIG = '__CONFIG__'
PROJECT = ('https://github.com/HARSHALK2598/cs224w-structural-optimizer', 'step/report')
PYG = ('https://github.com/HARSHALK2598/pytorch_geometric', 'contrib/structural-optimizer')
OUT = Path('/kaggle/working/results')
ROOT = Path('/tmp/proj')
LOG = open('/kaggle/working/run.log', 'a')


def sh(cmd, cwd=None):
    print('$', cmd, flush=True); LOG.write(f'$ {cmd}\n'); LOG.flush()
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    for line in p.stdout:
        if 'Embedding raw data' in line or 'Warning' in line:
            continue
        print(line, end='', flush=True); LOG.write(line); LOG.flush()
    if p.wait() != 0:
        raise RuntimeError(f'failed ({p.returncode}): {cmd}')


gpu = subprocess.run('nvidia-smi', shell=True, capture_output=True).returncode == 0
build = 'cu126' if gpu else 'cpu'
pip = f'{sys.executable} -m pip install -q'
sh(f'git clone -q --branch {PROJECT[1]} {PROJECT[0]} {ROOT}')
sh(f'git clone -q --branch {PYG[1]} {PYG[0]} {ROOT}/pytorch_geometric')
sh(f'{sys.executable} -m pip uninstall -q -y torchvision torchaudio || true')
sh(f'{pip} torch==2.12.0 --index-url https://download.pytorch.org/whl/{build}')
sh(f'{pip} pyg_lib==0.9.0 -f https://data.pyg.org/whl/torch-2.12.0+{build}.html')
sh(f'{pip} -e {ROOT}/pytorch_geometric')
sh(f'{pip} relbench==3.1.0 pytorch-frame==0.3.0 sentence-transformers==6.3.0 pyyaml pandas')
sh('git log --oneline -1', cwd=ROOT)
sh(f'{sys.executable} -c "import torch; print(torch.__version__, torch.cuda.is_available())"')
OUT.mkdir(parents=True, exist_ok=True)
sh(f'{sys.executable} experiments/sweep.py {CONFIG} --out {OUT} --save-checkpoints', cwd=ROOT)
print('DONE', flush=True)
