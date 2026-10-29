#!/usr/bin/env bash
# Rebuild the development workspace from scratch (idempotent).
#
# Usage (from the project repo root):  bash scripts/setup_env.sh [--data]
#
#   1. clones the PyG fork (branch contrib/structural-optimizer) into ./pytorch_geometric
#   2. clones the authors' reference code into ./reference (read-only, git-ignored)
#   3. creates ./.venv with CPU torch, PyG (editable), RelBench, test tools
#   4. with --data: downloads only the rel-f1 database (via scripts/download_data.py)
#
# Network hosts needed: github.com, pypi.org, files.pythonhosted.org,
# download.pytorch.org, download-r2.pytorch.org, data.pyg.org, huggingface.co, *.hf.co
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYG_FORK="https://github.com/HARSHALK2598/pytorch_geometric"
PYG_UPSTREAM="https://github.com/pyg-team/pytorch_geometric.git"
PYG_BRANCH="contrib/structural-optimizer"
AUTHORS_REPO="https://github.com/cy623/Structural_Optimizer_RDL.git"

# 1. PyG fork ---------------------------------------------------------------
if [ ! -d pytorch_geometric/.git ]; then
  git clone "$PYG_FORK" pytorch_geometric
  git -C pytorch_geometric remote add upstream "$PYG_UPSTREAM"
fi
if git -C pytorch_geometric ls-remote --exit-code --heads origin "$PYG_BRANCH" >/dev/null 2>&1; then
  git -C pytorch_geometric fetch -q origin "$PYG_BRANCH"
  git -C pytorch_geometric checkout -q -B "$PYG_BRANCH" "origin/$PYG_BRANCH"
else
  git -C pytorch_geometric checkout -q -B "$PYG_BRANCH"
fi

# 2. Authors' reference code (no license: read only, never commit or copy) --
[ -d reference/Structural_Optimizer_RDL ] || \
  git clone -q "$AUTHORS_REPO" reference/Structural_Optimizer_RDL

# 3. Python environment -----------------------------------------------------
command -v uv >/dev/null || pip install -q uv
[ -d .venv ] || uv venv -q .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
# torch is pinned to the newest version with pyg-lib wheels (needed for temporal
# neighbour sampling); keep TORCH_VERSION in sync with experiments/colab_run.ipynb.
TORCH_VERSION="2.12.0"
uv pip install -q "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
uv pip install -q pyg_lib -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
uv pip install -q -e "./pytorch_geometric[test]" \
  relbench pytorch-frame pytest flake8 matplotlib pandas nbformat \
  nbconvert ipykernel networkx sentence-transformers
python -c "import torch, torch_geometric, relbench; \
print('torch', torch.__version__, '| pyg', torch_geometric.__version__, \
'| relbench', relbench.__version__)"

# 4. Data (optional) --------------------------------------------------------
if [ "${1:-}" = "--data" ]; then
  python scripts/download_data.py --dataset rel-f1
fi

echo "Done. Activate with: source .venv/bin/activate"
