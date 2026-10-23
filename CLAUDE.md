# Working rules for this project

- **The paper is the specification.** Implement from the paper's equations (see
  `notes/paper_notes.md`), not from the authors' code. `reference/` has no license:
  read it to understand behaviour and to document deviations, but never copy code from
  it and never commit it.
- **Library code goes in the PyG fork** (`pytorch_geometric/`, branch
  `contrib/structural-optimizer`), under `torch_geometric/contrib/` with tests under
  `test/contrib/`. Nothing outside `contrib/` is modified, except the docs index and
  `CHANGELOG.md`.
- **Follow PyG style**: PyG-format docstrings (`r"""` with math, `Args:`, shapes,
  `Example:`), type hints, `reset_parameters()`, `__repr__`, flake8 clean. Match the
  surrounding code in `torch_geometric/nn/conv/hetero_conv.py` and `contrib/`.
- **Every PR item maps to the rubric checklist in `README.md`**; tick items there as
  they are done.
- **Experiments**: `experiments/train.py` is the single entry point; the same script
  runs a CPU smoke test here (`--smoke`) and full GPU runs on Colab. One CSV row per
  (task, model, seed) in `experiments/results/`. Never fabricate or adjust results.
- **Data**: only download what a step needs (currently `rel-f1`).
- **Before ending a session**: commit and push both repos; the container is ephemeral.
  Rebuild a fresh container with `bash scripts/setup_env.sh --data`.
- Commits are authored as Harshal Kulkarni; the user opens the upstream PR themselves.
