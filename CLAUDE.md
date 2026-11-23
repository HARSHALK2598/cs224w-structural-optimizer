# Working rules for this project

- **The paper is the specification** (arXiv:2606.08491). `reference/` (the authors' code)
  has no license: read it only to document deviations; never copy or commit it.
- **Library code goes in the PyG fork** (branch `contrib/structural-optimizer`), under
  `torch_geometric/contrib/`, tests under `test/contrib/`. Nothing outside `contrib/` is
  modified except the docs index and `CHANGELOG.md`.
- **PyG style**: `r"""` docstrings with math, `Args:`, shapes, `Example:`; type hints;
  `reset_parameters()`; `__repr__`; flake8 and yapf clean.
- **Experiments**: `experiments/train.py` is the single entry point (`--smoke` for a CPU
  check, full runs on Colab). One CSV row per run in `experiments/results/`, indexed in
  its README. Never fabricate or adjust results.
- **Docs stay short**: no long explanatory files; state results and settings only.
- **Data**: download only what a step needs.
- **Before ending a session**: commit and push both repos; rebuild a fresh container
  with `bash scripts/setup_env.sh --data`.
- Commits are authored as Harshal Kulkarni; the user opens the upstream PR themselves.
