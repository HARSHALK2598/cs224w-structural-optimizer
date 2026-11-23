# Learning the Graph for Relational Deep Learning

CS224W project: a PR to `torch_geometric.contrib` implementing the structural optimizer of
*What Makes a Desired Graph for Relational Deep Learning?* (Cheng & Luo, ICML 2026,
arXiv:2606.08491), plus a blog post. Author: Harshal Kulkarni.

| Repository | Contents |
|---|---|
| this one | experiments, configs, results, proposal |
| [`HARSHALK2598/pytorch_geometric`](https://github.com/HARSHALK2598/pytorch_geometric/tree/contrib/structural-optimizer), branch `contrib/structural-optimizer` | the library code and its tests |

## Layout

```
experiments/train.py       single entry point (Base, --model filter, --inject ...)
experiments/sweep.py       resumable grid runner (YAML configs in experiments/sweeps/)
experiments/colab_*.ipynb  Colab notebooks (GPU runs)
experiments/tests/         tests of the experiment code
experiments/results/       one CSV row per run; index in experiments/results/README.md
proposal/                  proposal.tex
scripts/                   setup_env.sh, download_data.py
```

## Setup

```bash
bash scripts/setup_env.sh --data      # CPU torch, the PyG fork (editable), RelBench, rel-f1
source .venv/bin/activate
pytest pytorch_geometric/test/contrib experiments/tests -q
python experiments/train.py --task driver-top3 --smoke
```

## Components (`torch_geometric/contrib/`, all merged)

| Class | Step | File |
|---|---|---|
| `HardConcreteGate` | A, C | `nn/models/hard_concrete.py` |
| `RelationalFilter` | A | `nn/models/relational_filter.py` |
| `VariationalBottleneck` | A | `nn/models/variational_bottleneck.py` |
| `KNNEdges`, `BehavioralEdges`, `TwoHopEdges` | B | `nn/models/edge_injection.py` |
| `TemporalContinuity` | B | `transforms/temporal_continuity.py` |
| `GatedHeteroConv` | C | `nn/conv/gated_hetero_conv.py` |
| `StructuralOptimizer` | A–D | `nn/models/structural_optimizer.py` |

Example: `examples/contrib/structural_optimizer_relbench.py`.

## Rubric checklist

PR (50)
- [x] Correctness: Tests 1–5; `test/contrib` (87 pass) and flake8 clean
- [x] Documentation: PyG docstrings (math, `Args:`, shapes, `Example:`) for every public class
- [ ] Integration: `HeteroData`, `NeighborLoader` ✓; docs index ✓; `CHANGELOG.md` ✓; other convolutions (GAT, GraphConv) and CUDA not tested yet
- [x] Ease of use: usage snippet; `examples/contrib/structural_optimizer_relbench.py`; every step behind a flag

Blog (25) + Medium (5)
- [ ] Motivation and method; implementation and deviations from the authors' code; code snippets; figures; published on Medium
