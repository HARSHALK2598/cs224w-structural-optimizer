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
experiments/colab_*.ipynb  Colab notebooks (GPU runs); colab_dataset.ipynb runs any task
experiments/tests/         tests of the experiment code
experiments/results/       one CSV row per run; index in experiments/results/README.md
proposal/                  proposal.tex
report/                    blog.md (the blog post, draft)
scripts/                   setup_env.sh, download_data.py
```

## Setup

```bash
bash scripts/setup_env.sh --data      # CPU torch, the PyG fork (editable), RelBench, rel-f1
source .venv/bin/activate
pytest pytorch_geometric/test/contrib experiments/tests -q
python experiments/train.py --task driver-top3 --smoke
```

## Components (`torch_geometric/contrib/`)

| Class | Step | File | Status |
|---|---|---|---|
| `HardConcreteGate` | A, C | `nn/models/hard_concrete.py` | merged |
| `RelationalFilter` | A | `nn/models/relational_filter.py` | merged |
| `KNNEdges`, `BehavioralEdges`, `TwoHopEdges` | B | `nn/models/edge_injection.py` | in review |
| `TemporalContinuity` | B | `transforms/temporal_continuity.py` | in review |
| `GatedHeteroConv` | C | `nn/conv/gated_hetero_conv.py` | in review |
| `StructuralOptimizer` | A–D | `nn/models/structural_optimizer.py` | planned |

## Rubric checklist

PR (50)
- [ ] Correctness: Test 1 (`E‖g‖₀` vs open rate) ✓; Test 2 (template edges, in review); Test 3 (gated sum, in review); Test 4 (all flags off = plain GNN); Test 5 (one rel-f1 epoch); `test/contrib` and flake8 clean
- [ ] Documentation: PyG docstrings (math, `Args:`, shapes, `Example:`) for every public class
- [ ] Integration: `HeteroData`, `NeighborLoader`, any convolution, CPU and CUDA; docs index; `CHANGELOG.md`
- [ ] Ease of use: usage snippet; `examples/contrib/structural_optimizer_relbench.py`; every step behind a flag

Blog (25) + Medium (5)
- [ ] Motivation and method; implementation and deviations from the authors' code; code snippets; figures; published on Medium
