# Learning the Graph for Relational Deep Learning

CS224W course project (Project 3, cutting-edge research): a PR to
`torch_geometric.contrib` implementing the structural optimizer of
*What Makes a Desired Graph for Relational Deep Learning?* (Cheng & Luo,
ICML 2026, arXiv:2606.08491), plus a Medium blog post.

Author: Harshal Kulkarni (`hsk8171@stanford.edu`) · Proposal: [`proposal/proposal.tex`](proposal/proposal.tex)

## Repositories

| Repo | What lives there |
|---|---|
| **This repo** (`HARSHALK2598/cs224w-structural-optimizer`) | proposal, notes, experiment scripts + configs, Colab notebook, results CSVs, blog draft + figures, milestone report |
| **PyG fork** (`HARSHALK2598/pytorch_geometric`, branch `contrib/structural-optimizer`) | the library code, tests, docs and example that become the PR to `pyg-team/pytorch_geometric` |
| `reference/Structural_Optimizer_RDL` (git-ignored, local only) | the authors' research code. **No license: read only, never copied or committed.** |

## Layout

```
proposal/      proposal.tex
notes/         paper_notes.md (equations + hyperparameters), author_code_audit.md
experiments/   train.py, configs/*.yaml, colab_run.ipynb, results/ (one CSV row per run)
blog/          draft.md, figures/
report/        milestone report draft
scripts/       setup_env.sh (rebuild workspace), download_data.py (rel-f1 only)
pytorch_geometric/   <- PyG fork clone (separate git repo, git-ignored here)
```

## Setup

```bash
git clone https://github.com/HARSHALK2598/cs224w-structural-optimizer
cd cs224w-structural-optimizer
bash scripts/setup_env.sh --data     # ~5 min; --data downloads rel-f1 only
source .venv/bin/activate
pytest pytorch_geometric/test/contrib -q
```

Where things run:
- **Cloud dev container (CPU):** library code, unit tests, flake8, one-iteration smoke runs.
- **Google Colab (GPU):** full experiments via `experiments/colab_run.ipynb`, which installs PyG
  from the fork branch and writes results and checkpoints to Google Drive.

## New PyG components (all under `torch_geometric/contrib/`)

| Class | Step | File |
|---|---|---|
| `HardConcreteGate` | A, C | `nn/models/hard_concrete.py` |
| `RelationalFilter` | A | `nn/models/relational_filter.py` |
| `TemporalContinuity` | B | `transforms/temporal_continuity.py` |
| `KNNEdges`, `BehavioralEdges` | B | `nn/models/edge_injection.py` |
| `GatedHeteroConv` | C | `nn/conv/gated_hetero_conv.py` |
| `StructuralOptimizer` | A–D | `nn/models/structural_optimizer.py` |

Two-hop template: existing `torch_geometric.transforms.AddMetaPaths` (no new code).

## Grading checklist (Project 3 rubric, 80 pts listed)

### PR — 50 pts
- [ ] **Correctness (15)**: no errors, nothing existing broken
  - [ ] Test 1: `E‖g‖₀` matches the empirical open rate of `HardConcreteGate`
  - [ ] Test 2: each template adds exactly the expected edges on a hand-made graph
  - [ ] Test 3: `GatedHeteroConv` equals the gated sum (train); gated-off relation contributes nothing (eval)
  - [ ] Test 4: all flags off ⇒ `StructuralOptimizer` matches the plain GNN exactly
  - [ ] Test 5: one epoch on `rel-f1` end to end (`@onlyOnline` / `@withPackage('relbench')`)
  - [ ] Full `test/contrib` and `test/nn` still pass; flake8 / PyG pre-commit clean
- [ ] **Documentation (15)**: every public class/function has a PyG-style docstring (math, `Args:`, shapes, returns, `Example:`); comments on the subtle parts (Hard-Concrete stretch, time-leakage guard, eval-time hard gates)
- [ ] **Ease of integration (10)**
  - [ ] Works with `HeteroData`, `NeighborLoader`, `AddMetaPaths`, `contrib.utils.from_relbench`
  - [ ] Works with any convolution (tested with `SAGEConv`, `GATConv`, `GraphConv`), CPU and CUDA
  - [ ] Pure-torch KNN fallback when `torch_cluster` is absent
  - [ ] Listed in the contrib docs index; `CHANGELOG.md` entry
- [ ] **Ease of use (10)**: 3-step usage snippet in the docstring; `examples/contrib/structural_optimizer_relbench.py`; every step behind a flag; sensible defaults

### Blog — 25 pts
- [ ] **Motivation & research method (10)**: information overload / semantic fragmentation, rooted in graph ML; method A–D explained with figures
- [ ] **Implementation (10)**: how each class maps to PyG / PyTorch primitives, design choices, the deviations from the authors' code
- [ ] **PyG code snippets (5)**: several, well-chosen, bug-free
- [ ] Figures: architecture diagram, gate heatmaps (which features/tables/templates survive), results table vs paper

### Medium — 5 pts
- [ ] Blog published on Medium, account name included in the submission

## Timeline

| Weeks | Work |
|---|---|
| 0 | Workspace, setup script, API skeletons, spec tests, Colab template |
| 1–2 | A (`HardConcreteGate`, `RelationalFilter`), `TemporalContinuity`, Tests 1–2, *Base* on `rel-f1` |
| 3–4 (milestone) | `KNNEdges`, `BehavioralEdges`, `GatedHeteroConv`, `StructuralOptimizer`, Tests 3–5, API docs, report draft |
| 5–7 | `rel-trial`, 3 models × 4 tasks × 5 seeds on Colab, authors' code on `rel-f1`, PR polish, blog, Medium |
