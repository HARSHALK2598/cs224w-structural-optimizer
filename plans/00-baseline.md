# Plan 00 — Baseline: RelBench's relational GraphSAGE on `rel-f1`

**Status:** draft, awaiting review
**Repo:** project repo only (`experiments/`). No change to the PyG fork.

## Goal

Measure the paper's **Base** model on `rel-f1`, reproducing exactly RelBench's official
training script for entity tasks, [`examples/gnn_entity.py`](https://github.com/snap-stanford/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/examples/gnn_entity.py), which is the
**source of truth** for this plan. The link is pinned to the version we compared
against, so it stays valid if RelBench changes the file later.
Later models are compared against these numbers.

**Scope:** `driver-top3`, `driver-dnf` (ROC-AUC ↑) and `driver-position` (MAE ↓,
NMAE also logged). 5 seeds (0–4), CPU in the dev container.

## Settings (from the official example)

| | Setting |
|---|---|
| Graph | `make_pkey_fkey_graph`, GloVe text embeddings, `remove_columns=task.hidden_columns()` |
| Sampling | `NeighborLoader`, `time_attr='time'`, `num_neighbors=[128, 64]`, `temporal_strategy='uniform'`, batch 512 |
| Model | `HeteroEncoder` (torch-frame ResNet) + `HeteroTemporalEncoder` → `HeteroGraphSAGE` (2 layers, 128 channels, `aggr='sum'`) → `MLP(num_layers=1)` |
| Loss | `BCEWithLogitsLoss` (binary), `L1Loss` (regression) |
| Optimisation | Adam, lr 0.005, 10 epochs, ≤ 2,000 steps per epoch |
| Selection | best val epoch on ROC-AUC (`>=`) / NMAE (`<=`), then a single test evaluation |
| Regression output | clip predictions to lie between the 2nd and 98th percentiles of the training labels |

## Changes to our training script

`experiments/train.py` is our own training script (written on day 1). It will train
all our models (Base now, the structural optimizer later), so it must first match the
[official example](https://github.com/snap-stanford/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/examples/gnn_entity.py) exactly. These are the differences found:

| # | Gap vs. the [official example](https://github.com/snap-stanford/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/examples/gnn_entity.py) | Fix |
|---|---|---|
| 1 | `remove_columns=task.hidden_columns()` not passed | pass it |
| 2 | Regression predictions not clipped | clip to the 2nd–98th percentile range of the training labels at inference |
| 3 | Tune metric is `task.metrics[0]`; ties keep the earlier epoch | explicit `roc_auc` / `nmae`; `>=` / `<=` |
| 4 | Regression logs only NMAE | also log MAE (val and test) |
| 5 | GloVe embedder fails on missing text | map `None`/NaN to `""` |
| 6 | `sentence-transformers` not installed | add to `setup_env.sh` |
| 7 | No run summary | `experiments/summarize.py`: mean ± std per (task, model) from `results.csv` |

Not ported (off by default in the example): stype JSON cache,
`--include_task_tables`, prediction-file export.

## Tests (`experiments/tests/`)

| Test | Expected |
|---|---|
| `test_config_precedence` | CLI > YAML > defaults |
| `test_graph_shape` | 9 node types, 26 edge types; each `rev_` edge is the transpose of its `f2p_` edge |
| `test_temporal_sampling_no_leakage` | every sampled timed node has time ≤ its seed's time |
| `test_regression_clamp` | `driver-position` val predictions lie within [P2, P98] of the train labels |
| `test_smoke_runs` | one-batch runs of `driver-top3` and `driver-position` succeed; nothing written to `results.csv` |
| `test_results_row_and_summary` | a 1-step run writes one row with the expected columns; `summarize.py` on a hand-made CSV gives the hand-computed mean ± std |

Tests that need `rel-f1` skip themselves if it isn't cached.

## Commits

1. Gaps 1–4, plus `test_config_precedence` and `test_regression_clamp`.
2. `test_graph_shape` and `test_temporal_sampling_no_leakage`.
3. Gaps 5–6 (GloVe).
4. Gap 7 (`summarize.py`), plus `test_results_row_and_summary` and `test_smoke_runs`.
   Time one seed before launching all runs.
5. Run 3 tasks × 5 seeds; commit `results.csv` and the summary table.

## Deliverable

| Task | Metric | Base-RelBench (ours, 5 seeds) | Paper's Base |
|---|---|---|---|
| driver-top3 | ROC-AUC ↑ | — | 81.48 ± 1.06 |
| driver-dnf | ROC-AUC ↑ | — | 74.53 ± 0.27 |
| driver-position | MAE ↓ | — | 3.88 ± 0.02 |

The paper's numbers are from Tables 1–3 of arXiv:2606.08491; its split sizes (Table 7)
match ours. Ours are reported without adjustment.

## Relation to the paper's Base

Same architecture, different training. The paper grid-searches lr ∈ {1e-4, 1e-3, 1e-2}
× hidden ∈ {64, 128, 256} on val and tunes on MAE. Its epochs, batch size and weight
decay are unstated; the authors' code uses 30 / 256 / 5e-4.

**Decision rule:** run Base-RelBench first. If the gap to the paper's Base matters
for our comparison, write a separate plan to re-create the paper's tuned version
(Base-paper).
