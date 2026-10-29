# Plan 00: Baseline

Before we can show that the structural optimizer helps, we need a trustworthy
number for the model it improves on. This plan trains the paper's **Base** model, a
standard relational GraphSAGE, on three `rel-f1` tasks, and compares the results with
the Base numbers reported in the paper.

## 1. What we run

We run RelBench's official training script for this kind of task,
[`examples/gnn_entity.py`](https://github.com/snap-stanford/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/examples/gnn_entity.py)
(linked at the exact version we use), **unmodified**. Its results are our Base
numbers, so no code of ours sits between the benchmark and the reported baseline.

The database is turned into a graph: one node type per table, one edge type per
foreign key. For each prediction, the model samples a 2-hop neighbourhood that
contains only rows from before the prediction time. It encodes each row's columns,
runs two GraphSAGE layers, and predicts from the target row's final embedding.

| | |
|---|---|
| Tasks | `driver-top3`, `driver-dnf` (classification, ROC-AUC ↑); `driver-position` (regression, MAE ↓) |
| Sampling | 128 then 64 neighbours per edge type; batches of 512 |
| Model | RelBench's row encoder → 2 GraphSAGE layers (128 channels) → linear head |
| Training | Adam, lr 0.005, 10 epochs; keep the epoch with the best validation score |
| Regression | predictions clipped to the 2nd–98th percentile range of the training labels |
| Runs | 5 seeds per task, reported as mean ± std |

## 2. Our training script

The official script cannot host the structural optimizer, because its model and loss
are fixed. Our own script, `experiments/train.py`, will train the optimizer, so it
must first reproduce the official Base numbers (the fixes are in PR #2). Its Base
runs are a reproduction check here; they become the control for the optimizer
comparison in a later plan (same pipeline, optimizer switched off).

## 3. How we check it

Unit tests in `experiments/tests/` cover the parts most likely to go wrong silently:

- the graph has the expected 9 node types and 26 edge types;
- sampling never includes rows from after the prediction time;
- regression predictions stay within the clipping range;
- a one-batch run works for both task types, and results are written and summarised
  correctly;
- **`train.py` reproduces the official script:** one seed per task falls within the
  official 5-seed mean ± 2 std.

## 4. Steps

Each step is one small commit, reviewed before it is merged:

1. Correctness fixes to `train.py`, with their tests.
2. Text features.
3. Run the official script for 3 tasks × 5 seeds and commit its results: these are
   the Base numbers.
4. Reporting (`summarize.py`), graph and sampling tests, and the check that
   `train.py` reproduces step 3.

## 5. Outcome and next decision

| Task | Metric | Base (official script) | Paper's Base |
|---|---|---|---|
| driver-top3 | ROC-AUC ↑ | — | 81.48 ± 1.06 |
| driver-dnf | ROC-AUC ↑ | — | 74.53 ± 0.27 |
| driver-position | MAE ↓ | — | 3.88 ± 0.02 |

The paper's numbers are from arXiv:2606.08491, Tables 1–3; its data splits match ours.
The paper uses the same architecture, but tunes the learning rate and hidden size on
validation data, while we use the official fixed settings. If our numbers fall
clearly short, a follow-up plan will re-create the paper's tuning before we compare
it against the structural optimizer.
