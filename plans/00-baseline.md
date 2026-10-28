# Plan 00: Baseline

Before we can show that the structural optimizer helps, we need a trustworthy
number for the model it improves on. This plan trains the paper's **Base** model, a
standard relational GraphSAGE, on three `rel-f1` tasks, and compares the results with
the Base numbers reported in the paper.

## 1. What we run

We reproduce RelBench's official training script for this kind of task,
[`examples/gnn_entity.py`](https://github.com/snap-stanford/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/examples/gnn_entity.py)
(linked at the exact version we compared against). It is our source of truth.

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

## 2. What changes in our code

Our training script, `experiments/train.py`, will train every model in this project,
so it must first match the official script exactly. Comparing the two shows these gaps:

1. **Correctness:** drop columns that RelBench marks as leaking the label; clip
   regression predictions as above; select epochs on the official metrics (ROC-AUC,
   NMAE).
2. **Reporting:** also log raw MAE, which is what the paper reports; add a
   `summarize.py` that turns the per-run results into mean ± std tables.
3. **Text features:** install the GloVe text encoder and handle empty text values.

## 3. How we check it

Unit tests in `experiments/tests/` cover the parts most likely to go wrong silently:

- the graph has the expected 9 node types and 26 edge types;
- sampling never includes rows from after the prediction time;
- regression predictions stay within the clipping range;
- a one-batch run works for both task types, and results are written and summarised
  correctly.

## 4. Steps

Each step is one small commit, reviewed before it is merged:

1. Correctness fixes, with their tests.
2. Graph and sampling tests.
3. Text features.
4. Reporting (`summarize.py`), the end-to-end tests, and timing one run.
5. Run all 3 tasks × 5 seeds and commit the results.

## 5. Outcome and next decision

| Task | Metric | Ours | Paper's Base |
|---|---|---|---|
| driver-top3 | ROC-AUC ↑ | — | 81.48 ± 1.06 |
| driver-dnf | ROC-AUC ↑ | — | 74.53 ± 0.27 |
| driver-position | MAE ↓ | — | 3.88 ± 0.02 |

The paper's numbers are from arXiv:2606.08491, Tables 1–3; its data splits match ours.
The paper uses the same architecture, but tunes the learning rate and hidden size on
validation data, while we use the official fixed settings. If our numbers fall
clearly short, a follow-up plan will re-create the paper's tuning before we compare
it against the structural optimizer.
