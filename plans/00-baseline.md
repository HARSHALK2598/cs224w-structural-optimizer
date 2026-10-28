# Plan 00: Baseline

To show that the structural optimizer helps, we first need a trustworthy number for
the paper's **Base** model: a standard relational GraphSAGE. We reproduce RelBench's
official script,
[`examples/gnn_entity.py`](https://github.com/snap-stanford/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/examples/gnn_entity.py),
on three `rel-f1` tasks, using its default settings (2 layers, 128 channels, Adam with
lr 0.005, 10 epochs) and 5 seeds per task.

## Approach

Our training script, `experiments/train.py`, will train every model in this project,
so we first align it with the official script. That means dropping columns that leak
the label, clipping regression outputs to the range of the training labels, selecting
epochs on the official metrics, logging MAE, and adding GloVe text features. Unit tests
check the graph's shape, that sampling never includes future rows, and that results are
logged and summarised correctly.

## Steps

Each step is one small, reviewed commit:

1. Correctness fixes, with tests.
2. Graph and sampling tests.
3. Text features.
4. Results summary and end-to-end tests.
5. Run 3 tasks × 5 seeds and commit the results.

## Outcome

| Task | Metric | Ours | Paper's Base |
|---|---|---|---|
| driver-top3 | ROC-AUC ↑ | — | 81.48 ± 1.06 |
| driver-dnf | ROC-AUC ↑ | — | 74.53 ± 0.27 |
| driver-position | MAE ↓ | — | 3.88 ± 0.02 |

The paper tunes the learning rate and hidden size, whereas we use fixed defaults. If our
numbers fall clearly short, a follow-up plan will re-create its tuning.
