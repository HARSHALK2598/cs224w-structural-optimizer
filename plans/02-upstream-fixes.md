# Plan 02: benchmark the two side fixes, then propose them upstream

Two fixes found while building the baseline (Plan 00) could help everyone who uses
PyTorch Frame or RelBench. Before proposing them, we check them on a GPU and on more
datasets than `rel-f1`, and collect the numbers maintainers would ask for.

## 1. What we know already

**NaN gradients in PyTorch Frame** (issue #6, our `--nan-safe-numerical`):
- `LinearEncoder` and `LinearBucketEncoder` compute `x · w + b` and then replace NaN
  outputs with 0. In the backward pass a missing value gives `NaN × 0 = NaN` in the
  column's weight gradient, so after one Adam step the whole column encodes to 0.
- PyTorch Frame has a setting that avoids this: `na_strategy`, which fills missing values
  *before* encoding (`MEAN`, `ZEROS`, …). It is a fixed fill-in, not a learned value.
  The default is `None` (no fill-in), and RelBench uses the default.
- Checked: with `na_strategy=None` both encoders get NaN gradients; with `MEAN` or `ZEROS`
  neither does. PyTorch Frame's current `master` has the same code as 0.3.0, so the bug
  is still there.
- The three behaviours differ:

| Option | A missing value becomes | Gradients |
|---|---|---|
| default (`None`), RelBench today | intended: the zero vector; in practice the column dies | broken |
| `na_strategy=MEAN` (existing) | the column mean, i.e. the same vector as an average value | fine |
| our fix (mask before the multiply) | the zero vector, as intended | fine |

**Encoder deduplication** (PR #5, our `--dedup-encoder`): with RelBench's disjoint
temporal sampling, a row appears once per seed that reaches it, about 3× on `driver-dnf`.
Encoding each distinct row once per batch gave 2.5× faster CPU training with the same
accuracy on `driver-top3`. The only behaviour change: copies of a row share one dropout
mask in training.

## 2. Benchmark 1: the NaN fix (for PyTorch Frame)

1. **Census, no training.** For each dataset, count the numerical columns with missing
   values and their missing rate, so we know which tasks the bug can affect.
2. **Accuracy and dead columns.** On the tasks where the census finds such columns, three
   arms × 5 seeds: default, `na_strategy=MEAN`, our fix. Record the task metric
   (validation and test) and the number of columns whose weights became NaN.
3. **Upstream deliverable:** an issue with the 25-line reproduction and the table, then a
   PR that masks before the multiply in the affected encoders, with tests. We first check
   every numerical encoder, not just these two.

## 3. Benchmark 2: deduplication (for RelBench)

1. **Speed and memory, on GPU (and CPU).** Per task: copies per row in a batch, time per
   epoch with dedup off and on (encoder, backward, total), peak GPU memory.
2. **Accuracy.** Dedup off against on, 5 seeds each, on the same GPU.
3. **Upstream deliverable:** an issue or PR to RelBench adding dedup as an option to
   `HeteroEncoder` (the place where sampled rows are encoded), with the tables, and the
   dropout caveat stated.

## 4. Datasets and tasks

Entity tasks only (the code path we have); RelBench's GNN example settings, 20 epochs.

| Dataset | Download | Tasks |
|---|---|---|
| rel-f1 | 1.4 MB (have it) | driver-dnf, driver-top3, driver-position |
| rel-event | 119 MB | user-repeat, user-ignore, user-attendance |
| rel-trial | 683 MB | study-outcome, study-adverse, site-success |
| rel-avito (optional) | 450 MB | user-visits, user-clicks, ad-ctr |

`rel-stack` (1.1 GB) and `rel-amazon` (7.7 GB) are left out. Downloads and runs happen on
Colab (T4), using the existing sweep tool and notebook; nothing large is downloaded here.

## 5. Code

- `experiments/benchmarks/nan_census.py`: missing-value census per dataset (step 2.1).
- `--na-strategy-mean` in `train.py`: the existing workaround as a third arm.
- `experiments/profiling/profile_step.py`: already measures time per stage; add peak GPU
  memory and run it per task.
- Sweep configs: one per benchmark; the notebook runs them unchanged.

## 6. Before proposing upstream

- Search both repositories' issues and PRs for earlier reports (from this session the
  GitHub search is blocked, so you would do this).
- Each upstream text is written here as a markdown draft for your review. You post it,
  under your name.

## 7. Questions for you

1. Datasets: rel-f1, rel-event and rel-trial, plus rel-avito if time allows. OK?
2. Order: the NaN benchmark first (one fix, clear bug), then dedup?
3. Compute: everything on your Colab T4. Fine, or do you have more GPU time?
