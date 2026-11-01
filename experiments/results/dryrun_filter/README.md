# Dry runs: making the gates work (Plan 01, before step 4)

Six short runs of `--model filter` on `driver-top3`, to check that training with gates
works before the 20 real runs. They changed two of the plan's settings. This page
explains each run in order: what we saw, what it means, and what we changed.

All runs: seed 0, 20 epochs, `--dedup-encoder`. Each is **one seed**, and runs on this
machine are not exactly repeatable, so the numbers show direction, not final results.
Decisions use validation scores only.

The reference is **Base** (the same model without gates, 5 seeds):
**validation ROC-AUC 0.786 ± 0.006**, test 0.792 ± 0.020.

## Background: what a gate does during training

Each column (and each table) has a gate with a learnable number, its logit θ. During
training, the gate is a random value between 0 and 1, redrawn every step, and the
column's vectors are multiplied by it. A larger θ makes the gate more often exactly 1
(column fully used). In evaluation, the gate is fixed: 1 if θ > 0, else 0 (column
removed). The penalty λ pushes every θ down. A column whose gate the model pushes
back up is one the task needs.

## Run 1: the plan's settings. The gates do not move.

Settings: gates start at θ = 2, trained with the model's learning rate (0.005),
λ = 1e-3.

**Seen:** after 20 epochs every gate is still open, and their open probabilities have
barely changed (0.968–0.975, against 0.973 at the start). Validation 0.766.

**Why:** the optimiser (Adam) moves each number by at most about its learning rate per
step. `driver-top3` has only 1,353 training examples, i.e. 3 steps per epoch, 60 steps
in total. So θ can move at most 0.005 × 60 ≈ 0.3, but it must go from 2 to below 0 to
close a gate. No value of λ can change that.

**Change 1:** the gates get their own learning rate, `--gate-lr` (0.1 in the runs
below). Now θ can move up to 6 in 20 epochs.

## Run 2: gates now close, but the score drops

Settings: as run 1, with gate learning rate 0.1.

**Seen:** the gates close fast: 9 of 45 columns and 5 of 8 tables are open at the best
epoch. But validation drops to 0.751 (test 0.687), and the columns that survive are
mostly names and IDs, while all numeric `results` columns (points, position, grid)
close.

**Question:** is the drop caused by the closed columns, or by something else?

## Run 3: no penalty (λ = 0). The score still drops.

Settings: as run 2, but λ = 0, so nothing pushes the gates down.

**Seen:** every gate stays open, so in evaluation this model is exactly Base. Yet
validation is still low: 0.752.

**Why:** the problem is in training, not in the closed columns. At θ = 2, a gate is:

| During a training step, the gate is... | at θ = 2 | at θ = 5 |
|---|---|---|
| exactly 1 (column fully used) | 60% of steps | 97% of steps |
| between 0 and 1 (column scaled down) | 37% | 3% |
| exactly 0 (column removed) | 3% | 0.1% |

So even "open" gates randomly weaken 40% of the columns in each step, like heavy
dropout on whole columns. The model is trained on noisy inputs but evaluated on clean
ones, and with only 60 steps it does not recover.

## Runs 4 and 5: two ways to remove the noise (λ = 0)

| Run | Change | Validation | Test |
|---|---|---|---|
| 4 | gates start more open: θ = 5 | **0.804** | 0.775 |
| 5 | lower temperature: t = 0.2 (samples closer to 0 or 1) | 0.757 | 0.716 |

**Starting at θ = 5 works:** validation is back at Base level (0.786 ± 0.006). An open
gate is now exactly 1 in 97% of steps.

**Lowering the temperature does not:** it makes each sample closer to 0 or 1, but at
θ = 2 it also makes "exactly 0" more frequent (8% of steps instead of 3%). That removes
columns even more often.

**Change 2:** gates start at θ = 5 (`--gate-init 5`). The temperature stays at 2/3.

## Run 6: both changes, with the penalty (λ = 1e-3)

**Seen:** validation 0.785 (test 0.778), within Base's range, and the gates close a
sensible set. At the best epoch, 21 of 45 columns and 5 of 8 tables are open:

| Closed | Why it makes sense |
|---|---|
| the tables `circuits`, `constructor_results`, `constructor_standings` | they are 3 hops from `drivers`, so a 2-layer model never receives anything from them. Only the penalty acts on their gates. |
| `results`: fastestLap, milliseconds, position, rank | mostly missing values; these are the columns the NaN-gradient bug ([#6](https://github.com/HARSHALK2598/cs224w-structural-optimizer/issues/6)) kills (found on `driver-dnf`), so they carry nothing |
| `results`: laps, statusId | not needed by the task, at least for this seed |

Kept: grid, points and positionOrder in `results`; points, position and wins in
`standings`; position in `qualifying`; and the driver and constructor identity
columns. These are what you would expect to predict a top-3 finish.

**One open point:** the gates were still closing at epoch 20. From θ = 5 at learning
rate 0.1, a gate needs about 50 steps (17 epochs) to reach 0. A gate learning rate of
0.2 should let them settle by about epoch 9 (not yet tested).

## Why we deviate from the paper's values

The paper does not give these three settings. It only tunes the overall learning
rate, over {1e-4, 1e-3, 1e-2}. The plan took them from the authors' code (gates start
at θ = 2 and share the model's learning rate) and from Louizos et al. (temperature
2/3). Those values assume many more training steps than `driver-top3` gives (the
authors use batch 256 and 30 epochs). On our task they fail for two reasons:

1. **Too few steps to move a gate.** With one shared learning rate, θ moves at most
   about 0.3, so no gate can close. → separate gate learning rate.
2. **Too much noise from half-open gates.** Starting at θ = 2 randomly weakens 40% of
   the columns in each step, which costs about 0.03 validation ROC-AUC even when no
   gate closes. → start at θ = 5.

Neither change alters the method (the same gates, the same penalty, the same
evaluation rule). They only make training reach the point the method describes.

## All runs

| Run | Gate lr | Gate init | Temp. | λ | Best epoch | Val | Test | Open columns / tables |
|---|---|---|---|---|---|---|---|---|
| Base (5 seeds) | – | – | – | – | 14–19 | 0.786 ± 0.006 | 0.792 ± 0.020 | – |
| 1 | 0.005 | 2 | 2/3 | 1e-3 | 13 | 0.766 | 0.750 | 45 / 8 |
| 2 | 0.1 | 2 | 2/3 | 1e-3 | 9 | 0.751 | 0.687 | 9 / 5 |
| 3 | 0.1 | 2 | 2/3 | 0 | 8 | 0.752 | 0.700 | 45 / 8 |
| 4 | 0.1 | 5 | 2/3 | 0 | 18 | 0.804 | 0.775 | 45 / 8 |
| 5 | 0.1 | 2 | 0.2 | 0 | 13 | 0.757 | 0.716 | 45 / 8 |
| 6 | 0.1 | 5 | 2/3 | 1e-3 | 19 | 0.785 | 0.778 | 21 / 5 |

Files: `results.csv` (one row per run), `logs/` (per-epoch scores and open gates),
`gates/` (every gate's state at the best epoch, by column name).
