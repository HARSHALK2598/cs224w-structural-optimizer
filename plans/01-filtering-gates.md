# Plan 01: Filtering gates (step A, without the bottleneck)

The paper's first idea is **filtering**: let the model learn which parts of the database
it should ignore. This plan adds the two learned switches that do this, a switch per
**column** of each table and a switch per **table**, trains them together with the
model, and checks on `driver-top3` that they do not hurt (and ideally help) the Base
model. The variational bottleneck, the other half of step A, is a separate later plan.

## 1. The switch: `HardConcreteGate`

A gate is a number $g \in [0, 1]$ controlled by one learnable logit $\theta$ (paper
Eq. 4, from Louizos et al. 2018):

- **Training:** a random sample that can be exactly 0 or exactly 1, yet still passes
  gradients to $\theta$:
  $g = \min(1, \max(0,\ \sigma(\tfrac{\log u - \log(1-u) + \theta}{t})(\zeta - \gamma) + \gamma)),\ u \sim U(0,1)$.
- **Evaluation:** a fixed 0/1 decision, $g = \mathbb{I}[\sigma(\theta) > \tau_g]$.
- **Cost of keeping it open** (used in the loss): the probability that it is non-zero,
  $\mathbb{E}\|g\|_0 = \sigma(\theta - t \log(-\gamma/\zeta))$, which is differentiable.

| Hyperparameter | Value | Source |
|---|---|---|
| stretch $\gamma$, $\zeta$ | −0.1, 1.1 | Louizos et al.; authors' code |
| temperature $t$ | 2/3, fixed | Louizos et al. (the authors anneal 2.0 → 0.3; possible later) |
| initial logit $\theta_0$ | 2.0 (≈ 0.88 open) | authors' code: gates start open, so training starts close to Base |
| eval threshold $\tau_g$ | 0.5 | paper Eq. 4 leaves it as a hyperparameter |

One sample is drawn per gate **per training step** and shared by all rows in the batch,
so a gate means "this column / table is used in this step".

## 2. Where the gates go in the pipeline

```
batch.tf_dict[T]                         raw columns of table T            [N_T rows, C_T columns]
  └ column encoders (torch-frame)        one 128-d vector per column        x_cols: [N_T, C_T, 128]
      × column gates g_T  (NEW)          g_T ∈ [0,1]^{C_T}, broadcast over rows and channels
  └ ResNet backbone + decoder            one vector per row                 [N_T, 128]
  + time encoding                        unchanged                           h0[T]: [N_T, 128]
      × table gate s_T  (NEW)            s_T ∈ [0,1], one scalar per table
  └ 2 GraphSAGE layers → head            unchanged
```

- **Column gates sit inside the row encoder**, between the per-column encoders and the
  ResNet, because that is the only place where columns still exist separately. A closed
  column contributes a zero vector, exactly like a missing value.
- **The table gate multiplies the whole row vector, after the time encoding.** Applied
  before it, a "closed" table would still send its time encoding to its neighbours.
  When $s_T = 0$, every row of T sends zero vectors into message passing.
- **The target table (`drivers` here) has no table gate:** closing it would remove the
  very rows being predicted. Its columns are still gated.
- `rel-f1` has 45 columns in 9 tables, so this adds **45 column gates + 8 table gates =
  53 learnable logits** (the model has 4.15 M parameters).

## 3. Decision for you: what a "column" gate gates

The paper is ambiguous here. Eq. 3 defines $g_T \in [0,1]^{d_T}$ on "the dimensionality
of the feature vector" $h^{(0)}_v$ and §4.1 says the gates "suppress noisy feature
dimensions", i.e. the 128 numbers of the encoded row. But it calls them
"column-level", and the authors' code gates **actual table columns**.

| | (a) Encoded dimensions (paper's equations) | (b) Table columns (authors' code) |
|---|---|---|
| What is gated | the 128 dimensions of $h^{(0)}$, per table | each of the table's columns |
| Number of gates | 9 × 128 = 1,152 | 45 |
| Interpretable? | no: dimensions have no meaning | yes: "the model dropped `results.rank`" |
| Placement | after the encoder | inside the encoder (§2) |

**Recommendation: (b) as the main method, (a) as one extra arm.** (b) gives the
interpretable result the blog needs ("which columns does the task need?"). The library
module is written once for both: it gates axis 1 of whatever it receives, `[N, C, 128]`
for (b) or `[N, 128]` for (a). The PR and blog will state the ambiguity.

## 4. The loss

$$\mathcal{L} = \mathcal{L}_\text{task} + \lambda \sum_T \sum_{c} \mathbb{E}\|g_{T,c}\|_0 + \lambda_s \sum_{T} \mathbb{E}\|s_T\|_0$$

- $\lambda$ (columns) and $\lambda_s$ (tables) are the new hyperparameters; the paper
  searches both in [1e-4, 1e-2] (Table 8). Larger values close more gates.
- Sums, not averages, as in the paper; with 45 columns the column term is at most 45λ.
- The paper's VIB term $\beta_\text{vib}$ is not added here (later plan).
- **"Worst case = Base":** with $\lambda = \lambda_s = 0$ and gates initialised open, the
  model may still close a gate if that lowers the task loss, but nothing pushes it to.
  A test also checks that with every gate open the model **is exactly** Base.

## 5. What changes, object by object

| Object | Change |
|---|---|
| `HeteroData` / mini-batches | **none**: gates live in the model, not the data |
| `x_cols` inside each table's encoder `[N_T, C_T, 128]` | multiplied by $g_T$ |
| `h0` (`x_dict`) `[N_T, 128]` | multiplied by $s_T$ |
| Model parameters | + 53 logits (`HardConcreteGate.logit`), trained by the same Adam optimiser |
| `model.train()` / `model.eval()` | switch gates between sampled and fixed 0/1 |
| Training loss (`train.train_epoch`) | `+ model.structural_loss()` |
| Training log, per epoch | penalty value; number of open column / table gates |
| `results.csv` | + `open_columns`, `open_tables`, `lambda_col`, `lambda_table` |
| New file per run | `gates_<task>_seed<N>.json`: each gate's open probability at the best epoch |
| `--dedup-encoder` | still works: the same gate values apply to every copy of a row |

## 6. Code and repositories

**PyG fork** (`contrib/structural-optimizer`, the library PR), files already sketched:
- `torch_geometric/contrib/nn/models/hard_concrete.py`: `HardConcreteGate`
  (`forward()`, `l0_penalty()`, eval-mode 0/1).
- `torch_geometric/contrib/nn/models/relational_filter.py`: `RelationalFilter` with
  per-type gates over axis 1 of its input and an optional per-type table gate;
  `penalties()` returns both sums. Not tied to RelBench or PyTorch Frame.

**Project repo** (integration and experiment):
- `experiments/train.py`: `--model filter`, `--gate-target {columns,dims}`,
  `--lambda-col`, `--lambda-table`. A `FilterModel(BaseModel)` runs each table's
  torch-frame encoder in its two halves, with the column gate in between (§2), then
  the table gate; `structural_loss()`; gate statistics in the log and `results.csv`.
- The project imports the library classes from the fork (installed in editable mode).

## 7. Tests

| Test | Checks |
|---|---|
| `HardConcreteGate` shapes and range | samples in [0, 1]; exact 0s and 1s occur in training |
| Expected-L0 formula | matches the empirical open rate over 100k samples (proposal test 1) |
| Evaluation mode | deterministic 0/1; threshold respected |
| Gradients | reach the logits through both the gate and the penalty |
| `RelationalFilter` | gates the right axis for `[N, C, d]` and `[N, d]`; closed table → zeros |
| **All gates open = Base** | with every gate forced to 1, `FilterModel` output equals `BaseModel` with the same weights |
| Training step | a large λ lowers the logits; λ = 0 leaves the penalty out of the loss |
| Target table | has no table gate |

## 8. Experiment on `driver-top3` (decided on validation)

Settings: `train.py`, 20 epochs, `--dedup-encoder`, 5 seeds, about 74 s per run.

| Arm | Gates | λ (columns), λ_s (tables) | Runs |
|---|---|---|---|
| Base | none | – | done: validation 0.787 ± 0.006 (arm C) |
| F0 | (b) columns + tables | 0, 0 | 5 |
| F1 | (b) | 1e-4, 1e-4 | 5 |
| F2 | (b) | 1e-3, 1e-3 | 5 |
| F3 | (b) | 1e-2, 1e-2 | 5 |
| D2 | (a) dims + tables | 1e-3, 1e-3 | 5 |

25 runs ≈ 30 minutes, in background batches of 5.

**Success:** at least one filtered arm is no worse than Base within noise (validation
≥ 0.781, i.e. Base minus one std), preferably better; and the gates close a non-trivial,
sensible set of columns or tables. We report validation for all arms and test only for
the λ chosen on validation.

## 9. Steps (each one PR, reviewed before merging)

1. **PyG fork:** `HardConcreteGate`, with tests.
2. **PyG fork:** `RelationalFilter` (column/dimension gates and table gates), with tests.
3. **Project repo:** `FilterModel`, flags, loss, logging, with the "all open = Base" test.
4. **Project repo:** run the 25 runs, commit results and a short analysis.

## 10. Out of scope

The variational bottleneck (VIB), edge injection and relation gating (steps B–D),
temperature annealing, other tasks.

## 11. Questions for you

1. Gate target: (b) table columns as the main method plus (a) as one arm (recommended),
   or follow the paper's equations, (a) only?
2. Table gate after the time encoding, and no table gate on the target table: OK?
3. Penalty as sums over gates, as in the paper: OK, or averages, so λ means the same
   for tables with few or many columns?
