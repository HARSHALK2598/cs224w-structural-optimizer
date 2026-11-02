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
| temperature $t$ | 2/3, fixed | Louizos et al. (the authors' code sets 2.0; its annealing to 0.3 is never called, so 2.0 throughout) |
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

### 2.1 One real row, from DataFrame to h⁽⁰⁾

The first row of `results` (the 1950 British Grand Prix, a driver who started 21st and
finished 11th), followed through each stage of the Base model, without gates.

**Stage 0: DataFrame row** (`db.table_dict['results'].df.iloc[0]`):

| resultId | raceId | driverId | constructorId | number | grid | position | positionOrder | points | laps | milliseconds | fastestLap | rank | statusId | date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 660 | 152 | 18 | 21 | 11 | 11 | 0 | 64 | NaN | NaN | NaN | 16 | 1950-05-13 |

**Stage 1: TensorFrame row** (`data['results'].tf`; materialisation, nothing learned).
The primary key becomes the node id and the 3 foreign keys become edges (both leave
the table), so 11 feature columns remain, grouped by stype:

| stype | columns | stored values for row 0 |
|---|---|---|
| numerical, `[N, 10]` | fastestLap, grid, laps, milliseconds, number, points, position, positionOrder, rank, statusId | `[NaN, 21, 64, NaN, 18, 0, 11, 11, NaN, 16]` |
| timestamp, `[N, 1, 7]` | date | `[1950, 4, 12, 5, 0, 0, 0]` (year, month−1, day−1, weekday, h, m, s) |

Other tables also have categorical columns (an integer index per value) and text
columns (a 300-d GloVe vector each, stored, not learned).

**Stage 2: column encoders** (learned; one small encoder per column $`c`$). Each value
$`x_{v,c}`$ (row $`v`$, column $`c`$) becomes one 128-d vector $`e_{v,c}`$. The formula
depends on the column's stype.

*Numerical* (learned $`w_c, b_c \in \mathbb{R}^{128}`$; $`\mu_c, \sigma_c`$ are the
column's training mean and std):

```math
e_{v,c} = \frac{x_{v,c} - \mu_c}{\sigma_c} \, w_c + b_c
```

*Categorical* (one learned embedding row per category):

```math
e_{v,c} = E_c[x_{v,c}]
```

*Text* (the stored 300-d GloVe vector, then a learned linear map
$`A_c \in \mathbb{R}^{128 \times 300}`$):

```math
e_{v,c} = A_c \, \mathrm{GloVe}(x_{v,c}) + b_c
```

*Timestamp* (each of the 7 parts gets a positional encoding, then a learned linear map):

```math
e_{v,c} = \sum_{k=1}^{7} W_{c,k} \, \mathrm{PE}(x_{v,c,k}) + b_c
```

*Missing value* (NaN, in a numerical, categorical or timestamp column):

```math
e_{v,c} = \mathbf{0}
```

(A missing text value is embedded as the empty string, so it is not zero.)

**Row 0, worked through.** With the training statistics:

```math
\text{grid: } \frac{21 - 11.22}{7.48} = 1.31 \quad\Rightarrow\quad e_{v,\mathrm{grid}} = 1.31 \, w_{\mathrm{grid}} + b_{\mathrm{grid}}
```

```math
\text{points: } \frac{0 - 1.09}{2.34} = -0.47 \quad\Rightarrow\quad e_{v,\mathrm{points}} = -0.47 \, w_{\mathrm{points}} + b_{\mathrm{points}}
```

milliseconds, fastestLap and rank are NaN, so their three vectors are $`\mathbf{0}`$.

Stacking the 11 column vectors gives the row's matrix (the batch is `[N, 11, 128]`):

```math
X_v = \begin{bmatrix} e_{v,1} \\ \vdots \\ e_{v,11} \end{bmatrix} \in \mathbb{R}^{11 \times 128}
```

**Stage 3: flatten and the first ResNet layer.** $`X_v`$ is flattened into one vector
of length $`11 \times 128 = 1408`$:

```math
\mathrm{vec}(X_v) \in \mathbb{R}^{1408}
```

The first block's linear layer has a weight matrix
$`W^{(0)} \in \mathbb{R}^{128 \times 1408}`$. Its 1408 columns split into 11 slices
$`W^{(0)}_c \in \mathbb{R}^{128 \times 128}`$, one per table column, so the layer is a
sum of one term per column:

```math
W^{(0)} \, \mathrm{vec}(X_v) = \sum_{c=1}^{11} W^{(0)}_c \, e_{v,c}
```

This is the first point where columns are mixed. After it, no part of the vector
belongs to one column.

**Stage 4: rest of the ResNet, then time.** Three more residual blocks and the decoder
give one vector per row:

```math
r_v \in \mathbb{R}^{128}
```

The time encoder adds how old the row is, in days, relative to the seed's prediction
time:

```math
\Delta t_v = \frac{t_{\mathrm{seed}} - t_v}{86400}
```

```math
h^{(0)}_v = r_v + \mathrm{Lin}_T\big(\mathrm{PE}(\Delta t_v)\big)
```

**Stage 5: message passing** (one GraphSAGE layer, simplified). For each relation
$`R`$, node $`v`$ sums its neighbours' vectors:

```math
m_{v,R} = \sum_{u \in \mathcal{N}_R(v)} h^{(0)}_u
```

Each relation has its own weights, and the relations are added up:

```math
z_v = \sum_{R} \Big( W^{\mathrm{nbr}}_R \, m_{v,R} + W^{\mathrm{self}}_R \, h^{(0)}_v \Big)
```

Then LayerNorm and ReLU:

```math
h^{(1)}_v = \mathrm{ReLU}\big(\mathrm{LN}(z_v)\big)
```

### 2.2 The two gates in this notation

**Column gate**, between stages 2 and 3, one number per column of table $`T`$:

```math
\tilde e_{v,c} = g_{T,c} \, e_{v,c}
```

Put into stage 3, each column's term is scaled by its gate:

```math
W^{(0)} \, \mathrm{vec}(\tilde X_v) = \sum_{c} g_{T,c} \, W^{(0)}_c \, e_{v,c}
```

So $`g_{T,c} = 0`$ deletes exactly column $`c`$'s term, for every row of $`T`$.

**Table gate**, after stage 4, one number per table:

```math
\tilde h^{(0)}_u = s_T \, h^{(0)}_u
```

Put into stage 5, the messages from table $`T`$ are scaled by its gate:

```math
\sum_{u \in \mathcal{N}_R(v)} \tilde h^{(0)}_u = s_T \sum_{u \in \mathcal{N}_R(v)} h^{(0)}_u \qquad (R \text{ a relation from } T)
```

So $`s_T = 0`$ deletes exactly table $`T`$'s first-layer messages. Both gates multiply,
so the loss has a gradient with respect to them, and the logits $`\theta`$ are trained.

### 2.3 Why these places, and not others

**Column gate: between the column encoders and the ResNet.**

- *Not on the raw values (stages 0–1).* Multiplying a value by 0 does not remove it, it
  changes it. For row 0, grid = 0 would mean "started from the pit lane", and the
  encoder would still output a non-zero vector:

  ```math
  \frac{0 - 11.22}{7.48} \, w_{\mathrm{grid}} + b_{\mathrm{grid}} = -1.50 \, w_{\mathrm{grid}} + b_{\mathrm{grid}} \neq \mathbf{0}
  ```

  A categorical index 0 is a real category. The only "absent" value is NaN, and
  switching a value to NaN is a hard on/off choice, which gives no gradient.
- *Not after the first ResNet layer (stage 3 onwards).* The sum above has already
  mixed all 11 columns, so there is nothing per-column left to switch off. Gating the
  128 numbers of $`r_v`$ or $`h^{(0)}_v`$ is option (a) in §3: it gates mixtures of
  columns.
- *Between stages 2 and 3,* a closed gate gives $`\tilde e_{v,c} = \mathbf{0}`$, the
  same vector a missing numerical, categorical or timestamp value gets. The bias
  $`b_c`$ is removed too. A closed column then looks to the model exactly like a column
  that is always empty, which the model already handles.

**Table gate: on $`h^{(0)}`$, after the time encoding.**

- *Not before the time encoding* (on $`r_v`$). Then the row vector would be

  ```math
  h^{(0)}_u = s_T \, r_u + \mathrm{Lin}_T\big(\mathrm{PE}(\Delta t_u)\big)
  ```

  and with $`s_T = 0`$ each neighbour still receives the sum of the time terms: how
  many rows of $`T`$ it has and how recent they are. The table would not be closed.
- *On $`h^{(0)}`$*, the paper's Eq. 5 position: $`\tilde h_v = s_T \, \hat h_v`$,
  applied to the column-filtered output. We read $`\hat h_v`$ as including the time
  encoding, for the reason above.
- **What a closed table still does:** it stays in the graph as a path. In layer 1 its
  own nodes still receive messages from their neighbours, and in layer 2 they pass
  those on. For example, with `results` closed, `drivers` still learns about `races`
  through `results` rows, but nothing from the `results` columns or dates. Removing
  the table as a path too would mean gating its messages in every layer, which is
  closer to relation gating (step C). Question 2 in §11.
- **No gate on the target table** (`drivers`): with $`s = 0`$ the seed's own vector
  $`h^{(0)}_v`$ would be zero. Its columns are still gated.

`rel-f1` has 45 columns in 9 tables, so this adds **45 column gates + 8 table gates =
53 learnable logits** (the Base model has 5.08 M parameters; see `notes/notation.md`).

## 3. What a "column" gate gates (decided: table columns)

The paper is ambiguous here. Eq. 3 defines $g_T \in [0,1]^{d_T}$ on "the dimensionality
of the feature vector" $h^{(0)}_v$ and §4.1 says the gates "suppress noisy feature
dimensions", i.e. the 128 numbers of the encoded row. But it calls them
"column-level".

*Correction (after reading the authors' code in full):* an earlier version of this plan
said the authors' code gates actual table columns. It does not. It creates one logit per
column, but then repeats that vector across the 128 channels of $h^{(0)}$ (channel $k$
uses gate $k \bmod C_T$), after the time encoding. Closing "column $c$" zeros channels
$c, c + C_T, c + 2C_T, \dots$, which mix all columns. That is neither (a) nor (b), and
in their evaluation the column gates are not applied at all (§12).

| | (a) Encoded dimensions (paper's equations) | (b) Table columns (our choice) |
|---|---|---|
| What is gated | the 128 dimensions of $h^{(0)}$, per table | each of the table's columns |
| Number of gates | 9 × 128 = 1,152 | 45 |
| Interpretable? | no: dimensions have no meaning | yes: "the model dropped `results.rank`" |
| Placement | after the encoder | inside the encoder (§2) |

**Decision: (b) only, for now.** It gives the interpretable result the blog needs
("which columns does the task need?"). The library module gates axis 1 of whatever it
receives, so (a) stays possible later without changing it. The PR and blog will state
the ambiguity.

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
- `experiments/train.py`: `--model filter`,
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

*Revised 2026-11-02, after the dry runs (`experiments/results/dryrun_filter/README.md`)
and a review of the selection risks. Replaces the earlier F0–F3 design.*

**Fixed before tuning, as chosen in the dry runs** (not tuned further): gate start
θ = 5, gate learning rate 0.1, temperature 2/3. The paper gives no values for these.

**Same for every run:** 30 epochs (gates were still closing at epoch 20),
`--dedup-encoder`, and `--nan-safe-numerical`, so that closing the columns broken by the
NaN bug (#6) is not mistaken for learned relevance.

| Stage | What | Settings × seeds | Runs |
|---|---|---|---|
| A | Base on the paper's grid: learning rate {1e-4, 1e-3, 1e-2} × hidden size {64, 128, 256} | 9 × 3 (seeds 0–2) | 27 |
| B | Filter at A's learning rate and hidden size: λ {1e-4, 1e-3, 1e-2} × λ_s {1e-4, 1e-3, 1e-2} | 9 × 3 (seeds 0–2) | 27 |
| C | A's winner and B's winner, 5 fresh seeds (10–14) | 2 × 5 | 10 |

**Rules:**
- Each stage picks the setting with the best **mean validation** ROC-AUC over its seeds.
- Base and the filter get the same tuning budget: 9 settings each.
- Test is computed for every run but **looked at once**, for the stage C winners.
- The report states how many configurations were tried in total: the 7 dry runs, 9 in
  A and 9 in B.
- No dry-run test score (for example run 7's 0.806) influences any choice.

**Reported alongside the scores** (from stage B's 27 runs):
1. Open gates against λ: the check that λ controls sparsity (§13).
2. Validation against λ: the paper's inverted-U claim (Fig. 2).
3. Open gates per epoch, to show whether the gates have settled before the best epoch.
   If they have not, the fallback is a λ warm-up (λ = 0 for the first epochs, then on).

**Targets from the paper** (test ROC-AUC, Table 1): Base 81.48 ± 1.06, FullFilter
84.66 ± 0.61.

## 9. Steps (each one PR, reviewed before merging)

1. **PyG fork:** `HardConcreteGate`, with tests.
2. **PyG fork:** `RelationalFilter` (column/dimension gates and table gates), with tests.
3. **Project repo:** `FilterModel`, flags, loss, logging, with the "all open = Base" test.
4. **Project repo:** run stages A–C (§8), commit results and a short analysis.

## 10. Out of scope

The variational bottleneck (VIB), edge injection and relation gating (steps B–D),
temperature annealing, other tasks.

## 11. Decisions

1. ~~Gate target~~ **Answered: (b) table columns only.**
2. **Answered:** table gate after the time encoding; no table gate on the target table;
   a closed table loses its own content but stays a path (paper Eq. 5; removing paths
   is step C).
3. **Answered:** sums over gates, as in the paper, with a separate $`\lambda_s`$ for
   tables.

## 12. Comparison with the paper and the authors' code

Checked line by line against the paper (arXiv:2606.08491, §3.1, §4.1, Eq. 3–5, 19 and
Appendix C, Table 8) and the authors' repository (`cy623/Structural_Optimizer_RDL`, commit
`68c9b57`, `model.py`, `train.py`, `main.py`; read only, nothing copied).

### Gates and loss

| Item | Paper | Authors' code | Ours |
|---|---|---|---|
| Training sample | Hard-Concrete, Eq. 4 | same formula | same formula |
| Temperature | "τ controls the temperature", no value | 2.0; the annealing hook (`update_anneal`) is never defined, so it stays 2.0 | 2/3 (`--gate-temperature`) |
| Start logit | not given | 2.0 | 2.0 default; dry runs suggest 5 (`--gate-init`) |
| Column gate target | $g_{T} \in [0,1]^{d_T}$ on $h^{(0)}$, "feature dimensions" (Eq. 3) | one logit per column, tiled over the 128 channels of $h^{(0)}$ | one gate per table column, inside the encoder: decision (b) |
| Table gate | $s_T$ for every node type on the filtered $h$ (Eq. 5) | every node type, target included, after the time encoding | every node type except the target, after the time encoding |
| Evaluation gate | hard: $\mathbb{I}[\sigma(\theta) > \tau_g]$ | evaluation path (`apply_pruning=True`): column gates **not applied**; closed tables' edges removed; no soft scaling | hard, $\sigma(\theta) > 0.5$, for both kinds |
| Penalty | $\lambda \sum \mathbb{E}\lVert g\rVert_0 + \lambda_s \sum \mathbb{E}\lVert s\rVert_0$ (Eq. 19) | same, `lambda_w` (columns) and `lambda_g` (tables); gates exist only if `lambda_g > 0` | same, `--lambda-col` and `--lambda-table` |
| Bottleneck (VIB) | yes, $\beta_\text{vib}$ | training only; skipped in evaluation | not yet (later plan) |
| Optimiser | Adam | Adam, one learning rate for all, weight decay 5e-4 (also on the gate logits) | Adam, no weight decay; separate gate learning rate (`--gate-lr`) |

### Training and evaluation setup

| Item | Paper | Authors' code | Ours |
|---|---|---|---|
| Learning rate | grid {1e-4, 1e-3, 1e-2} | 0.005 | 0.005 (RelBench example) |
| Hidden size | grid {64, 128, 256} | 128 | 128 |
| λ, λ_s | searched in [1e-4, 1e-2] | 1e-4 | sweep {1e-4, 1e-3, 1e-2} |
| Selection | grid search on validation | best validation epoch | best validation epoch; settings on validation |
| Seeds | 5 | 2 (`for seed in [0, 1]`) | 5 for reported results (3 in the sweep) |
| Epochs, batch | not given | 30, 256 | 20, 512 (RelBench example; 20 from Plan 00) |
| Neighbours | not given | 128, 64 | 128, 64 |
| Hardware | one A30 GPU | – | CPU here; GPU on Colab |

### What this means

1. **We follow the paper's equations**, except for two deliberate choices: column gates on
   table columns (decision (b), for interpretable results) and no table gate on the target
   (closing it would zero the seeds' own vectors).
2. **The authors' code does not implement the paper's filtering at evaluation time.** Their
   column gates only act during training, and only on channel groups, not columns. So
   neither our numbers nor theirs can be read as "the paper's column filtering, exactly
   as published". We will state this in the PR and the blog.
3. **Settings the paper leaves open** (temperature, start logit, gate learning rate,
   epochs, batch size) are ours to choose on validation. The sweep does that for the gate
   settings.
4. **Not yet matched:** the paper's grid over learning rate and hidden size (for Base and
   for the optimizer). Our Base uses RelBench's defaults, which partly explains Base 0.79
   vs the paper's 0.815 (Plan 00).

## 13. What the paper's theory requires, and how the code guarantees it

The paper's theory (§3.4, Def. 3.1, Thm. 3.2, Prop. H.1, App. G) is stated for gates in
general, not for one gate target. Each claim needs certain properties from the code;
this lists them for our column version (b), with the test that checks each one.

| # | Paper | What the code must do | Ours (b) | Checked by |
|---|---|---|---|---|
| 1 | Thm. 3.2: gates are keep/drop variables $M_j$ with $\pi_j = P(M_j = 1)$ | each gate defines a keep/drop event with a computable probability | $M_j = \mathbb{I}[g_j > 0]$, $\pi_j = \sigma(\theta_j - t\log(-\gamma/\zeta))$ (`prob_open`) | closed form = empirical rate over 100k samples (`test_hard_concrete_gate_prob_open`) |
| 2 | Eq. 11, Thm. 3.2: the penalty is $\mu = \sum_j \pi_j = \mathbb{E}\lVert M\rVert_0$ | penalty = sum of open probabilities, differentiable | `l0_penalties()` = $\sum$ `prob_open`; loss adds $\lambda \sum + \lambda_s \sum$ | `test_relational_filter_l0_penalties`, `test_filter_model_penalty` |
| 3 | App. G: $\ell_0$, not $\ell_1$: discrete keep/drop | gates can be exactly 0 or 1 | stretched and clipped samples; hard 0/1 in evaluation | `test_hard_concrete_gate_training_samples`, `test_hard_concrete_gate_eval` |
| 4 | Prop. H.1 (filtering analogue): the gated family contains Base, so gates cannot make the best task risk worse | all gates open ⇒ exactly Base | yes | `test_filter_model_all_open_is_base` (one batch), `test_filter_model_open_trains_like_base` (training, exact), test (b) (5 seeds × 20 epochs) |
| 5 | Def. 3.1: a gate setting $\phi$ selects a graph view | a closed gate removes exactly its component | closed column ⇒ the prediction does not depend on that column's values; closed table ⇒ it does not depend on the table's columns or timestamps | `test_filter_model_closed_column_removes_its_information`, `test_filter_model_closed_table_removes_its_information` |
| 6 | Eq. 4: one deterministic structure at test time | evaluation uses hard 0/1 gates | yes, $\sigma(\theta) > 0.5$ | `test_hard_concrete_gate_eval`, `test_filter_model_closed_gates` |
| 7 | Def. 3.1: gates $\phi$ and weights $\theta$ optimised together | one loss, one optimiser for both | yes (Adam, with a separate gate learning rate) | `test_make_optimizer`, `test_filter_model_penalty` |

**Not guaranteed by code, to check in experiments:**

- **λ controls sparsity** (the knob behind Thm. 3.2 and the inverted-U of §3.3, App. D.1).
  At a stationary point a larger λ means fewer open gates. But Adam scales every update
  to about the learning rate, and `driver-top3` has only 60 steps, so training may never
  reach that point. The sweep must show the number of open gates falling as λ rises.
- **Thm. 3.2 is a bound, not a guarantee of better scores.** It says fewer expected open
  gates means a smaller space of structures, not that the chosen structure predicts
  better. The inverted-U is empirical.

**Where our deviations change the meaning, but not the guarantees:**

- **Gate target (b).** The theorem holds for any set of $J$ gates. Ours has $J = 45 + 8 =
  53$ (one per column or table), against $9 \times 128 + 9 = 1{,}161$ for the paper's
  equations. The paper's own wording for the gates in Thm. 3.2 is "attribute-, type- and
  template-level selection", which matches columns (attributes) and tables (types).
- **A closed `date` column** removes the date as a feature only. The row's timestamp still
  reaches the model through the time encoding, and through temporal sampling. Removing it
  needs the table gate (row 5).
- **A closed table stays a path**: its rows still pass on what they receive from
  neighbours (§2.3, decision 2). Row 5 checks that its own content is removed.
- **No table gate on the target table:** the theorem does not depend on it.
- **Gate settings** (temperature, start logit, gate learning rate) do not appear in the
  theory. They change only how training reaches a solution.
