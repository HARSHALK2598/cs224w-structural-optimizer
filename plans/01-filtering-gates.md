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

### 2.1 One real row, from DataFrame to $h^{(0)}$

The first row of `results` (the 1950 British Grand Prix, a driver who started 21st and
finished 11th), followed through each stage of the Base model, without gates.

**Stage 0: DataFrame row** (`db.table_dict['results'].df.iloc[0]`):

| resultId | raceId | driverId | constructorId | number | grid | position | positionOrder | points | laps | milliseconds | fastestLap | rank | statusId | date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 660 | 152 | 18 | 21 | 11 | 11 | 0 | 64 | NaN | NaN | NaN | 16 | 1950-05-13 |

**Stage 1: TensorFrame row** (`data['results'].tf`; materialisation, nothing learned).
The primary key becomes the node id and the 3 foreign keys become edges (both leave
the table), so $C_\text{results} = 11$ feature columns remain, grouped by stype:

| stype | columns | stored values for row 0 |
|---|---|---|
| numerical, `[N, 10]` | fastestLap, grid, laps, milliseconds, number, points, position, positionOrder, rank, statusId | `[NaN, 21, 64, NaN, 18, 0, 11, 11, NaN, 16]` |
| timestamp, `[N, 1, 7]` | date | `[1950, 4, 12, 5, 0, 0, 0]` (year, month−1, day−1, weekday, h, m, s) |

Other tables also have categorical columns (an integer index per value) and text
columns (a 300-d GloVe vector each, stored, not learned).

**Stage 2: column encoders** (learned; one small encoder per column $c$). Each column
value becomes one 128-d vector $e_{v,c}$:

$$
e_{v,c} =
\begin{cases}
\dfrac{x_{v,c} - \mu_c}{\sigma_c}\, w_c + b_c & \text{numerical } (w_c, b_c \in \mathbb{R}^{128})\\[4pt]
E_c[\,x_{v,c}\,] & \text{categorical (embedding row)}\\[2pt]
A_c\, \text{GloVe}(x_{v,c}) + b_c & \text{text } (A_c \in \mathbb{R}^{128 \times 300})\\[2pt]
\textstyle\sum_{k=1}^{7} W_{c,k}\, \text{PE}(x_{v,c,k}) + b_c & \text{timestamp (7 parts)}\\[2pt]
\mathbf{0} & \text{if } x_{v,c} \text{ is missing (NaN)}
\end{cases}
$$

(A missing text value is embedded as the empty string, so it is not $\mathbf{0}$.)
For row 0, using the training statistics: grid → $\frac{21 - 11.22}{7.48} = 1.31$,
so $e_{v,\text{grid}} = 1.31\, w_\text{grid} + b_\text{grid}$; points →
$\frac{0 - 1.09}{2.34} = -0.47$; milliseconds, fastestLap and rank are missing →
$\mathbf{0}$. Stacking the 11 vectors gives the row's matrix

$$X_v = [\,e_{v,1};\ \dots;\ e_{v,11}\,] \in \mathbb{R}^{11 \times 128}, \qquad \text{batch: } [N, 11, 128].$$

**Stage 3: flatten and the first ResNet layer.** $X_v$ is flattened to
$\text{vec}(X_v) \in \mathbb{R}^{1408}$ ($11 \times 128$). The first block's linear layer
$W^{(0)} \in \mathbb{R}^{128 \times 1408}$ splits into one $128 \times 128$ slice per column, so

$$W^{(0)} \text{vec}(X_v) = \sum_{c=1}^{11} W^{(0)}_c\, e_{v,c}.$$

This is the first point where columns are mixed. After it, no part of the vector
belongs to one column.

**Stage 4: rest of the ResNet, then time.** Three more residual blocks and the decoder
give one vector $r_v \in \mathbb{R}^{128}$. The time encoder adds how old the row is,
relative to the seed's prediction time $t_\text{seed}$:

$$h^{(0)}_v = r_v + \text{Lin}_T\big(\text{PE}(\Delta t_v)\big), \qquad \Delta t_v = \frac{t_\text{seed} - t_v}{86400} \text{ days}.$$

**Stage 5: message passing** (simplified; one GraphSAGE layer, sum aggregation, summed
over relations $R$). A node $v$ receives the sum of its neighbours' vectors:

$$h^{(1)}_v = \text{ReLU}\Big(\text{LN}\Big(\sum_{R} \big[ W^\text{nbr}_R \sum_{u \in \mathcal{N}_R(v)} h^{(0)}_u + W^\text{self}_R\, h^{(0)}_v \big]\Big)\Big).$$

### 2.2 The two gates in this notation

$$\tilde e_{v,c} = g_{T,c}\; e_{v,c} \quad \text{(column gate, between stages 2 and 3)}$$
$$\tilde h^{(0)}_u = s_T\; h^{(0)}_u \quad \text{(table gate, after stage 4)}$$

Substituting into stages 3 and 5:

$$W^{(0)} \text{vec}(\tilde X_v) = \sum_{c} g_{T,c}\, W^{(0)}_c\, e_{v,c}, \qquad \sum_{u \in \mathcal{N}_R(v),\ u \in T} \tilde h^{(0)}_u = s_T \sum_{u \in \mathcal{N}_R(v),\ u \in T} h^{(0)}_u.$$

So $g_{T,c} = 0$ deletes exactly column $c$'s term, for every row of $T$, and
$s_T = 0$ deletes exactly table $T$'s first-layer messages. The gates multiply, so
$\partial \mathcal{L}/\partial g$ exists and the logits $\theta$ get a gradient.

### 2.3 Why these places, and not others

**Column gate: between the column encoders and the ResNet.**

- *Not on the raw values (stages 0–1).* Multiplying a value by 0 does not remove it, it
  changes it. For row 0, grid = 0 would mean "started from the pit lane", and the
  encoder would still output $\frac{0 - 11.22}{7.48}\, w_\text{grid} + b_\text{grid} \neq \mathbf{0}$.
  A categorical index 0 is a real category. The only "absent" value is NaN, and
  switching to NaN is a hard on/off choice, which gives no gradient.
- *Not after the first ResNet layer (stage 3 onwards).* The sum above has already
  mixed all 11 columns, so there is nothing per-column left to switch off. Gating the
  128 numbers of $r_v$ or $h^{(0)}_v$ is option (a) in §3: it gates mixtures of columns.
- *Between stages 2 and 3, $g_{T,c} = 0$ gives $\tilde e_{v,c} = \mathbf{0}$*, the same
  vector a missing numerical, categorical or timestamp value gets. The bias $b_c$ is removed too. A closed column then
  looks to the model exactly like a column that is always empty, which the model has
  already learned to handle.

**Table gate: on $h^{(0)}$, after the time encoding.**

- *Not before the time encoding* (on $r_v$). Then
  $h^{(0)}_u = s_T\, r_u + \text{Lin}_T(\text{PE}(\Delta t_u))$, and with $s_T = 0$
  each neighbour still receives $\sum_u \text{Lin}_T(\text{PE}(\Delta t_u))$: how many
  rows of $T$ it has and how recent they are. The table would not be closed.
- *On $h^{(0)}$*, the paper's Eq. 5 position ($\tilde h_v = s_{T}\, \hat h_v$, applied
  to the column-filtered output). We read $\hat h_v$ as including the time encoding,
  for the reason above.
- **What a closed table still does:** it stays in the graph as a path. In layer 1 its
  own nodes still receive messages from their neighbours, and in layer 2 they pass
  those on. For example, with `results` closed, `drivers` still learns about `races`
  through `results` rows, but nothing from the `results` columns or dates. Removing
  the table as a path too would mean gating its messages in every layer, which is
  closer to relation gating (step C). Question 2 in §11.
- **No gate on the target table** (`drivers`): with $s = 0$ the seed's own vector
  $h^{(0)}_v$ would be zero. Its columns are still gated.

`rel-f1` has 45 columns in 9 tables, so this adds **45 column gates + 8 table gates =
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
2. Table gate after the time encoding, and no table gate on the target table: OK? And
   is "closed table = no own content, but still a path" (§2.3) what we want for now?
3. Penalty as sums over gates, as in the paper: OK, or averages, so λ means the same
   for tables with few or many columns?
