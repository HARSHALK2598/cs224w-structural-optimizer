# Plan 00 — Baseline: RelBench's relational GraphSAGE on `rel-f1`

**Status:** draft, awaiting review
**Repo:** project repo only (`experiments/`). No change to the PyG fork.

## 1. Goal

Measure the paper's **Base** model, RelBench's standard heterogeneous GraphSAGE built
from unmodified PyG components, on `rel-f1`. Every later model (filtering, full
optimizer) is compared against these numbers, so the baseline must:

- match RelBench's official example (`examples/gnn_entity.py` in
  `snap-stanford/relbench`, read at commit `91c1f64`) in every modelling choice;
- be reproducible: 5 fixed seeds, every run logged as one CSV row;
- be understood end to end. Sections 3–8 are the study material.

**Scope.** Three `rel-f1` tasks: `driver-top3` and `driver-dnf` (binary
classification), and `driver-position` (regression). `rel-trial` is left for a later
plan (it still has to be downloaded and sized). `rel-f1` is small enough that all
3 tasks × 5 seeds run **on the CPU in this container**, so Colab isn't needed here.

## 2. Pipeline at a glance

```
relational DB (9 tables)
  └─ make_pkey_fkey_graph ──► HeteroData: 9 node types, 26 edge types, TensorFrame features, timestamps
training table rows (driverId, t, label)
  └─ NeighborLoader (temporal, 2 hops: 128, 64) ──► one disjoint subgraph per (driverId, t)
       └─ HeteroEncoder + HeteroTemporalEncoder ──► h⁽⁰⁾ for every sampled row
            └─ HeteroGraphSAGE (2 layers) ──► h⁽²⁾
                 └─ MLP head on the seed driver ──► ŷ
                      └─ BCE / L1 loss → Adam; select best epoch on val; evaluate on test
```

## 3. Dataset: `rel-f1` as a graph

**Database.** 9 tables, 13 foreign keys. It downloads to `data/hf/` (2.1 MB) through
`relbench.load_dataset('rel-f1')`, from the Hugging Face repo
`stanford-star/relbench-v1/rel-f1`.

| Table | Rows | Time column | Foreign keys → |
|---|---:|---|---|
| races | 820 | date | circuits |
| circuits | 77 | — | — |
| drivers | 857 | — | — |
| constructors | 211 | — | — |
| results | 20,323 | date | races, drivers, constructors |
| qualifying | 4,082 | date | races, drivers, constructors |
| standings | 28,115 | date | races, drivers |
| constructor_results | 9,408 | date | races, constructors |
| constructor_standings | 10,170 | date | races, constructors |

**Graph construction** (`relbench.modeling.graph.make_pkey_fkey_graph`). This builds a
heterogeneous graph $G=(V,E)$ with a node-type map $\phi: V \to \mathcal T$ (the
tables) and an edge-type map $\psi: E \to \mathcal R$:

- **Nodes.** Each row $v$ of table $T$ becomes a node with $\phi(v)=T$. The primary
  key is the node index (RelBench asserts that keys are $0..|T|-1$).
- **Edges.** Each foreign-key column $c$ of $T_{\text{src}}$ that points to
  $T_{\text{dst}}$ produces two edge types:
  $(T_{\text{src}},\ \texttt{f2p\_}c,\ T_{\text{dst}})$ (row → referenced row) and
  $(T_{\text{dst}},\ \texttt{rev\_f2p\_}c,\ T_{\text{src}})$.
  13 FKs give $|\mathcal R| = 26$. Dangling keys are dropped, and each `edge_index`
  is sorted with `torch_geometric.utils.sort_edge_index`.
- **Features.** Primary- and foreign-key columns are removed. The remaining columns
  are stored as a `torch_frame.TensorFrame` in `data[T].tf`, with per-column statistics
  (mean, std, categories, …) in `col_stats_dict`. Column types come from
  `relbench.modeling.utils.get_stype_proposal`: numerical, categorical, timestamp, or
  text (names, nationalities).
- **Time.** Tables with a time column get `data[T].time` (Unix seconds). `drivers`,
  `circuits` and `constructors` have none: they are static.
- `remove_columns=task.hidden_columns()` drops columns that would leak the label.
  This is empty for all three `rel-f1` tasks, but we pass it anyway to match the
  official example.

**Tasks** (`dataset.load_task(name)`). Each task is a *training table* of rows
$(\text{driverId}, t, y)$. The label $y$ is computed by SQL over the window
$(t, t+\Delta]$; the model may only use database rows with timestamp $\le t$.

| Task | Type | Label | Δ | Train / Val / Test rows | Metric |
|---|---|---|---|---|---|
| driver-top3 | binary | qualifies in the top 3 in a race in the next Δ | 30 d | 1,353 / 588 / 726 | ROC-AUC ↑ |
| driver-dnf | binary | does not finish a race in the next Δ | 30 d | 11,411 / 566 / 702 | ROC-AUC ↑ |
| driver-position | regression | mean finishing position over the next Δ | 60 d | 7,453 / 499 / 760 | NMAE ↓ (and MAE) |

**Splits are temporal.** Train seeds have $t <$ 2005-01-01, val seeds fall in
[2005-01-01, 2010-01-01), and test seeds have $t \ge$ 2010-01-01. Test labels are
hidden: `task.get_table('test')` has no label column, and `task.evaluate(pred)`
scores the predictions against the hosted labels.

> Note: the `driver-dnf` SQL carries a known RelBench issue
> (stanford-star/relbench#402). Its "active driver" filter has no upper time bound.
> RelBench keeps it as-is for comparability, and so do we.

## 4. Sampling: temporal neighbour sampling

**Seeds.** A mini-batch is $B=512$ training-table rows. Each row $(v_i, t_i)$
becomes a *seed*: node $v_i$ of `drivers` with seed time $t_i$
(`get_node_train_table_input` → `input_nodes`, `input_time`).

**Neighbourhood** (`torch_geometric.loader.NeighborLoader`, backed by
`pyg_lib.sampler`):

- **Two hops**, with fan-out $[128, 64]$ (`num_neighbors=[128 // 2**i for i in range(2)]`).
  The fan-out applies **per edge type**: a node can contribute up to 128 neighbours
  for each incoming relation at hop 1, and up to 64 at hop 2.
- **The temporal constraint** (`time_attr='time'`): a neighbour $u$ may be sampled
  for seed $i$ only if $t_u \le t_i$. This is what prevents leaking the future.
  Nodes of static tables have no time and are always eligible. *(We will check this
  with the test in §9.)*
- **Strategy** `temporal_strategy='uniform'`: sample uniformly among the eligible
  neighbours (rather than the most recent ones).
- **Disjoint subgraphs.** Temporal sampling forces `disjoint=True`: each seed gets
  its own copy of every sampled node, because the same row can be eligible for one
  seed and not for another. `batch[T].batch` maps each sampled node to its seed, and
  the seeds are the first $B$ nodes of `drivers` in the batch.
- **Labels** are attached afterwards by `AttachTargetTransform`, via
  `batch['drivers'].input_id`.
- Train batches are shuffled; val and test batches are not.

## 5. Model (the paper's *Base*)

Notation: $T(v)$ is the table of node $v$, $C_T$ the feature columns of $T$, and
$d=128$ the channel size. $\mathcal N_r(v)$ are the sampled $r$-neighbours of $v$.

### 5.1 Column encoders → row embedding (`relbench.modeling.nn.HeteroEncoder`)
There is one `torch_frame.nn.ResNet` per table. Each column $c$ is first embedded to
$\mathbb R^d$ by its type's encoder (`torch_frame.nn.StypeWiseFeatureEncoder`):

| stype | encoder | $e_{v,c} \in \mathbb R^d$ |
|---|---|---|
| numerical | `LinearEncoder` | $\mathbf w_c \cdot \frac{x_{v,c}-\mu_c}{\sigma_c} + \mathbf b_c$ |
| categorical | `EmbeddingEncoder` | $\mathbf E_c[x_{v,c}]$ (a lookup table; slot 0 for missing values) |
| timestamp | `TimestampEncoder` | sinusoidal encoding of year/month/day/… → linear |
| text | `LinearEmbeddingEncoder` | $\mathbf W_c\,\mathrm{GloVe}(x_{v,c})$, $\mathbf W_c \in \mathbb R^{d\times 300}$ |

Text is embedded once, before training, as the average 300-d GloVe vector
(`sentence-transformers/average_word_embeddings_glove.6B.300d`), and cached.

Columns are concatenated, $\mathbf x_v = [e_{v,c_1};\dots;e_{v,c_{|C_T|}}] \in
\mathbb R^{d|C_T|}$, then passed through 4 residual blocks and a decoder:

$$
\mathrm{Block}(\mathbf x) = \mathrm{Drop}\big(\mathrm{ReLU}(\mathrm{LN}(W_2\,
\mathrm{Drop}(\mathrm{ReLU}(\mathrm{LN}(W_1\mathbf x))))) \big) + S\mathbf x,
\qquad
\mathrm{Enc}_T(\mathbf x_v) = W_o\,\mathrm{ReLU}(\mathrm{LN}(\mathrm{Block}^{4}(\mathbf x_v)))
$$

Dropout is 0.2. $S$ is a linear shortcut in the first block (where the input size
$d|C_T| \ne d$) and the identity afterwards.

### 5.2 Relative-time encoding (`HeteroTemporalEncoder`, `torch_geometric.nn.PositionalEncoding`)
For a node $v$ of a timed table, sampled for seed $i$:

$$
\Delta_v = \frac{t_i - t_v}{86400}\ \text{(days)},\qquad
\mathrm{PE}(\Delta)_{2k} = \sin(\Delta\,\omega_k),\ \mathrm{PE}(\Delta)_{2k+1} =
\cos(\Delta\,\omega_k),
$$

where $\omega_k$ is log-spaced from 1 down to $10^{-4}$. Then
$\mathbf h^{(0)}_v = \mathrm{Enc}_{T(v)}(\mathbf x_v) + W_{T}\,\mathrm{PE}(\Delta_v)$.
Static tables get no time term. This $\mathbf h^{(0)}$ is exactly the input that the
paper's filtering step (A) will later gate.

### 5.3 Message passing (`relbench.modeling.nn.HeteroGraphSAGE`)
There are $L=2$ layers. Each layer is a `torch_geometric.nn.HeteroConv` holding one
`torch_geometric.nn.SAGEConv((d, d), d, aggr='sum')` per edge type, combined with
`aggr='sum'`:

$$
\mathbf m^{(l)}_{v,r} = W^{(l)}_{r,1} \sum_{u \in \mathcal N_r(v)} \mathbf h^{(l)}_u
+ W^{(l)}_{r,2}\,\mathbf h^{(l)}_v + \mathbf b^{(l)}_r,
\qquad
\mathbf h^{(l+1)}_v = \mathrm{ReLU}\Big(\mathrm{LN}^{(l)}_{T(v)}\Big(\sum_{r \in
\mathcal R:\ r \text{ ends in } T(v)} \mathbf m^{(l)}_{v,r}\Big)\Big)
$$

The norm $\mathrm{LN}$ is `torch_geometric.nn.LayerNorm(d, mode='node')`, one per
node type per layer. Note that `HeteroConv` only updates node types that are the
destination of some relation. All 9 are, thanks to the `rev_` edges.

### 5.4 Head (`torch_geometric.nn.MLP(d, out_channels=1, num_layers=1)`)
With one layer, the MLP is a single linear map, so its `norm='batch_norm'` argument
has no effect: $\hat y_i = \mathbf w^\top \mathbf h^{(2)}_{v_i} + b$. It is applied
to the $B$ seed nodes only (`x_dict['drivers'][:B]`).

## 6. Loss, optimisation and model selection

- **Binary tasks** (`torch.nn.BCEWithLogitsLoss`):
  $\mathcal L = -\frac1B \sum_i \big[y_i \log\sigma(\hat y_i) + (1-y_i)\log(1-\sigma(\hat y_i))\big]$.
  Predictions are $\sigma(\hat y)$.
- **Regression** (`torch.nn.L1Loss`): $\mathcal L = \frac1B\sum_i |\hat y_i - y_i|$.
  At inference, $\hat y$ is clamped to the [2nd, 98th] percentiles of the training
  labels (`torch.clamp`). This guards against outlier predictions, as in the official
  example.
- **Optimiser:** `torch.optim.Adam`, lr 0.005, no weight decay. 10 epochs, at most
  2,000 steps per epoch, batch 512.
- **Model selection:** after every epoch, evaluate on val with the *tune metric*
  (ROC-AUC ↑ / NMAE ↓). Keep the best state (`>=` / `<=`, so ties go to the later
  epoch), reload it, then evaluate once on test.
- **Metrics** (`task.evaluate` → `relbench.metrics`): ROC-AUC; NMAE $=\mathrm{MAE}/s$,
  where $s$ is RelBench's hosted standard deviation of the task's regression target.
  We also log raw MAE for regression, since the paper (and RelBench v1) report MAE.
- **Seeds** 0–4 through `torch_geometric.seed.seed_everything`. Report mean ± std.

## 7. PyG / PyTorch building blocks used

| Component | Function / class | From |
|---|---|---|
| DB → graph | `make_pkey_fkey_graph`, `get_stype_proposal` | RelBench (builds `HeteroData`) |
| Graph container | `HeteroData`, `utils.sort_edge_index` | PyG |
| Tabular features | `TensorFrame`, `nn.ResNet`, stype encoders | PyTorch Frame |
| Text features | `SentenceTransformer` (GloVe), `TextEmbedderConfig` | sentence-transformers, PyTorch Frame |
| Sampling | `loader.NeighborLoader` (temporal), `pyg_lib` sampler | PyG |
| Train-table input / labels | `get_node_train_table_input`, `AttachTargetTransform` | RelBench |
| Time encoding | `nn.PositionalEncoding` (inside `HeteroTemporalEncoder`) | PyG |
| Message passing | `nn.HeteroConv`, `nn.SAGEConv`, `nn.LayerNorm` | PyG (via RelBench's `HeteroGraphSAGE`) |
| Head | `nn.MLP` | PyG |
| Loss / optimiser | `BCEWithLogitsLoss`, `L1Loss`, `optim.Adam`, `torch.clamp` | PyTorch |
| Seeding | `seed.seed_everything` | PyG |
| Evaluation | `task.evaluate`, `relbench.metrics` | RelBench |

## 8. What changes in `experiments/train.py`

The current script (day 1) already follows §§3–6, except for these gaps against the
official example:

| # | Gap | Fix |
|---|---|---|
| 1 | `remove_columns=task.hidden_columns()` not passed | pass it |
| 2 | Regression predictions not clamped | clamp to train-label P2/P98 at inference |
| 3 | Tune metric is `task.metrics[0]`; ties keep the earlier epoch | explicit `roc_auc` / `nmae`; `>=` / `<=` |
| 4 | Regression logs only NMAE | also log MAE (val and test) |
| 5 | GloVe embedder fails on missing text (NaN) | map `None`/NaN to `""`, as in the example |
| 6 | `sentence-transformers` not installed | add it to `setup_env.sh` (the GloVe model is ~480 MB, from Hugging Face) |
| 7 | No summary of runs | `experiments/summarize.py`: mean ± std per (task, model) from `results.csv` |

Deliberately not ported: the stype JSON cache, `--include_task_tables`, and the
prediction-file export. These are options the official example has off by default.

## 9. Tests (`experiments/tests/`, run with `pytest experiments/tests -q`)

| Test | What it checks | Expected |
|---|---|---|
| `test_config_precedence` | CLI > YAML > defaults in `parse_args` | hand-written YAML sets `lr=0.01`; CLI `--lr 0.02` wins; untouched keys keep their defaults |
| `test_graph_shape` | `make_pkey_fkey_graph` on `rel-f1` | 9 node types, 26 edge types, every `rev_` edge is the transpose of its `f2p_` edge, node counts match the table above |
| `test_temporal_sampling_no_leakage` | the core no-future-information guarantee | for one val batch: every sampled node $u$ of a timed table has $t_u \le$ seed time of `batch[T].batch[u]`; the seed drivers are the first $B$ nodes |
| `test_regression_clamp` | §6 clamp | on `driver-position`, all val predictions lie in [P2, P98] of the train labels |
| `test_smoke_runs` | end to end, one batch, `--text-embedder none` | `driver-top3` and `driver-position` both run; predictions have one value per val row; no row written to `results.csv` |
| `test_results_row_and_summary` | logging and aggregation | a 1-epoch / 1-step run writes one row with the expected columns; `summarize.py` on a hand-made 3-row CSV gives the hand-computed mean ± std |

The `rel-f1`-dependent tests skip themselves if the dataset isn't cached.

## 10. Commits (small; each reviewed before it is merged)

1. `train.py`: hidden columns, regression clamp, explicit tune metric with ties, MAE
   logging (gaps 1–4), plus `test_config_precedence` and `test_regression_clamp`.
2. Sampling and graph tests only: `test_graph_shape`, `test_temporal_sampling_no_leakage`.
3. GloVe text embedder hardening, plus `sentence-transformers` in `setup_env.sh` (gaps 5–6).
4. `summarize.py` and `test_results_row_and_summary` (gap 7), plus `test_smoke_runs`.
5. **Run the baseline:** 3 tasks × 5 seeds, GloVe, CPU. Commit `results.csv` and the
   summary table (§11). No code changes.

## 11. Deliverable

A table in `notes/status.md` and `experiments/results/`. We report our numbers next to
the paper's as they are, with no adjustment:

| Task | Metric | Ours: Base (mean ± std, 5 seeds) | Paper: Base |
|---|---|---|---|
| driver-top3 | ROC-AUC ↑ | — | 81.48 ± 1.06 |
| driver-dnf | ROC-AUC ↑ | — | 74.53 ± 0.27 |
| driver-position | MAE ↓ (NMAE) | — | 3.88 ± 0.02 |

Paper numbers: Tables 1–3 of arXiv:2606.08491, mean ± std over 5 seeds. The paper's
train/val/test row counts (Table 7) match ours exactly.

Run time is not known yet. It will be measured in commit 4 with one seed and
reported before launching all 15 runs.

## 12. How this baseline relates to the paper's Base

This plan builds **Base-RelBench**: RelBench's official example with its fixed
defaults. The paper's Base is the same architecture (Appendix B: "relational
GraphSAGE … on the original heterogeneous graph"), but trained differently:

| | Base-RelBench (this plan) | Paper's Base |
|---|---|---|
| lr / hidden size | 0.005 / 128, fixed | grid search on val: lr ∈ {1e-4, 1e-3, 1e-2} × hidden ∈ {64, 128, 256} (Table 8, App. C) |
| epochs / batch / weight decay | 10 / 512 / 0 | not stated in the paper; authors' code uses 30 / 256 / 5e-4 |
| regression metric | tuned on NMAE, MAE also logged | MAE |
| encoder | RelBench ResNet `HeteroEncoder` | "same heterogeneous encoder" (Eq. 1 writes it as linear; authors' code uses the RelBench ResNet) |

**Decision rule (agreed):** run Base-RelBench first and compare it with the paper's
Base. Only if the gap matters for our comparison do we write a separate plan to
re-create the paper's tuned version (**Base-paper**).

## 13. Questions for you

1. ~~Tasks~~: resolved. The paper uses exactly these three `rel-f1` tasks.
2. ~~Paper numbers~~: resolved, see §11.
3. **Review granularity:** is one plan for the whole baseline milestone, with a
   review per commit (§10), OK? Or do you want a separate plan file per commit?
4. **Extra floor (optional):** RelBench also has trivial baselines (majority class or
   mean prediction, `examples/trivial_entity.py`). Add them as a sanity floor in the
   results table? It costs about one small commit.
