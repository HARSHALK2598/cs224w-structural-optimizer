# Plan 04: Structural injection (step B)

The paper's second idea is **injection**: add edges the database schema does not have,
so that the GNN can reach related rows in fewer hops. This plan builds the paper's
four edge templates, makes sure none of them lets a prediction see the future, and adds
them to the Base model as extra relation types. It then repeats the paper's probe
(Section 3.3, Figure 1): Base + one template at a time.

The per-template **gates** (step C) and their penalty (step D) are Plan 05. Here every
injected relation is simply on, as in the paper's probe experiments. Everything is
written for any RelBench dataset; rel-f1 is the first one run, and the numbers below
are rel-f1's.

## 1. The four templates (paper Section 3.2)

| Template | Paper | Adds an edge $u \to v$ when … | Intended for (paper's finding) |
|---|---|---|---|
| **Type-wise KNN** | Eq. 6 | $u$ is among the $n$ rows of $v$'s table whose initial embedding $h^{(0)}$ is most cosine-similar to $v$'s | classification |
| **Two-hop shortcut** | Eq. 7 | there is a path $u \xrightarrow{r_1} w \xrightarrow{r_2} v$ through some row $w$ | (helps some tasks; hurts temporal ones) |
| **Behavioural similarity** | Eq. 8 | the mean embedding of $u$'s neighbours along a relation $r$ is cosine-similar to $v$'s, above $\tau_{\text{sim}}$ | recommendation |
| **Temporal continuity** | Eq. 9 | $u$ and $v$ are records of the same entity and $u$ comes just before $v$ in time | regression |

Each template has **instances**, one per place in the schema it can apply (paper
Eqs. 15–16): one KNN instance per table, one two-hop instance per pair of relations
$A \xrightarrow{r_1} B \xrightarrow{r_2} C$, and so on. Every instance becomes **its own
relation type** in the graph, with its own convolution (and, in Plan 05, its own gate).

The paper's settings (Table 8): KNN neighbours $n \in \{1, 5, 10, 15, 20\}$; temporal
chain length $m \in \{1, 3, 5, 7, 9\}$. $\tau_{\text{sim}}$ is not given.

## 2. The rule that matters most: no edge may leak the future

Our loader samples **in time**: for a seed row predicted at time $t_s$, it only samples
rows with time $\le t_s$. PyG makes temporal sampling **disjoint**: a batch of 512 seeds
is 512 separate subgraphs, each cut at its own seed time (`batch_dict` says which
subgraph each node belongs to; `seed_time` gives its time).

- **Edges inside one subgraph are safe:** everything there is from the seed's past.
- **Edges across subgraphs can leak:** a node from a seed predicted *later* carries,
  after one GNN layer, information from that later seed's past, which is the earlier
  seed's future.
- **Edges built once on the whole database can leak too:** a two-hop shortcut
  `drivers → constructors` through `results` is created by a *future* race result,
  and `drivers` and `constructors` have no time column, so the sampler cannot filter
  it. The shortcut would tell the model which team a driver will race for.

**Our rule** (applied to every template built per batch):

```math
u \to v \ \text{is allowed only if}\ \ t_s(\text{subgraph of } u) \le t_s(\text{subgraph of } v),\quad u \neq v .
```

Then everything $u$ can pass on is from $v$'s past. Edges within one subgraph satisfy
it trivially.

| Template | Built | Why this is leak-free |
|---|---|---|
| Temporal continuity | **once, on the database** (a dataset transform) | it only links records older → newer (or equal time); the sampler already drops records newer than the seed |
| Two-hop shortcut | **per batch**, from the sampled edges | it composes edges the sampler already cut at the seed's time; both hops lie in one subgraph |
| KNN | **per batch**, on $h^{(0)}$ | the rule above |
| Behavioural | **per batch**, on $h^{(0)}$ | the rule above |

KNN and behavioural must be per batch anyway, since $h^{(0)}$ is learned. The proposal
planned two-hop as a one-off `AddMetaPaths` transform. The leak above rules that out,
so two-hop becomes per batch too (a change from the proposal, to state in the blog).

**The authors' code** builds KNN and two-hop per batch, KNN with no time check across
subgraphs, and never passes the injected edges to a convolution (the new relation types
have none), so they have no effect on its predictions.

## 3. Each template in detail, with rel-f1's numbers

One training batch of `driver-top3` (512 seeds, 128 → 64 neighbours) holds:
`races` 31,901 nodes, `results` 28,112, `standings` 28,126, `qualifying` 11,760,
`constructors` 1,295, `drivers` 512 (one per seed), others 0.

### 3.1 Temporal continuity (dataset transform, `TemporalContinuity`)

For a dated record table $R$ with a foreign key to an entity table $E$: group $R$'s rows
by the entity, sort each group by time (ties by row id), and link each record to the
$m$ records before it: $v_{t-j} \to v_t$ for $j = 1, \ldots, m$ ($m = 1$ is Eq. 9).

- **Which instances.** rel-f1 has 13 dated-record → entity keys. In 5 of them the entity
  is itself a dated event (`races`): there, all records of a group share one timestamp
  (e.g. 19,503 of 19,503 consecutive `results` pairs per race have equal time), so
  "temporal order" means nothing. **Proposal:** instantiate only when the entity table
  has no time column (a lasting entity: `drivers`, `constructors`, `circuits`). That
  gives **8 instances** on rel-f1, e.g. `results` of one driver, `standings` of one
  constructor, `races` at one circuit.
- **$m$**: the paper never defines "chain length". We read it as "link each record to
  its previous $m$", which is Eq. 9 for $m = 1$ (**our interpretation, to state**).
- **Size:** at most $m$ × rows of $R$ edges per instance (e.g. ≤ 19,517 × $m$ for
  `results` by driver).

### 3.2 Two-hop shortcut (per batch, `TwoHopEdges`)

For relations $A \xrightarrow{r_1} B \xrightarrow{r_2} C$, compose the batch's sampled
edges: $u \to v$ for every $u \to w \to v$. Duplicate pairs become one edge.

- **Which instances.** rel-f1 has 92 relation pairs; 66 with $A \neq C$; 36 distinct
  $(A, C)$. Round trips $A \to B \to A$ (e.g. driver → result → same driver) add little.
  **Proposal:** all pairs with $A \neq C$, one instance per pair $(r_1, r_2)$: **66
  instances** (each one a relation type with its own convolution).
- **Cost:** a sparse product of two sampled edge lists per instance, per batch.

### 3.3 Type-wise KNN (per batch, `KNNEdges`)

For a table $T$: cosine similarity of $h^{(0)}$ between the batch's $T$-nodes; for each
$v$, the $n$ most similar allowed $u$ (the rule in §2), $u \to v$.

- **Which $h^{(0)}$.** The Base model's input to the GNN: the encoded row + relative-time
  encoding (with Plan 01's gates, after them). Similarities are computed without
  gradient (choosing neighbours is not differentiable); messages along the chosen edges
  are differentiable as usual.
- **Which instances.** The authors' code uses every table. Cost is the issue: a batch has
  ~30,000 `races` rows, so all pairs is ~$10^9$ similarities per table per batch.
  **Proposal:** one instance per table, but **on by default only for the task's entity
  table** (`drivers`: 512 × 512 per batch, cheap), with a flag to add others; other tables
  use a chunked top-$n$ so memory stays bounded.
- **Note:** a seed's prediction now depends on which other seeds share its batch. In
  evaluation the batch order is fixed, so results stay reproducible.

### 3.4 Behavioural similarity (per batch, `BehavioralEdges`)

For a table $A$ and a relation $B \xrightarrow{r} A$: each $A$-node's mean $h^{(0)}$ over
its $B$-neighbours in the batch; link $u \to v$ when their cosine similarity exceeds
$\tau_{\text{sim}}$, keeping at most $n$ per $v$, under the rule in §2.

- **Which instances.** One per relation into a table: 26 on rel-f1. **Proposal:** like
  KNN, on by default for the entity table only (`drivers` via `results`, `standings`,
  `qualifying`: 3 instances).
- **$\tau_{\text{sim}}$**: not given by the paper. **Proposal:** 0.9 with a cap of $n$
  per node (the existing skeleton's default), swept in Plan 07 if it matters.

## 4. Where it plugs in

```
database ──TemporalContinuity──▶ graph (+ temporal relations)
                                   │ NeighborLoader (temporal, one subgraph per seed)
                                   ▼
batch ──encoder──▶ h⁽⁰⁾ ──(Plan 01 gates)──▶ KNNEdges / BehavioralEdges / TwoHopEdges
                                                    │ add per-batch relations
                                                    ▼
                         HeteroConv: one SAGEConv per relation (schema + injected)
                                                    ▼
                                                  head
```

- Injected relation types must be **known when the model is built**, so each gets a
  `SAGEConv` in `HeteroConv` (Base sums the relations). The templates expose their
  edge types up front; the model adds them to its relation list.
- `train.py` gets flags: `--inject temporal knn twohop behavioral` (any subset),
  `--knn-n`, `--temporal-m`, `--behavioral-threshold`, `--inject-tables` (default: the
  task's entity table, for KNN/behavioural). Without `--inject`, nothing changes.

## 5. Code (PyG fork, `contrib/`)

| Class | Kind | Status |
|---|---|---|
| `contrib.transforms.TemporalContinuity` | dataset transform | skeleton exists; implement, add $m$ and the lasting-entity rule |
| `contrib.nn.models.TwoHopEdges` | per-batch module | new |
| `contrib.nn.models.KNNEdges` | per-batch module | skeleton exists; implement with the §2 rule (pure-PyTorch top-$n$, chunked; `torch_cluster` if present) |
| `contrib.nn.models.BehavioralEdges` | per-batch module | skeleton exists; implement |
| helper `allowed_by_time(batch_u, batch_v, seed_time)` | function | new: the §2 rule, shared by all three |

Each with PyG-style docstrings (math, `Args`, shapes, `Example`), `__repr__`, and tests
in `test/contrib/`.

## 6. Tests

1. **Exact edges on a hand-made graph** (the PR's Test 2), one per template: a few rows
   with known times and keys, and the full expected edge list written out by hand.
2. **No leakage:** a batch of two subgraphs with seed times $t_1 < t_2$; no injected edge
   may run from the $t_2$ subgraph into the $t_1$ subgraph, for every template.
3. **Temporal continuity:** ties broken by row id; no edges for an entity table with a
   time column; $m$ = 1, 2 and more than a group's size.
4. **Model:** with `--inject` off, outputs equal today's Base exactly; with it on, every
   injected relation has a convolution and changes the output.
5. **End to end:** one epoch on `rel-f1` per template.

## 7. Experiment: the paper's probe (Figure 1)

Base vs Base + one template, on rel-f1's three tasks (`driver-top3`, `driver-dnf`:
classification; `driver-position`: regression), RelBench's settings (as Base), 5 seeds,
validation for choosing, test reported once. Strengths: KNN $n \in \{5, 10\}$, temporal
$m \in \{1, 5\}$ (from the paper's grid), two-hop and behavioural at their defaults.

**What we look for:** the paper's pattern, KNN helping classification and temporal
continuity helping regression, and two-hop hurting some tasks. We report our numbers
next to the paper's Figure 1 without adjustment. This doubles as the "Base + B only"
ablation for the blog.

Runs: 3 tasks × (1 Base + 6 template settings) × 5 seeds = 105 short rel-f1 runs, on
Colab via `sweep.py`, CPU dry run first.

## 8. Decisions for you

1. **Temporal continuity instances:** only for entity tables without a time column
   (8 on rel-f1), as proposed in §3.1? Or every dated-record → entity key (13)?
2. **Chain length $m$:** our reading, "link to the previous $m$ records"?
3. **Two-hop instances:** all relation pairs with $A \neq C$ (66 on rel-f1)? Or one per
   distinct $(A, C)$ (36), merging parallel paths?
4. **KNN and behavioural:** on for the task's entity table only by default, others
   behind a flag (§3.3)?
5. **$\tau_{\text{sim}} = 0.9$** with a cap of $n$ per node?
6. **Two-hop per batch** instead of the proposal's `AddMetaPaths`, because of the leak
   in §2?

## 9. Out of scope

Template gates, the hard 0/1 selection at test time (Eqs. 17–18) and the $\lambda_k$
penalty (Plan 05); the variational bottleneck and the `StructuralOptimizer` wrapper
(Plan 06); the full evaluation (Plan 07).
