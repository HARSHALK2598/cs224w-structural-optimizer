# Plan 05: Template gates (steps C and D)

Plan 04 adds the injected relations and keeps every one of them on. The paper does not:
each injected relation (template instance) $E_k$ gets a learned switch $g_k$, so the
model can keep the templates that help a task and drop the ones that hurt (paper
Section 4.2). Our quick check showed why this matters: on `driver-top3`, KNN alone
lowered validation ROC-AUC from 0.7849 to 0.7522 (Plan 04 probe, tuned settings).

This plan adds those switches (step C) and their penalty in the loss (step D). It is
implementation only, in two small PRs; the evaluation comes after all the code
(Plan 07).

## 1. What the paper specifies

**Gated message (Eq. 17).** For template $E_k$, the aggregated message into $v$ at
layer $l$ is scaled by its gate:

```math
z^{(l)}_{v,k} = g_k \cdot \mathrm{AGG}\{ z^{(l)}_u : u \in N_k(v) \}
```

and it is summed with the schema relations' messages, as in Base (Eq. 19).

**The gate.** "The same Hard-Concrete relaxation as in Eqs. (3)–(5)": the switch from
Plan 01 (`HardConcreteGate`). In training, a random sample that can be exactly 0 or 1
and passes gradients to its logit; in evaluation, a fixed 0/1.

**Test time (Eq. 18).** Only open templates are used:

```math
E' = E \cup \{ E_k \mid g_k > \tau_k \} .
```

**Penalty (Eq. 11, step D).** The loss pays for every template expected to be open:

```math
\mathcal{L} = \mathcal{L}_{\text{task}} + (\text{Plan 01's gate terms}) + \lambda_k \sum_k \mathbb{E}\|g_k\|_0 ,
\qquad \lambda_k \in [10^{-4}, 10^{-2}] \ \text{(Table 8)}.
```

## 2. What exactly the gate multiplies (decision 1)

Our model gives every relation a `SAGEConv`, whose output for node $v$ is

```math
W_{\text{nbr}} \cdot \mathrm{mean}\{ z_u : u \in N(v) \} \; + \; W_{\text{root}} \, z_v \; + \; b .
```

Eq. 17 gates only the **neighbour part**. The root term $W_{\text{root}} z_v$ and the
bias $b$ are added for *every* node $v$ of the target type, **even one with no edges of
this relation**. So today (Plan 04), adding an injected relation changes every driver's
embedding through an extra root term and bias, even drivers that receive no KNN edge.
That is not "adding edges", and gating it would gate the node's own features too.

**Proposal:** injected relations use `SAGEConv(root_weight=False, bias=False)`, so
their output is exactly $W_k \cdot \mathrm{mean}\{z_u\}$, which is 0 for a node with no
such edges. Then:

- the gate multiplies exactly Eq. 17's message;
- a closed gate (or a relation with no edges) changes nothing: the model is Base;
- the schema relations keep their root terms and bias, as in Base.

This also changes Plan 04's ungated injection (merged code), in the same PR, so that
injection means only "messages along the new edges". The quick KNN check would then be
rerun in the evaluation (Plan 07).

## 3. The gates

| Choice | Proposal | Why |
|---|---|---|
| How many | **one gate per template instance** $E_k$ (e.g. 66 for two-hop on rel-f1) | the paper: "every $E_k \in M_{\text{all}}$" |
| Across layers | **shared**: the same $g_k$ in every GNN layer | Eq. 17's $g_k$ has no layer index |
| Sample | one sample per gate per training step, shared by all nodes | as Plan 01's gates |
| Evaluation | $g_k = \mathbb{I}[\sigma(\theta_k) > 0.5]$ | as Plan 01 ($\tau_k$ = 0.5) |
| Start, learning rate | logit 5 (open), gate learning rate 0.1, temperature 2/3 | Plan 01's tuned settings: with the model's learning rate, gates barely move in our number of steps |
| Penalty | $\lambda_k \sum_k \mathbb{E}\|g_k\|_0$, $\mathbb{E}\|g\|_0 = \sigma(\theta - t \log(-\gamma/\zeta))$ | Eq. 11 |

**Closed gates cost nothing (Eq. 18).** When a gate is 0 (always in evaluation when
closed; sometimes exactly 0 in training), its edges are not built at all: no KNN search,
no two-hop composition. For temporal relations, which the loader samples, the
convolution is skipped.

**The gates learn from the task.** The gate multiplies a message that reaches the
prediction, so the task loss pushes $g_k$ up when the template helps and down when it
hurts. In the authors' code the template gate only feels the $L_0$ penalty (Plan 04
§9), so it closes regardless.

## 4. Code, as two PRs

**05-1, fork: `contrib.nn.conv.GatedHeteroConv`** (the proposal's class for step C).

- Wraps one convolution per relation, like `HeteroConv` (sum over relations), plus a
  `HardConcreteGate` for each relation listed as gated.
- `forward(x_dict, edge_index_dict)`: each gated relation's output is multiplied by its
  gate; relations whose gate is 0 are skipped.
- `open_edge_types()`: the gated relations currently open, so a caller can skip building
  closed ones' edges; `l0_penalty()`: $\sum_k \mathbb{E}\|g_k\|_0$.
- PyG docstring (math, args, shapes, example), `reset_parameters`, `__repr__`; tests in
  `test/contrib/nn/conv/`.

**05-2, project: training with template gates.**

- Our GNN becomes RelBench's `HeteroGraphSAGE` (same norms, ReLU) built on
  `GatedHeteroConv`; with no gated relations it is RelBench's model exactly (tested).
- Injected relations: `SAGEConv(root_weight=False, bias=False)` (decision 1).
- Flags: `--template-gates` (gate the injected relations), `--lambda-template`
  ($\lambda_k$); the gate start, learning rate and temperature reuse Plan 01's
  `--gate-init`, `--gate-lr`, `--gate-temperature`.
- Logs and saves each template's open probability and 0/1 state at the best epoch, as
  Plan 01 does for columns and tables; `open_templates` in `results.csv`.

## 5. Tests

1. **The PR's Test 3, analytic:** two relations on a hand-made graph, one gated; with
   the gate set by hand, the output equals Agg$_r$ + $g \cdot$ Agg$_k$ exactly; in
   evaluation a closed gate's relation contributes nothing and an open one its full
   message.
2. **No gated relations:** `GatedHeteroConv` equals `HeteroConv`; our GNN equals
   RelBench's `HeteroGraphSAGE` (same weights, same output).
3. **A closed or edge-less injected relation leaves Base unchanged** (decision 1).
4. **Gates learn from the task:** after one backward pass on a rel-f1 batch, each open
   template gate's logit has a gradient from the task loss alone ($\lambda_k$ = 0).
5. **Closed templates are not built:** with a gate forced shut in evaluation, its KNN /
   two-hop module is not called.
6. **End to end:** one epoch on rel-f1 with all four templates gated; CPU dry run of the
   sweep runner.

## 6. Decisions for you

1. **Gate the neighbour message only:** injected relations without root term and bias,
   which also changes Plan 04's ungated injection (§2)?
2. **One gate per template instance, shared across layers** (§3)?
3. **Gate settings from Plan 01** (start 5, gate learning rate 0.1, threshold 0.5) as
   defaults, with $\lambda_k$ swept later in Plan 07?
4. **Skip building closed templates' edges** (§3)?

## 7. Out of scope

The variational bottleneck and the `StructuralOptimizer` wrapper (Plan 06); all sweeps
and the evaluation on several datasets (Plan 07).
