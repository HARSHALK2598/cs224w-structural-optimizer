# Learning the Graph for Relational Deep Learning, in PyG

<!-- Skeleton of the CS224W blog post (Medium). Each section lists what goes in it;
     numbers already measured are filled in, gaps are marked TODO. -->

*Harshal Kulkarni, CS224W project. Code:
[PyG `contrib` PR](https://github.com/HARSHALK2598/pytorch_geometric/tree/contrib/structural-optimizer),
[experiments](https://github.com/HARSHALK2598/cs224w-structural-optimizer).*

**Main assumption, stated up front.** All numbers below follow the authors' evaluation
protocol: the model reads the *whole* database, including rows after RelBench's test cutoff,
with only rows strictly earlier than each prediction time sampled. RelBench's standard
protocol cuts the database at the test timestamp. Under the standard cutoff, the authors'
released driver-top3 weights drop from 90.33 to 83.98 ROC-AUC (Section 6).

## 1. Motivation

- Relational deep learning (RDL) turns a database into a graph mechanically: rows are nodes,
  primary–foreign-key links are edges. Lectures 13–16 (RelBench, RelGNN, RT).
- The paper's claim: this schema graph is often a poor graph for message passing.
  - *Information overload*: irrelevant tables and columns add noise.
  - *Semantic fragmentation*: related entities are many hops apart or unlinked.
- Its answer: learn the graph for the task, end to end, on an unchanged GNN.
- Why PyG: no PyG component learns at training time which features, node types or relation
  types to keep. Explainers only do it post hoc, per instance.
- FIGURE 1: rel-f1 schema graph → filtered graph → graph with injected edges.

## 2. The method (paper Eqs. 3–9, 14, 17–21)

- Base model: encoder → heterogeneous GraphSAGE → MLP head (RelBench's model).
- Step A, filtering: column gates g_T and table gates s_T (Hard-Concrete, L0 penalty).
- Step B, edge injection, 4 templates:
  - temporal continuity: each record → its previous m records of the same entity;
  - two-hop shortcuts: A → B → C becomes A → C;
  - KNN: each node → its n most similar nodes of the same table;
  - behavioural: nodes that interacted with similar rows.
- Step C, template gates: one Hard-Concrete switch per injected relation type.
- Step D, variational bottleneck per node type; loss = task + λ·L0 + β·KL (Eq. 21).
- At test time: a gate is open if σ(logit) > threshold.
- Connections to the course:
  - two-hop shortcuts ≈ RelGNN's atomic routes;
  - strictly-earlier sampling ≈ RT's rule;
  - per-relation weights as in R-GCN;
  - template gates ≈ InGram's relation importance.

## 3. Implementation in PyG (`torch_geometric.contrib`)

- Table: class → step → file.
  - `HardConcreteGate`: steps A and C.
  - `RelationalFilter`: step A.
  - `KNNEdges`, `BehavioralEdges`, `TwoHopEdges`: step B.
  - `TemporalContinuity` (transform): step B.
  - `GatedHeteroConv`: step C.
  - `VariationalBottleneck`: step D.
  - `StructuralOptimizer`: steps A–D, with every step behind a flag.
- Design choices:
  - Injected edges are built per batch, inside each seed's subgraph, so no rows from the
    future leak across seeds.
  - Closed relations are skipped, not multiplied by 0, so pruning saves compute.
  - Template convolutions have no root weight or bias.
  - With every flag off, the model is exactly the plain GNN (Test 4).
- CODE SNIPPET: building `StructuralOptimizer` on rel-f1, from
  `examples/contrib/structural_optimizer_relbench.py`.
- Tests:
  - Tests 1–5: L0 expectation, template edges, gated sum, all-off = plain GNN, and one
    rel-f1 epoch end to end.
  - `test/contrib`: 87 passed, 11 skipped. Experiment tests: 56 passed.
- TODO: CUDA, and GAT/GraphConv as the convolution.

## 4. Deviations from the authors' code

| | Paper | Authors' code | Ours |
|---|---|---|---|
| Column gate | g ∈ [0,1]^d on h⁽⁰⁾ ("feature dimensions") | per embedding channel | per raw column (per-channel: TODO option) |
| Gate temperature | one τ | annealed (2.0 → 0.3) | fixed 2/3 |
| Two-hop round trips A → B → A | — | included | excluded |
| KNN / behavioural tables | — | every table | the task's table (`--inject-tables all` optional) |
| Gate threshold | τ_g | best on validation of 6 | same (`--gate-thresholds`) |

- Speed: about 25 s per epoch for ours vs about 2 min for theirs (CPU, default config).
  Ours builds edges in one masked matrix operation per batch; theirs loops per seed on a
  single thread.

## 5. Reproducing the paper on rel-f1 driver-top3 (their protocol)

Test ROC-AUC, mean ± std over seeds.

| Model | Paper | Released weights | Their code, default config (1 seed) | Ours, default config (3 seeds) | Ours, released config (3 seeds) |
|---|---|---|---|---|---|
| Base | 81.48 ± 1.06 | – | 83.12 | 82.48 ± 5.75 | 90.22 ± 0.32 |
| Full model | 86.82 | 90.33 | 84.52 | 88.06 ± 0.83 | 89.90 ± 0.99 |

- "Released config" means the settings stored in the released checkpoint. They differ from
  the repo's defaults:
  - 1 layer, hidden 64, 16 neighbours, `last` sampling, lr 3e-3;
  - λ_w = 0.1, KNN k = 25.
  - λ_w, lr and k fall outside the paper's own search ranges (Table 8).
- Findings:
  1. Our implementation reaches the released weights' score (89.90 vs 90.33).
  2. Base with the released config does as well, so the configuration explains the jump from
     about 82 to 90. TODO: one-setting-at-a-time ablation (sampling, layers, neighbours, lr).
  3. The gates barely train, theirs and ours: they move about 0.03 from their initial value.
     Pruning comes from the validation-chosen threshold falling between nearly equal gates.
     - The released model at 0.86 keeps 3 tables (qualifying, results, standings), no
       injected edges, and no driver features.
     - Our seed 1 at 0.78 keeps the same 3 tables.
     - Our seeds 0 and 2 prune almost nothing, with the same score.
- FIGURE 2: gate values over epochs, with the thresholds. Data: `gates/*.json` and logs.
- FIGURE 3: bar chart of the table above.

## 6. The evaluation protocol (the main assumption)

- RelBench: `get_db(upto_test_timestamp=True)`, so test inputs stop at the test cutoff.
- Authors' code: `get_db(upto_test_timestamp=False)` plus strictly-earlier sampling.
- Audit (rel-f1 driver-top3):
  - No sampled row is newer than its seed, in any split.
  - There are no post-cutoff rows for train or val.
  - For test, 39% of sampled rows (55,858 of 142,093) are after the cutoff.
  - 642 of 726 test seeds receive the results rows that define an earlier test label of the
    same driver.
- Effect on test ROC-AUC:

| Test ROC-AUC | Their protocol | RelBench cutoff |
|---|---|---|
| Released weights | 90.33 | 83.98 |
| Their code, full model | 84.52 | 78.35 |
| Ours, full model (default config) | 88.06 ± 0.83 | 75.11 ± 2.6 |
| Ours, Base (default config) | 82.48 ± 5.75 | 78.42 ± 0.9 |

- It is a test-input protocol choice, not training-label leakage. We cannot tell whether the
  paper's own numbers used it.
- FIGURE 4: timeline of a test seed, its sampled rows, and the cutoff.

## 7. Other databases

- TODO: score the 8 other released checkpoints (2 rel-amazon, 2 rel-event, 2 rel-stack,
  rel-hm, rel-f1 driver-dnf), then run ours with each checkpoint's config.
- Measured so far, rel-f1 driver-dnf released weights: 80.41 (their protocol) vs 72.02
  (cutoff).

## 8. Takeaways and limitations

- TODO after Section 7.

## References

- Cheng & Luo, *What Makes a Desired Graph for Relational Deep Learning?*, ICML 2026.
- Fey et al., RDL; Robinson et al., RelBench; Louizos et al., L0 / Hard-Concrete;
  Alemi et al., VIB.
- TODO: RelGNN, RT, R-GCN, InGram.
