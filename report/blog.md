# Learning the Graph for Relational Deep Learning, in PyG

<!-- Draft of the CS224W blog post (Medium). TODO marks open items. -->

*Harshal Kulkarni, CS224W project. Code:
[PyG `contrib` PR](https://github.com/HARSHALK2598/pytorch_geometric/tree/contrib/structural-optimizer),
[experiments](https://github.com/HARSHALK2598/cs224w-structural-optimizer).*

## Names used in this post

| Name | Meaning |
|---|---|
| **GraphSAGE baseline** | RelBench's standard model: an encoder for each table's columns, a heterogeneous GraphSAGE GNN over the database graph, and an MLP head. Nothing is learned about the graph itself. |
| **Structural optimizer** | The paper's method: the GraphSAGE baseline plus four additions (Section 2). |
| **Paper's code** | The authors' GitHub repository, commit 030b07c. |
| **Released model** | The trained driver-top3 model file the authors published. |
| **Repo settings** | The default training settings in the paper's code: 2 GNN layers, hidden size 128, 64 sampled neighbours, learning rate 1e-3. |
| **Checkpoint settings** | The settings saved inside the released model file: 1 GNN layer, hidden size 64, 16 neighbours, learning rate 3e-3, a 100× larger column-gate penalty. |
| **Tuned settings** | Settings we chose on validation (Section 4): 1 GNN layer and, for the structural optimizer, switch penalty 1e-4; otherwise the repo settings. |
| **Full-history test** | The paper's test setup: a prediction at time *t* may use every database row before *t*. |
| **RelBench cutoff test** | RelBench's standard setup: every test prediction uses only rows before the start of the test period. |

**Main assumption.** Unless marked otherwise, every number uses the full-history test,
because that is how the paper's code evaluates. Section 5 shows why this matters.

## 1. Motivation

- Relational deep learning turns a database into a graph mechanically: rows become nodes,
  and foreign-key links become edges (lectures 13–16).
- The paper argues this graph is often a poor fit for message passing:
  - irrelevant tables and columns add noise;
  - related rows can be many hops apart.
- Its answer is to learn the graph for each task, end to end.
- No PyG component does this at training time. Explainers learn masks only after training,
  one prediction at a time.
- FIGURE 1: the rel-f1 database graph, the same graph after filtering, and after adding
  edges.

## 2. The structural optimizer in four additions

1. **Table and column switches.** Each table and each column gets a learned on/off switch
   (a Hard-Concrete gate). A penalty pushes switches off unless they help.
2. **Added edges**, of four kinds:
   - *temporal*: each record links to the same entity's previous records;
   - *two-hop shortcut*: A → B → C becomes A → C;
   - *KNN*: each driver links to its most similar drivers;
   - *behavioural*: drivers link when their linked rows look alike.
3. **Edge-type switches.** One on/off switch per added edge type, with its own penalty.
4. **Bottleneck.** Each node's embedding is compressed with a variational information
   bottleneck.

At test time, a switch counts as on if its learned probability is above a threshold. The
paper's code tries six thresholds (0.5 to 0.9) and keeps the one that scores best on
validation.

Course connections:
- two-hop shortcuts ≈ RelGNN's atomic routes;
- the rule that a prediction sees only earlier rows ≈ RT's sampling;
- edge-type switches ≈ R-GCN's per-relation weights and InGram's relation importance.

## 3. Implementation in PyG (`torch_geometric.contrib`)

| Class | Addition |
|---|---|
| `HardConcreteGate` | the switches (1 and 3) |
| `RelationalFilter` | table and column switches (1) |
| `KNNEdges`, `BehavioralEdges`, `TwoHopEdges`, `TemporalContinuity` | added edges (2) |
| `GatedHeteroConv` | message passing with edge-type switches (3) |
| `VariationalBottleneck` | bottleneck (4) |
| `StructuralOptimizer` | everything, each addition behind its own flag |

- With every flag off, `StructuralOptimizer` is exactly the GraphSAGE baseline (Test 4).
- A switched-off edge type is skipped entirely, so pruning saves compute.
- `test/contrib`: 87 passed, 11 skipped. Experiment tests: 56 passed.
- CODE SNIPPET: `examples/contrib/structural_optimizer_relbench.py`.
- TODO: CUDA, and GAT/GraphConv as the convolution.

## 4. Results on rel-f1 driver-top3, and why

Test ROC-AUC, mean ± std over 3 seeds unless marked.

| Model | Repo settings | Checkpoint settings | Tuned settings | Paper |
|---|---|---|---|---|
| GraphSAGE baseline, our PyG implementation | 82.48 ± 5.75 | 90.22 ± 0.32 | **87.86 ± 0.83** | 81.48 ± 1.06 |
| Structural optimizer, our PyG implementation | 88.06 ± 0.83 | 89.90 ± 0.99 | **88.48 ± 1.05** | 86.82 |
| GraphSAGE baseline, paper's code (1 seed) | 83.12 | – | – | |
| Structural optimizer, paper's code (1 seed) | 84.52 | – | – | |
| Released model | – | 90.33 | – | |

Our implementation reproduces the released model (89.90 vs 90.33). It also beats the
paper's 86.82. But once both models are tuned, the structural optimizer is only 0.6 points
above the baseline, within one standard deviation. The rest of this section explains each
part of the table from how the method works.

### 4.1 One GNN layer is what matters

| GraphSAGE baseline | Validation | Test |
|---|---|---|
| 2 layers (repo settings) | 79.60 ± 0.57 | 82.48 ± 5.75 |
| 1 layer (tuned settings) | 81.63 ± 0.30 | 87.86 ± 0.83 |

- The task asks whether a driver finishes in the top 3 next month. The best evidence is the
  driver's own recent results, standings and qualifying rows, all one hop from the driver.
- A second layer adds what those rows link to: races, circuits, constructors. That makes
  each subgraph much larger and adds weakly related information. With 1,353 training rows,
  the model overfits it, so test scores swing between 78 and 89 across seeds.
- Most of the jump from the repo settings to the checkpoint settings is this one change.
  Sampling the most recent neighbours instead of random ones barely mattered (81.03 vs 81.63
  validation).

### 4.2 The switches can barely move, so they cannot learn to prune

- A training run is only 300 optimizer steps: 1,353 rows at batch 256 is 6 steps per
  epoch, times 50 epochs.
- Adam moves each weight by at most about the learning rate per step.
- A switch starts at logit 2.0, a probability of 0.88. At learning rate 1e-3 it can move at
  most 0.3 in logit, so it can't fall below 0.846. To switch off at threshold 0.5 it would
  need logit 0, about 2,000 steps.
- Measured:
  - With tuned settings, every switch ended between 0.848 and 0.878. All stayed on at every
    penalty we tried.
  - With checkpoint settings, learning rate 3e-3 allows a floor of 0.75. The lowest switch
    reached 0.753.
- So pruning, when it happens, comes from the threshold search. The penalty pushes unhelpful
  switches down by a few hundredths. A threshold such as 0.86 then lands in that small gap
  and switches them off.
  - The released model at 0.86 keeps only the qualifying, results and standings tables, and
    no added edges.
  - Our seed 1 with checkpoint settings picked 0.78 and kept the same three tables.
  - Our seeds 0 and 2 picked thresholds that kept almost everything, and scored the same.
- The tables that survive are exactly the one-hop tables from 4.1. The switches and the
  layer result point to the same signal, but the switches are too weak to find it
  reliably.

### 4.3 Why the structural optimizer is close to the baseline

- With every switch on, the structural optimizer is the baseline plus added edges plus the
  bottleneck.
- With one GNN layer, only edges pointing into the driver can change its prediction:
  - temporal edges connect results to results, and that information would need a second
    layer to reach the driver;
  - two-hop shortcuts are built from sampled paths of length two, and a one-layer subgraph
    has none.
- So only KNN and behavioural driver-to-driver edges are active. We counted every edge in
  every test batch: all 64,480 added edges were of these two kinds.
- Those edges carry other drivers' encoded columns, such as nationality and date of birth.
  That says little about next month's result. The bottleneck weight (1e-3) is too small to
  change much.
- With two layers (repo settings), the structural optimizer did beat the baseline, 88.06 vs
  82.48. But that baseline's test scores spread from 78 to 89, so the gap is mostly the
  baseline's instability, not a gain from the added structure.

### 4.4 What we did not tune

The checkpoint settings also change the learning rate (3e-3), hidden size (64), neighbours
(16) and the penalties. We tuned only the number of layers, the sampling order and one
shared penalty, so the remaining 2 points between our tuned baseline (87.86) and the
checkpoint settings (90.22) come from these untuned settings. Several checkpoint values
fall outside the paper's own search ranges (its Table 8): column penalty 0.1, learning rate
3e-3, KNN k 25.

## 5. The test setup is the main assumption

- Test predictions are monthly, from 2010-03 to 2013-03. The 726 test rows cover the same
  42 drivers about 17 times each.
- **Full-history test:** a prediction in 2012 sees every result before 2012. That includes
  the races that decide that driver's earlier test labels.
- **RelBench cutoff test:** a prediction in 2012 sees only rows before 2010.
- Every prediction uses only data from before its own date, so no prediction sees its own
  answer. We checked:
  - no sampled row is newer than its prediction;
  - no added edge brings in a later prediction's subgraph.
- But under the full-history test, later predictions know recent form. Section 4.1 shows
  recent form is exactly the signal that matters, so this is an easier task than RelBench's,
  and its scores are not comparable with RelBench numbers.

| Test ROC-AUC (repo settings unless marked) | Full-history test | RelBench cutoff test |
|---|---|---|
| Released model (checkpoint settings) | 90.33 | 83.98 |
| Structural optimizer, paper's code (1 seed) | 84.52 | 78.35 |
| Structural optimizer, our implementation | 88.06 ± 0.83 | 75.11 ± 2.62 |
| GraphSAGE baseline, our implementation | 82.48 ± 5.75 | 78.42 ± 0.89 |

- Under the cutoff, the structural optimizer falls below the baseline. A likely reason: its
  extra parameters fit recent-form signals that are stale by up to three years under the
  cutoff. We did not test this further.
- FIGURE 2: one driver's test predictions on a timeline, with the rows each prediction can
  see under the two setups.

## 6. Differences from the paper's code

| | Paper's text | Paper's code | Our PyG implementation |
|---|---|---|---|
| Column switch | one per "feature dimension" | one per embedding channel | one per database column |
| Switch temperature | one value | lowered during training | fixed at 2/3 |
| Two-hop round trips A → B → A | not stated | included | excluded |
| KNN and behavioural edges | not stated | within each prediction's own subgraph | also from other predictions at the same or an earlier date |
| Column statistics, full-history test | not stated | rows before the first training date | all rows |
| Speed, CPU, repo settings | – | about 2 min per epoch | about 25 s per epoch |

Our per-column switches can say which database columns matter. Per-channel switches
cannot, because each embedding channel mixes all of a table's columns.

## 7. Other databases

- TODO: score the 8 other released models, and run our implementation on each.
- So far, the rel-f1 driver-dnf released model scores 80.41 (full-history test) vs 72.02
  (RelBench cutoff test).

## 8. Takeaways

1. Our PyG implementation reproduces the released model on driver-top3, given the same
   settings and test setup.
2. On this task, the gains come from the training settings, mainly using one GNN layer, and
   from the full-history test. The four additions add little over a tuned baseline.
3. The switches cannot learn to prune within 300 training steps. Their pruning comes from a
   threshold search over nearly equal values, so it changes from seed to seed.
4. TODO after Section 7: whether this holds on other databases.

## References

- Cheng & Luo, *What Makes a Desired Graph for Relational Deep Learning?*, ICML 2026.
- Fey et al., Relational Deep Learning; Robinson et al., RelBench; Louizos et al., L0
  regularisation (Hard-Concrete); Alemi et al., Deep Variational Information Bottleneck.
- TODO: RelGNN, RT, R-GCN, InGram.
