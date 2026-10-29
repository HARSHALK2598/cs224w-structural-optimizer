# Profiling the Base model, and encoder deduplication

Where does training time go for RelBench's relational GraphSAGE, and how can it be cut
without changing the model? Measured on `rel-f1` / `driver-dnf` (11,411 training
targets, 23 batches of 512 per epoch), 4 CPU threads.

```bash
python experiments/profiling/profile_step.py --task driver-dnf                  # as RelBench
python experiments/profiling/profile_step.py --task driver-dnf --dedup-encoder  # deduplicated
```
Raw outputs: [`results/driver-dnf_dedup-off.txt`](results/driver-dnf_dedup-off.txt),
[`results/driver-dnf_dedup-on.txt`](results/driver-dnf_dedup-on.txt).

## 1. Where the time goes

| Stage of a training step | Time per epoch | Share |
|---|---|---|
| Sample mini-batch (`NeighborLoader`) | 0.5 s | 1% |
| **Row encoder** (one torch-frame ResNet per table) | **28.3 s** | **53%** |
| Time encoding | 0.5 s | 1% |
| GNN (2 GraphSAGE layers) | 6.0 s | 11% |
| Head + loss | 0.0 s | 0% |
| **Backward pass** | **17.6 s** | **33%** |
| Adam step | 0.4 s | 1% |
| **Epoch** | **53.4 s** | |

Sampling is negligible; the cost is dense matrix multiplication (`aten::mm` and
`aten::addmm` are over half of all operator time), mostly in the row encoder and its
backward pass. On a GPU this part would speed up the most.

## 2. Why the row encoder is expensive: duplicated rows

Each target gets its own copy of every row it samples (temporal sampling is
disjoint), so popular rows are encoded many times per batch:

| Table | Sampled per batch | Distinct rows | Copies per row |
|---|---|---|---|
| races | 23,799 | 727 | 32.7× |
| constructors | 1,383 | 119 | 11.6× |
| results | 20,529 | 9,458 | 2.2× |
| standings | 22,597 | 10,691 | 2.1× |
| **All tables** | **~71,000** | **~22,600** | **3.1×** |

## 3. Encoder deduplication (`--dedup-encoder`, off by default)

`BaseModel.encode_rows` encodes each distinct row of a batch once and copies its
embedding to every position (`n_id.unique(return_inverse=True)`, then indexing).

- **Forward:** identical results; the time encoding is still applied per copy (it
  depends on each target's prediction time).
- **Backward:** the indexing step sums the gradients of all copies into the one
  encoded row, which is exactly the gradient the copies would have produced
  separately. Measured: max. relative difference 7×10⁻⁷ (float32), 10⁻¹⁵ (float64).
- **Only difference: dropout.** The encoder uses dropout 0.2; with deduplication all
  copies of a row share one dropout mask in training. Evaluation is unaffected.
- The encoder uses LayerNorm (per row), not BatchNorm, so batch statistics are not
  affected.

Tests in `experiments/tests/test_dedup_encoder.py`: each distinct row is encoded
once; outputs are identical in evaluation mode; gradients are identical with dropout
off.

**Speed (one epoch, `driver-dnf`):**

| | Off | On | Speed-up |
|---|---|---|---|
| Row encoder | 28.3 s | 8.6 s | 3.3× |
| Backward pass | 17.6 s | 8.4 s | 2.1× |
| **Epoch** | **53.4 s** | **24.3 s** | **2.2×** |

**Still to verify:** that full training with deduplication reaches the same scores
as without (shared dropout masks could change training slightly). To be done after
the baseline runs.

## 4. Sampling randomness

Fan-out is 128 per relation at hop 1 and 64 at hop 2. For each training target, the
number of eligible rows (older than the prediction time) per relation into `drivers`:

| Relation | Eligible rows (median / max) | Targets sampled randomly (> 128 eligible) |
|---|---|---|
| qualifying | 0 / 98 | 0% |
| standings | 33 / 259 | 7.8% |
| results | 27 / 256 | 6.8% |

At hop 2 every row links to exactly one race / driver / constructor, so nothing is
sampled. Over 90% of targets therefore see the same subgraph in every epoch.

## 5. Found along the way: numerical columns with missing values stop learning

PyTorch Frame (0.3.0, the latest release) encodes a numerical value as
`x · w + b` and replaces a NaN result with 0. In the backward pass the weight
gradient is `x × (incoming gradient)`; for a missing value that is `NaN × 0 = NaN`,
so the column's whole weight row gets a NaN gradient. Adam then writes NaN into the
weights, and from then on the column encodes to 0 for **every** row.

On `driver-dnf`, 5 of the database's 22 numerical columns are affected, all in
`results`: `fastestLap` (90% missing), `rank` (90%), `milliseconds` (79%),
`position` (49%) after the first step, and `number` during the first epoch.

This happens in the unmodified library, with or without deduplication, so the
official RelBench runs are affected too. It is not changed here.
