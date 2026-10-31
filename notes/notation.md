# Notation: the Base model, from database to prediction

One notation for the whole pipeline, used by the plans, the notebooks and the blog.
Everything here describes the unmodified RelBench model (`train.BaseModel`) on
`rel-f1` / `driver-top3`. Numbers in brackets are this dataset's values.

Conventions: $`d = 128`$ is the hidden size. Bold lower case is a vector, upper case a
matrix. $`[\cdot\,;\,\cdot]`$ stacks, $`\mathrm{vec}`$ flattens row by row.

---

## 1. Sets and indices

| Symbol | Meaning | `rel-f1` |
|---|---|---|
| $`\mathcal{T}`$ | tables (= node types) | 9 |
| $`V_T`$ | rows of table $`T`$ (= nodes of type $`T`$) | 77 (circuits) … 28,115 (standings) |
| $`\phi(v)`$ | the table that node $`v`$ belongs to | |
| $`\mathcal{C}_T`$, $`C_T = \lvert\mathcal{C}_T\rvert`$ | feature columns of $`T`$ (not keys, not the time column) | 45 in total; $`C_\mathrm{results} = 11`$ |
| $`t_v`$ | timestamp of row $`v`$ (only for tables with a time column) | 6 of 9 tables |
| $`\mathcal{R}`$ | relations: one per foreign key and direction, $`R = (T_\mathrm{src} \to T_\mathrm{dst})`$ | 13 keys × 2 = 26 |
| $`\mathcal{N}_R(v)`$ | neighbours of $`v`$ along relation $`R`$ | |
| $`(v^\ast_i, t^\ast_i, y_i)`$ | training example $`i`$: seed driver, prediction time, label | 1,353 training examples |

---

## 2. Preprocessing (once, before training; nothing learned)

**Graph.** Each row becomes a node. Each foreign-key value becomes two edges, one in
each direction. Primary and foreign keys are then dropped from the features.

**Column statistics** (constants, computed once per column $`c`$):

| Column type | Constants |
|---|---|
| numerical | mean $`\mu_c`$, std $`\sigma_c`$ |
| categorical | number of categories $`K_c`$ |
| timestamp | earliest year $`Y_c`$ |
| text | none |

**Stored inputs** (the TensorFrame). Each cell becomes one stored value $`x_{v,c}`$:

| Column type | Stored $`x_{v,c}`$ | Missing value |
|---|---|---|
| numerical | a float | NaN |
| categorical | an index in $`\{0, \dots, K_c - 1\}`$ | −1 |
| timestamp | 7 integers: year, month, day, weekday, hour, minute, second | NaN |
| text | $`\mathbf{q}_{v,c} = \mathrm{GloVe}(\text{text}) \in \mathbb{R}^{300}`$ (frozen GloVe) | empty string |

These stored values never change during training.

---

## 3. Mini-batch (every step; nothing learned)

Take $`B = 512`$ training examples. For each example $`i`$, sample a subgraph around
$`v^\ast_i`$:

- up to 2 hops ($`L = 2`$);
- at most 128 neighbours per relation at hop 1, 64 at hop 2;
- only rows with $`t_u \le t^\ast_i`$ (no information from the future).

Each example gets its **own copy** of every row it reaches (disjoint sampling). For a
sampled node $`u`$, $`b(u) = i`$ records which example it belongs to, and its age is

```math
\Delta t_u = \frac{t^\ast_{b(u)} - t_u}{86400} \quad \text{(days)}
```

The batch holds, per table, the sliced stored values $`x_{u,c}`$, the edges, $`t_u`$
and $`b(u)`$.

**The running example:** the first training batch of `driver-top3` (RelBench defaults:
$`B = 512`$, fan-out [128, 64], $`d = 128`$). Measured:

| Table | Rows in batch $`N_T`$ | Distinct rows | $`C_T`$ | Flattened $`C_T \cdot d`$ |
|---|---|---|---|---|
| drivers (the 512 seeds) | 512 | 82 | 6 | 768 |
| results | 28,230 | 5,051 | 11 | 1,408 |
| standings | 28,229 | 5,079 | 4 | 512 |
| qualifying | 11,422 | 1,820 | 3 | 384 |
| races | 31,548 | 392 | 5 | 640 |
| constructors | 1,268 | 39 | 3 | 384 |
| circuits, constructor_results, constructor_standings | 0 (3 hops away) | 0 | | |
| **Total** | **101,209** | | | |

Edges: 243,256 in 11 of the 26 relations; for example, results → drivers has 28,230
edges (about 55 `results` rows per seed).

---

## 4. Trainable parameters

### 4a. Column encoders $`\Theta_\mathrm{col}`$ (one per column $`c`$)

| Column type | Parameters | Size |
|---|---|---|
| numerical | $`\mathbf{w}_c, \mathbf{b}_c \in \mathbb{R}^{d}`$ | $`2d`$ |
| categorical | embedding table $`E_c \in \mathbb{R}^{(K_c + 1) \times d}`$; row 0 (missing) is fixed at zero | $`(K_c + 1)\, d`$ |
| text | $`A_c \in \mathbb{R}^{d \times 300}`$, $`\mathbf{b}_c \in \mathbb{R}^{d}`$ | $`301\, d`$ |
| timestamp | $`W_{c,k} \in \mathbb{R}^{d \times 8}`$ for each part $`k = 1..7`$, $`\mathbf{b}_c \in \mathbb{R}^{d}`$ | $`57\, d`$ |

↳ *Example:* `results` (10 numerical + 1 timestamp): 10 × 256 + 7 × 8 × 128 + 128 = 9,856 parameters.

### 4b. Row encoder $`\Theta_\mathrm{row}`$ (one ResNet per table $`T`$)

Four residual blocks $`\ell = 1..4`$. Block $`\ell`$ has two linear layers, each with a
LayerNorm:

```math
U^{(\ell)}_1 \in \mathbb{R}^{d \times n_\ell}, \qquad U^{(\ell)}_2 \in \mathbb{R}^{d \times d}, \qquad n_1 = C_T\, d, \quad n_{2,3,4} = d
```

Block 1 also has a linear shortcut, because its input is wider than $`d`$:

```math
P \in \mathbb{R}^{d \times C_T d}
```

The decoder is a LayerNorm and one more linear layer:

```math
D \in \mathbb{R}^{d \times d}
```

↳ *Example:* `results`: $`U^{(1)}_1, P`$ are 128 × 1408; the other 7 $`U`$'s and $`D`$ are 128 × 128; 495,104 parameters in all.

(Each linear layer also has a bias; each LayerNorm has a scale and a shift in
$`\mathbb{R}^d`$.)

### 4c. Time encoder $`\Theta_\mathrm{time}`$ (one per table with a time column)

```math
M_T \in \mathbb{R}^{d \times d}, \qquad \mathbf{m}_T \in \mathbb{R}^{d}
```

↳ *Example:* 128 × 128 + 128 = 16,512 per table; × 6 tables = 99,072.

The positional encoding $`\mathrm{PE}`$ itself is fixed sines and cosines, not learned.

### 4d. GNN $`\Theta_\mathrm{gnn}`$ (2 layers)

For each layer $`\ell = 1, 2`$ and each relation $`R`$ (one GraphSAGE convolution):

```math
W^{\mathrm{nbr}}_{\ell,R} \in \mathbb{R}^{d \times d}, \qquad \boldsymbol{\beta}_{\ell,R} \in \mathbb{R}^{d}, \qquad W^{\mathrm{self}}_{\ell,R} \in \mathbb{R}^{d \times d}
```

↳ *Example:* (128 × 128 + 128 + 128 × 128) = 32,896 per relation; × 26 relations × 2 layers = 1,710,592.

For each layer and each table: a LayerNorm with scale and shift in $`\mathbb{R}^d`$.

### 4e. Head $`\Theta_\mathrm{head}`$

One linear layer:

```math
\mathbf{w}_\mathrm{out} \in \mathbb{R}^{d}, \qquad b_\mathrm{out} \in \mathbb{R}
```

↳ *Example:* 128 + 1 = 129 parameters.

### Parameter count (measured on the model)

| Part | Parameters | Share |
|---|---|---|
| Column encoders, all tables | 576,384 | 11% |
| Row encoders (ResNets), all tables | 2,686,464 | 53% |
| Time encoders (6 tables) | 99,072 | 2% |
| GNN (2 layers × 26 relations, plus LayerNorms) | 1,715,200 | 34% |
| Head | 129 | 0% |
| **Total** | **5,077,249** | |

The biggest single ResNet is `results` (495,104), because its first layer and shortcut
take $`11 \times 128 = 1408`$ inputs.

---

## 5. Forward pass (one training step)

Steps 1–4 run for every sampled node $`u`$; steps 5–6 use the graph.

**Step 1: column encoders.** Each stored value becomes a vector
$`\mathbf{e}_{u,c} \in \mathbb{R}^d`$.

Numerical:

```math
\mathbf{e}_{u,c} = \frac{x_{u,c} - \mu_c}{\sigma_c}\, \mathbf{w}_c + \mathbf{b}_c
```

↳ *Example:* `results.grid`: $`x`$ is `[28230]`, $`\mathbf{w}_c, \mathbf{b}_c`$ are `[128]` → $`\mathbf{e}`$ is `[28230, 128]`; all 10 numerical columns → `[28230, 10, 128]`.

Categorical:

```math
\mathbf{e}_{u,c} = E_c[\,x_{u,c} + 1\,]
```

↳ *Example:* `races.year` ($`K_c = 60`$): $`E_c`$ is `[61, 128]`, $`x`$ is `[31548]` → $`\mathbf{e}`$ is `[31548, 128]`.

Text:

```math
\mathbf{e}_{u,c} = A_c\, \mathbf{q}_{u,c} + \mathbf{b}_c
```

↳ *Example:* `drivers.surname`: $`\mathbf{q}`$ is `[512, 300]`, $`A_c`$ is `[128, 300]` → `[512, 128]`; all 5 text columns (stored as `[512, 1500]`) → `[512, 5, 128]`.

Timestamp (part $`k`$ of the date is $`x_{u,c,k}`$; the year uses a positional
encoding of $`x_{u,c,1} - Y_c`$, the other parts a cyclic encoding):

```math
\mathbf{e}_{u,c} = \sum_{k=1}^{7} W_{c,k}\, \mathrm{enc}_k(x_{u,c,k}) + \mathbf{b}_c
```

↳ *Example:* `results.date`: $`x`$ is `[28230, 7]`, encodings are `[28230, 7, 8]`, each $`W_{c,k}`$ is `[128, 8]`, summed over 7 parts → `[28230, 128]`.

Missing value (numerical, categorical or timestamp):

```math
\mathbf{e}_{u,c} = \mathbf{0}
```

↳ *Example:* row 0 of `results` has NaN `milliseconds`, so its vector is 128 zeros.

**Step 2: stack and flatten.**

```math
X_u = [\,\mathbf{e}_{u,1};\ \dots;\ \mathbf{e}_{u,C_T}\,] \in \mathbb{R}^{C_T \times d}, \qquad \mathbf{a}^{(0)}_u = \mathrm{vec}(X_u) \in \mathbb{R}^{C_T d}
```

↳ *Example:* `results`: 11 vectors of `[28230, 128]` → $`X`$ is `[28230, 11, 128]` → $`\mathbf{a}^{(0)}`$ is `[28230, 1408]`.

**Step 3: ResNet.** Each block, for $`\ell = 1..4`$ (with dropout 0.2 after each
ReLU during training):

```math
\mathbf{z} = \mathrm{ReLU}\big(\mathrm{LN}(U^{(\ell)}_1 \mathbf{a}^{(\ell-1)}_u)\big)
```

↳ *Example:* block 1, `results`: `[28230, 1408]` × $`U^{(1)\top}_1`$ `[1408, 128]` → $`\mathbf{z}`$ is `[28230, 128]`.

```math
\mathbf{a}^{(\ell)}_u = \mathrm{ReLU}\big(\mathrm{LN}(U^{(\ell)}_2 \mathbf{z})\big) + S_\ell(\mathbf{a}^{(\ell-1)}_u)
```

↳ *Example:* block 1: `[28230, 128]` × `[128, 128]` plus shortcut `[28230, 1408]` × `[1408, 128]` → `[28230, 128]`; blocks 2–4 keep `[28230, 128]`.

The shortcut is $`S_1(\mathbf{a}) = P\mathbf{a}`$ for block 1 and the identity
$`S_\ell(\mathbf{a}) = \mathbf{a}`$ for blocks 2–4. Then the decoder:

```math
\mathbf{r}_u = D\, \mathrm{ReLU}\big(\mathrm{LN}(\mathbf{a}^{(4)}_u)\big)
```

↳ *Example:* `[28230, 128]` × $`D^\top`$ `[128, 128]` → $`\mathbf{r}`$ is `[28230, 128]`. With `--dedup-encoder`, steps 1–3 run on the 5,051 distinct rows (`[5051, 1408]` → `[5051, 128]`), then copy back to `[28230, 128]`.

**Step 4: add the time encoding.** For tables with a time column:

```math
\mathbf{h}^{(0)}_u = \mathbf{r}_u + M_{\phi(u)}\, \mathrm{PE}(\Delta t_u) + \mathbf{m}_{\phi(u)}
```

↳ *Example:* `results`: $`\Delta t`$ is `[28230]` → PE `[28230, 128]` → × $`M^\top`$ `[128, 128]` → `[28230, 128]`, added to $`\mathbf{r}`$ → $`\mathbf{h}^{(0)}`$ is `[28230, 128]`.

For tables without one (drivers, constructors, circuits):

```math
\mathbf{h}^{(0)}_u = \mathbf{r}_u
```

↳ *Example:* `drivers`: $`\mathbf{h}^{(0)} = \mathbf{r}`$ is `[512, 128]`. All tables together: 101,209 rows of 128.

**Step 5: two GraphSAGE layers.** For $`\ell = 1, 2`$, every node $`v`$ sums its
neighbours along each incoming relation:

```math
\mathbf{m}_{v,R} = \sum_{u \in \mathcal{N}_R(v)} \mathbf{h}^{(\ell-1)}_u
```

↳ *Example:* layer 1, results → drivers: 28,230 edges sum `[28230, 128]` into `[512, 128]` (as a matrix: `[512, 28230]` × `[28230, 128]`).

The relations are combined by summing, each with its own weights:

```math
\mathbf{s}_v = \sum_{R \,:\, T_\mathrm{dst} = \phi(v)} \Big( W^{\mathrm{nbr}}_{\ell,R}\, \mathbf{m}_{v,R} + \boldsymbol{\beta}_{\ell,R} + W^{\mathrm{self}}_{\ell,R}\, \mathbf{h}^{(\ell-1)}_v \Big)
```

↳ *Example:* drivers have 3 incoming relations (results, standings, qualifying): 3 × (`[512, 128]` × `[128, 128]` twice) summed → `[512, 128]`.

```math
\mathbf{h}^{(\ell)}_v = \mathrm{ReLU}\big(\mathrm{LN}_{\ell,\phi(v)}(\mathbf{s}_v)\big)
```

↳ *Example:* $`\mathbf{h}^{(1)}`$: drivers `[512, 128]`, results `[28230, 128]`, … (every table); layer 2 repeats this → $`\mathbf{h}^{(2)}`$ drivers `[512, 128]`.

**Step 6: head and loss.** Only the seed nodes are read out:

```math
\hat{y}_i = \sigma\big(\mathbf{w}_\mathrm{out}^\top \mathbf{h}^{(2)}_{v^\ast_i} + b_\mathrm{out}\big)
```

↳ *Example:* `[512, 128]` × $`\mathbf{w}_\mathrm{out}`$ `[128]` + $`b_\mathrm{out}`$ → `[512]` (the code returns `[512, 1]`), one probability per seed.

```math
\mathcal{L}_\mathrm{task} = -\frac{1}{B} \sum_{i=1}^{B} \Big[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \Big]
```

↳ *Example:* 512 predictions and 512 labels → one number. 1,353 training examples → 3 steps per epoch (512, 512, 329 seeds).

Backpropagation computes the gradient of $`\mathcal{L}_\mathrm{task}`$ for all of
$`\Theta_\mathrm{col}, \Theta_\mathrm{row}, \Theta_\mathrm{time}, \Theta_\mathrm{gnn}, \Theta_\mathrm{head}`$,
and Adam updates them all. The stored inputs from §2 are never updated.

---

## 6. Two consequences, in this notation

**Why the encoder runs every step.** $`\mathbf{r}_u`$ depends on $`\Theta_\mathrm{col}`$
and $`\Theta_\mathrm{row}`$, which change after every Adam step. So $`\mathbf{r}_u`$
must be recomputed each step. Only the inputs $`x_{u,c}`$ and $`\mathbf{q}_{u,c}`$ are
computed once.

**Why `--dedup-encoder` is exact.** Steps 1–3 depend only on the stored row, not on
which example the copy belongs to. So all copies of a row have the same
$`\mathbf{r}_u`$, and it is enough to compute it once per distinct row. Step 4 differs
between copies (through $`\Delta t_u`$) and is still done per copy.

---

## 7. Where the planned changes enter (Plan 01)

| Change | Where | In this notation |
|---|---|---|
| Column gate $`g_{T,c}`$ | between steps 1 and 2 | $`\tilde{\mathbf{e}}_{u,c} = g_{T,c}\, \mathbf{e}_{u,c}`$ |
| Table gate $`s_T`$ | after step 4 | $`\tilde{\mathbf{h}}^{(0)}_u = s_{\phi(u)}\, \mathbf{h}^{(0)}_u`$ |
| Loss | step 6 | $`\mathcal{L} = \mathcal{L}_\mathrm{task} + \lambda \sum_{T,c} \mathbb{E}\lVert g_{T,c}\rVert_0 + \lambda_s \sum_T \mathbb{E}\lVert s_T \rVert_0`$ |
| New parameters | $`\Theta_\mathrm{gate}`$ | one logit per gate: 45 + 8 = 53 |
