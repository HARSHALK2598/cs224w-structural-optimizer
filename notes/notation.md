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

---

## 4. Trainable parameters

### 4a. Column encoders $`\Theta_\mathrm{col}`$ (one per column $`c`$)

| Column type | Parameters | Size |
|---|---|---|
| numerical | $`\mathbf{w}_c, \mathbf{b}_c \in \mathbb{R}^{d}`$ | $`2d`$ |
| categorical | embedding table $`E_c \in \mathbb{R}^{(K_c + 1) \times d}`$; row 0 (missing) is fixed at zero | $`(K_c + 1)\, d`$ |
| text | $`A_c \in \mathbb{R}^{d \times 300}`$, $`\mathbf{b}_c \in \mathbb{R}^{d}`$ | $`301\, d`$ |
| timestamp | $`W_{c,k} \in \mathbb{R}^{d \times 8}`$ for each part $`k = 1..7`$, $`\mathbf{b}_c \in \mathbb{R}^{d}`$ | $`57\, d`$ |

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

(Each linear layer also has a bias; each LayerNorm has a scale and a shift in
$`\mathbb{R}^d`$.)

### 4c. Time encoder $`\Theta_\mathrm{time}`$ (one per table with a time column)

```math
M_T \in \mathbb{R}^{d \times d}, \qquad \mathbf{m}_T \in \mathbb{R}^{d}
```

The positional encoding $`\mathrm{PE}`$ itself is fixed sines and cosines, not learned.

### 4d. GNN $`\Theta_\mathrm{gnn}`$ (2 layers)

For each layer $`\ell = 1, 2`$ and each relation $`R`$ (one GraphSAGE convolution):

```math
W^{\mathrm{nbr}}_{\ell,R} \in \mathbb{R}^{d \times d}, \qquad \boldsymbol{\beta}_{\ell,R} \in \mathbb{R}^{d}, \qquad W^{\mathrm{self}}_{\ell,R} \in \mathbb{R}^{d \times d}
```

For each layer and each table: a LayerNorm with scale and shift in $`\mathbb{R}^d`$.

### 4e. Head $`\Theta_\mathrm{head}`$

One linear layer:

```math
\mathbf{w}_\mathrm{out} \in \mathbb{R}^{d}, \qquad b_\mathrm{out} \in \mathbb{R}
```

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

Categorical:

```math
\mathbf{e}_{u,c} = E_c[\,x_{u,c} + 1\,]
```

Text:

```math
\mathbf{e}_{u,c} = A_c\, \mathbf{q}_{u,c} + \mathbf{b}_c
```

Timestamp (part $`k`$ of the date is $`x_{u,c,k}`$; the year uses a positional
encoding of $`x_{u,c,1} - Y_c`$, the other parts a cyclic encoding):

```math
\mathbf{e}_{u,c} = \sum_{k=1}^{7} W_{c,k}\, \mathrm{enc}_k(x_{u,c,k}) + \mathbf{b}_c
```

Missing value (numerical, categorical or timestamp):

```math
\mathbf{e}_{u,c} = \mathbf{0}
```

**Step 2: stack and flatten.**

```math
X_u = [\,\mathbf{e}_{u,1};\ \dots;\ \mathbf{e}_{u,C_T}\,] \in \mathbb{R}^{C_T \times d}, \qquad \mathbf{a}^{(0)}_u = \mathrm{vec}(X_u) \in \mathbb{R}^{C_T d}
```

**Step 3: ResNet.** Each block, for $`\ell = 1..4`$ (with dropout 0.2 after each
ReLU during training):

```math
\mathbf{z} = \mathrm{ReLU}\big(\mathrm{LN}(U^{(\ell)}_1 \mathbf{a}^{(\ell-1)}_u)\big)
```

```math
\mathbf{a}^{(\ell)}_u = \mathrm{ReLU}\big(\mathrm{LN}(U^{(\ell)}_2 \mathbf{z})\big) + S_\ell(\mathbf{a}^{(\ell-1)}_u)
```

The shortcut is $`S_1(\mathbf{a}) = P\mathbf{a}`$ for block 1 and the identity
$`S_\ell(\mathbf{a}) = \mathbf{a}`$ for blocks 2–4. Then the decoder:

```math
\mathbf{r}_u = D\, \mathrm{ReLU}\big(\mathrm{LN}(\mathbf{a}^{(4)}_u)\big)
```

**Step 4: add the time encoding.** For tables with a time column:

```math
\mathbf{h}^{(0)}_u = \mathbf{r}_u + M_{\phi(u)}\, \mathrm{PE}(\Delta t_u) + \mathbf{m}_{\phi(u)}
```

For tables without one (drivers, constructors, circuits):

```math
\mathbf{h}^{(0)}_u = \mathbf{r}_u
```

**Step 5: two GraphSAGE layers.** For $`\ell = 1, 2`$, every node $`v`$ sums its
neighbours along each incoming relation:

```math
\mathbf{m}_{v,R} = \sum_{u \in \mathcal{N}_R(v)} \mathbf{h}^{(\ell-1)}_u
```

The relations are combined by summing, each with its own weights:

```math
\mathbf{s}_v = \sum_{R \,:\, T_\mathrm{dst} = \phi(v)} \Big( W^{\mathrm{nbr}}_{\ell,R}\, \mathbf{m}_{v,R} + \boldsymbol{\beta}_{\ell,R} + W^{\mathrm{self}}_{\ell,R}\, \mathbf{h}^{(\ell-1)}_v \Big)
```

```math
\mathbf{h}^{(\ell)}_v = \mathrm{ReLU}\big(\mathrm{LN}_{\ell,\phi(v)}(\mathbf{s}_v)\big)
```

**Step 6: head and loss.** Only the seed nodes are read out:

```math
\hat{y}_i = \sigma\big(\mathbf{w}_\mathrm{out}^\top \mathbf{h}^{(2)}_{v^\ast_i} + b_\mathrm{out}\big)
```

```math
\mathcal{L}_\mathrm{task} = -\frac{1}{B} \sum_{i=1}^{B} \Big[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \Big]
```

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
