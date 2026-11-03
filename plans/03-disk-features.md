# Plan 03: keep table features on disk, for datasets larger than memory

Plan 02's dry runs showed that two RelBench datasets do not fit in Colab's 12.7 GB of RAM.
This plan lets a run keep some or all table features on disk (memory-mapped) and read
only the rows each batch needs. Plan 02 needs it for `rel-trial`; it would also be a
standalone contribution upstream.

The plan has two parts, each a separate optimisation:

- **Part 1, done:** keep table features on disk. Build them chunk by chunk, embed each
  distinct text once, and choose `none` / `text` / `all`.
- **Part 2, done:** read the raw tables one at a time, so that the raw database never has
  to fit in memory at once.

# Part 1: features on disk (implemented)

## 1. What we measured

| Dataset | With text features | Without text features |
|---|---|---|
| rel-f1 | 3.3 GB peak, fits | – |
| rel-trial | out of memory (12.8 GB); the text vectors alone are over 5 GB | 7.1 GB peak, fits |
| rel-event | out of memory (13.3 GB) | out of memory (13.5 GB), while reading the raw tables |

**Memory-mapping works on RelBench's cache files.** A 2.7 GB `rel-trial` table opens in
under 0.1 s and adds 0 GB of memory. Taking 30,000 random rows (about one batch) takes
1.7–3.3 s. The memory used for those rows is disk cache, which the system can free.

## 2. What it does

One option, `--features-on-disk`:

| Setting | In RAM | On disk | For |
|---|---|---|---|
| `none` (default) | everything | – | small datasets; as today |
| `text` | numbers, categories, dates | text vectors | the text is too big, the rest fits (`rel-trial`) |
| `all` | – | every feature | even the non-text features are too big |

**Building:** each table is converted and saved on its own, then freed, so the build never
holds every table's features at once. This is the same conversion RelBench does; its own
builder keeps every finished table in memory.

**Loading:** the cached tables open memory-mapped; the parts the setting keeps in RAM are
copied into memory.

**Training:** unchanged. The sampler slices each table by the batch's row ids, and that
slicing works the same on memory-mapped tensors.

**On Colab:** the cache lives on the runtime's local disk (112 GB), not on Drive, which
is too slow for random reads. It can be copied from Drive at the start of a session.

## 3. Where the code goes

**First, in this project** (`experiments/`), so Plan 02 can use it and it is tested on
real data:
- `train.load_graph(..., features_on_disk=...)`;
- a table-by-table builder;
- `--features-on-disk` in `train.py` and in sweep configs.

**Then upstream, as two separate PRs:**

| Repository | Change |
|---|---|
| PyTorch Frame (`pyg-team/pytorch-frame`) | `torch_frame.load(path, mmap=False)`, passed to `torch.load`; documented and tested; `Dataset.materialize` can pass it through |
| RelBench (`snap-stanford/relbench`) | `make_pkey_fkey_graph(..., features_on_disk=...)`: table-by-table build and memory-mapped loading |

## 4. Checks

1. **Same results.** On `rel-f1`, `none`, `text` and `all` give identical batches and
   identical model outputs (exact equality), so the setting changes only memory and speed.
2. **Memory.** `rel-trial` with `text`: peak RAM measured here, which must stay well
   under 12.7 GB, both while building and while training.
3. **Speed.** Time per epoch for `none` against `text` on `rel-f1`, and for `text` on
   `rel-trial`. This is reported, not a pass/fail.
4. **Unit tests:** the builder writes one file per table; the loader returns memory-mapped
   tensors for the chosen parts and in-memory tensors for the rest; `torch.load` with
   `mmap=True` also works with `weights_only=True`, which PyTorch Frame uses.

## 5. Out of scope

- `rel-event`: it runs out of memory while reading the raw tables, before any features are
  built. That is Part 2.
- Sharing memory-mapped tensors with data-loader worker processes (`num_workers > 0`). We
  use 0 workers.

## 6. Decisions

1. The default is `none`: nothing changes for anyone who doesn't ask for it.
2. Plan 02 waits for this plan, so its larger datasets can run.

## 7. What was implemented, and results

Branch `step/03-1-disk-features`, in `experiments/disk_features.py`, with
`--features-on-disk` in `train.py` and the sweep runner.

1. **Tables built one at a time.** Each table is converted and saved as `<table>.pt`, then
   freed before the next. Its steps are those of RelBench's builder.
2. **Chunked conversion.** Column statistics come from the whole table, as in PyTorch
   Frame. The rows are then converted 50,000 at a time, and each chunk is written straight
   into a file on disk, so a table's features are never all in memory.
3. **Distinct texts embedded once.** Within a chunk, each text column embeds each distinct
   text once and copies its vector to every row with that text (`DistinctTexts`).
   21% of `rel-trial`'s text cells are distinct, and 1% of `rel-event`'s.
4. **Memory-mapped loading.** Cached tables open from disk. With `text`, everything except
   the text vectors is copied into memory; with `all`, nothing is.

**Results** (`rel-trial`, `study-outcome`, `--features-on-disk text`):

| | Whole-table build | Chunked | Chunked + distinct texts |
|---|---|---|---|
| Build peak (memory the system can't drop) | 11.8 GB | 6.1 GB | 6.2 GB |
| Training memory (cache built) | 4.0 GB | 4.0 GB | 4.0 GB |
| Build + 1 epoch, all 15 tables | – | 550 s | 376 s |

- **Memory is freed between tables:** it stays flat at 3.7–3.9 GB after every table. That
  is mostly the raw tables, which RelBench reads all at once (Part 2).
- **Exactness:** our tables equal RelBench's (8 tests on `rel-f1`, with and without text).
  On `rel-trial`'s longer texts, text vectors differ by up to 1.2e-7, which is float32
  rounding: PyTorch Frame's own output changes by exactly as much when only its
  `batch_size` changes.
- **Not done:**
  - distinct texts found across the whole column, not per chunk (closer to 5× faster
    on `rel-trial`);
  - background workers that overlap the disk reads with training.

# Part 2: read the raw tables one at a time (implemented)

## 8. Why

RelBench's `get_db()` reads every raw table into memory before anything else. For
`rel-event` that alone takes 9–13 GB, so it fails even without text features. Every step
of `get_db()` needs only one table, except one: checking a foreign key needs the row count
of the table it points to, after the date cut.

## 9. What it does

Active only with `--features-on-disk text` or `all`; `none` keeps RelBench's path.

1. **Order the tables** using the schema's foreign keys, so each table comes after the
   tables it points to (`studies` before `outcomes`).
2. **One table at a time, in that order:**
   1. read the table's file;
   2. apply the test-date cut, if the table has a date;
   3. check its primary keys are 0…n−1 (RelBench's check);
   4. check its foreign keys: a key at or past the referenced table's row count becomes
      missing, and gets no edge (RelBench's rule);
   5. guess its column types (RelBench's `get_stype_proposal`);
   6. build its features with Part 1, skipping this if cached;
   7. keep only its edges, dates and row count, and free the raw table.
3. **Save the graph structure** (edges, dates, counts) in the cache. Later runs load it with
   the memory-mapped features and never read the raw tables again.
4. **Per task:** drop the task's hidden columns after loading, as now.

At any moment, memory holds one raw table plus the graph structure built so far.

## 10. Special cases

| Case | Example | How it is handled |
|---|---|---|
| Dated table, referenced by others | `studies` | cut by date; its surviving row count is kept for the tables that point to it |
| Dateless table | `facilities` | kept whole (RelBench's cut skips it); row count = its length |
| Foreign key into a dated table that lost rows | `outcomes.study_id` → a removed study | set to missing, no edge |
| Foreign key into a dateless table | → `facilities` | checked against its full length |
| Foreign key already missing in the data | empty `study_id` | stays missing, no edge |
| Table nobody points to | `user_friends` | no count needed; built the same way |
| Table pointing to itself | `rel-stack` `posts.ParentId` | its own count is known once its cut is done |
| Two tables pointing at each other (a loop) | none known | detected while ordering; a quick extra pass reads only those tables' date columns to get their counts |
| No feature columns once keys are removed | link tables | constant `__const__` feature (RelBench's rule) |
| Primary keys not 0…n−1 after the cut | rows not in date order | the same error RelBench raises |
| Some tables already cached | rerun after a crash | features skipped; edges, dates and counts rebuilt (cheap) |
| Task's hidden columns | label-leaking columns | dropped after loading, as now |
| Task tables (labels) | `study-outcome` splits | must come ready-made from the download; **to check first**. A dataset that computes them from the full database is not supported, and we say so. |

## 11. Checks

1. **Same graph as RelBench's `get_db()` + `make_pkey_fkey_graph`**, on `rel-f1` and
   `rel-trial`:
   - edges, dates, row counts and column statistics identical;
   - features identical (text vectors within float32 rounding);
   - model output identical on a batch.
2. **Unit tests on a tiny hand-made database:** a dated table that loses rows, a dateless
   table, missing keys, a self-reference, a loop, a table with no features.
3. **`rel-event`:** peak memory while building and while training, which must stay under
   12.7 GB.

## 12. Before building

- Confirmed: every dataset stores one Parquet file per table, and a `manifest.yaml` with
  each table's primary key, date column and foreign keys.
- Confirmed: all 30 tasks of the 7 datasets come with ready-made `train` / `val` / `test`
  labels, so no task needs the whole database.
