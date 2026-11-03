# Plan 03: keep table features on disk, for datasets larger than memory

Plan 02's dry runs showed that two RelBench datasets do not fit in Colab's 12.7 GB of RAM.
This plan lets a run keep some or all table features on disk (memory-mapped) and read
only the rows each batch needs. Plan 02 needs it for `rel-trial`; it would also be a
standalone contribution upstream.

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
  built. That needs a different change (reading the source files table by table) and is
  left out.
- Sharing memory-mapped tensors with data-loader worker processes (`num_workers > 0`). We
  use 0 workers.

## 6. Decisions

1. The default is `none`: nothing changes for anyone who doesn't ask for it.
2. Plan 02 waits for this plan, so its larger datasets can run.
