"""Keep table features on disk, for databases larger than memory (Plan 03).

RelBench's ``make_pkey_fkey_graph`` converts every table into a PyTorch Frame
``TensorFrame`` and keeps all of them in memory, which needs more than 12.7 GB
for ``rel-trial`` with text features. Here, each table is converted on its own
and saved to the cache, and the cached tables are then opened memory-mapped:
a tensor stays on disk, and only the rows a batch reads are loaded.

``features_on_disk`` chooses what stays on disk:

    'none'  everything in memory (RelBench's behaviour)
    'text'  only the text vectors (stype ``embedding``) on disk
    'all'   every feature on disk

The model and training code are unchanged: the sampler slices each table by
the batch's row ids, which works the same on memory-mapped tensors.
"""
import contextlib
import gc
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch_frame
from relbench.modeling.graph import remove_pkey_fkey
from torch import Tensor
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import MultiEmbeddingTensor
from torch_frame.data.stats import compute_col_stats

from torch_geometric.data import HeteroData

FEATURES_ON_DISK = ('none', 'text', 'all')


def build_tables(db, col_to_stype_dict: Dict[str, Dict], cache_dir: Path,
                 text_embedder_cfg: Optional[TextEmbedderConfig] = None,
                 rows_per_chunk: int = 50_000, log=print) -> None:
    r"""Converts every table of :obj:`db` that is not cached yet into a
    ``TensorFrame`` file in :obj:`cache_dir`, one table at a time. The files
    are the ones ``make_pkey_fkey_graph`` writes and reads, built with the same
    per-table steps. Each table is converted :obj:`rows_per_chunk` rows at a
    time (see :func:`materialize_in_chunks`), and each distinct text of a
    chunk's column is embedded once (see :class:`DistinctTexts`)."""
    if text_embedder_cfg is not None:
        # One embedder call per column and chunk, so that repeated texts
        # across the whole chunk are found:
        text_embedder_cfg = TextEmbedderConfig(
            text_embedder=DistinctTexts(text_embedder_cfg.text_embedder),
            batch_size=None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for table_name, table in db.table_dict.items():
        path = cache_dir / f'{table_name}.pt'
        if path.exists():
            continue
        log(f'building {table_name} ({len(table.df):,} rows)')
        col_to_stype = dict(col_to_stype_dict[table_name])
        remove_pkey_fkey(col_to_stype, table)
        df = table.df
        if len(col_to_stype) == 0:  # As RelBench: a constant feature.
            col_to_stype = {'__const__': torch_frame.numerical}
            fkeys = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({'__const__': np.ones(len(df)), **fkeys})
        dataset = torch_frame.data.Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        )
        materialize_in_chunks(dataset, path, rows_per_chunk)
        del dataset
        gc.collect()
        log(f'built {table_name}; process memory now {_memory_gb():.1f} GB')


class DistinctTexts:
    r"""Wraps a text embedder so that each distinct text of a call is
    embedded once, and its vector copied to every position with that text.

    Each row gets the vector of its text, as before. The vectors can differ in
    the last float32 bit (up to ~1e-7) because texts are batched differently:
    the GloVe embedder rounds slightly differently depending on which texts
    share a batch, and PyTorch Frame's own output changes the same way when
    only its ``batch_size`` changes. In RelBench most text cells repeat (21%
    of ``rel-trial``'s and 1% of ``rel-event``'s are distinct), so far fewer
    texts are embedded.
    """
    def __init__(self, text_embedder):
        self.text_embedder = text_embedder

    def __call__(self, texts):
        codes, distinct = pd.factorize(pd.Series(texts, dtype=object))
        vectors = self.text_embedder(list(distinct))
        return vectors[torch.from_numpy(codes)]


def materialize_in_chunks(dataset: torch_frame.data.Dataset, path: Path,
                          rows_per_chunk: int) -> None:
    r"""Does what ``dataset.materialize(path=path)`` does, without holding the
    whole table's features in memory.

    The column statistics come from the whole table, as in ``materialize``,
    so every chunk maps values (e.g. category indices) the same way. The
    rows are then converted chunk by chunk, each chunk written straight into
    a file-backed tensor, and the result is saved in the usual format.
    Conversion maps each row on its own, so the result is that of converting
    the whole table at once (text vectors: up to float32 rounding from
    batching, see :class:`DistinctTexts`).

    Uses ``Dataset``'s private converter, since PyTorch Frame has no public
    chunked conversion.
    """
    df = dataset.df
    for col, stype in dataset.col_to_stype.items():  # As materialize, step 1.
        dataset._col_stats[col] = compute_col_stats(
            df[col], stype, sep=dataset.col_to_sep.get(col, None),
            time_format=dataset.col_to_time_format.get(col, None))
    converter = dataset._get_tensorframe_converter()

    num_rows, feat_dict, col_names_dict, files = len(df), {}, None, []
    for start in range(0, num_rows, rows_per_chunk):
        tf = converter(df.iloc[start:start + rows_per_chunk])
        if col_names_dict is None:  # Allocate the outputs on disk.
            col_names_dict = tf.col_names_dict
            for stype, feat in tf.feat_dict.items():
                values = feat if isinstance(feat, Tensor) else feat.values
                out = _file_tensor(path, f'{stype.value}', num_rows,
                                   values.shape[1:], values.dtype)
                files.append(out[1])
                feat_dict[stype] = (out[0] if isinstance(feat, Tensor) else
                                    MultiEmbeddingTensor(
                                        num_rows=num_rows,
                                        num_cols=feat.num_cols,
                                        values=out[0], offset=feat.offset))
        end = start + tf.num_rows
        for stype, feat in tf.feat_dict.items():
            out = feat_dict[stype]
            if isinstance(feat, Tensor):
                out[start:end] = feat
            else:
                out.values[start:end] = feat.values
        del tf

    dataset._tensor_frame = torch_frame.TensorFrame(feat_dict, col_names_dict)
    dataset._update_col_stats()  # As materialize, step 3 (text dims).
    torch_frame.save(dataset._tensor_frame, dataset._col_stats, str(path))
    dataset._tensor_frame = None
    del feat_dict
    gc.collect()
    for file in files:
        os.remove(file)


def _memory_gb() -> float:
    r"""This process's memory that is not backed by files (Linux), in GB."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('RssAnon:'):
                return int(line.split()[1]) / 1e6
    return float('nan')


def _file_tensor(path: Path, name: str, num_rows: int, shape, dtype):
    r"""An uninitialised tensor of :obj:`num_rows` rows, stored in a
    temporary file next to :obj:`path` instead of in memory."""
    file = path.with_name(f'{path.stem}.{name}.tmp')
    numel = num_rows * int(torch.Size(shape).numel())
    tensor = torch.from_file(str(file), shared=True, size=numel, dtype=dtype)
    return tensor.view(num_rows, *shape), file


@contextlib.contextmanager
def memory_mapped_loading() -> Iterator[None]:
    r"""Within this context, ``torch.load`` (and so ``torch_frame.load``)
    opens files memory-mapped instead of reading them into memory."""
    load = torch.load

    def mmap_load(*args, **kwargs):
        kwargs.setdefault('mmap', True)
        return load(*args, **kwargs)

    torch.load = mmap_load
    try:
        yield
    finally:
        torch.load = load


def is_on_disk(tensor: Tensor) -> bool:
    r"""Whether :obj:`tensor`'s memory is a mapped file (Linux only: reads
    ``/proc/self/maps``, the process's list of memory regions)."""
    address = tensor.untyped_storage().data_ptr()
    with open('/proc/self/maps') as f:
        for line in f:
            fields = line.split()
            start, end = (int(x, 16) for x in fields[0].split('-'))
            if start <= address < end:
                return len(fields) >= 6 and fields[5].startswith('/')
    return False


def keep_in_memory(data: HeteroData, features_on_disk: str) -> None:
    r"""Copies into memory the features that :obj:`features_on_disk` does not
    keep on disk: everything except the text vectors for ``'text'``,
    nothing for ``'all'``."""
    if features_on_disk == 'all':
        return
    for node_type in data.node_types:
        tf = data[node_type].tf
        for stype, feat in tf.feat_dict.items():
            if stype != torch_frame.embedding:
                tf.feat_dict[stype] = feat.clone()
