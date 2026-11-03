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
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch_frame
from relbench.modeling.graph import remove_pkey_fkey
from torch import Tensor
from torch_frame.config import TextEmbedderConfig

from torch_geometric.data import HeteroData

FEATURES_ON_DISK = ('none', 'text', 'all')


def build_tables(db, col_to_stype_dict: Dict[str, Dict], cache_dir: Path,
                 text_embedder_cfg: Optional[TextEmbedderConfig] = None,
                 log=print) -> None:
    r"""Converts every table of :obj:`db` that is not cached yet into a
    ``TensorFrame`` file in :obj:`cache_dir`, one table at a time, freeing
    each before the next. The files are the ones ``make_pkey_fkey_graph``
    writes and reads, built with the same per-table steps."""
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
        torch_frame.data.Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=str(path))
        gc.collect()


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
