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

With ``'text'`` or ``'all'``, the raw tables are also read one at a time
(:func:`build_graph`), instead of all at once by RelBench's ``get_db()``, and
the graph structure is cached, so later runs read no raw table at all.
"""
import contextlib
import gc
import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch_frame
from relbench.base import Database, Table
from relbench.load import _coerce_string_dtype
from relbench.modeling.graph import drop_tensor_frame_columns, remove_pkey_fkey
from relbench.modeling.utils import to_unix_time
from torch import Tensor
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import MultiEmbeddingTensor
from torch_frame.data.stats import StatType, compute_col_stats

from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

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
    with fast_numerical_stats():
        for col, stype in dataset.col_to_stype.items():  # As materialize, 1.
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


_FLATTENED_STATS = (StatType.MEAN, StatType.STD, StatType.QUANTILES)


@contextlib.contextmanager
def fast_numerical_stats() -> Iterator[None]:
    r"""Within this context, PyTorch Frame's mean, standard deviation and
    quantiles of a plain numerical column skip a slow, value-by-value copy.

    ``StatType.compute`` first flattens the values with
    ``np.hstack(np.hstack(values))``, which handles list-valued columns but
    walks a plain 1-D number array one value at a time in Python: about 6 s
    per statistic for a column of 3 million rows. For such an array the
    result is the array itself, so it is used directly and the statistics
    are identical. Every other column (list-valued, pandas' own dtypes) goes
    through PyTorch Frame's code unchanged."""
    compute = StatType.compute

    def fast_compute(self, ser, sep=None):
        values = ser.values
        if (self not in _FLATTENED_STATS or not isinstance(values, np.ndarray)
                or values.ndim != 1 or values.dtype.kind not in 'iuf'):
            return compute(self, ser, sep)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return compute(self, ser, sep)  # PyTorch Frame's NaN result.
        if self == StatType.MEAN:
            return np.mean(finite).item()
        if self == StatType.STD:
            return np.std(finite).item()
        return np.quantile(finite, q=[0, 0.25, 0.5, 0.75, 1]).tolist()

    StatType.compute = fast_compute
    try:
        yield
    finally:
        StatType.compute = compute


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


# Part 2: read the raw tables one at a time. ---------------------------------
#
# RelBench's ``dataset.get_db()`` reads every table into memory, cuts each at
# the test date, then checks the keys. Each of these steps needs one table,
# except the foreign-key check, which needs the row count (after the cut) of
# the table a key points to. So the row counts are found first, from the
# Parquet files' footers and date columns, and the tables are then read one at
# a time, in RelBench's order.

GRAPH_FILE = 'graph.pt'


def _parquet_path(dataset, name: str) -> Path:
    return Path(dataset.db_dir) / f'{name}.parquet'


def check_supported(dataset) -> None:
    r"""Raises an error unless :obj:`dataset` reads its tables the way
    :func:`read_table` does: from a manifest and one Parquet file per table,
    as RelBench's hosted datasets do."""
    from relbench.load import RelBenchDataset
    if (not isinstance(dataset, RelBenchDataset)
            or type(dataset).make_db is not RelBenchDataset.make_db
            or type(dataset).get_db is not RelBenchDataset.get_db):
        raise ValueError(
            f"Reading tables one at a time needs a dataset stored as a "
            f"manifest and one Parquet file per table (got {dataset!r})")


def read_table(dataset, name: str,
               columns: Optional[List[str]] = None) -> Table:
    r"""Table :obj:`name` as ``dataset.get_db()`` has it before its keys are
    checked: read as RelBench's loader reads it, then cut at the test date
    (tables without a date are kept whole). With :obj:`columns`, only those
    columns are read."""
    spec = dataset.manifest.tables[name]
    df = pd.read_parquet(_parquet_path(dataset, name), columns=columns)
    table = Table(df=_coerce_string_dtype(df),
                  fkey_col_to_pkey_table=dict(spec.fkeys),
                  pkey_col=spec.pkey, time_col=spec.time_col)
    return table.upto(dataset.test_timestamp)


def row_counts(dataset) -> Dict[str, int]:
    r"""Every table's number of rows after the test-date cut, as in
    ``dataset.get_db()``. A table without a date keeps all its rows, whose
    number is in its Parquet file's footer; for a table with a date, only the
    date column is read."""
    counts = {}
    for name, spec in dataset.manifest.tables.items():
        if spec.time_col is None:
            path = _parquet_path(dataset, name)
            counts[name] = pq.ParquetFile(path).metadata.num_rows
        else:
            counts[name] = len(read_table(dataset, name, [spec.time_col]))
    return counts


def validate_table(dataset, name: str, table: Table,
                   counts: Dict[str, int]) -> None:
    r"""RelBench's ``validate_and_correct_db`` for one table, in place: its
    primary keys must be 0..n-1, and a foreign key at or past the referenced
    table's row count becomes missing. RelBench's own function is run, with
    each referenced table stood in by an empty table of its row count, which
    is all the check reads from it."""
    table_dict = {
        ref: Table(df=pd.DataFrame(index=pd.RangeIndex(counts[ref])),
                   fkey_col_to_pkey_table={})
        for ref in table.fkey_col_to_pkey_table.values() if ref != name
    }
    table_dict[name] = table
    dataset.validate_and_correct_db(Database(table_dict))


def table_edges(name: str, table: Table, counts: Dict[str, int]
                ) -> List[Tuple[Tuple[str, str, str], Tensor]]:
    r"""The edges ``make_pkey_fkey_graph`` adds for :obj:`table`'s foreign
    keys, in its order: for each key, row -> referenced row (``f2p_<key>``)
    and the reverse (``rev_f2p_<key>``), leaving out missing keys."""
    edges = []
    for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
        pkey_index = table.df[fkey_name]
        mask = ~pkey_index.isna()
        fkey_index = torch.arange(len(pkey_index))
        pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
        fkey_index = fkey_index[torch.from_numpy(mask.values)]
        if not (pkey_index < counts[pkey_table_name]).all():
            raise RuntimeError(f"Table '{name}' has foreign keys "
                               f"'{fkey_name}' past '{pkey_table_name}'")
        edge_index = torch.stack([fkey_index, pkey_index], dim=0)
        edges.append(((name, f'f2p_{fkey_name}', pkey_table_name),
                      sort_edge_index(edge_index)))
        edge_index = torch.stack([pkey_index, fkey_index], dim=0)
        edges.append(((pkey_table_name, f'rev_f2p_{fkey_name}', name),
                      sort_edge_index(edge_index)))
    return edges


def build_graph(dataset, cache_dir: Path,
                propose_stypes: Callable[[Database], Dict[str, Dict]],
                text_embedder_cfg: Optional[TextEmbedderConfig] = None,
                rows_per_chunk: int = 50_000, log=print) -> None:
    r"""Builds what ``make_pkey_fkey_graph(dataset.get_db(), ...)`` builds,
    reading one raw table at a time, and saves it in :obj:`cache_dir`: each
    table's features (``<table>.pt``, as Part 1 builds them) and the graph
    structure (:obj:`GRAPH_FILE`: row counts, dates and edges).

    1. Every table's row count after the test-date cut (:func:`row_counts`).
    2. Then for each table, in the manifest's order (RelBench's):
       read it (only its key and date columns if its features are cached),
       cut it at the test date, check its keys (:func:`validate_table`),
       build its features if not cached (column types from
       :obj:`propose_stypes`, called with a database of this one table), and
       keep its dates and edges; then free it.

    Memory holds one raw table at a time, plus the dates and edges so far.
    """
    check_supported(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = row_counts(dataset)
    graph = dict(manifest=dataset.manifest.to_dict(), num_rows=counts,
                 time={}, edges={})
    for name, spec in dataset.manifest.tables.items():
        cached = (cache_dir / f'{name}.pt').exists()
        columns = None
        if cached:  # Only the keys and the date are needed:
            columns = list(dict.fromkeys(
                c for c in [spec.pkey, spec.time_col, *spec.fkeys] if c))
        table = read_table(dataset, name, columns)
        if len(table) != counts[name]:
            raise RuntimeError(f"Table '{name}' has {len(table)} rows after "
                               f"the cut, but {counts[name]} were counted")
        validate_table(dataset, name, table, counts)
        if not cached:
            one_table = Database({name: table})
            build_tables(one_table, propose_stypes(one_table), cache_dir,
                         text_embedder_cfg, rows_per_chunk, log)
            del one_table
        if spec.time_col is not None:
            graph['time'][name] = torch.from_numpy(
                to_unix_time(table.df[spec.time_col]))
        graph['edges'][name] = table_edges(name, table, counts)
        del table
        gc.collect()
        log(f'{name}: {counts[name]:,} rows; process memory now '
            f'{_memory_gb():.1f} GB')
    path = cache_dir / GRAPH_FILE
    torch.save(graph, path.with_suffix('.tmp'))
    os.replace(path.with_suffix('.tmp'), path)  # Never a partial file.


def load_cached_graph(dataset, cache_dir: Path, remove_columns=None
                      ) -> Optional[Tuple[HeteroData, Dict[str, Dict]]]:
    r"""The graph ``make_pkey_fkey_graph`` returns, from the files
    :func:`build_graph` saved, without reading any raw table: features
    memory-mapped, dates and edges in memory. :obj:`remove_columns` (a task's
    hidden columns) are dropped as RelBench drops them. Returns :obj:`None`
    if the cache is missing or was built from a different manifest."""
    path = cache_dir / GRAPH_FILE
    if not path.exists():
        return None
    graph = torch.load(path)
    names = list(graph['num_rows'])
    if (graph['manifest'] != dataset.manifest.to_dict() or not all(
            (cache_dir / f'{name}.pt').exists() for name in names)):
        return None

    remove_columns = [tuple(pair) for pair in (remove_columns or [])]
    data, col_stats_dict = HeteroData(), {}
    for name in names:  # As make_pkey_fkey_graph, in its order:
        with memory_mapped_loading():
            tf, col_stats = torch_frame.load(str(cache_dir / f'{name}.pt'))
        if tf.num_rows != graph['num_rows'][name]:
            raise RuntimeError(
                f"Cached features of '{name}' have {tf.num_rows} rows, but "
                f"the table has {graph['num_rows'][name]}; delete {cache_dir}")
        drop = {col for table, col in remove_columns if table == name}
        if drop:
            tf, col_stats = drop_tensor_frame_columns(tf, col_stats, drop)
        data[name].tf = tf
        col_stats_dict[name] = col_stats
        if name in graph['time']:
            data[name].time = graph['time'][name]
        for edge_type, edge_index in graph['edges'][name]:
            data[edge_type].edge_index = edge_index
    data.validate()
    return data, col_stats_dict
