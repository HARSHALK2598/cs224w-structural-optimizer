r"""Plan 03 Part 2: reading the raw tables one at a time gives the graph that
RelBench builds from the whole database (``dataset.get_db()`` and
``make_pkey_fkey_graph``).

Main checks on rel-f1 and rel-trial; a small hand-made database
(``tiny_database.py``) adds the cases those two do not contain: foreign keys
into rows the date cut removes, a table pointing to itself, and two tables
pointing at each other.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
import torch_frame
from conftest import GLOVE_CACHED, ROOT, requires_rel_f1
from relbench.load import RelBenchDataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config import TextEmbedderConfig

import disk_features
import tiny_database
import train

RELTRIAL_CACHED = any((ROOT / 'data' / 'hf' / 'hub').glob(
    'datasets--stanford-star--relbench-v1/snapshots/*/rel-trial/'
    'manifest.yaml'))
requires_rel_trial = pytest.mark.skipif(
    not RELTRIAL_CACHED, reason='rel-trial not downloaded')

QUIET = dict(log=lambda _: None)


def no_text(db):
    r"""RelBench's column types without text, as ``--text-embedder none``."""
    return train.propose_stypes(db, 'none')


def tensors(feat):
    return [feat] if isinstance(feat, torch.Tensor) else [feat.values]


def assert_same_graph(ours, our_stats, ref, ref_stats):
    r"""Same node and edge types, in the same order, and the same features,
    column statistics, dates and edges."""
    assert ours.node_types == ref.node_types
    assert ours.edge_types == ref.edge_types
    for node_type in ref.node_types:
        a, b = ours[node_type], ref[node_type]
        assert set(a.keys()) == set(b.keys()), node_type
        assert a.tf.num_rows == b.tf.num_rows, node_type
        assert a.tf.col_names_dict == b.tf.col_names_dict, node_type
        for stype in b.tf.feat_dict:
            for x, y in zip(tensors(a.tf.feat_dict[stype]),
                            tensors(b.tf.feat_dict[stype])):
                assert x.dtype == y.dtype and x.shape == y.shape
                assert torch.equal(x, y) or torch.allclose(
                    x, y, rtol=0, atol=0, equal_nan=True), (node_type, stype)
        if 'time' in b:
            assert a.time.dtype == b.time.dtype
            assert torch.equal(a.time, b.time), node_type
    for edge_type in ref.edge_types:
        x, y = ours[edge_type].edge_index, ref[edge_type].edge_index
        assert x.dtype == y.dtype and torch.equal(x, y), edge_type
    assert repr(our_stats) == repr(ref_stats)


def relbench_graph(dataset, cfg=None, propose=no_text,
                   remove_columns=None, cache_dir=None, seed=0):
    np.random.seed(seed)  # Column types are guessed from random samples.
    db = dataset.get_db()
    return make_pkey_fkey_graph(db, propose(db), cfg, cache_dir=cache_dir,
                                remove_columns=remove_columns)


def our_graph(dataset, cache_dir, cfg=None, propose=no_text,
              remove_columns=None, seed=0, rows_per_chunk=50_000):
    np.random.seed(seed)
    disk_features.build_graph(dataset, cache_dir, propose, cfg,
                              rows_per_chunk=rows_per_chunk, **QUIET)
    return disk_features.load_cached_graph(dataset, cache_dir,
                                           remove_columns)


# The small hand-made database ----------------------------------------------


@pytest.fixture
def tiny(tmp_path):
    return RelBenchDataset(tiny_database.write_dataset(tmp_path / 'tiny'))


def test_tiny_has_the_cases(tiny):
    raw = {n: pd.read_parquet(tiny.db_dir / f'{n}.parquet')
           for n in tiny.manifest.tables}
    db = tiny.get_db()
    # Foreign keys into rows the cut removed become missing:
    for table, key in [('visits', 'study_id'), ('notes', 'parent_id'),
                       ('a', 'b_id'), ('b', 'a_id')]:
        kept = raw[table].loc[db.table_dict[table].df.index, key]
        assert db.table_dict[table].df[key].isna().sum() > kept.isna().sum()
    # Self-reference and loop, in the schema:
    assert tiny.manifest.tables['notes'].fkeys['parent_id'] == 'notes'
    assert tiny.manifest.tables['a'].fkeys['b_id'] == 'b'
    assert tiny.manifest.tables['b'].fkeys['a_id'] == 'a'
    # A table pointing to one that comes later in the order:
    names = list(tiny.manifest.tables)
    assert names.index('visits') < names.index('studies')


def test_tiny_row_counts(tiny):
    db = tiny.get_db()
    assert disk_features.row_counts(tiny) == {
        name: len(table) for name, table in db.table_dict.items()
    }


@pytest.mark.parametrize('text', [False, True])
@pytest.mark.parametrize('hidden', [None, [('studies', 'size'),
                                           ('links', 'study_id')]])
def test_tiny_same_graph_as_relbench(tiny, tmp_path, text, hidden):
    cfg = (TextEmbedderConfig(text_embedder=tiny_database.embed,
                              batch_size=None) if text else None)
    propose = get_stype_proposal if text else no_text
    ref = relbench_graph(tiny, cfg, propose, hidden)
    # Small chunks, so that tables are converted in several pieces:
    ours = our_graph(tiny, tmp_path / 'cache', cfg, propose, hidden,
                     rows_per_chunk=16)
    assert_same_graph(*ours, *ref)
    assert ('studies', 'f2p_study_id', 'studies') not in ours[0].edge_types
    assert ('notes', 'f2p_parent_id', 'notes') in ours[0].edge_types
    assert ours[0]['links'].tf.col_names_dict == {
        torch_frame.numerical: ['__const__']
    }
    names = sorted(p.name for p in (tmp_path / 'cache').iterdir())
    assert names == sorted([f'{n}.pt' for n in tiny.manifest.tables] +
                           [disk_features.GRAPH_FILE])  # No temporary files.


def test_tiny_cached_graph_reads_no_raw_table(tiny, tmp_path, monkeypatch):
    first = our_graph(tiny, tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError('a raw table was read')

    monkeypatch.setattr(pd, 'read_parquet', fail)
    monkeypatch.setattr(pq, 'ParquetFile', fail)
    again = disk_features.load_cached_graph(tiny, tmp_path)
    assert_same_graph(*again, *first)


def test_tiny_cached_features_read_only_keys(tiny, tmp_path, monkeypatch):
    first = our_graph(tiny, tmp_path)
    (tmp_path / disk_features.GRAPH_FILE).unlink()  # Features stay cached.
    read, read_parquet = [], pd.read_parquet

    def spy(path, columns=None, **kwargs):
        read.append((Path(path).stem, columns))
        return read_parquet(path, columns=columns, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', spy)
    again = our_graph(tiny, tmp_path)
    assert_same_graph(*again, *first)
    tables = tiny.manifest.tables
    # Row counts: only the dates of the tables that have one ...
    counts = [(n, [s.time_col]) for n, s in tables.items() if s.time_col]
    assert read[:len(counts)] == counts
    # ... then each table once, only its keys and date, in the manifest order:
    assert read[len(counts):] == [
        (n, [c for c in [s.pkey, s.time_col, *s.fkeys] if c])
        for n, s in tables.items()
    ]


def test_tiny_primary_keys_not_in_date_order(tmp_path):
    tables, specs = tiny_database.make_tables()
    studies = tables['studies']
    studies['start'] = studies['start'].values[::-1]  # Cut leaves gaps.
    tiny = RelBenchDataset(tiny_database.write_dataset(tmp_path / 't', tables,
                                                       specs))
    with pytest.raises(RuntimeError) as relbench_error:
        tiny.get_db()
    with pytest.raises(RuntimeError) as our_error:
        disk_features.build_graph(tiny, tmp_path / 'cache', no_text,
                                  **QUIET)
    assert str(our_error.value) == str(relbench_error.value)


def test_tiny_cache_of_another_manifest_is_not_used(tiny, tmp_path):
    our_graph(tiny, tmp_path)
    tiny.manifest.test_timestamp = '2020-01-01'
    assert disk_features.load_cached_graph(tiny, tmp_path) is None


def test_tiny_cached_features_with_wrong_rows_are_rejected(tiny, tmp_path):
    our_graph(tiny, tmp_path / 'a')
    shutil.copy(tmp_path / 'a' / 'sites.pt', tmp_path / 'a' / 'lonely.pt')
    with pytest.raises(RuntimeError, match="'lonely'"):
        disk_features.load_cached_graph(tiny, tmp_path / 'a')


def test_unsupported_dataset(tmp_path):
    class OwnDatabase(RelBenchDataset):
        def make_db(self):
            raise NotImplementedError

    tiny = OwnDatabase(tiny_database.write_dataset(tmp_path / 'tiny'))
    with pytest.raises(ValueError, match='one Parquet file per table'):
        disk_features.build_graph(tiny, tmp_path / 'cache',
                                  get_stype_proposal, **QUIET)


# rel-f1 and rel-trial ---------------------------------------------------


@requires_rel_f1
@pytest.mark.parametrize('text', [False, True])
def test_rel_f1_same_graph_as_relbench(rel_f1, tmp_path, text):
    if text and not GLOVE_CACHED:
        pytest.skip('GloVe model not downloaded')
    name = 'glove' if text else 'none'
    cfg = train.text_embedder_cfg(name, torch.device('cpu'))

    def propose(db):
        return train.propose_stypes(db, name)

    hidden = rel_f1.load_task('driver-top3').hidden_columns()
    ref = relbench_graph(rel_f1, cfg, propose, hidden)
    ours = our_graph(rel_f1, tmp_path, cfg, propose, hidden,
                     rows_per_chunk=5000)
    assert_same_graph(*ours, *ref)


@requires_rel_f1
def test_rel_f1_row_counts(rel_f1):
    assert disk_features.row_counts(rel_f1) == {
        name: len(table) for name, table in rel_f1.get_db().table_dict.items()
    }


@pytest.fixture(scope='module')
def rel_trial():
    from relbench import load_dataset
    return load_dataset('rel-trial')


@requires_rel_trial
def test_rel_trial_same_graph_as_relbench(rel_trial, tmp_path):
    # Built from scratch on both sides, without text (quicker):
    hidden = rel_trial.load_task('study-outcome').hidden_columns()
    ref = relbench_graph(rel_trial, remove_columns=hidden)
    ours = our_graph(rel_trial, tmp_path, remove_columns=hidden)
    assert_same_graph(*ours, *ref)


@requires_rel_trial
def test_rel_trial_with_text_same_graph_as_relbench(rel_trial, tmp_path):
    # The GloVe features built in Part 1 (checked there) are reused by both;
    # the graph structure is built from the raw tables by each.
    cache = ROOT / 'data' / 'materialized' / 'rel-trial' / 'glove'
    names = list(rel_trial.manifest.tables)
    if not all((cache / f'{n}.pt').exists() for n in names):
        pytest.skip('rel-trial GloVe features not built')
    for name in rel_trial.manifest.tables:
        (tmp_path / f'{name}.pt').symlink_to(cache / f'{name}.pt')
    # RelBench's side opens the files memory-mapped too: in memory, they
    # need more than this machine has (the reason for Plan 03).
    cfg = train.text_embedder_cfg('glove', torch.device('cpu'))
    with disk_features.memory_mapped_loading():
        ref = relbench_graph(rel_trial, cfg, get_stype_proposal,
                             cache_dir=str(tmp_path))
    ours = our_graph(rel_trial, tmp_path, cfg)
    assert_same_graph(*ours, *ref)
