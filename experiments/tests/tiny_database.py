"""A small hand-made RelBench dataset with every case the one-table-at-a-time
reader must handle (Plan 03, section 10), written as RelBench hosts datasets:
a manifest and one Parquet file per table."""
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.manifest import DatasetManifest, TableSpec

TEST_TIMESTAMP = pd.Timestamp('2021-01-01')
WORDS = ['heart', 'lung', 'kidney', 'brain', 'liver', 'skin', 'bone']


def _dates(n, after, rng):
    r"""Sorted dates, the last :obj:`after` of them past the test date, so
    that the cut keeps primary keys 0..k-1."""
    before = pd.to_datetime('2015-01-01') + pd.to_timedelta(
        np.sort(rng.integers(0, 2000, n - after)), unit='D')
    later = TEST_TIMESTAMP + pd.to_timedelta(
        np.sort(rng.integers(1, 300, after)), unit='D')
    return pd.DatetimeIndex(list(before) + list(later))


def _text(n, rng):
    return [' '.join(rng.choice(WORDS, rng.integers(2, 6))) + ' study'
            for _ in range(n)]


def make_tables(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_studies, n_sites, n_notes, n_a, n_b = 80, 40, 60, 50, 45
    studies = pd.DataFrame({
        'study_id': np.arange(n_studies),
        'start': _dates(n_studies, after=12, rng=rng),
        'title': _text(n_studies, rng),
        'size': np.where(rng.random(n_studies) < 0.2, np.nan,
                         rng.integers(10, 500, n_studies).astype(float)),
        'phase': rng.choice(['I', 'II', 'III'], n_studies),
    })
    # Missing titles, including a run of 20 in a row, so that some chunks of
    # 16 rows have only missing texts:
    missing = (rng.random(n_studies) < 0.2) | (np.arange(n_studies) >= 40) & (
        np.arange(n_studies) < 60)
    studies['title'] = studies['title'].where(~missing, None)
    sites = pd.DataFrame({  # No date: never cut.
        'site_id': np.arange(n_sites),
        'beds': rng.integers(5, 900, n_sites).astype(float),
        'country': pd.array(rng.choice(['us', 'fr', 'jp', 'in'], n_sites),
                            dtype='string'),  # RelBench turns it to object.
    })
    n_visits = 150
    visit_study = rng.integers(0, n_studies, n_visits).astype(float)
    visit_study[rng.random(n_visits) < 0.1] = np.nan  # Already missing.
    visits = pd.DataFrame({  # Points to studies, which come later.
        'study_id': visit_study,  # Some point to studies the cut removes.
        'site_id': rng.integers(0, n_sites, n_visits),
        'date': pd.to_datetime('2016-01-01') + pd.to_timedelta(
            rng.integers(0, 2500, n_visits), unit='D'),  # Unsorted, no pkey.
        'cost': rng.normal(100, 30, n_visits),
        'kind': rng.choice(['screen', 'dose', 'follow'], n_visits),
    })
    links = pd.DataFrame({  # Only foreign keys: a constant feature.
        'study_id': rng.integers(0, n_studies, 70),
        'site_id': rng.integers(0, n_sites, 70),
    })
    parent = np.full(n_notes, np.nan)
    parent[1:] = [rng.integers(0, n_notes) for _ in range(1, n_notes)]
    notes = pd.DataFrame({  # Points to itself, including to removed notes.
        'note_id': np.arange(n_notes),
        'parent_id': parent,
        'study_id': rng.integers(0, n_studies, n_notes),
        'written': _dates(n_notes, after=8, rng=rng),
        'length': rng.integers(1, 2000, n_notes).astype(float),
    })
    a = pd.DataFrame({  # a and b point at each other: a loop.
        'a_id': np.arange(n_a),
        'b_id': rng.integers(0, n_b, n_a),
        'time': _dates(n_a, after=7, rng=rng),
        'score': rng.normal(0, 1, n_a),
    })
    b = pd.DataFrame({
        'b_id': np.arange(n_b),
        'a_id': rng.integers(0, n_a, n_b),
        'time': _dates(n_b, after=9, rng=rng),
        'weight': rng.normal(5, 2, n_b),
    })
    lonely = pd.DataFrame({  # Nobody points to it.
        'lonely_id': np.arange(30),
        'value': rng.normal(0, 1, 30),
        'label': rng.choice(['x', 'y'], 30),
    })
    tables = dict(visits=visits, studies=studies, sites=sites, links=links,
                  notes=notes, a=a, b=b, lonely=lonely)
    specs = dict(
        visits=TableSpec(None, 'date', {'study_id': 'studies',
                                        'site_id': 'sites'}),
        studies=TableSpec('study_id', 'start', {}),
        sites=TableSpec('site_id', None, {}),
        links=TableSpec(None, None, {'study_id': 'studies',
                                     'site_id': 'sites'}),
        notes=TableSpec('note_id', 'written', {'parent_id': 'notes',
                                               'study_id': 'studies'}),
        a=TableSpec('a_id', 'time', {'b_id': 'b'}),
        b=TableSpec('b_id', 'time', {'a_id': 'a'}),
        lonely=TableSpec('lonely_id', None, {}),
    )
    return tables, specs


def write_dataset(root: Path, tables=None, specs=None) -> Path:
    if tables is None:
        tables, specs = make_tables()
    (root / 'db').mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(root / 'db' / f'{name}.parquet', index=False)
    DatasetManifest(name='tiny', val_timestamp='2020-06-01',
                    test_timestamp=str(TEST_TIMESTAMP.date()),
                    tables=specs).save(root / 'manifest.yaml')
    return root


def embed(texts):
    r"""A stand-in text embedder: a few numbers per text. Missing values
    become empty texts, as in ``GloveTextEmbedding``."""
    import torch
    texts = ['' if t is None or (isinstance(t, float) and t != t) else t
             for t in texts]
    return torch.tensor([[float(len(t)), float(t.count('e')),
                          float(ord(t[0])) if t else 0.0] for t in texts])
