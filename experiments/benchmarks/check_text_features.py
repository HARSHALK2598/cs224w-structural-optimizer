"""Checks our text features against RelBench's, column by column (Plan 03).

For every text column of a dataset (after the test-date cut), the column's
GloVe vectors are built twice:

    relbench  PyTorch Frame's ``Dataset(...).materialize()``, in memory, as
              ``make_pkey_fkey_graph`` builds them
    ours      ``build_tables``: chunked, each distinct text embedded once,
              written to disk

and compared row by row. One column is built at a time, so memory holds one
column's vectors. Results go to ``experiments/results/text_check/``:

    python experiments/benchmarks/check_text_features.py rel-trial rel-event

A row whose vector differs by more than ``--tolerance`` counts as wrong;
GloVe's float32 rounding between batchings is about 1e-7.
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
os.environ.setdefault('HF_HOME', str(EXPERIMENTS.parent / 'data' / 'hf'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_frame  # noqa: E402
from relbench import load_dataset  # noqa: E402
from relbench.base import Database, Table  # noqa: E402

import disk_features  # noqa: E402
import train  # noqa: E402

FIELDS = ['dataset', 'table', 'column', 'rows', 'missing', 'distinct',
          'max_abs_diff', 'rows_off', 'missing_rows_off', 'minutes']


def check_column(df, column, cfg, tmp: Path, rows_per_chunk: int):
    ser = df[[column]]
    stypes = {column: torch_frame.text_embedded}
    ref = torch_frame.data.Dataset(df=ser, col_to_stype=stypes,
                                   col_to_text_embedder_cfg=cfg).materialize()
    ref = ref.tensor_frame.feat_dict[torch_frame.embedding].values
    path = tmp / 'one'
    shutil.rmtree(path, ignore_errors=True)
    disk_features.build_tables(
        Database({'one': Table(df=ser, fkey_col_to_pkey_table={})}),
        {'one': stypes}, path, cfg, rows_per_chunk, log=lambda _: None)
    with disk_features.memory_mapped_loading():
        tf, _ = torch_frame.load(str(path / 'one.pt'))
    ours = tf.feat_dict[torch_frame.embedding].values
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    diff = (ours - ref).abs().amax(dim=1)
    return diff, ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--tables', nargs='+', default=None,
                        help='Only these tables (default: all).')
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--rows-per-chunk', type=int, default=50_000)
    parser.add_argument('--out', default=str(
        EXPERIMENTS / 'results' / 'text_check' / 'results.csv'))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = train.text_embedder_cfg('glove', torch.device('cpu'))
    tmp = Path(tempfile.mkdtemp(prefix='text_check_'))
    try:
        for name in args.datasets:
            dataset = load_dataset(name)
            counts = disk_features.row_counts(dataset)
            for table_name in dataset.manifest.tables:
                if args.tables and table_name not in args.tables:
                    continue
                table = disk_features.read_table(dataset, table_name)
                disk_features.validate_table(dataset, table_name, table,
                                             counts)
                np.random.seed(0)
                stypes = train.propose_stypes(
                    Database({table_name: table}), 'glove')[table_name]
                for column, stype in stypes.items():
                    if stype != torch_frame.text_embedded:
                        continue
                    start = time.time()
                    df = table.df
                    diff, _ = check_column(df, column, cfg, tmp,
                                           args.rows_per_chunk)
                    texts = df[column].astype(str)
                    missing = texts.isna().to_numpy()
                    off = (diff > args.tolerance).numpy()
                    row = dict(
                        dataset=name, table=table_name, column=column,
                        rows=len(df), missing=int(missing.sum()),
                        distinct=int(texts.nunique(dropna=False)),
                        max_abs_diff=f'{diff.max().item():.3g}',
                        rows_off=int(off.sum()),
                        missing_rows_off=int((off & missing).sum()),
                        minutes=round((time.time() - start) / 60, 2))
                    print(row, flush=True)
                    new = not out.exists()
                    with open(out, 'a', newline='') as f:
                        writer = csv.DictWriter(f, FIELDS)
                        if new:
                            writer.writeheader()
                        writer.writerow(row)
                del table
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
