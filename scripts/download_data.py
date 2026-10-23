"""Download a single RelBench database (default: rel-f1) and print a summary.

Only the requested dataset's folder (database parquet files + its task labels) is
fetched from the Hugging Face Hub; nothing else from RelBench is downloaded. Files are
cached under ``data/hf`` (git-ignored) unless ``HF_HOME`` is already set.

Usage:
    python scripts/download_data.py --dataset rel-f1
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'hf'))

from relbench import load_dataset  # noqa: E402 (HF_HOME must be set first)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='rel-f1')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    db = dataset.get_db()
    print(f'{args.dataset}: {dataset.dataset_dir}')
    print(f'val={dataset.val_timestamp.date()}  '
          f'test={dataset.test_timestamp.date()}')
    for name, table in db.table_dict.items():
        print(f'  {name:<24} {len(table.df):>8} rows  '
              f'{len(table.df.columns):>3} cols  '
              f'fkeys={table.fkey_col_to_pkey_table}')
    print('tasks:', ', '.join(dataset.get_task_names()))


if __name__ == '__main__':
    main()
