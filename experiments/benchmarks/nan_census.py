"""Missing-value census of RelBench's numerical columns (Plan 02, step 2.1).

PyTorch Frame's default numerical encoders turn a missing value into a NaN
weight gradient, after which the whole column encodes to 0 (issue #6). This
lists, per dataset, every column RelBench encodes as numerical and how often
it is missing, so we know which tasks the bug can affect. No training.

    python experiments/benchmarks/nan_census.py rel-f1 rel-event rel-trial

Writes experiments/results/nan_census/<dataset>.csv and prints a summary.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402  (sets HF_HOME)
from relbench.modeling.utils import get_stype_proposal  # noqa: E402


def census(name: str) -> pd.DataFrame:
    db = train.load_dataset(name).get_db()
    stypes = get_stype_proposal(db)
    rows = []
    for table_name, table in db.table_dict.items():
        df = table.df
        for col, stype in stypes[table_name].items():
            if stype != torch_frame.numerical:
                continue
            missing = df[col].isna()
            rows.append(dict(dataset=name, table=table_name, column=col,
                             rows=len(df), missing=int(missing.sum()),
                             missing_frac=round(float(missing.mean()), 4)))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    args = parser.parse_args()
    out = train.ROOT / 'experiments' / 'results' / 'nan_census'
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        df = census(name)
        df.to_csv(out / f'{name}.csv', index=False)
        hit = df[df.missing > 0]
        print(f'{name}: {len(df)} numerical columns, {len(hit)} with missing '
              f'values (in {hit.table.nunique()} of '
              f'{df.table.nunique()} tables)', flush=True)
        if len(hit):
            print(hit.sort_values('missing_frac', ascending=False)
                  .to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
