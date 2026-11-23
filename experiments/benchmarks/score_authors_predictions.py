"""Scores the authors' released predictions on RelBench's test labels.

The authors' code (github.com/cy623/Structural_Optimizer_RDL, run unmodified
and not part of this repository) writes test probabilities with
`main.py --mode predict` but never scores them. This joins each
`<output_root>/predictions/<dataset>__<task>.csv` to the task's
`test.parquet` labels on the task's key columns and computes ROC-AUC.

    python experiments/benchmarks/score_authors_predictions.py \\
        --predictions <output_root>/predictions --data_root <their data/raw>
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

# The paper's test ROC-AUC (%) of its full model ("Ours"). Table 2, and
# Table 9 for rel-stack and rel-amazon. Table 9's user-churn (72.18) does not
# say which dataset it is for, so it is not assigned to one here.
PAPER = {
    ('rel-trial', 'study-outcome'): 72.35,
    ('rel-f1', 'driver-dnf'): 78.21,
    ('rel-f1', 'driver-top3'): 86.82,
    ('rel-event', 'user-repeat'): 82.36,
    ('rel-event', 'user-ignore'): 81.25,
    ('rel-avito', 'user-clicks'): 67.53,
    ('rel-avito', 'user-visits'): 70.15,
    ('rel-stack', 'user-engagement'): 91.62,
    ('rel-stack', 'user-badge'): 90.47,
    ('rel-amazon', 'item-churn'): 84.17,
}
FIELDS = ['dataset', 'task', 'rows', 'positive_rate', 'test_roc_auc',
          'paper_test_roc_auc', 'difference']


def score(prediction: Path, data_root: Path):
    dataset, task = prediction.stem.split('__')
    import yaml
    task_dir = data_root / dataset / 'tasks' / task
    manifest = yaml.safe_load((task_dir / 'manifest.yaml').read_text())
    keys = [manifest['entity_col'], manifest['time_col']]
    target = manifest['target_col']
    labels = pd.read_parquet(task_dir / 'test.parquet')
    pred = pd.read_csv(prediction)
    labels[keys[1]] = pd.to_datetime(labels[keys[1]])
    pred[keys[1]] = pd.to_datetime(pred[keys[1]])
    merged = labels.merge(pred, on=keys, suffixes=('', '_prob'),
                          validate='one_to_one')
    if len(merged) != len(labels) or len(pred) != len(labels):
        raise ValueError(f'{prediction.name}: {len(pred)} predictions for '
                         f'{len(labels)} test rows ({len(merged)} matched)')
    auc = 100 * roc_auc_score(merged[target], merged[target + '_prob'])
    paper = PAPER.get((dataset, task))
    return dict(dataset=dataset, task=task, rows=len(merged),
                positive_rate=round(float(merged[target].mean()), 4),
                test_roc_auc=round(auc, 2), paper_test_roc_auc=paper,
                difference='' if paper is None else round(auc - paper, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).parents[1]
                        / 'results' / 'authors_weights' / 'results.csv')
    args = parser.parse_args()
    rows = [score(p, args.data_root)
            for p in sorted(args.predictions.glob('*__*.csv'))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)


if __name__ == '__main__':
    main()
