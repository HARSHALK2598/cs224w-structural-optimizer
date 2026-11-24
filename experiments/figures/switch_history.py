r"""Every switch's probability sigmoid(logit) after each training epoch, for
one structural-optimizer run with the tuned settings (blog Figure 3).

    python experiments/figures/switch_history.py  # writes switch_history.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'results/switch_history'
ARGV = [
    '--model', 'filter', '--seed', '0', '--epochs', '50', '--lr', '0.001',
    '--weight-decay', '0.005', '--batch-size', '256', '--num-neighbors', '64',
    '--channels', '128', '--num-layers', '1', '--temporal-strategy',
    'uniform', '--dedup-encoder', '--nan-safe-numerical', '--full-db',
    '--strict-time', '--lambda-col', '0.0001', '--lambda-table', '0.0001',
    '--lambda-template', '0.0001', '--inject',
    'temporal,twohop,knn,behavioral', '--template-gates', '--beta-vib',
    '0.001', '--knn-n', '10', '--temporal-m', '5', '--behavioral-threshold',
    '0.8', '--gate-init', '2.0'
]


def sigmoids(model):
    s = model.gate_summary()
    out = {f'table/{t}': v['sigmoid'] for t, v in s['tables'].items()}
    out.update({f'column/{t}/{c}': v['sigmoid']
                for t, cols in s['columns'].items() for c, v in cols.items()})
    out.update({f'edge/{e}': v['sigmoid']
                for e, v in s['templates'].items()})
    return out


def main():
    args = train.parse_args(ARGV)
    train.seed_everything(args.seed)
    device = torch.device('cpu')
    dataset = train.load_dataset(args.dataset)
    task = dataset.load_task(args.task)
    data, col_stats = train.load_graph(dataset, task, args.text_embedder,
                                       device, args.full_db)
    data = train.prepare_graph(data, args)
    spec = train.task_spec(task)
    loaders = train.make_loaders(data, task, args)
    model = train.build_model(args, data, col_stats, spec.out_channels,
                              task.entity_table)
    optimizer = train.make_optimizer(model, args)
    history = [dict(epoch=0, **sigmoids(model))]
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train.train_epoch(model, loaders['train'], optimizer, spec,
                          task.entity_table, device,
                          args.max_steps_per_epoch)
        history.append(dict(epoch=epoch, **sigmoids(model)))
        lowest = min(list(history[-1].values())[1:])
        print(f'epoch {epoch:02d}  min {lowest:.3f}'
              f'  ({time.time() - start:.0f} s)', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'switch_history.json').write_text(
        json.dumps(dict(argv=ARGV, history=history), indent=1))


if __name__ == '__main__':
    main()
