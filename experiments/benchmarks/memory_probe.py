r"""Peak memory of each step of one run, printed as it goes (so a run killed
for memory still shows the last step it finished).

    python experiments/benchmarks/memory_probe.py <sweep config>

Steps: the database, the graph (with feature statistics), the temporal
edges, one sampled training batch, and one training step of the first
structural-optimizer arm.
"""
import gc
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep  # noqa: E402
import train  # noqa: E402

START = time.time()


def memory_gb():
    fields = {}
    for line in open('/proc/self/status'):
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            fields[key] = int(value.split()[0]) / 1e6  # kB -> GB
    return fields['VmRSS'], fields['VmHWM']


def report(step, extra=''):
    gc.collect()
    now, peak = memory_gb()
    print(f'[{time.time() - START:7.0f} s] {step:34s} now {now:6.2f} GB  '
          f'peak {peak:6.2f} GB  {extra}', flush=True)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    run = next(r for r in sweep.expand(cfg)
               if r['overrides'].get('model') == 'filter')
    args = train.parse_args(sweep.to_argv({**cfg['fixed'],
                                           **run['overrides']}))
    total = open('/proc/meminfo').readline().split()[1]
    print(f'machine memory {int(total) / 1e6:.1f} GB | {args.dataset} '
          f'{args.task} | full_db {args.full_db}', flush=True)
    report('start')

    dataset = train.load_dataset(args.dataset)
    task = dataset.load_task(args.task)
    db = dataset.get_db(upto_test_timestamp=not args.full_db)
    rows = {name: len(t.df) for name, t in db.table_dict.items()}
    report('database loaded', f'{sum(rows.values()):,} rows')
    largest = sorted(rows.items(), key=lambda kv: -kv[1])
    print('  rows per table:', dict(largest), flush=True)
    del db
    gc.collect()

    data, col_stats = train.load_graph(dataset, task, args.text_embedder,
                                       torch.device('cpu'), args.full_db,
                                       not args.all_rows_col_stats)
    edges = sum(e.size(1) for e in data.edge_index_dict.values())
    report('graph built', f'{data.num_nodes:,} nodes, {edges:,} edges')

    data = train.prepare_graph(data, args)
    edges = sum(e.size(1) for e in data.edge_index_dict.values())
    report('temporal edges added', f'{edges:,} edges')

    loaders = train.make_loaders(data, task, args)
    batch = next(iter(loaders['train']))
    report('one training batch sampled',
           f'{batch.num_nodes:,} nodes in the batch')

    spec = train.task_spec(task)
    model = train.build_model(args, data, col_stats, spec.out_channels,
                              task.entity_table)
    optimizer = train.make_optimizer(model, args)
    model.train()
    pred = model(batch, task.entity_table).view(-1)
    loss = spec.loss_fn(pred, batch[task.entity_table].y.float())
    (loss + model.structural_loss()).backward()
    optimizer.step()
    report('one training step done')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
