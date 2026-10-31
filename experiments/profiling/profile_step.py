"""Profiles one training epoch of the Base model on a rel-f1 task.

1. Wall-clock time per stage of every training step.
2. Duplicate rows per batch (what --dedup-encoder removes).
3. torch.profiler: the most expensive operators over a few steps.
4. How often hop-1 sampling is actually random (eligible rows > fan-out).

Usage:
    python experiments/profiling/profile_step.py --task driver-dnf
    python experiments/profiling/profile_step.py --task driver-dnf \
        --dedup-encoder
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'hf'))
sys.path.insert(0, str(ROOT / 'experiments'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from relbench import load_dataset  # noqa: E402
from torch_geometric.seed import seed_everything  # noqa: E402

import train  # noqa: E402

cli = argparse.ArgumentParser()
cli.add_argument('--task', default='driver-dnf')
cli.add_argument('--dedup-encoder', action='store_true')
cli = cli.parse_args()
TASK = cli.task
cpu = torch.device('cpu')
seed_everything(0)
print('torch threads:', torch.get_num_threads())

t0 = time.perf_counter()
dataset = load_dataset('rel-f1')
task = dataset.load_task(TASK)
data, col_stats = train.load_graph(dataset, task, 'glove', cpu)
spec = train.task_spec(task)
args = train.parse_args([])
loaders = train.make_loaders(data, task, args)
model = train.BaseModel(data, col_stats, 128, 1, 2, 'sum',
                        dedup_encoder=cli.dedup_encoder)
print('dedup_encoder:', cli.dedup_encoder)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
print(f'setup (graph from cache, loaders, model): '
      f'{time.perf_counter() - t0:.1f}s')
print(f'{TASK}: {len(task.get_table("train").df)} train targets, '
      f'{len(loaders["train"])} batches/epoch\n')

# ---- 1. per-stage wall clock over one epoch --------------------------------
stages = defaultdict(float)
sizes, distinct = [], []
entity = task.entity_table
model.train()
it = iter(loaders['train'])
epoch_start = time.perf_counter()
while True:
    t = time.perf_counter()
    try:
        batch = next(it)
    except StopIteration:
        break
    stages['1 sample mini-batch (NeighborLoader)'] += time.perf_counter() - t
    sizes.append(sum(batch[nt].num_nodes for nt in batch.node_types))
    distinct.append(sum(batch[nt].n_id.unique().numel()
                        for nt in batch.node_types))

    t = time.perf_counter()
    seed_time = batch[entity].seed_time
    x = model.encode_rows(batch)
    stages['2 row encoder (ResNet per table)'] += time.perf_counter() - t

    t = time.perf_counter()
    rel = model.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
    for k, v in rel.items():
        x[k] = x[k] + v
    stages['3 time encoding'] += time.perf_counter() - t

    t = time.perf_counter()
    x = model.gnn(x, batch.edge_index_dict)
    stages['4 GNN (2 GraphSAGE layers)'] += time.perf_counter() - t

    t = time.perf_counter()
    pred = model.head(x[entity][:seed_time.size(0)]).view(-1)
    loss = spec.loss_fn(pred, batch[entity].y.float())
    stages['5 head + loss'] += time.perf_counter() - t

    t = time.perf_counter()
    opt.zero_grad()
    loss.backward()
    stages['6 backward pass'] += time.perf_counter() - t

    t = time.perf_counter()
    opt.step()
    stages['7 optimizer step (Adam)'] += time.perf_counter() - t
epoch = time.perf_counter() - epoch_start

print(f'one training epoch: {epoch:.1f}s, {len(sizes)} steps, '
      f'{np.mean(sizes):,.0f} sampled nodes per batch on average, '
      f'{np.mean(distinct):,.0f} distinct rows '
      f'({np.mean(sizes) / np.mean(distinct):.1f}x duplication)')
for name, sec in sorted(stages.items()):
    print(f'  {name:40s} {sec:6.1f}s  {100 * sec / epoch:5.1f}%')

t = time.perf_counter()
train.predict(model, loaders['val'], spec, entity, cpu)
print(f'validation pass ({len(task.get_table("val").df)} targets): '
      f'{time.perf_counter() - t:.1f}s\n')

# ---- 2. operator-level profile over 3 steps --------------------------------
from torch.profiler import ProfilerActivity, profile  # noqa: E402

it = iter(loaders['train'])
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        batch = next(it)
        loss = spec.loss_fn(model(batch, entity).view(-1),
                            batch[entity].y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
print('top operators over 3 steps (model forward + backward only):')
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=12,
                                max_name_column_width=40))

# ---- 3. how random is hop-1 sampling? --------------------------------------
fanout = args.num_neighbors
targets = task.get_table('train').df
t_target = targets[task.time_col].astype('int64').to_numpy() // 10**9
d_target = targets[task.entity_col].to_numpy()
print(f'hop 1 (rows -> {entity}), fan-out {fanout} per relation:')
for (src, rel, dst) in data.edge_types:
    if dst != entity:
        continue
    ei = data[src, rel, dst].edge_index
    row_time = data[src].time[ei[0]].numpy()
    owner = ei[1].numpy()
    eligible = np.zeros(len(targets), dtype=int)
    by_driver = defaultdict(list)
    for d, tt in zip(owner, row_time):
        by_driver[d].append(tt)
    by_driver = {d: np.sort(v) for d, v in by_driver.items()}
    for i, (d, tt) in enumerate(zip(d_target, t_target)):
        if d in by_driver:
            eligible[i] = np.searchsorted(by_driver[d], tt, side='right')
    over = eligible > fanout
    print(f'  {src:12s} eligible rows per target: median '
          f'{np.median(eligible):5.0f}, max {eligible.max():4d} | targets '
          f'with > {fanout} (random): {over.sum():5d} / {len(targets)} '
          f'({100 * over.mean():.1f}%)')
print('hop 2: every hop-1 row links to exactly one race / driver / '
      'constructor, so hop 2 never needs to sample.')
