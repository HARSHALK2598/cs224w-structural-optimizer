"""One run of the A/B comparison (Plan 03): RelBench's in-memory path against
our tables-one-at-a-time, features-on-disk path, on the same model.

    A_relbench  --features-on-disk none   get_db() + make_pkey_fkey_graph,
                                          everything in memory
    B_ours      --features-on-disk text   tables read one at a time, text
                                          vectors on disk, memory-mapped

Each call trains and evaluates one (arm, seed) with train.py's own steps and
settings, and appends one row to the results file: accuracy, the time of
each part (loading the graph, every epoch, one prediction pass) and peak
memory. Each arm has its own cache folder, so its first run builds the graph
from the raw tables and later runs load it.

    python experiments/benchmarks/ab_compare.py --arm A_relbench --seed 0 \\
        --dataset rel-arxiv --task paper-citation

Run each call in a fresh process (as the notebook does), so that memory and
timing are not carried over from another run.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / 'benchmarks'))
os.environ.setdefault('HF_HOME', str(EXPERIMENTS.parent / 'data' / 'hf'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import train  # noqa: E402
from graph_memory import MemorySampler  # noqa: E402

ARMS = {'A_relbench': 'none', 'B_ours': 'text'}
FIELDS = ['arm', 'features_on_disk', 'dataset', 'task', 'seed', 'device',
          'gpu', 'metric', 'val', 'test', 'best_epoch', 'epochs',
          'cache_was_built', 'load_s', 'epoch_s_mean', 'epoch_s_first',
          'fit_s', 'predict_val_s', 'total_s', 'peak_anon_gb',
          'peak_file_gb', 'gpu_peak_gb', 'pandas', 'torch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', default='rel-arxiv')
    parser.add_argument('--task', default='paper-citation')
    parser.add_argument('--epochs', type=int, default=None,
                        help="Default: train.py's (RelBench's) setting.")
    parser.add_argument('--text-embedder', default='glove')
    parser.add_argument('--cache-root', default=str(
        EXPERIMENTS.parent / 'data' / 'materialized_ab'))
    parser.add_argument('--out', default=str(
        EXPERIMENTS / 'results' / 'ab_compare' / 'results.csv'))
    args = parser.parse_args()

    features_on_disk = ARMS[args.arm]
    argv = ['--dataset', args.dataset, '--task', args.task, '--seed',
            str(args.seed), '--text-embedder', args.text_embedder,
            '--features-on-disk', features_on_disk]
    if args.epochs is not None:
        argv += ['--epochs', str(args.epochs)]
    targs = train.parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = (Path(args.cache_root) / args.dataset / args.text_embedder /
                 args.arm)
    was_built = not any(cache_dir.glob('*.pt'))

    sampler = MemorySampler()
    start = time.time()
    train.seed_everything(targs.seed)
    dataset = train.load_dataset(targs.dataset)
    task = dataset.load_task(targs.task)
    data, col_stats = train.load_graph(dataset, task, targs.text_embedder,
                                       device, features_on_disk, cache_dir)
    load_s = time.time() - start
    print(f'[{args.arm} seed {args.seed}] graph loaded in {load_s:.1f} s '
          f'({"built from the raw tables" if was_built else "from cache"})',
          flush=True)

    # The two paths draw different amounts of NumPy's random numbers while
    # loading (RelBench's always guesses column types from a random sample;
    # ours does not when the cache is built), and training draws from it
    # too. Reseeding here starts both arms' training from the same state.
    train.seed_everything(targs.seed)
    spec = train.task_spec(task)
    loaders = train.make_loaders(data, task, targs)
    model = train.build_model(targs, data, col_stats, spec.out_channels,
                              task.entity_table).to(device)
    epoch_ends = [time.time()]

    def log(line):  # fit() logs one line per epoch, after validation.
        if line.startswith('epoch'):
            epoch_ends.append(time.time())
        print(f'[{args.arm} seed {args.seed}] {line}', flush=True)

    fit_start = time.time()
    out = train.fit(model, loaders, task, spec, targs, device, log=log)
    fit_s = time.time() - fit_start
    epochs = np.diff(epoch_ends)

    predict_start = time.time()
    train.predict(model, loaders['val'], spec, task.entity_table, device)
    predict_val_s = time.time() - predict_start
    total_s = time.time() - start
    memory = sampler.stop()

    metric = spec.tune_metric
    row = dict(
        arm=args.arm, features_on_disk=features_on_disk,
        dataset=args.dataset, task=args.task, seed=args.seed,
        device=device.type,
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else '',
        metric=metric, val=round(out['val'][metric], 6),
        test=round(out['test'][metric], 6), best_epoch=out['best_epoch'],
        epochs=len(epochs), cache_was_built=was_built,
        load_s=round(load_s, 1),
        epoch_s_mean=round(float(epochs.mean()), 1) if len(epochs) else '',
        epoch_s_first=round(float(epochs[0]), 1) if len(epochs) else '',
        fit_s=round(fit_s, 1), predict_val_s=round(predict_val_s, 1),
        total_s=round(total_s, 1), peak_anon_gb=memory['peak_anon_gb'],
        peak_file_gb=memory['peak_file_gb'],
        gpu_peak_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                     if device.type == 'cuda' else ''),
        pandas=pd.__version__, torch=torch.__version__)
    print(row, flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    main()
