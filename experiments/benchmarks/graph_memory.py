"""Peak memory of building a dataset's graph: RelBench's path (all tables at
once) against reading the tables one at a time (Plan 03 Part 2).

Each measurement runs in a fresh process and reports that process's peak
resident memory (``ru_maxrss``), so one does not affect the other.

    python experiments/benchmarks/graph_memory.py rel-event
    python experiments/benchmarks/graph_memory.py rel-trial --steps get_db

Steps:
    get_db     RelBench's ``dataset.get_db()`` alone (no features)
    relbench   ``get_db()`` + ``make_pkey_fkey_graph``, features built fresh
    ours       ``build_graph`` (features built fresh) + ``load_cached_graph``
    ours-cached  ``load_cached_graph`` from the files ``ours`` saved

Results are appended to ``experiments/results/graph_memory/results.csv``.
"""
import argparse
import csv
import json
import multiprocessing as mp
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

STEPS = ('get_db', 'relbench', 'ours', 'ours-cached')
FIELDS = ['dataset', 'text_embedder', 'step', 'peak_gb', 'minutes',
          'num_nodes', 'num_edges']


def _measure(step, dataset_name, text_embedder, cache_dir, queue):
    import torch

    import disk_features
    import train
    from relbench import load_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph

    dataset = load_dataset(dataset_name)
    cpu = torch.device('cpu')
    start = time.time()
    data = None
    if step == 'get_db':
        db = dataset.get_db()
        sizes = {n: len(t) for n, t in db.table_dict.items()}
    elif step == 'relbench':
        db = dataset.get_db()
        data, _ = make_pkey_fkey_graph(
            db, train.propose_stypes(db, text_embedder),
            train.text_embedder_cfg(text_embedder, cpu), cache_dir=None)
    elif step == 'ours':
        disk_features.build_graph(
            dataset, Path(cache_dir),
            lambda db: train.propose_stypes(db, text_embedder),
            train.text_embedder_cfg(text_embedder, cpu))
        data, _ = disk_features.load_cached_graph(dataset, Path(cache_dir))
    else:
        data, _ = disk_features.load_cached_graph(dataset, Path(cache_dir))
        disk_features.keep_in_memory(data, 'text')
    minutes = (time.time() - start) / 60
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if data is not None:
        sizes = dict(num_nodes=sum(data[n].tf.num_rows
                                   for n in data.node_types),
                     num_edges=sum(data[e].edge_index.size(1)
                                   for e in data.edge_types))
    else:
        sizes = dict(num_nodes=sum(sizes.values()), num_edges='')
    queue.put(dict(peak_gb=round(peak_gb, 2), minutes=round(minutes, 2),
                   **sizes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--text-embedder', default='none')
    parser.add_argument('--steps', nargs='+', default=list(STEPS),
                        choices=STEPS)
    parser.add_argument('--out', default=str(
        EXPERIMENTS / 'results' / 'graph_memory' / 'results.csv'))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = tempfile.mkdtemp(prefix='graph_memory_')
    ctx = mp.get_context('spawn')  # A fresh process per step.
    try:
        for step in args.steps:
            queue = ctx.Queue()
            proc = ctx.Process(target=_measure, args=(
                step, args.dataset, args.text_embedder, cache_dir, queue))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                result = dict(peak_gb=f'failed (exit {proc.exitcode})')
            else:
                result = queue.get()
            row = dict(dataset=args.dataset, text_embedder=args.text_embedder,
                       step=step, **result)
            print(json.dumps(row), flush=True)
            new = not out.exists()
            with open(out, 'a', newline='') as f:
                writer = csv.DictWriter(f, FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
