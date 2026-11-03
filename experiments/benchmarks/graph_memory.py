"""Peak memory of building a dataset's graph: RelBench's path (all tables at
once) against reading the tables one at a time (Plan 03 Part 2), and of
training on it.

Each measurement runs in a fresh process and reports that process's peak
resident memory (``ru_maxrss``), so one does not affect the other.

    python experiments/benchmarks/graph_memory.py rel-event
    python experiments/benchmarks/graph_memory.py rel-trial --steps get_db
    python experiments/benchmarks/graph_memory.py rel-event --task \\
        user-repeat --text-embedder glove --cache-dir train \\
        --steps ours ours-cached train

Steps:
    get_db       RelBench's ``dataset.get_db()`` alone (no features)
    relbench     ``get_db()`` + ``make_pkey_fkey_graph``, features built fresh
    ours         ``build_graph`` + ``load_cached_graph`` (features built,
                 unless the cache already has them: see ``cache_files``)
    ours-cached  ``load_cached_graph`` from the files ``ours`` saved
    train        ``train.py``'s steps with ``--features-on-disk text``: load
                 the graph, train ``--epochs`` epochs, evaluate (needs
                 ``--task`` and ``--cache-dir train``)

``--cache-dir`` is a temporary folder by default, deleted at the end;
``train`` uses ``train.py``'s own cache folder, kept, so that ``train`` (and
later runs) find what ``ours`` built.

Results are appended to ``experiments/results/graph_memory/results.csv``.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
# Where train.py keeps downloads; set before anything imports RelBench:
os.environ.setdefault('HF_HOME', str(EXPERIMENTS.parent / 'data' / 'hf'))

STEPS = ('get_db', 'relbench', 'ours', 'ours-cached', 'train')
FIELDS = ['dataset', 'text_embedder', 'step', 'peak_gb', 'minutes',
          'num_nodes', 'num_edges', 'task', 'cache_files', 'gpu_peak_gb',
          'val', 'device', 'peak_anon_gb', 'peak_file_gb']


class MemorySampler:
    r"""Peaks of this process's memory, sampled every 0.1 s, split into the
    kinds of ``/proc/self/status``: ``RssAnon`` (ordinary memory, which the
    system cannot drop) and ``RssFile`` + ``RssShmem`` (pages of files mapped
    into memory, e.g. memory-mapped features, which it can drop or write
    back). ``ru_maxrss`` (``peak_gb``) is the peak of their sum."""
    def __init__(self):
        import threading
        self.anon = self.file = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _read(self):
        kb = {}
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(('RssAnon:', 'RssFile:', 'RssShmem:')):
                    kb[line.split(':')[0]] = int(line.split()[1])
        return kb.get('RssAnon', 0), kb.get('RssFile', 0) + kb.get(
            'RssShmem', 0)

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(0.1)

    def sample(self):
        anon, file = self._read()
        self.anon, self.file = max(self.anon, anon), max(self.file, file)

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.sample()
        return dict(peak_anon_gb=round(self.anon / 1e6, 2),
                    peak_file_gb=round(self.file / 1e6, 2))


def _measure(step, dataset_name, text_embedder, cache_dir, queue,
             task_name=None, epochs=1):
    import torch

    import disk_features
    import train
    from relbench import load_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph

    dataset = load_dataset(dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_files = len(list(Path(cache_dir).glob('*.pt')))
    sampler = MemorySampler()
    start = time.time()
    data, extra = None, {}
    if step == 'get_db':
        db = dataset.get_db()
        sizes = {n: len(t) for n, t in db.table_dict.items()}
    elif step == 'relbench':
        db = dataset.get_db()
        data, _ = make_pkey_fkey_graph(
            db, train.propose_stypes(db, text_embedder),
            train.text_embedder_cfg(text_embedder, device), cache_dir=None)
    elif step == 'ours':
        disk_features.build_graph(
            dataset, Path(cache_dir),
            lambda db: train.propose_stypes(db, text_embedder),
            train.text_embedder_cfg(text_embedder, device))
        data, _ = disk_features.load_cached_graph(dataset, Path(cache_dir))
    elif step == 'ours-cached':
        data, _ = disk_features.load_cached_graph(dataset, Path(cache_dir))
        disk_features.keep_in_memory(data, 'text')
    else:  # train, as train.py's main() does:
        args = train.parse_args([
            '--dataset', dataset_name, '--task', task_name,
            '--text-embedder', text_embedder, '--features-on-disk', 'text',
            '--epochs', str(epochs)])
        train.seed_everything(args.seed)
        task = dataset.load_task(task_name)
        data, col_stats = train.load_graph(dataset, task, text_embedder,
                                           device, 'text')
        spec = train.task_spec(task)
        loaders = train.make_loaders(data, task, args)
        model = train.build_model(args, data, col_stats, spec.out_channels,
                                  task.entity_table).to(device)
        out = train.fit(model, loaders, task, spec, args, device,
                        log=lambda line: print(line, flush=True))
        extra['val'] = round(out['val'][spec.tune_metric], 4)
        if device.type == 'cuda':
            extra['gpu_peak_gb'] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2)
    minutes = (time.time() - start) / 60
    extra.update(sampler.stop())
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if data is not None:
        sizes = dict(num_nodes=sum(data[n].tf.num_rows
                                   for n in data.node_types),
                     num_edges=sum(data[e].edge_index.size(1)
                                   for e in data.edge_types))
    else:
        sizes = dict(num_nodes=sum(sizes.values()), num_edges='')
    queue.put(dict(peak_gb=round(peak_gb, 2), minutes=round(minutes, 2),
                   cache_files=cache_files, device=device.type, **sizes,
                   **extra))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--task', default=None, help='For the train step.')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--text-embedder', default='none')
    parser.add_argument('--steps', nargs='+', default=list(STEPS[:4]),
                        choices=STEPS)
    parser.add_argument('--cache-dir', default=None,
                        help="'train' for train.py's cache folder (kept); "
                        "default: a temporary folder (deleted).")
    parser.add_argument('--out', default=str(
        EXPERIMENTS / 'results' / 'graph_memory' / 'results.csv'))
    args = parser.parse_args()
    if 'train' in args.steps and (args.task is None
                                  or args.cache_dir != 'train'):
        parser.error("the train step needs --task and --cache-dir train")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_dir == 'train':
        cache_dir = str(EXPERIMENTS.parent / 'data' / 'materialized' /
                        args.dataset / args.text_embedder)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = tempfile.mkdtemp(prefix='graph_memory_')
    ctx = mp.get_context('spawn')  # A fresh process per step.
    try:
        for step in args.steps:
            queue = ctx.Queue()
            proc = ctx.Process(target=_measure, args=(
                step, args.dataset, args.text_embedder, cache_dir, queue,
                args.task, args.epochs))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                result = dict(peak_gb=f'failed (exit {proc.exitcode})')
            else:
                result = queue.get()
            row = dict(dataset=args.dataset, text_embedder=args.text_embedder,
                       step=step, task=args.task if step == 'train' else '',
                       **result)
            print(json.dumps(row), flush=True)
            new = not out.exists()
            if not new:  # Older files have fewer columns: keep one header.
                with open(out) as f:
                    header = f.readline().strip().split(',')
                if header != FIELDS:
                    with open(out) as f:
                        rows = list(csv.DictReader(f))
                    with open(out, 'w', newline='') as f:
                        writer = csv.DictWriter(f, FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
            with open(out, 'a', newline='') as f:
                writer = csv.DictWriter(f, FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
    finally:
        if args.cache_dir != 'train':
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
