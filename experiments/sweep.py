"""Runs a grid of `train.py` settings x seeds in one process, resumably.

The dataset is loaded once; every run then trains and evaluates one model,
exactly as `python experiments/train.py <args>` would.

Examples:
    # the full sweep (GPU if available)
    python experiments/sweep.py experiments/sweeps/gates_driver-top3.yaml

    # a quick check: 2 runs of 2 epochs, into a separate folder
    python experiments/sweep.py experiments/sweeps/gates_driver-top3.yaml \\
        --max-runs 2 --epochs 2 --out experiments/results/sweeps/check

Writes to `experiments/results/sweeps/<name>/` (or `--out`):
    results.csv      one row per finished run: run_id, device, train.py columns
    summary.csv      one row per setting: mean and std over seeds, best first
    logs/<run>.log   per-epoch loss, validation score and open gates
    gates/<run>.json every gate's state at the best epoch (--model filter)
    failures.log     tracebacks of runs that crashed (the sweep continues)
    env.json         versions, device and git commits of both repositories

Finished runs (by run_id in results.csv) are skipped, so an interrupted
sweep continues where it stopped when started again.
"""
import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

SWEEP_FIELDS = ['run_id', 'device'] + train.RESULT_FIELDS
# Short names for run ids; 'lambda' sets both lambda_col and lambda_table.
SHORT = dict(gate_lr='glr', gate_init='gi', gate_temperature='gt',
             lambda_col='lc', lambda_table='lt', lambda_='lam', lr='lr',
             channels='ch')


def expand(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""All runs of a sweep config, seed by seed: every setting runs once
    before any setting runs again, so a partial sweep covers all settings."""
    runs = []
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            grid = arm.get('grid', {})
            keys = list(grid.keys())
            for values in itertools.product(*(grid[k] for k in keys)):
                setting = {'model': arm['model']}
                for key, value in zip(keys, values):
                    if key == 'lambda':
                        setting.update(lambda_col=value, lambda_table=value)
                    else:
                        setting[key] = value
                # The run name lists the grid values (one 'lam' for 'lambda'):
                name = '_'.join([arm['model']] + [
                    f"{SHORT.get(k if k != 'lambda' else 'lambda_', k)}"
                    f"{v:g}" if isinstance(v, (int, float)) else
                    f"{SHORT.get(k, k)}-{v}"
                    for k, v in zip(keys, values)
                ])
                runs.append(dict(run_id=f'{name}_seed{seed}',
                                 setting=name, seed=seed,
                                 overrides=dict(setting, seed=seed)))
    return runs


def to_argv(options: Dict[str, Any]) -> List[str]:
    r"""Turns :obj:`{'gate_lr': 0.1, 'dedup_encoder': True}` into
    :obj:`['--gate-lr', '0.1', '--dedup-encoder']`."""
    argv = []
    for key, value in options.items():
        flag = '--' + key.replace('_', '-')
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            argv += [flag, str(value)]
    return argv


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # Not a git checkout.
        return 'unknown'


def env_info(device: torch.device) -> Dict[str, Any]:
    import relbench
    import torch_frame
    import torch_geometric
    return dict(
        started=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_geometric=torch_geometric.__version__,
        torch_frame=torch_frame.__version__,
        relbench=relbench.__version__,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        cpu_threads=torch.get_num_threads(),
        project_commit=git_commit(train.ROOT),
        pyg_commit=git_commit(Path(torch_geometric.__file__).parents[1]),
    )


def finished_runs(path: Path) -> set:
    if not path.exists():
        return set()
    import csv
    with open(path, newline='') as f:
        return {row['run_id'] for row in csv.DictReader(f)}


def run_sweep(cfg: Dict[str, Any], out: Path,
              max_runs: Optional[int] = None,
              epochs: Optional[int] = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = env_info(device)
    (out / 'env.json').write_text(json.dumps(env, indent=2))
    print(f"device {env['device']} {env['gpu'] or ''} | torch "
          f"{env['torch']} | project {env['project_commit'][:8]} | "
          f"pyg {env['pyg_commit'][:8]}", flush=True)

    fixed = dict(cfg.get('fixed', {}))
    if epochs is not None:
        fixed['epochs'] = epochs
    runs = expand(cfg)
    done = finished_runs(out / 'results.csv')
    todo = [r for r in runs if r['run_id'] not in done]
    if max_runs is not None:
        todo = todo[:max_runs]
    print(f'{len(runs)} runs in the sweep, {len(done)} already finished, '
          f'{len(todo)} to run now', flush=True)
    if not todo:
        return

    # Each (dataset, task) is loaded once; its runs are done together.
    def data_key(run) -> tuple:
        args = train.parse_args(to_argv({**fixed, **run['overrides']}))
        return args.dataset, args.task, args.text_embedder

    keys = [data_key(r) for r in todo]
    order = sorted(range(len(todo)), key=lambda j: keys.index(keys[j]))
    todo = [todo[j] for j in order]
    loaded_key = None

    start = time.time()
    for i, run in enumerate(todo, start=1):
        args = train.parse_args(to_argv({**fixed, **run['overrides']}))
        if data_key(run) != loaded_key:
            loaded_key = data_key(run)
            print(f'loading {loaded_key[0]} / {loaded_key[1]}', flush=True)
            dataset = train.load_dataset(args.dataset)
            task = dataset.load_task(args.task)
            data, col_stats_dict = train.load_graph(dataset, task,
                                                    args.text_embedder,
                                                    device)
            spec = train.task_spec(task)
        log_path = out / 'logs' / f"{run['run_id']}.log"
        with open(log_path, 'w') as log_file:

            def log(line: str) -> None:
                log_file.write(line + '\n')
                log_file.flush()

            log(f"run {run['run_id']}: {' '.join(to_argv(vars(args)))}")
            try:
                train.seed_everything(args.seed)
                loaders = train.make_loaders(data, task, args)
                model = train.build_model(args, data, col_stats_dict,
                                          spec.out_channels,
                                          task.entity_table).to(device)
                run_start = time.time()
                result = train.fit(model, loaders, task, spec, args, device,
                                   log=log)
                minutes = (time.time() - run_start) / 60
            except Exception:
                trace = traceback.format_exc()
                log(trace)
                with open(out / 'failures.log', 'a') as f:
                    f.write(f"== {run['run_id']}\n{trace}\n")
                print(f"[{i}/{len(todo)}] {run['run_id']}: FAILED "
                      f"(see failures.log)", flush=True)
                continue

        row = train.result_row(args, spec, result, minutes)
        row.update(run_id=run['run_id'], device=env['device'])
        train.write_result(str(out / 'results.csv'), row, SWEEP_FIELDS)
        if result['gates'] is not None:
            train.save_gates(out / 'gates' / f"{run['run_id']}.json",
                             result['gates'])
        elapsed = (time.time() - start) / 60
        eta = elapsed / i * (len(todo) - i)
        gates = ('' if result['gates'] is None else
                 f"  open {result['gates']['open_columns']} cols / "
                 f"{result['gates']['open_tables']} tables")
        print(f"[{i}/{len(todo)}] {run['run_id']}: val "
              f"{result['val'][spec.tune_metric]:.4f}  test "
              f"{result['test'][spec.tune_metric]:.4f}  best epoch "
              f"{result['best_epoch']}{gates}  ({minutes:.1f} min, "
              f"~{eta:.0f} min left)", flush=True)
        del model, loaders
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def summarize(out: Path, metric: Optional[str] = None):
    r"""One row per setting (all seeds pooled), best validation first;
    written to :obj:`summary.csv` and returned as a DataFrame."""
    import pandas as pd
    df = pd.read_csv(out / 'results.csv')
    df['setting'] = df['run_id'].str.replace(r'_seed\d+$', '', regex=True)
    metric = metric or df['tune_metric'].iloc[0]
    agg = dict(seeds=('seed', 'count'),
               val_mean=(f'val_{metric}', 'mean'),
               val_std=(f'val_{metric}', 'std'),
               test_mean=(f'test_{metric}', 'mean'),
               test_std=(f'test_{metric}', 'std'),
               best_epoch=('best_epoch', 'mean'),
               minutes=('minutes', 'mean'))
    for column in ['open_columns', 'open_tables']:
        if column in df and df[column].notna().any():
            agg[column] = (column, 'mean')
    summary = df.groupby('setting').agg(**agg).reset_index()
    summary = summary.sort_values('val_mean', ascending=False)
    summary.round(4).to_csv(out / 'summary.csv', index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Sweep YAML file.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output folder (default: '
                        'experiments/results/sweeps/<name>).')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Stop after this many runs (for checks).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the epochs of every run (checks).')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = Path(args.out) if args.out else (train.ROOT / 'experiments' /
                                           'results' / 'sweeps' / cfg['name'])
    run_sweep(cfg, out, args.max_runs, args.epochs)
    if (out / 'results.csv').exists():
        print(summarize(out).round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
