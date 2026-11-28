r"""Starts a sweep config as a private Kaggle notebook.

    python experiments/kaggle/launch.py <sweep config> [--gpu]

The Kaggle token is sent by the environment's network proxy; kg.py holds only
a placeholder. Results are the notebook's output (kg.output / download).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--command', default='',
                        help='A script (and arguments) to run instead of '
                        'the sweep, e.g. a benchmark.')
    parser.add_argument('--name', default=None,
                        help='The notebook name (default: from the config).')
    parser.add_argument('--project-branch', default='step/report')
    parser.add_argument('--pyg-branch', default='contrib/structural-optimizer')
    args = parser.parse_args()
    template = (Path(__file__).parent / 'run_sweep_kernel.py').read_text()
    code = (template.replace('__CONFIG__', args.config)
            .replace('__PROJECT_BRANCH__', args.project_branch)
            .replace('__PYG_BRANCH__', args.pyg_branch)
            .replace('__COMMAND__', args.command))
    slug = args.name or ('relbench-' +
                         Path(args.config).stem.replace('_', '-').lower())
    print(slug, kg.push(slug, code, gpu=args.gpu, title=slug))


if __name__ == '__main__':
    main()
