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
    args = parser.parse_args()
    template = (Path(__file__).parent / 'run_sweep_kernel.py').read_text()
    code = template.replace('__CONFIG__', args.config)
    slug = 'relbench-' + Path(args.config).stem.replace('_', '-').lower()
    print(slug, kg.push(slug, code, gpu=args.gpu, title=slug))


if __name__ == '__main__':
    main()
