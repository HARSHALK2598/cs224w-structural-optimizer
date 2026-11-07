import ast
from pathlib import Path

import pytest

import train

OFFICIAL = (Path(__file__).resolve().parents[2] / 'reference' / 'relbench' /
            'examples' / 'gnn_entity.py')


def official_defaults():
    r"""Reads the `default=` of every `parser.add_argument('--x', ...)` in the
    official script, without running it."""
    defaults = {}
    for node in ast.walk(ast.parse(OFFICIAL.read_text())):
        if (isinstance(node, ast.Call)
                and getattr(node.func, 'attr', None) == 'add_argument'):
            name = node.args[0].value.lstrip('-')
            for kw in node.keywords:
                if kw.arg == 'default' and isinstance(kw.value, ast.Constant):
                    defaults[name] = kw.value.value
    return defaults


@pytest.mark.skipif(not OFFICIAL.exists(),
                    reason='RelBench repo not cloned (scripts/setup_env.sh)')
def test_defaults_match_the_official_script():
    official = official_defaults()
    # Settings that define the Base model and its training:
    for key in ['lr', 'epochs', 'batch_size', 'channels', 'aggr', 'num_layers',
                'num_neighbors', 'max_steps_per_epoch']:
        assert train.DEFAULTS[key] == official[key], key
    # make_loaders() always samples with temporal_strategy='uniform':
    assert official['temporal_strategy'] == 'uniform'
    assert official['gnn'] == 'sage'  # BaseModel is the GraphSAGE variant
