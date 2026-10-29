import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

RELF1_CACHED = any((ROOT / 'data' / 'hf' / 'hub').glob(
    'datasets--stanford-star--relbench-v1/snapshots/*/rel-f1/manifest.yaml'))

GLOVE_CACHED = any((ROOT / 'data' / 'hf' / 'hub').glob(
    'models--sentence-transformers--average_word_embeddings_glove.6B.300d/'
    'snapshots/*/modules.json'))

requires_glove = pytest.mark.skipif(
    not GLOVE_CACHED, reason='GloVe model not downloaded (~1GB)')

requires_rel_f1 = pytest.mark.skipif(
    not RELF1_CACHED,
    reason='rel-f1 not downloaded (run scripts/download_data.py)')


@pytest.fixture(scope='session')
def rel_f1():
    from relbench import load_dataset
    return load_dataset('rel-f1')
