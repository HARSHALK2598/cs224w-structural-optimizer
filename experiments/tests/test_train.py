from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import requires_rel_f1

import train


def test_config_precedence(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('lr: 0.01\nepochs: 3\n')
    args = train.parse_args(['--config', str(config), '--lr', '0.02'])
    assert args.lr == 0.02  # CLI beats YAML
    assert args.epochs == 3  # YAML beats defaults
    assert args.batch_size == 512  # untouched default


def test_is_better_ties_go_to_the_later_epoch():
    assert train.is_better(0.7, None, higher_is_better=True)
    assert train.is_better(0.8, 0.8, higher_is_better=True)
    assert not train.is_better(0.79, 0.8, higher_is_better=True)
    assert train.is_better(0.5, 0.5, higher_is_better=False)
    assert not train.is_better(0.6, 0.5, higher_is_better=False)


def test_mae():
    assert train.mae(np.array([1., 2., 3.]), np.array([2., 2., 5.])) == 1.0


def test_load_graph_drops_hidden_columns(monkeypatch):
    captured = {}

    def fake_graph(db, **kwargs):
        captured.update(kwargs)
        return 'data', 'col_stats'

    monkeypatch.setattr(train, 'make_pkey_fkey_graph', fake_graph)
    monkeypatch.setattr(train, 'get_stype_proposal', lambda db: {})
    dataset = SimpleNamespace(name_or_path='rel-x', get_db=lambda: 'db')
    task = SimpleNamespace(hidden_columns=lambda: [('drivers', 'dob')])

    train.load_graph(dataset, task, 'none', torch.device('cpu'))
    assert captured['remove_columns'] == [('drivers', 'dob')]


@requires_rel_f1
def test_task_spec_classification(rel_f1):
    spec = train.task_spec(rel_f1.load_task('driver-top3'))
    assert isinstance(spec.loss_fn, torch.nn.BCEWithLogitsLoss)
    assert (spec.tune_metric, spec.higher_is_better) == ('roc_auc', True)
    assert [m.__name__ for m in spec.metrics] == ['roc_auc']
    assert spec.clip is None


@requires_rel_f1
def test_task_spec_regression(rel_f1):
    spec = train.task_spec(rel_f1.load_task('driver-position'))
    assert isinstance(spec.loss_fn, torch.nn.L1Loss)
    assert (spec.tune_metric, spec.higher_is_better) == ('nmae', False)
    assert [m.__name__ for m in spec.metrics] == ['nmae', 'mae']
    # 2nd / 98th percentiles of the 7,453 training labels (positions 1-39):
    assert spec.clip == (2.0, 31.0)


@requires_rel_f1
def test_evaluate_regression_reports_nmae_and_mae(rel_f1):
    task = rel_f1.load_task('driver-position')
    spec = train.task_spec(task)
    labels = task.get_table('val').df[task.target_col].to_numpy()

    perfect = train.evaluate(task, spec, labels, 'val')
    assert perfect == {'nmae': 0.0, 'mae': 0.0}
    metrics = train.evaluate(task, spec, labels + 1.0, 'val')
    assert metrics['mae'] == pytest.approx(1.0)
    assert metrics['nmae'] == pytest.approx(1.0 / task.nmae_std)


@requires_rel_f1
@pytest.mark.parametrize('bias', [1000.0, -1000.0])
def test_regression_predictions_are_clipped(rel_f1, bias):
    task = rel_f1.load_task('driver-position')
    spec = train.task_spec(task)
    args = train.parse_args(['--num-neighbors', '4', '--batch-size', '64'])
    data, col_stats = train.load_graph(rel_f1, task, 'none',
                                       torch.device('cpu'))
    loader = train.make_loaders(data, task, args)['val']
    model = train.BaseModel(data, col_stats, 128, 1, 2, 'sum')

    # Force a constant, far-out-of-range output from the head:
    head = model.head.lins[0]
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.constant_(head.bias, bias)

    pred = train.predict(model, loader, spec, task.entity_table,
                         torch.device('cpu'), max_steps=1)
    low, high = spec.clip
    assert len(pred) == 64
    assert np.all(pred == (high if bias > 0 else low))
