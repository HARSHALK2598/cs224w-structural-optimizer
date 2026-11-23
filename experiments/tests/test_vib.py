r"""The variational bottleneck in training (step A)."""
import pytest
import torch
from conftest import requires_rel_f1

import train


@pytest.fixture(scope='module')
def f1(rel_f1):
    task = rel_f1.load_task('driver-top3')
    data, col_stats = train.load_graph(rel_f1, task, 'none',
                                       torch.device('cpu'))
    return task, data, col_stats


def args_for(*argv):
    return train.parse_args(['--text-embedder', 'none', '--batch-size', '64',
                             '--num-neighbors', '16', *argv])


@requires_rel_f1
@pytest.mark.parametrize('model', ['base', 'filter'])
def test_vib(f1, model):
    task, data, col_stats = f1
    args = args_for('--model', model, '--beta-vib', '0.01')
    torch.manual_seed(0)
    gated = train.build_model(args, data, col_stats, 1, 'drivers')
    torch.manual_seed(0)
    plain = train.build_model(args_for('--model', model), data, col_stats, 1,
                              'drivers')
    assert plain.vib is None and gated.vib is not None
    # Created last: every other module starts as without the bottleneck.
    for key, value in plain.state_dict().items():
        assert torch.equal(gated.state_dict()[key], value), key
    assert all(key.startswith('vib.') for key in set(gated.state_dict()) -
               set(plain.state_dict()))

    torch.manual_seed(0)
    batch = next(iter(train.make_loaders(data, task, args)['train']))
    gated.train()
    out = gated(batch, 'drivers')
    kl = gated._kl
    assert kl.item() > 0
    penalty = gated.structural_loss()
    if model == 'base':
        assert torch.allclose(penalty, 0.01 * kl)
    (out.sum() + penalty).backward()
    assert gated.vib.logvar['drivers'].weight.grad.abs().sum() > 0

    gated.eval()  # The mean in evaluation: deterministic.
    with torch.no_grad():
        assert torch.equal(gated(batch, 'drivers'), gated(batch, 'drivers'))


def test_vib_result_field():
    args = train.parse_args(['--beta-vib', '0.001'])
    assert 'beta_vib' in train.RESULT_FIELDS
    spec = train.TaskSpec(None, 1, 'roc_auc', True, [], None)
    row = train.result_row(args, spec, dict(best_epoch=1, val={}, test={},
                                            gates=None), 0.0)
    assert row['beta_vib'] == 0.001
