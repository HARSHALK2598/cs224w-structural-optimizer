r"""Template gates in training (steps C and D)."""
import pytest
import torch
from conftest import requires_rel_f1
from relbench.modeling.nn import HeteroGraphSAGE

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


def first_batch(data, task, args):
    torch.manual_seed(0)
    return next(iter(train.make_loaders(data, task, args)['train']))


@requires_rel_f1
def test_gnn_without_injection_is_relbench(f1):
    _, data, _ = f1
    torch.manual_seed(0)
    ours = train.GatedGraphSAGE(data.node_types, data.edge_types, [], 16,
                                'sum', 2)
    torch.manual_seed(0)
    theirs = HeteroGraphSAGE(data.node_types, data.edge_types, 16, 'sum', 2)
    assert ours.state_dict().keys() == theirs.state_dict().keys()
    for key, value in theirs.state_dict().items():
        assert torch.equal(ours.state_dict()[key], value), key
    x_dict = {nt: torch.randn(5, 16) for nt in data.node_types}
    edges = {et: torch.randint(0, 5, (2, 7)) for et in data.edge_types}
    out, expected = ours(x_dict, edges), theirs(x_dict, edges)
    for key in expected:
        assert torch.equal(out[key], expected[key]), key


def gated_model(f1, *argv):
    task, data, col_stats = f1
    args = args_for('--inject', 'knn,twohop', '--template-gates', *argv)
    run_data = train.prepare_graph(data, args)
    torch.manual_seed(0)
    model = train.build_model(args, run_data, col_stats, 1, 'drivers')
    return task, run_data, args, model


@requires_rel_f1
def test_closed_templates_give_base(f1):
    task, run_data, args, model = gated_model(f1)
    assert model.gnn.gate.num_gates == 1 + 66  # KNN + two-hop instances
    base = train.build_model(args_for(), f1[1], f1[2], 1, 'drivers')
    # Same weights for everything Base has; injected relations add their
    # own convolutions only:
    missing, unexpected = model.load_state_dict(base.state_dict(),
                                                strict=False)
    assert unexpected == []
    assert all('gnn.convs' in key or 'gnn.gate' in key for key in missing)

    with torch.no_grad():
        model.gnn.gate.logit.fill_(-10.0)  # Every template closed.
    model.eval(), base.eval()
    batch = first_batch(run_data, task, args)
    with torch.no_grad():
        assert torch.allclose(model(batch, 'drivers'),
                              base(batch, 'drivers'), atol=1e-6)


@requires_rel_f1
def test_closed_templates_are_not_built(f1, monkeypatch):
    task, run_data, args, model = gated_model(f1)
    with torch.no_grad():
        model.gnn.gate.logit.fill_(-10.0)
        model.gnn.gate.logit[0] = 10.0  # Only KNN open.
    built = []
    for template in [*model.injection.knn, *model.injection.two_hop]:
        monkeypatch.setattr(
            template, 'forward', lambda *a, t=template, **k:
            (built.append(t.edge_type), (t.edge_type, torch.empty(
                (2, 0), dtype=torch.long)))[1])
    model.eval()
    with torch.no_grad():
        model(first_batch(run_data, task, args), 'drivers')
    assert built == [model.injection.knn[0].edge_type]


@requires_rel_f1
def test_template_gates_learn_from_the_task(f1):
    task, run_data, args, model = gated_model(f1, '--gate-init', '0')
    model.train()
    torch.manual_seed(3)
    pred = model(first_batch(run_data, task, args), 'drivers')
    pred.sum().backward()  # Task loss only (lambda_template = 0).
    grad = model.gnn.gate.logit.grad
    assert grad is not None and grad.abs().sum() > 0


@requires_rel_f1
def test_penalty_summary_and_optimizer(f1):
    _, _, args, model = gated_model(f1, '--lambda-template', '0.01',
                                    '--gate-lr', '0.1', '--gate-init', '5')
    penalty = model.structural_loss()
    expected = 0.01 * model.gnn.gate.l0_penalty()
    assert torch.allclose(penalty, expected) and penalty.requires_grad
    summary = model.gate_summary()
    assert summary['open_templates'] == 67
    assert len(summary['templates']) == 67
    optimizer = train.make_optimizer(model, args)
    assert optimizer.param_groups[1]['lr'] == 0.1
    assert optimizer.param_groups[1]['params'][0] is model.gnn.gate.logit


@requires_rel_f1
def test_filter_with_template_gates(f1):
    task, data, col_stats = f1
    args = args_for('--model', 'filter', '--inject', 'knn',
                    '--template-gates', '--lambda-col', '0.001',
                    '--lambda-template', '0.01')
    model = train.build_model(args, data, col_stats, 1, 'drivers')
    summary = model.gate_summary()
    assert {'open_columns', 'open_tables', 'open_templates'} <= set(summary)
    col, table = model.filter.l0_penalties()
    expected = (0.001 * col + 0.01 * model.gnn.gate.l0_penalty())
    assert torch.allclose(model.structural_loss(), expected)
    assert len(train.make_optimizer(
        model, train.parse_args(['--gate-lr', '0.1'])).param_groups) == 2
