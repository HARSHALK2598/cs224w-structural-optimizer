r"""Structural injection in training (plans/04-structural-injection.md)."""
import pytest
import torch
from conftest import requires_rel_f1

import sweep
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


def test_unknown_template_is_rejected():
    with pytest.raises(SystemExit):
        train.parse_args(['--inject', 'knn,shortcut'])


@requires_rel_f1
def test_no_injection_is_base(f1):
    task, data, col_stats = f1
    args = args_for()
    assert train.prepare_graph(data, args) is data
    assert train.make_injection(args, data, 'drivers') is None
    torch.manual_seed(0)
    model = train.build_model(args, data, col_stats, 1, 'drivers')
    torch.manual_seed(0)
    base = train.BaseModel(data, col_stats, args.channels, 1,
                           args.num_layers, args.aggr)
    assert model.injection is None and model.num_injected_relations == 0
    batch = first_batch(data, task, args)
    model.eval(), base.eval()
    with torch.no_grad():
        assert torch.equal(model(batch, 'drivers'), base(batch, 'drivers'))


@requires_rel_f1
def test_temporal_relations(f1):
    _, data, _ = f1
    before = list(data.edge_types)
    out = train.prepare_graph(data, args_for('--inject', 'temporal'))
    added = [et for et in out.edge_types if et[1].startswith('next__')]
    # Records of the lasting entities (no time column): drivers,
    # constructors, circuits (plans/04, section 3.1):
    assert {(et[0], et[1]) for et in added} == {
        ('races', 'next__f2p_circuitId'),
        ('qualifying', 'next__f2p_driverId'),
        ('qualifying', 'next__f2p_constructorId'),
        ('constructor_standings', 'next__f2p_constructorId'),
        ('standings', 'next__f2p_driverId'),
        ('constructor_results', 'next__f2p_constructorId'),
        ('results', 'next__f2p_driverId'),
        ('results', 'next__f2p_constructorId'),
    }
    assert data.edge_types == before  # The loaded graph is unchanged.
    for edge_type in added:  # Older (or same time) -> newer only:
        src, dst = out[edge_type].edge_index
        time = out[edge_type[0]].time
        assert (time[src] <= time[dst]).all()


@requires_rel_f1
def test_instance_counts(f1):
    _, data, _ = f1
    injection = train.make_injection(
        args_for('--inject', 'knn,behavioral,twohop'), data, 'drivers')
    assert len(injection.knn) == 1  # drivers only, by default
    assert {t.via[0] for t in injection.behavioral} == {
        'results', 'standings', 'qualifying'
    }
    assert len(injection.two_hop) == 66
    every = train.make_injection(
        args_for('--inject', 'knn', '--inject-tables', 'all'), data,
        'drivers')
    assert len(every.knn) == len(data.node_types)


@requires_rel_f1
@pytest.mark.parametrize('inject', ['temporal', 'twohop', 'knn',
                                    'behavioral'])
def test_injected_edges_reach_message_passing(f1, inject):
    task, data, col_stats = f1
    args = args_for('--inject', inject, '--behavioral-threshold', '0.5')
    run_data = train.prepare_graph(data, args)
    torch.manual_seed(0)
    model = train.build_model(args, run_data, col_stats, 1, 'drivers')
    assert model.num_injected_relations > 0
    injected = [et for et in run_data.edge_types if et[1].startswith('next__')]
    if model.injection is not None:
        injected += model.injection.edge_types
    assert len(injected) == model.num_injected_relations
    # Each one has a convolution in every layer:
    assert len(model.gnn.convs[0].convs) == len(data.edge_types) + len(
        injected)

    batch = first_batch(run_data, task, args)
    model.train()
    out = model(batch, 'drivers')
    out.sum().backward()
    # The injected relations' convolutions take part: their weights get a
    # gradient (in the authors' code, injected edges have no convolution).
    grads = [
        model.gnn.convs[layer].convs[et].lin_l.weight.grad
        for layer in range(args.num_layers) for et in injected
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


@requires_rel_f1
def test_no_future_in_a_real_batch(f1):
    task, data, col_stats = f1
    args = args_for('--inject', 'knn,behavioral,twohop',
                    '--behavioral-threshold', '0.0', '--inject-tables',
                    'drivers,results')
    model = train.build_model(args, data, col_stats, 1, 'drivers')
    batch = first_batch(data, task, args)
    x_dict = model.encode(batch, 'drivers')
    seed_time = batch['drivers'].seed_time
    edges = model.injection(x_dict, batch, 'drivers')
    assert sum(e.size(1) for e in edges.values()) > 0
    for (src, rel, dst), edge_index in edges.items():
        t_src = seed_time[batch[src].batch[edge_index[0]]]
        t_dst = seed_time[batch[dst].batch[edge_index[1]]]
        assert (t_src <= t_dst).all(), rel


def test_sweep_run_names_with_text_values():
    cfg = dict(seeds=[0], arms=[
        dict(model='base'),
        dict(model='base', grid=dict(inject=['knn', 'twohop,knn'],
                                     knn_n=[5])),
        dict(model='base', fixed=dict(inject='temporal'),
             grid=dict(temporal_m=[1, 5])),
    ])
    runs = sweep.expand(cfg)
    assert [r['run_id'] for r in runs] == [
        'base_seed0', 'base_inject-knn_n5_seed0',
        'base_inject-twohop+knn_n5_seed0', 'base_m1_inject-temporal_seed0',
        'base_m5_inject-temporal_seed0'
    ]
    assert runs[3]['overrides']['inject'] == 'temporal'
    assert sweep.to_argv(runs[1]['overrides'])[:4] == [
        '--model', 'base', '--inject', 'knn'
    ]
