import pytest
import torch
from conftest import requires_rel_f1

import train


@pytest.fixture(scope='module')
def setup(rel_f1):
    torch.manual_seed(0)
    task = rel_f1.load_task('driver-top3')
    data, col_stats = train.load_graph(rel_f1, task, 'none',
                                       torch.device('cpu'))
    args = train.parse_args(['--batch-size', '64', '--num-neighbors', '16'])
    batch = next(iter(train.make_loaders(data, task, args)['train']))
    return task, data, col_stats, batch


def make(setup, **kwargs):
    task, data, col_stats, _ = setup
    torch.manual_seed(0)
    return train.FilterModel(data, col_stats, 128, 1, 2, 'sum',
                             task.entity_table, **kwargs)


@requires_rel_f1
def test_filter_model_gates(setup):
    task, data, _, batch = setup
    model = make(setup)
    gates = model.filter.column_gates
    assert sum(g.num_gates for g in gates.values()) == sum(
        data[nt].tf.num_cols for nt in data.node_types)
    # Every table has a table gate, except the target table:
    assert sorted(model.filter.table_gates.keys()) == sorted(
        nt for nt in data.node_types if nt != task.entity_table)

    # Gate names follow the order of the encoder's columns:
    for node_type, tf in batch.tf_dict.items():
        _, col_names = model.encoder.encoders[node_type].encoder(tf)
        assert model.col_names[node_type] == col_names, node_type


@requires_rel_f1
@pytest.mark.parametrize('dedup', [False, True])
def test_filter_model_all_open_is_base(setup, dedup):
    task, data, col_stats, batch = setup
    model = make(setup, dedup_encoder=dedup)
    base = train.BaseModel(data, col_stats, 128, 1, 2, 'sum', dedup)
    base.load_state_dict(model.state_dict(), strict=False)
    model.eval()
    base.eval()
    # Initial logits (2.0) are above the threshold: every gate is open.
    with torch.no_grad():
        out = model(batch, task.entity_table)
        expected = base(batch, task.entity_table)
    assert torch.equal(out, expected)


@requires_rel_f1
def test_filter_model_closed_gates(setup):
    task, _, _, batch = setup
    model = make(setup)
    model.eval()
    with torch.no_grad():
        model.filter.table_gates['results'].logit.fill_(-10.0)
        x_dict = model.encode(batch, task.entity_table)
    assert torch.equal(x_dict['results'],
                       torch.zeros_like(x_dict['results']))
    assert x_dict['races'].abs().sum() > 0

    # A closed column encodes like a column whose values are all missing:
    with torch.no_grad():
        model.filter.column_gates['results'].logit.fill_(-10.0)
        tf = batch['results'].tf
        x = model.encode_table('results', tf)
        resnet = model.encoder.encoders['results']
        zeros = torch.zeros(tf.num_rows, tf.num_cols * 128)
        expected = resnet.decoder(resnet.backbone(zeros))
    assert torch.allclose(x, expected)


@requires_rel_f1
def test_filter_model_penalty(setup):
    task, _, _, batch = setup
    spec = train.task_spec(task)

    model = make(setup)  # lambda = 0: the penalty adds nothing.
    assert model.structural_loss().item() == 0.0

    model = make(setup, lambda_col=10.0, lambda_table=10.0)
    assert model.structural_loss().item() > 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    train.train_epoch(model, [batch], optimizer, spec, task.entity_table,
                      torch.device('cpu'), max_steps=1)
    # A large lambda pushes every gate towards closed:
    for gate in list(model.filter.column_gates.values()) + list(
            model.filter.table_gates.values()):
        assert (gate.logit < 2.0).all()


@requires_rel_f1
def test_filter_model_gate_summary(setup):
    model = make(setup)
    summary = model.gate_summary()
    assert summary['open_columns'] == sum(
        len(cols) for cols in model.col_names.values())
    assert summary['open_tables'] == len(model.filter.table_gates)
    assert set(summary['columns']['results']) == set(
        model.col_names['results'])
    assert 0.9 < summary['tables']['races']['prob_open'] < 1.0


@requires_rel_f1
def test_make_optimizer(setup):
    model = make(setup)
    gates = {id(p) for p in model.filter.parameters()}

    args = train.parse_args(['--lr', '0.005'])  # gate_lr = 0: one group.
    optimizer = train.make_optimizer(model, args)
    assert len(optimizer.param_groups) == 1

    args = train.parse_args(['--lr', '0.005', '--gate-lr', '0.1'])
    optimizer = train.make_optimizer(model, args)
    rest, gate_group = optimizer.param_groups
    assert rest['lr'] == 0.005 and gate_group['lr'] == 0.1
    assert {id(p) for p in gate_group['params']} == gates
    assert not gates & {id(p) for p in rest['params']}
    assert len(rest['params']) + len(gate_group['params']) == len(
        list(model.parameters()))


@requires_rel_f1
def test_filter_model_gate_options(setup):
    model = make(setup, gate_init=5.0, gate_temperature=0.2)
    for gate in list(model.filter.column_gates.values()) + list(
            model.filter.table_gates.values()):
        assert (gate.logit == 5.0).all() and gate.temperature == 0.2
