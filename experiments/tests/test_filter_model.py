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


def force_open(model):
    r"""Fixes every gate of :obj:`model` at exactly 1, drawing no random
    numbers (so dropout masks match a Base model's)."""
    def ones(gate):
        def forward(num_samples=None):
            size = (gate.num_gates, ) if num_samples is None else (
                num_samples, gate.num_gates)
            return torch.ones(size)
        return forward

    for gate in list(model.filter.column_gates.values()) + list(
            model.filter.table_gates.values()):
        gate.forward = ones(gate)


@requires_rel_f1
@pytest.mark.parametrize('dedup', [False, True])
def test_filter_model_open_trains_like_base(setup, dedup):
    r"""Test (a): with every gate fixed at 1, training the gated model gives
    the same loss at every step and the same weights as training Base."""
    task, data, col_stats, _ = setup
    spec = train.task_spec(task)
    args = train.parse_args(['--batch-size', '64', '--num-neighbors', '16',
                             '--gate-lr', '0.1'])
    torch.manual_seed(0)
    batches = [b for b, _ in zip(train.make_loaders(data, task, args)['train'],
                                 range(4))]
    val_batch = next(iter(train.make_loaders(data, task, args)['val']))

    model = make(setup, dedup_encoder=dedup)
    force_open(model)
    base = train.BaseModel(data, col_stats, 128, 1, 2, 'sum', dedup)
    base.load_state_dict(model.state_dict(), strict=False)

    # With deduplication, the backward pass sums the gradients of a row's
    # copies in parallel, in a varying order; one thread makes it exact
    # (Base against Base differs by ~1e-8 otherwise).
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    losses = {}
    for name, m, optimizer in [
        ('base', base, torch.optim.Adam(base.parameters(), lr=args.lr)),
        ('filter', model, train.make_optimizer(model, args)),
    ]:
        torch.manual_seed(1)  # The same dropout masks for both models.
        losses[name] = [
            train.train_epoch(m, [b], optimizer, spec, task.entity_table,
                              torch.device('cpu'), max_steps=1)
            for b in batches
        ]
    torch.set_num_threads(threads)
    assert losses['filter'] == losses['base']

    base_state = base.state_dict()
    for key, value in model.state_dict().items():
        if key.startswith('filter.'):
            continue
        # Exactly equal; NaN weights (issue #6) occur in both models alike.
        assert torch.allclose(value, base_state[key], rtol=0, atol=0,
                              equal_nan=True), key

    model.eval()
    base.eval()
    with torch.no_grad():
        assert torch.equal(model(val_batch, task.entity_table),
                           base(val_batch, task.entity_table))


@requires_rel_f1
def test_filter_model_gates_open_option(setup):
    task, data, col_stats, batch = setup
    model = make(setup, gates_open=True)
    with torch.no_grad():  # Closed logits have no effect when gates are open.
        for gate in model.filter.column_gates.values():
            gate.logit.fill_(-10.0)
    base = train.BaseModel(data, col_stats, 128, 1, 2, 'sum')
    base.load_state_dict(model.state_dict(), strict=False)
    model.eval()
    base.eval()
    with torch.no_grad():
        assert torch.equal(model(batch, task.entity_table),
                           base(batch, task.entity_table))


def perturbed(batch, node_type, col=None):
    r"""A copy of :obj:`batch` with new random values in one numerical
    column of :obj:`node_type` (or in all its numerical columns and its
    timestamps if :obj:`col` is :obj:`None`)."""
    import copy

    import torch_frame
    out = copy.copy(batch)
    store = out[node_type]
    tf = copy.copy(store.tf)
    tf.feat_dict = dict(tf.feat_dict)
    num = tf.feat_dict[torch_frame.numerical].clone()
    names = tf.col_names_dict[torch_frame.numerical]
    cols = [names.index(col)] if col is not None else range(len(names))
    for c in cols:
        num[:, c] = torch.randn(num.size(0)) * 100
    tf.feat_dict[torch_frame.numerical] = num
    store.tf = tf
    if col is None:
        store.time = store.time - torch.randint(1, 10**8, store.time.shape)
    return out


@requires_rel_f1
def test_filter_model_closed_column_removes_its_information(setup):
    r"""A closed column gate removes that column from the model: changing
    the column's values leaves the prediction unchanged (and changes it when
    the gate is open)."""
    task, _, _, batch = setup
    model = make(setup)
    model.eval()
    idx = model.col_names['results'].index('grid')
    other = perturbed(batch, 'results', col='grid')
    with torch.no_grad():
        assert not torch.equal(model(batch, task.entity_table),
                               model(other, task.entity_table))  # Open.
        model.filter.column_gates['results'].logit[idx] = -10.0
        assert torch.equal(model(batch, task.entity_table),
                           model(other, task.entity_table))  # Closed.


@requires_rel_f1
def test_filter_model_closed_table_removes_its_information(setup):
    r"""A closed table gate removes the table's own content (its columns and
    its timestamps) from the model; the table's rows still pass on what they
    receive from their neighbours (a closed table stays a path)."""
    task, _, _, batch = setup
    model = make(setup)
    model.eval()
    other = perturbed(batch, 'standings')
    with torch.no_grad():
        assert not torch.equal(model(batch, task.entity_table),
                               model(other, task.entity_table))  # Open.
        model.filter.table_gates['standings'].logit.fill_(-10.0)
        assert torch.equal(model(batch, task.entity_table),
                           model(other, task.entity_table))  # Closed.
