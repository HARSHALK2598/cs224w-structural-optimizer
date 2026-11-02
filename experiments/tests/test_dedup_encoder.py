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
    model = train.BaseModel(data, col_stats, 128, 1, 2, 'sum')
    return task, batch, model


def run(model, batch, task, dedup):
    model.dedup_encoder = dedup
    return model(batch, task.entity_table).view(-1)


@requires_rel_f1
def test_batch_has_duplicate_rows(setup):
    _, batch, _ = setup
    n_id = batch['races'].n_id
    assert n_id.unique().numel() < n_id.numel()


@requires_rel_f1
def test_dedup_encodes_each_distinct_row_once(setup):
    task, batch, model = setup
    rows_in = {}

    def count_rows(node_type):
        def hook(module, inputs, output):
            rows_in[node_type] = inputs[0].num_rows
        return hook

    hooks = [
        enc.register_forward_hook(count_rows(nt))
        for nt, enc in model.encoder.encoders.items()
    ]
    try:
        model.eval()
        with torch.no_grad():
            run(model, batch, task, dedup=True)
    finally:
        for hook in hooks:
            hook.remove()
    for nt in batch.node_types:
        assert rows_in[nt] == batch[nt].n_id.unique().numel(), nt


@requires_rel_f1
def test_dedup_gives_identical_outputs_in_eval(setup):
    task, batch, model = setup
    model.eval()
    with torch.no_grad():
        ref = run(model, batch, task, dedup=False)
        out = run(model, batch, task, dedup=True)
    assert torch.allclose(out, ref, atol=1e-5)


@requires_rel_f1
def test_dedup_gives_identical_gradients_without_dropout(setup):
    task, batch, model = setup
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    old_p = [m.p for m in dropouts]
    for m in dropouts:
        m.p = 0.0  # dropout is the only source of difference in training
    try:
        model.train()
        y = batch[task.entity_table].y.float()
        grads = []
        for dedup in [False, True]:
            model.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                run(model, batch, task, dedup), y)
            loss.backward()
            grads.append({
                n: p.grad.clone()
                for n, p in model.named_parameters() if p.grad is not None
            })
    finally:
        for m, p in zip(dropouts, old_p):
            m.p = p
    ref, out = grads
    assert ref.keys() == out.keys()
    for name in ref:
        # Some numerical-column weights get NaN gradients from missing values
        # with or without dedup (see experiments/profiling/README.md), so
        # NaNs must sit in the same places and everything else must match.
        assert torch.allclose(out[name], ref[name], rtol=1e-4, atol=1e-6,
                              equal_nan=True), name
