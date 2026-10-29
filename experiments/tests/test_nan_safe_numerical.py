import pytest
import torch
import torch_frame
from conftest import requires_rel_f1
from torch_frame.data.stats import StatType

import train

STATS = [{StatType.MEAN: 0.0, StatType.STD: 1.0}] * 2
# Column 0 is always present; column 1 is missing in one of three rows.
FEAT = torch.tensor([[1.0, 2.0], [0.5, float('nan')], [-1.0, 3.0]])


def encoder_pair():
    torch.manual_seed(0)
    original = torch_frame.nn.LinearEncoder(4, STATS, torch_frame.numerical)
    safe = train.NaNSafeLinearEncoder(4, STATS, torch_frame.numerical)
    safe.load_state_dict(original.state_dict())
    return original, safe


def test_forward_is_unchanged():
    original, safe = encoder_pair()
    assert torch.equal(safe(FEAT), original(FEAT))
    assert torch.equal(safe(FEAT)[1, 1], torch.zeros(4))  # missing -> 0


def test_gradients_are_finite_and_unchanged_for_present_values():
    original, safe = encoder_pair()
    original(FEAT).sum().backward()
    safe(FEAT).sum().backward()
    assert torch.isnan(original.weight.grad[1]).all()  # the bug (issue #6)
    assert torch.isfinite(safe.weight.grad).all()
    assert torch.equal(safe.weight.grad[0], original.weight.grad[0])
    assert torch.equal(safe.bias.grad, original.bias.grad)


@pytest.fixture(scope='module')
def batch_setup(rel_f1):
    torch.manual_seed(0)
    task = rel_f1.load_task('driver-top3')
    data, col_stats = train.load_graph(rel_f1, task, 'none',
                                       torch.device('cpu'))
    args = train.parse_args(['--batch-size', '64', '--num-neighbors', '16'])
    batch = next(iter(train.make_loaders(data, task, args)['train']))
    return task, data, col_stats, batch


def nan_parameters_after_one_step(batch_setup, nan_safe):
    task, data, col_stats, batch = batch_setup
    torch.manual_seed(0)
    model = train.BaseModel(data, col_stats, 128, 1, 2, 'sum',
                            nan_safe_numerical=nan_safe)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(batch, task.entity_table).view(-1),
        batch[task.entity_table].y.float())
    loss.backward()
    opt.step()
    return [n for n, p in model.named_parameters() if torch.isnan(p).any()]


@requires_rel_f1
def test_batch_contains_missing_numerical_values(batch_setup):
    batch = batch_setup[-1]
    feat = batch['results'].tf.feat_dict[torch_frame.numerical]
    assert torch.isnan(feat).any()


@requires_rel_f1
def test_original_encoder_gets_nan_weights(batch_setup):
    nan_params = nan_parameters_after_one_step(batch_setup, nan_safe=False)
    assert 'encoder.encoders.results.encoder.encoder_dict.numerical.weight' \
        in nan_params


@requires_rel_f1
def test_nan_safe_encoder_keeps_all_parameters_finite(batch_setup):
    assert nan_parameters_after_one_step(batch_setup, nan_safe=True) == []
