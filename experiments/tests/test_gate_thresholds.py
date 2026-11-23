r"""Gate thresholds chosen on validation, as in the authors' code."""
import torch
from torch_geometric.contrib.nn.models import HardConcreteGate

import train


class Gated(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = HardConcreteGate(2, init_logit=0.0)
        self.b = HardConcreteGate(1, init_logit=0.0)
        with torch.no_grad():  # sigmoids 0.6, 0.8 and 0.88:
            self.a.logit.copy_(torch.tensor([0.6, 0.8]).logit())
            self.b.logit.copy_(torch.tensor([0.88]).logit())


class Task:
    entity_table = 'users'


def test_validate_picks_best_threshold_and_shares_passes(monkeypatch):
    model = Gated().eval()
    spec = train.TaskSpec(None, 1, 'roc_auc', True, ['roc_auc'], None)
    passes = []

    def fake_predict(model, *args, **kwargs):  # Score = number open.
        passes.append(1)
        return sum(int(g().sum()) for g in train.gates_of(model))

    monkeypatch.setattr(train, 'predict', fake_predict)
    monkeypatch.setattr(train, 'evaluate',
                        lambda task, spec, pred, split: {'roc_auc': -pred})
    # Open gates: 0.5 -> 3, 0.7 -> 2, 0.78 -> 2, 0.86 -> 1, 0.9 -> 0.
    best, val = train.validate(model, {'val': None}, Task(), spec, 'cpu',
                               [0.5, 0.7, 0.78, 0.86, 0.9])
    assert best == 0.9 and val == {'roc_auc': 0}
    assert len(passes) == 4  # 0.7 and 0.78 give the same gates.
    assert all(g.threshold == 0.9 for g in train.gates_of(model))
    assert int(model.a().sum()) == 0 and int(model.b().sum()) == 0


def test_thresholds_and_strategy_parse():
    args = train.parse_args(['--gate-thresholds', '0.5,0.86',
                             '--temporal-strategy', 'last'])
    assert args.gate_thresholds == '0.5,0.86'
    assert args.temporal_strategy == 'last'
    assert train.parse_args([]).gate_thresholds == ''
