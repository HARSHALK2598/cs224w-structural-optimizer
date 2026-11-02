"""Single entry point for all experiments.

Reproduces RelBench's official entity-task script (examples/gnn_entity.py at
commit 91c1f64) for the Base model; see plans/00-baseline.md.

Models:
    base    RelBench's relational GraphSAGE (the paper's *Base*).
    filter  Base + column and table gates (step A without the bottleneck;
            plans/01-filtering-gates.md).
    full    Base + the full optimizer (A-D).    [not implemented yet]

Examples:
    # one train batch + one val batch, nothing written to results/
    python experiments/train.py --task driver-top3 --smoke --text-embedder none

    # a full run, appends one row to experiments/results/results.csv
    python experiments/train.py --task driver-top3 --model base --seed 0

    # with column and table gates
    python experiments/train.py --task driver-top3 --model filter \
        --lambda-col 1e-3 --lambda-table 1e-3 --gate-lr 0.1 --dedup-encoder
"""
import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'hf'))

import torch_frame  # noqa: E402
from relbench import load_dataset  # noqa: E402
from relbench.base import TaskType  # noqa: E402
from relbench.modeling.graph import (  # noqa: E402
    get_node_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.nn import (  # noqa: E402
    HeteroEncoder,
    HeteroGraphSAGE,
    HeteroTemporalEncoder,
)
from relbench.modeling.utils import get_stype_proposal  # noqa: E402
from torch import Tensor  # noqa: E402

from torch_geometric.contrib.nn.models import RelationalFilter  # noqa: E402
from torch_geometric.loader import NeighborLoader  # noqa: E402
from torch_geometric.nn import MLP  # noqa: E402
from torch_geometric.seed import seed_everything  # noqa: E402

DEFAULTS: Dict[str, Any] = dict(
    dataset='rel-f1',
    task='driver-top3',
    model='base',
    seed=0,
    epochs=10,
    lr=0.005,
    batch_size=512,
    channels=128,
    num_layers=2,
    num_neighbors=128,
    aggr='sum',
    max_steps_per_epoch=2000,
    text_embedder='glove',  # 'glove' (RelBench default) or 'none'
    num_workers=0,
    lambda_col=0.0,  # --model filter: weight of the column-gate penalty
    lambda_table=0.0,  # --model filter: weight of the table-gate penalty
    gate_lr=0.0,  # --model filter: learning rate of the gates (0: --lr)
    gate_init=2.0,  # --model filter: initial gate logit
    gate_temperature=2. / 3.,  # --model filter: Hard-Concrete temperature
)

# Columns of results.csv; metrics a task does not report are left empty.
RESULT_FIELDS = [
    'dataset', 'task', 'model', 'seed', 'tune_metric', 'best_epoch',
    'val_roc_auc', 'test_roc_auc', 'val_nmae', 'test_nmae', 'val_mae',
    'test_mae', 'epochs', 'minutes', 'text_embedder', 'lambda_col',
    'lambda_table', 'gate_lr', 'gate_init', 'gate_temperature',
    'open_columns', 'open_tables'
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='YAML file whose keys override the defaults.')
    for key, value in DEFAULTS.items():
        parser.add_argument(f'--{key.replace("_", "-")}', type=type(value),
                            default=None)
    parser.add_argument('--dedup-encoder', action='store_true',
                        help='Encode each distinct row once per batch '
                        '(see experiments/profiling/README.md).')
    parser.add_argument('--smoke', action='store_true',
                        help='Run one train and one val batch, write nothing.')
    parser.add_argument('--results', type=str,
                        default=str(ROOT / 'experiments/results/results.csv'))
    args = parser.parse_args(argv)

    # Precedence: CLI flag > YAML config > DEFAULTS.
    cfg = dict(DEFAULTS)
    if args.config is not None:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for key in DEFAULTS:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for key, value in cfg.items():
        setattr(args, key, value)
    return args


class GloveTextEmbedding:
    r"""Averaged 300-d GloVe word embeddings, RelBench's default text encoder.

    Matches the official example's `text_embedder.py`. Requires
    `sentence-transformers`; the model (~1GB on disk) is downloaded from the
    Hugging Face Hub on first use. Missing values (None / NaN) become empty
    strings, which the tokenizer expects.
    """
    MODEL = 'sentence-transformers/average_word_embeddings_glove.6B.300d'

    def __init__(self, device: torch.device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL, device=device)

    @staticmethod
    def clean(sentences) -> List[str]:
        out = []
        for s in sentences:
            if s is None or (isinstance(s, float) and s != s):  # None / NaN
                out.append('')
            else:
                out.append(s if isinstance(s, str) else str(s))
        return out

    def __call__(self, sentences) -> Tensor:
        return torch.from_numpy(self.model.encode(self.clean(sentences)))


def text_embedder_cfg(name: str, device: torch.device):
    if name == 'none':
        return None
    if name == 'glove':
        return torch_frame.config.TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device), batch_size=256)
    raise ValueError(f"Unknown text embedder '{name}'")


def load_graph(dataset, task, text_embedder: str, device: torch.device):
    r"""Builds the database graph, without the columns that leak the task's
    label (:meth:`task.hidden_columns`)."""
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    if text_embedder == 'none':  # Drop text columns (quick CPU runs).
        col_to_stype_dict = {
            table: {
                col: st
                for col, st in cols.items() if st != torch_frame.text_embedded
            }
            for table, cols in col_to_stype_dict.items()
        }
    # Hidden columns are dropped after materialisation, so one cache per
    # dataset serves every task (as in the official example).
    name = Path(str(dataset.name_or_path)).name
    cache_dir = ROOT / 'data' / 'materialized' / name / text_embedder
    return make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg(text_embedder, device),
        cache_dir=str(cache_dir),
        remove_columns=task.hidden_columns(),
    )


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    r"""Mean absolute error, the regression metric the paper reports."""
    return float(np.abs(true - pred).mean())


def clip_bounds(task) -> Tuple[float, float]:
    r"""The 2nd and 98th percentiles of the training labels. Regression
    predictions are clipped to this range at inference time."""
    labels = task.get_table('train').df[task.target_col].to_numpy()
    low, high = np.percentile(labels, [2, 98])
    return float(low), float(high)


class TaskSpec(NamedTuple):
    r"""Everything that depends on the task type."""
    loss_fn: torch.nn.Module
    out_channels: int
    tune_metric: str  # Selects the best epoch on the validation split.
    higher_is_better: bool
    metrics: List[Callable]  # Reported on the validation and test splits.
    clip: Optional[Tuple[float, float]]  # Regression output range.


def task_spec(task) -> TaskSpec:
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return TaskSpec(torch.nn.BCEWithLogitsLoss(), 1, 'roc_auc', True,
                        list(task.metrics), None)
    if task.task_type == TaskType.REGRESSION:
        return TaskSpec(torch.nn.L1Loss(), 1, 'nmae', False,
                        list(task.metrics) + [mae], clip_bounds(task))
    raise NotImplementedError(f'Task type {task.task_type} not supported')


def is_better(value: float, best: Optional[float],
              higher_is_better: bool) -> bool:
    r"""Whether an epoch's validation score replaces the best so far. Ties go
    to the later epoch, as in the official example."""
    if best is None:
        return True
    return value >= best if higher_is_better else value <= best


def make_loaders(data, task, args) -> Dict[str, NeighborLoader]:
    loaders = {}
    for split in ['train', 'val', 'test']:
        table_input = get_node_train_table_input(table=task.get_table(split),
                                                 task=task)
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=[
                args.num_neighbors // 2**i for i in range(args.num_layers)
            ],
            time_attr='time',
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=args.batch_size,
            temporal_strategy='uniform',
            shuffle=split == 'train',
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    return loaders


class BaseModel(torch.nn.Module):
    r"""RelBench's relational GraphSAGE: encoder -> hetero GNN -> MLP head.

    The encoder output (column encoder + relative-time encoding) is
    :math:`h^{(0)}`, the input that the structural optimizer's filtering step
    (A) will gate.

    With :obj:`dedup_encoder=True`, a row sampled several times in one batch
    is encoded once and its embedding copied to every position. Outputs are
    identical in evaluation mode, and gradients are identical without
    dropout; in training, all copies of a row share one dropout mask.
    """
    def __init__(self, data, col_stats_dict, channels: int, out_channels: int,
                 num_layers: int, aggr: str, dedup_encoder: bool = False):
        super().__init__()
        self.dedup_encoder = dedup_encoder
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if 'time' in data[nt]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(channels, out_channels=out_channels, norm='batch_norm',
                        num_layers=1)

    def encode_table(self, node_type: str, tf) -> Tensor:
        r"""Encodes the rows :obj:`tf` of one table into
        :obj:`[num_rows, channels]`."""
        return self.encoder.encoders[node_type](tf)

    def encode_rows(self, batch) -> Dict[str, Tensor]:
        r"""Runs the row encoder (one ResNet per table) on the batch."""
        if not self.dedup_encoder:
            return {
                node_type: self.encode_table(node_type, tf)
                for node_type, tf in batch.tf_dict.items()
            }
        x_dict = {}
        for node_type, tf in batch.tf_dict.items():
            # `n_id` is the database row of every sampled node; `inverse`
            # maps each node to its distinct row.
            n_id = batch[node_type].n_id
            unique, inverse = n_id.unique(return_inverse=True)
            # One batch position per distinct row (any copy works):
            first = torch.empty_like(unique).scatter_(
                0, inverse, torch.arange(n_id.numel(), device=n_id.device))
            x = self.encode_table(node_type, tf[first])
            # Indexing copies the embedding to every position; in backward,
            # it sums the gradients of all copies into the encoded row.
            x_dict[node_type] = x[inverse]
        return x_dict

    def encode(self, batch, entity_table: str) -> Dict[str, Tensor]:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encode_rows(batch)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        return x_dict

    def forward(self, batch, entity_table: str) -> Tensor:
        x_dict = self.encode(batch, entity_table)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        num_seeds = batch[entity_table].seed_time.size(0)
        return self.head(x_dict[entity_table][:num_seeds])


class FilterModel(BaseModel):
    r"""Base with column and table gates (plans/01-filtering-gates.md).

    Each table's row encoder runs in two halves, with the column gates in
    between: the column encoders give :obj:`[N, C, channels]`, the gates
    scale each column, then the ResNet mixes the columns. The table gates
    scale :math:`h^{(0)}` (after the time encoding) of every table except
    :obj:`entity_table`. With every gate open, the model is Base.
    """
    def __init__(self, data, col_stats_dict, channels: int, out_channels: int,
                 num_layers: int, aggr: str, entity_table: str,
                 lambda_col: float = 0.0, lambda_table: float = 0.0,
                 dedup_encoder: bool = False, gate_init: float = 2.0,
                 gate_temperature: float = 2. / 3.):
        super().__init__(data, col_stats_dict, channels, out_channels,
                         num_layers, aggr, dedup_encoder)
        self.lambda_col = lambda_col
        self.lambda_table = lambda_table
        # Gate c of a table belongs to its c-th encoded column; PyTorch Frame
        # orders the columns by stype:
        self.col_names = {
            node_type: [
                col for stype in data[node_type].tf.stypes
                for col in data[node_type].tf.col_names_dict[stype]
            ]
            for node_type in data.node_types
        }
        self.filter = RelationalFilter(
            num_columns={
                node_type: len(cols)
                for node_type, cols in self.col_names.items()
            },
            table_types=[nt for nt in data.node_types if nt != entity_table],
            init_logit=gate_init,
            temperature=gate_temperature,
        )

    def encode_table(self, node_type: str, tf) -> Tensor:
        resnet = self.encoder.encoders[node_type]
        x, _ = resnet.encoder(tf)  # [N, C, channels]
        x = self.filter.gate_columns(node_type, x)
        # As in ResNet.forward (`math.prod` also handles empty tables):
        x = x.view(x.size(0), math.prod(x.shape[1:]))
        return resnet.decoder(resnet.backbone(x))

    def encode(self, batch, entity_table: str) -> Dict[str, Tensor]:
        return self.filter.gate_tables(super().encode(batch, entity_table))

    def structural_loss(self) -> Tensor:
        r"""The penalty term :math:`\lambda \sum \mathbb{E}\|g\|_0 +
        \lambda_s \sum \mathbb{E}\|s\|_0` of the loss."""
        col_penalty, table_penalty = self.filter.l0_penalties()
        return (self.lambda_col * col_penalty +
                self.lambda_table * table_penalty)

    def gate_summary(self) -> Dict[str, Any]:
        r"""Each gate's open probability and evaluation-mode state."""
        def threshold(gate) -> Tensor:  # The evaluation-mode 0/1 state.
            return gate.logit.sigmoid() > gate.threshold

        columns, tables = {}, {}
        with torch.no_grad():
            for node_type, gate in self.filter.column_gates.items():
                prob, is_open = gate.prob_open(), threshold(gate)
                columns[node_type] = {
                    col: dict(prob_open=round(float(p), 4), open=bool(o))
                    for col, p, o in zip(self.col_names[node_type], prob,
                                         is_open)
                }
            for node_type, gate in self.filter.table_gates.items():
                tables[node_type] = dict(
                    prob_open=round(float(gate.prob_open()[0]), 4),
                    open=bool(threshold(gate)[0]))
        return dict(
            open_columns=sum(c['open'] for cols in columns.values()
                             for c in cols.values()),
            open_tables=sum(t['open'] for t in tables.values()),
            columns=columns, tables=tables)


def build_model(args, data, col_stats_dict, out_channels: int,
                entity_table: Optional[str] = None):
    if args.model == 'base':
        return BaseModel(data, col_stats_dict, args.channels, out_channels,
                         args.num_layers, args.aggr, args.dedup_encoder)
    if args.model == 'filter':
        return FilterModel(data, col_stats_dict, args.channels, out_channels,
                           args.num_layers, args.aggr, entity_table,
                           args.lambda_col, args.lambda_table,
                           args.dedup_encoder, args.gate_init,
                           args.gate_temperature)
    raise NotImplementedError(f"Model '{args.model}' is not implemented yet")


def train_epoch(model, loader, optimizer, spec: TaskSpec, entity_table: str,
                device: torch.device, max_steps: int) -> float:
    r"""Runs one epoch (at most :obj:`max_steps` batches) and returns the
    mean task loss per seed (without any structural penalty)."""
    model.train()
    total_loss = total_count = 0
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, entity_table).view(-1)
        loss = spec.loss_fn(pred, batch[entity_table].y.float())
        if hasattr(model, 'structural_loss'):  # The gates' L0 penalty.
            (loss + model.structural_loss()).backward()
        else:
            loss.backward()
        optimizer.step()
        # Reports the task loss only, comparable across models:
        total_loss += loss.item() * pred.numel()
        total_count += pred.numel()
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict(model, loader, spec: TaskSpec, entity_table: str,
            device: torch.device, max_steps: Optional[int] = None):
    r"""Predictions for every seed of :obj:`loader`, in loader order:
    probabilities for classification, clipped values for regression."""
    model.eval()
    preds = []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        pred = model(batch.to(device), entity_table).view(-1)
        if spec.clip is not None:
            pred = pred.clamp(*spec.clip)
        else:
            pred = pred.sigmoid()
        preds.append(pred.cpu())
    return torch.cat(preds).numpy()


def evaluate(task, spec: TaskSpec, pred, split: str) -> Dict[str, float]:
    r"""Scores predictions on :obj:`'val'` or :obj:`'test'` with the task's
    metrics (plus MAE for regression)."""
    # For the test split, task.evaluate loads the labels itself.
    table = task.get_table('val') if split == 'val' else None
    return task.evaluate(pred, table, metrics=spec.metrics)


def make_optimizer(model, args) -> torch.optim.Optimizer:
    r"""Adam over all parameters. With :obj:`args.gate_lr > 0`, the gate
    logits of :class:`FilterModel` get their own learning rate: Adam moves a
    logit by at most about the learning rate per step, so with few steps per
    epoch the gates could not close otherwise."""
    if getattr(model, 'filter', None) is None or args.gate_lr <= 0:
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    gate_ids = {id(p) for p in model.filter.parameters()}
    rest = [p for p in model.parameters() if id(p) not in gate_ids]
    return torch.optim.Adam([
        dict(params=rest),
        dict(params=list(model.filter.parameters()), lr=args.gate_lr),
    ], lr=args.lr)


def fit(model, loaders, task, spec: TaskSpec, args, device: torch.device,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    r"""Trains for :obj:`args.epochs` epochs, keeps the epoch with the best
    validation score, and evaluates it once on the test split."""
    optimizer = make_optimizer(model, args)
    entity_table = task.entity_table
    best, best_state, best_epoch, history = None, None, None, []
    best_gates = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders['train'], optimizer, spec,
                           entity_table, device, args.max_steps_per_epoch)
        val = evaluate(
            task, spec,
            predict(model, loaders['val'], spec, entity_table, device), 'val')
        gates = model.gate_summary() if hasattr(model,
                                                'gate_summary') else None
        history.append(dict(epoch=epoch, loss=loss, **val))
        log(f'epoch {epoch:02d}  loss {loss:.4f}  ' +
            '  '.join(f'val {k} {v:.4f}' for k, v in val.items()) +
            ('' if gates is None else
             f'  penalty {model.structural_loss().item():.4f}'
             f'  open columns {gates["open_columns"]}'
             f'  open tables {gates["open_tables"]}'))
        if is_better(val[spec.tune_metric], best, spec.higher_is_better):
            best, best_epoch = val[spec.tune_metric], epoch
            best_state, best_val = copy.deepcopy(model.state_dict()), val
            best_gates = gates

    model.load_state_dict(best_state)
    test = evaluate(
        task, spec,
        predict(model, loaders['test'], spec, entity_table, device), 'test')
    log(f'best epoch {best_epoch}: ' +
        '  '.join(f'val {k} {v:.4f}' for k, v in best_val.items()) + '  ' +
        '  '.join(f'test {k} {v:.4f}' for k, v in test.items()))
    return dict(best_epoch=best_epoch, val=best_val, test=test,
                history=history, gates=best_gates)


def write_result(path: str, row: Dict[str, Any]) -> None:
    r"""Appends one row to :obj:`path`, writing the header for a new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file:
        with open(path, newline='') as f:
            header = next(csv.reader(f), None)
        if header != RESULT_FIELDS:
            raise ValueError(f"'{path}' has different columns than "
                             f"RESULT_FIELDS; write to a new file")
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, restval='')
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(args.dataset)
    task = dataset.load_task(args.task)
    data, col_stats_dict = load_graph(dataset, task, args.text_embedder,
                                      device)
    spec = task_spec(task)
    loaders = make_loaders(data, task, args)
    model = build_model(args, data, col_stats_dict, spec.out_channels,
                        task.entity_table).to(device)

    if args.smoke:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss = train_epoch(model, loaders['train'], optimizer, spec,
                           task.entity_table, device, max_steps=1)
        pred = predict(model, loaders['val'], spec, task.entity_table, device,
                       max_steps=1)
        print(f'[smoke] train loss {loss:.4f}, val pred shape {pred.shape} '
              f'-> OK')
        return

    start = time.time()
    out = fit(model, loaders, task, spec, args, device)
    row = dict(dataset=args.dataset, task=args.task, model=args.model,
               seed=args.seed, tune_metric=spec.tune_metric,
               best_epoch=out['best_epoch'], epochs=args.epochs,
               minutes=round((time.time() - start) / 60, 2),
               text_embedder=args.text_embedder)
    for split in ['val', 'test']:
        for name, value in out[split].items():
            row[f'{split}_{name}'] = value
    if out['gates'] is not None:  # --model filter
        row.update(lambda_col=args.lambda_col, lambda_table=args.lambda_table,
                   gate_lr=args.gate_lr or args.lr, gate_init=args.gate_init,
                   gate_temperature=round(args.gate_temperature, 4),
                   open_columns=out['gates']['open_columns'],
                   open_tables=out['gates']['open_tables'])
        # Every gate's state at the best epoch, next to results.csv:
        path = Path(args.results).parent / 'gates' / (
            f'{args.task}_seed{args.seed}_lc{args.lambda_col:g}'
            f'_lt{args.lambda_table:g}_glr{args.gate_lr or args.lr:g}'
            f'_gi{args.gate_init:g}_gt{args.gate_temperature:.2g}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out['gates'], indent=2))
    write_result(args.results, row)


if __name__ == '__main__':
    main()
