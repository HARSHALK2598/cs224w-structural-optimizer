"""Single entry point for all experiments (CPU smoke test here, full runs on Colab).

Models:
    base    RelBench's relational GraphSAGE (the paper's *Base*).
    filter  Base + filtering (step A).                 [needs RelationalFilter]
    full    Base + the full structural optimizer (A-D). [needs StructuralOptimizer]

Examples:
    # one train batch + one val batch on CPU, nothing written to results/
    python experiments/train.py --task driver-top3 --model base --smoke \
        --text-embedder none

    # a full run (Colab), appends one row to experiments/results/results.csv
    python experiments/train.py --config experiments/configs/rel-f1.yaml \
        --task driver-top3 --model base --seed 0
"""
import argparse
import copy
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict

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
    text_embedder='glove',  # 'glove' (paper / RelBench default) or 'none'
    num_workers=0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='YAML file whose keys override the defaults.')
    for key, value in DEFAULTS.items():
        parser.add_argument(f'--{key.replace("_", "-")}', type=type(value),
                            default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='Run one train and one val batch, write nothing.')
    parser.add_argument('--results', type=str,
                        default=str(ROOT / 'experiments/results/results.csv'))
    args = parser.parse_args()

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

    RelBench 3 no longer ships this helper, so it is re-created here. Requires
    `sentence-transformers` (installed in the Colab notebook); the model
    (~480MB) is downloaded from the Hugging Face Hub on first use.
    """
    def __init__(self, device: torch.device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            'sentence-transformers/average_word_embeddings_glove.6B.300d',
            device=device)

    def __call__(self, sentences) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


def text_embedder_cfg(name: str, device: torch.device):
    if name == 'none':
        return None
    if name == 'glove':
        return torch_frame.config.TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device), batch_size=256)
    raise ValueError(f"Unknown text embedder '{name}'")


class BaseModel(torch.nn.Module):
    r"""RelBench's relational GraphSAGE: encoder -> hetero GNN -> MLP head.

    The encoder output (column encoder + relative-time encoding) is
    :math:`h^{(0)}`, the input that the structural optimizer's filtering step
    (A) will gate.
    """
    def __init__(self, data, col_stats_dict, channels: int, out_channels: int,
                 num_layers: int, aggr: str):
        super().__init__()
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

    def encode(self, batch, entity_table: str) -> Dict[str, Tensor]:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
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


def build_model(args, data, col_stats_dict, out_channels: int):
    if args.model == 'base':
        return BaseModel(data, col_stats_dict, args.channels, out_channels,
                         args.num_layers, args.aggr)
    # 'filter' and 'full' wrap BaseModel's encoder in
    # torch_geometric.contrib.nn.StructuralOptimizer (Phase 1-2).
    raise NotImplementedError(f"Model '{args.model}' is not implemented yet")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(args.dataset)
    task = dataset.load_task(args.task)
    db = dataset.get_db()

    col_to_stype_dict = get_stype_proposal(db)
    if args.text_embedder == 'none':  # Drop text columns (CPU smoke runs).
        col_to_stype_dict = {
            table: {
                col: st
                for col, st in cols.items() if st != torch_frame.text_embedded
            }
            for table, cols in col_to_stype_dict.items()
        }
    cache_dir = ROOT / 'data' / 'materialized' / args.dataset / args.text_embedder
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg(args.text_embedder, device),
        cache_dir=str(cache_dir),
    )

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        loss_fn, out_channels = torch.nn.BCEWithLogitsLoss(), 1
    elif task.task_type == TaskType.REGRESSION:
        loss_fn, out_channels = torch.nn.L1Loss(), 1
    else:
        raise NotImplementedError(f'Task type {task.task_type} not supported')

    loaders = {}
    for split in ['train', 'val', 'test']:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
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

    model = build_model(args, data, col_stats_dict, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_steps = 1 if args.smoke else args.max_steps_per_epoch
    entity_table = task.entity_table

    def train_epoch() -> float:
        model.train()
        total_loss = total_count = 0
        for step, batch in enumerate(loaders['train']):
            if step >= max_steps:
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, entity_table).view(-1)
            loss = loss_fn(pred, batch[entity_table].y.float())
            # 'filter' / 'full': loss = loss + model.structural_loss()  (step D)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_count += pred.numel()
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def predict(split: str) -> Tensor:
        model.eval()
        preds = []
        for step, batch in enumerate(loaders[split]):
            if args.smoke and step >= 1:
                break
            batch = batch.to(device)
            pred = model(batch, entity_table).view(-1)
            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = pred.sigmoid()
            preds.append(pred.cpu())
        return torch.cat(preds).numpy()

    epochs = 1 if args.smoke else args.epochs
    metric = task.metrics[0].__name__
    higher_is_better = task.task_type == TaskType.BINARY_CLASSIFICATION
    best_val, best_state = None, None
    start = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_epoch()
        val_pred = predict('val')
        if args.smoke:
            print(f'[smoke] epoch {epoch}: train loss {loss:.4f}, '
                  f'val pred shape {val_pred.shape} -> OK')
            return
        val = task.evaluate(val_pred, task.get_table('val'))[metric]
        print(f'epoch {epoch:02d}  loss {loss:.4f}  val {metric} {val:.4f}')
        better = best_val is None or (val > best_val if higher_is_better else
                                      val < best_val)
        if better:
            best_val, best_state = val, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test = task.evaluate(predict('test'))[metric]
    print(f'best val {metric} {best_val:.4f}  test {metric} {test:.4f}')

    path = Path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(dataset=args.dataset, task=args.task, model=args.model,
               seed=args.seed, metric=metric, val=best_val, test=test,
               epochs=epochs, minutes=round((time.time() - start) / 60, 2),
               text_embedder=args.text_embedder)
    new_file = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    main()
