import pytest
import torch
import torch_frame
from conftest import GLOVE_CACHED, requires_glove, requires_rel_f1
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal

import disk_features
import train


def tensors(feat):
    r"""The tensors that hold a feature (a plain tensor, or the values of a
    multi-embedding/multi-nested tensor)."""
    return [feat] if isinstance(feat, torch.Tensor) else [feat.values]


def assert_same_features(data_a, data_b):
    for node_type in data_a.node_types:
        tf_a, tf_b = data_a[node_type].tf, data_b[node_type].tf
        assert tf_a.col_names_dict == tf_b.col_names_dict, node_type
        for stype in tf_a.feat_dict:
            for a, b in zip(tensors(tf_a.feat_dict[stype]),
                            tensors(tf_b.feat_dict[stype])):
                assert torch.equal(a, b) or torch.allclose(
                    a, b, rtol=0, atol=0, equal_nan=True), (node_type, stype)


@requires_rel_f1
@pytest.mark.parametrize('text', [False, True])
def test_build_tables_matches_relbench(rel_f1, tmp_path, text):
    if text and not GLOVE_CACHED:
        pytest.skip('GloVe model not downloaded')
    db = rel_f1.get_db()
    stypes = get_stype_proposal(db)
    if not text:  # Quicker.
        stypes = {
            t: {c: s
                for c, s in cols.items() if s != torch_frame.text_embedded}
            for t, cols in stypes.items()
        }
    cfg = train.text_embedder_cfg('glove' if text else 'none',
                                  torch.device('cpu'))
    # Small chunks, so that most tables are converted in several pieces:
    disk_features.build_tables(db, stypes, tmp_path, cfg, rows_per_chunk=1000,
                               log=lambda _: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f'{t}.pt' for t in db.table_dict)  # No temporary files left.

    # RelBench reads the files we built, and gets what it builds itself.
    # Exact here; on longer texts, text vectors can differ by float32
    # rounding (~1e-7), as they do between PyTorch Frame batch sizes:
    ours, our_stats = make_pkey_fkey_graph(db, stypes, cfg,
                                           cache_dir=str(tmp_path))
    relbench, relbench_stats = make_pkey_fkey_graph(db, stypes, cfg,
                                                    cache_dir=None)
    assert_same_features(ours, relbench)
    assert repr(our_stats) == repr(relbench_stats)


@requires_rel_f1
def test_memory_mapped_loading_restores_torch_load():
    load = torch.load
    with disk_features.memory_mapped_loading():
        assert torch.load is not load
    assert torch.load is load


@pytest.fixture(scope='module')
def graphs(rel_f1):
    task = rel_f1.load_task('driver-top3')
    cpu = torch.device('cpu')
    return task, {
        setting: train.load_graph(rel_f1, task, 'glove', cpu, setting)
        for setting in disk_features.FEATURES_ON_DISK
    }


@requires_rel_f1
@requires_glove
def test_features_on_disk_same_features(graphs):
    _, by_setting = graphs
    data_none = by_setting['none'][0]
    for setting in ['text', 'all']:
        assert_same_features(by_setting[setting][0], data_none)


@requires_rel_f1
@requires_glove
def test_features_on_disk_where_features_live(graphs):
    _, by_setting = graphs
    tf = {s: by_setting[s][0]['drivers'].tf for s in by_setting}
    text, numbers = torch_frame.embedding, torch_frame.timestamp
    on_disk = disk_features.is_on_disk
    assert not on_disk(tf['none'].feat_dict[text].values)
    assert not on_disk(tf['none'].feat_dict[numbers])
    assert on_disk(tf['text'].feat_dict[text].values)
    assert not on_disk(tf['text'].feat_dict[numbers])
    assert on_disk(tf['all'].feat_dict[text].values)
    assert on_disk(tf['all'].feat_dict[numbers])


@requires_rel_f1
@requires_glove
def test_features_on_disk_same_model_output(graphs):
    task, by_setting = graphs
    args = train.parse_args(['--batch-size', '64', '--num-neighbors', '16'])
    data_none = by_setting['none'][0]
    # The sampler is not seeded by torch within one process, so one batch is
    # sampled and each setting's features are sliced with its row ids, as the
    # loader does:
    batch = next(iter(train.make_loaders(data_none, task, args)['val']))
    outputs = {}
    for setting, (data, col_stats) in by_setting.items():
        for node_type in batch.node_types:
            batch[node_type].tf = data[node_type].tf[batch[node_type].n_id]
        torch.manual_seed(0)
        model = train.BaseModel(data, col_stats, 128, 1, 2, 'sum')
        model.eval()
        with torch.no_grad():
            outputs[setting] = model(batch, task.entity_table)
    assert torch.equal(outputs['text'], outputs['none'])
    assert torch.equal(outputs['all'], outputs['none'])


@requires_rel_f1
def test_features_on_disk_rejects_unknown_setting(rel_f1):
    task = rel_f1.load_task('driver-top3')
    with pytest.raises(ValueError, match='features_on_disk'):
        train.load_graph(rel_f1, task, 'none', torch.device('cpu'), 'some')


def test_distinct_texts():
    calls = []

    def embed(texts):  # A stand-in embedder: one vector per text.
        calls.append(list(texts))
        return torch.tensor([[float(len(t)), float(ord(t[0]))] for t in texts])

    texts = ['us', 'france', 'us', 'nan', 'us', 'france']
    out = disk_features.DistinctTexts(embed)(texts)
    assert torch.equal(out, embed(texts))  # Same vectors as embedding all.
    assert calls[0] == ['us', 'france', 'nan']  # Each text embedded once.
