# Project status / handoff

_Last updated: 2026-10-23 (end of day 1)_

## How we work (agreed)
This is a learning project: for every component we go **understand → check yourself
→ design together → write code → test**, and nothing goes into the PR that Harshal
can't explain. Slow down rather than rush through tasks.

**Open questions for the next session:**
1. How much code does Harshal write himself (all, with review/hints) vs. review
   drafts?
2. Start with a walkthrough of what already exists, or straight into the paper's
   ideas?
3. Check the CS224W guidelines doc for the AI-use / disclosure policy (the Google Doc
   could not be opened from the container; only the rubric PDF was shared).

## Where everything lives
| What | Where |
|---|---|
| Project repo | `HARSHALK2598/cs224w-structural-optimizer` (`main`) |
| PyG fork | `HARSHALK2598/pytorch_geometric`, branch `contrib/structural-optimizer` |
| Authors' code | not in git (no license); `setup_env.sh` re-clones it to `reference/` |
| rel-f1 data, venv | not in git; `setup_env.sh --data` recreates them (~5 min) |

## Resume in a fresh container
```bash
cd /home/user && git clone https://github.com/HARSHALK2598/cs224w-structural-optimizer
cd cs224w-structural-optimizer && bash scripts/setup_env.sh --data
source .venv/bin/activate
python experiments/train.py --task driver-top3 --model base --smoke --text-embedder none
```
Network hosts the environment must allow (already added): `download.pytorch.org`,
`download-r2.pytorch.org`, `data.pyg.org`, `huggingface.co`, `*.hf.co`
(plus GitHub/PyPI, allowed by default).

## Done
- Workspace, `README.md` (rubric checklist), `CLAUDE.md` (working rules), `.gitignore`.
- `scripts/setup_env.sh` (one-command rebuild), `scripts/download_data.py` (rel-f1 only, 2.1 MB).
- Environment: torch 2.12.0 CPU (pinned: newest version with `pyg-lib` wheels),
  `pyg-lib` 0.9.0, PyG 2.9.0 editable from the fork, RelBench 3.0.2.
- `experiments/train.py`: Base (RelBench relational GraphSAGE); `--smoke` passes on
  rel-f1 / driver-top3 in ~16 s on CPU.
- Fork branch: skeletons (docstrings + constructors; methods raise
  `NotImplementedError`) for `HardConcreteGate`, `RelationalFilter`, `KNNEdges`,
  `BehavioralEdges`, `TemporalContinuity`. **Not yet reviewed with Harshal**; treat
  them as drafts to walk through and revise together.

## Remaining in Phase 0
- [ ] Walk through everything above together (learning-first)
- [ ] `GatedHeteroConv`, `StructuralOptimizer` skeletons; register classes in contrib `__init__` files
- [ ] Spec tests 1–5 under `test/contrib/`
- [ ] `notes/paper_notes.md`, `notes/author_code_audit.md`
- [ ] `experiments/colab_run.ipynb` (must install `pyg-lib` matching Colab's torch)
- [ ] Fix 9 flake8 line-length warnings (train.py, download_data.py, relational_filter.py)

## Findings to remember
- Authors' gates: the training path (`model.py`) uses a Hard-Concrete `L0Gate`
  (γ=-0.1, ζ=1.1, init logit 2.0), but in eval it returns the *soft* stretched
  `sigmoid(logit/τ)`, not a hard 0/1. (`gates.py` has a separate Gumbel-softmax
  `StructureGate`; the day-1 note wrongly said the code uses Gumbel-softmax gates.)
- Authors gate **raw table columns**; our proposal gates **encoder embedding dims**
  of h⁽⁰⁾. Check against the paper before implementing `RelationalFilter`.
- Authors' defaults: channels 128, 2 layers, 128 neighbours, lr 0.005, 30 epochs,
  λ = β_vib = 1e-4, Hard-Concrete τ annealed 2.0 → 0.3, init logit 2.0, knn_k 10.
- RelBench 3 loads datasets from Hugging Face (`stanford-star/relbench-v1/rel-f1`)
  and dropped its GloVe helper (re-created in `train.py`). The authors' code pins
  RelBench 1.1.0, so running it needs a separate environment (on Colab).
- rel-f1 tasks: driver-top3, driver-dnf (classification); driver-position,
  qualifying-position, results-position (regression); driver-circuit-compete.
- `nn.knn(cosine=True)` needs `torch-cluster`; plan a pure-torch fallback.
- PyG contrib docs are generated from each package's `classes` list.
