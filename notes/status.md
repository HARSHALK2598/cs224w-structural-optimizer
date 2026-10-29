# Project status / handoff

_Last updated: 2026-10-29 (end of day 3)_

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

## Day 3 summary (start here next)
- **Plan 00 revised (Option A, merged in PR #3):** Base numbers come from RelBench's
  official `examples/gnn_entity.py`, run **unmodified** (3 rel-f1 tasks × 5 seeds).
  `train.py` must reproduce them (one seed per task within mean ± 2 std); its Base
  runs later become the control for the optimizer.
- **Step 1 merged (PR #2):** `train.py` split into functions (`load_graph`,
  `task_spec`, `train_epoch`, `predict`, `evaluate`, `fit`), label-leak columns
  dropped, regression clipping, explicit tune metric with ties to later epoch, MAE
  logged; 9 unit tests in `experiments/tests/`.
- **Step 2 open (PR #4):** GloVe text features (`sentence-transformers`, model ~1 GB
  on disk in `data/hf`, already downloaded here). Harshal reviews, then merges.
- **Walkthroughs:** Part 1 and Part 2 have Does/In/Out notes; Part 2 runs
  `train.py`'s real functions.
- **Next: step 3, the training.** Clone RelBench at `91c1f64` (examples/ is not in
  the pip package), run `gnn_entity.py` unmodified, time one run, then 15 runs. The
  official script prints only NMAE for regression: compute MAE from its saved
  prediction CSV.
- **Parked:** Colab setup (needs: install torch 2.12 + matching pyg-lib, clone the
  private repo via a token in Colab Secrets or make it public).
- **Merging:** this session cannot merge PRs that have no GitHub review; Harshal
  merges on GitHub.

## Day 2 summary
- **Workflow agreed:** small commits with unit tests. For each one: a short plan in
  `plans/` reviewed as a GitHub PR → implementation → review → merge. Plans hold only
  the implementation spec; learning happens in chat and the walkthrough notebooks.
- **Baseline first:** Base-RelBench (RelBench's official
  `examples/gnn_entity.py` settings) on rel-f1 driver-top3 / driver-dnf /
  driver-position, 5 seeds. Compare with the paper's Base (81.48 / 74.53 / MAE 3.88),
  then decide whether to re-create the paper's tuned Base.
- **Plan 00 under review:** PR #1 (`plan/00-baseline`), all review comments so far
  addressed. Open questions: one plan per milestone or per commit? Add trivial
  baselines?
- **Walkthrough 1 done** (`walkthroughs/01_data_graph_sampling.ipynb`): data, graph,
  sampling. Findings: 3 tables are 3 hops from drivers and never reach a 2-layer
  model; 55/1353 train seeds have no history.
- **Next:** finish the plan 00 review → commit 1 of the plan; walkthrough 2 (encoders,
  one GraphSAGE layer by hand, loss, one optimiser step).
- The paper PDF is at arxiv.org/pdf/2606.08491 (arxiv.org is now allowed).

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
