# Results

All on `rel-f1` / `driver-top3`, ROC-AUC, mean ± std over seeds. Choose on validation; test
read once (stage C). Each folder has `results.csv` (one row per run); sweep folders also
have `summary.csv`, `env.json` (versions, device, commits) and per-run `logs/`.

| Folder | What | Runs | Settings | Validation | Test |
|---|---|---|---|---|---|
| `base_dedup_encoder` | Base with `--dedup-encoder` (same accuracy, 2.5× faster) | 5 seeds | 20 epochs, RelBench defaults, CPU | 0.7865 ± 0.0070 | 0.7923 ± 0.0197 |
| `base_nan_safe_numerical` | Base with `--nan-safe-numerical` | 5 seeds | as above | 0.7833 ± 0.0093 | 0.7826 ± 0.0208 |
| `filter_gates_open_exact` | Base twice and gates-fixed-at-1 once, seed 0 (CPU runs are not bit-for-bit repeatable; exactness is a unit test) | 3 | 20 epochs, CPU | 0.7884, 0.7884 / 0.7864 | 0.7713, 0.7606 / 0.7701 |
| `sweeps/filter_gates_open_5seeds` | the same over 5 seeds (Base / gates open) | 10 | 20 epochs, CPU | 0.7849 ± 0.0161 / 0.7853 ± 0.0177 | 0.7888 ± 0.0228 / 0.7925 ± 0.0161 |
| `filter_dryruns` | filter dry runs that set gate start 5 and gate lr 0.1 | 7 | seed 0, 20 epochs, CPU | – | – |
| `sweeps/stageA_base_driver-top3` | Base: lr {1e-4, 1e-3, 1e-2} × hidden {64, 128, 256} | 27 | 3 seeds, 30 epochs, CPU | best lr 1e-2, hidden 64: 0.7880 ± 0.0053 | – |
| `sweeps/stageB_filter_driver-top3` | filter at stage A's best: λ × λ_s ∈ {1e-4, 1e-3, 1e-2}² | 27 | 3 seeds, 30 epochs, T4 | best λ = λ_s = 1e-4: 0.7932 ± 0.0154 | – |
| `sweeps/stageC_final_driver-top3` | the two winners on fresh seeds 10–14 | 10 | 30 epochs, T4 | Base 0.7849 ± 0.0081 / filter 0.7901 ± 0.0142 (p = 0.50) | Base 0.7657 ± 0.0396 / filter 0.7719 ± 0.0411 |
| `sweeps/probe_quick_knn_driver-top3` | Base / + KNN (n = 10), RelBench defaults (lr 5e-3, hidden 128); did not converge (best epoch = last) | 6 | seeds 0–2, 10 epochs, CPU | 0.7708 ± 0.0337 / 0.6932 ± 0.0619 | 0.7779 ± 0.0187 / 0.7718 ± 0.0438 |
| `sweeps/probe_knn_stageC_driver-top3` | + KNN (n = 10) with stage C's settings; compare with stage C's Base | 5 | seeds 10–14, 30 epochs, CPU | 0.7522 ± 0.0134 (Base 0.7849 ± 0.0081, Welch p = 0.003) | 0.7842 ± 0.0216 |

Paper (Table 1, test): Base 0.8148 ± 0.0106, filtering 0.8466 ± 0.0061.

All runs from stage A on use `--dedup-encoder --nan-safe-numerical`; gated runs use gate
start 5, gate lr 0.1, temperature 2/3.
