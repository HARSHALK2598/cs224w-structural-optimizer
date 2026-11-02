# Stage B: filter at Base's lr 0.01 / hidden 64, lambda x lambda_s (Plan 01 §8)

27 runs (9 settings × seeds 0–2), 30 epochs, `--dedup-encoder --nan-safe-numerical`,
gate start 5, gate lr 0.1, temperature 2/3. Run on Colab (Tesla T4, about 0.4 min per run,
project commit `34f7716`). Uploaded as a zip and unpacked here. `check/` is Colab's
2-run quick check.

**Mean validation ROC-AUC (3 seeds):**

| λ ↓ / λ_s → | 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|
| 1e-4 | **0.7932 ± 0.0154** | 0.7896 | 0.7879 |
| 1e-3 | 0.7914 | 0.7873 | 0.7853 |
| 1e-2 | 0.7864 | 0.7866 | 0.7817 |

All 9 settings lie within 0.011, inside the seed spread. Selected on validation:
**λ = λ_s = 1e-4**.

**Mean open columns (of 45) by epoch:**

| Epoch | λ = 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|
| 1–12 | 45 | 45 | 45 |
| 16 | 31.1 | 29.1 | 20.7 |
| 30 | 30.6 | 25.3 | 12.6 |

- **λ controls sparsity by the end of training**, as Plan 01 §13 requires. But the gates
  only start closing at epoch 13–14, and the median best epoch is 16, so the selected
  models are taken while the gates are still closing (about 34 of 45 columns open at the
  best epoch, at every λ). In 21 of 27 runs, the best epoch comes before the open count
  stops changing.
- **Validation against λ** is flat or slightly decreasing; no inverted U in this range.
- **With the NaN fix,** `results.fastestLap` closed in 9 of 27 runs and `rank` in 7;
  `milliseconds` and `position` never closed. Their closure in the dry runs was partly the
  NaN bug.
- **The 3 tables a 2-layer model never reaches** closed in 16 of 27 runs.

**Protocol note:** `summary.csv` contains test means, and they were printed once during
this analysis. They were not used for any choice. Selection is on validation only, and the
reported test numbers are those of stage C.
