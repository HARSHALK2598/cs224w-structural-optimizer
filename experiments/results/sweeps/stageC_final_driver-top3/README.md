# Stage C: final comparison (Plan 01 §8)

Stage A's winner (Base, lr 0.01, hidden 64) against stage B's winner (filter,
λ = λ_s = 1e-4) on 5 fresh seeds (10–14). Run on Colab (Tesla T4, project commit
`d7c3102`), both arms in one session. 30 epochs, `--dedup-encoder
--nan-safe-numerical`, gate start 5, gate lr 0.1, temperature 2/3. Test was read once,
here.

| Seed | Base val | Filter val | Base test | Filter test | Filter: open columns / tables |
|---|---|---|---|---|---|
| 10 | 0.7835 | 0.7982 | 0.7581 | 0.7204 | 30 / 5 |
| 11 | 0.7871 | 0.7677 | 0.8077 | 0.8060 | 30 / 5 |
| 12 | 0.7950 | 0.7922 | 0.7225 | 0.7342 | 32 / 5 |
| 13 | 0.7865 | 0.8052 | 0.7345 | 0.8018 | 31 / 5 |
| 14 | 0.7726 | 0.7872 | 0.8057 | 0.7971 | 30 / 5 |
| **Mean ± std** | **0.7849 ± 0.0081** | **0.7901 ± 0.0142** | **0.7657 ± 0.0396** | **0.7719 ± 0.0411** | |
| Paper (Table 1, test) | | | 0.8148 ± 0.0106 | 0.8466 ± 0.0061 | |

**Difference (filter − Base).** Validation is +0.005 (Welch p = 0.50, paired by seed
p = 0.51). Test is +0.006 (Welch p = 0.81, paired p = 0.74). There is no detectable
difference.

**What the gates removed at the best epoch.** Every filter run closed the 3 tables a
2-layer model never reaches (and their 13 columns), which does not change any prediction.
Beyond those, it closed only `races.round` (4 of 5 runs), `results.fastestLap` (2) and
`races.year` (1).

**Configurations tried in total:** 7 dry runs, 9 Base settings (stage A) and 9 filter
settings (stage B), then these 2 on 5 fresh seeds.

**Notes:**
- Test varies far more across seeds (std 0.04, from 0.72 to 0.81) than the paper reports
  (0.006–0.011). The test split has 726 rows.
- Our Base is 4.9 points below the paper's Base on test, with the paper's tuning grid.
