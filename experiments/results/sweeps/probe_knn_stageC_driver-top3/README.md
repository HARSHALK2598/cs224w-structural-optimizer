# KNN (n = 10) against Base, driver-top3, Plan 01's stage C settings

Settings (both): lr 0.01, hidden 64, 30 epochs, `--dedup-encoder`,
`--nan-safe-numerical`, seeds 10-14. Base is stage C's
(`../stageC_final_driver-top3`, Colab GPU), not rerun; KNN ran on CPU here.

| | Val ROC-AUC | Test ROC-AUC | Best epochs |
|---|---|---|---|
| Base (stage C) | 0.7849 ± 0.0081 | 0.7657 ± 0.0396 | 22, 15, 14, 15, 20 |
| Base + KNN, n = 10 | 0.7522 ± 0.0134 | 0.7842 ± 0.0216 | 15, 25, 27, 30, 9 |

Means ± sample std over the 5 seeds, from `results.csv`.

Correction: the commit that added these results (e6f5e5c) gives the KNN
validation std as 0.0137; the value is 0.0134, as above.
