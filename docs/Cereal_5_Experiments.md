# Cereal 5 — Experiments

Oct 26, 2026 · Anas Alsirafy

Five steps were run: a monolithic baseline, threshold tuning on it, then a specialist redesign that was checked on one fold, adjusted, and run in full. The monolithic model with tuned thresholds remains the best result.

## Experiment log

| Step | What was done | Why | Macro-F1 (OOF) | Outcome |
| --- | --- | --- | --- | --- |
| 1 | Monolithic ConvNeXt-Base, 8 sigmoids, weighted BCE, threshold 0.5 | Baseline | 0.432 (micro 0.492, Hamming 0.264) | Rare labels over-predicted: high recall, low precision |
| 2 | Per-label threshold search on step 1's OOF predictions | Fix the recall bias from weighted BCE | 0.468 (micro 0.543, Hamming 0.204) | Best maize result |
| 3 | Specialist pipeline, fold 1 only | One shared head blurs 8 decision boundaries; SR has 39 positives against 2,316 negatives | 0.334 (micro 0.356) | Already below monolithic |
| 4 | Compared S5's validation AUC by phase | Check whether fine-tuning helps S5 | Frozen 0.8026; top-30 0.7830; full 0.7919 | Fine-tuning removed from S5 |
| 5 | Specialist pipeline, all 5 folds | Full evaluation | 0.260 at 0.5; 0.283 tuned | Clearly worse than monolithic; open |

## What the specialist design assumed

- Binary specialists S1–S4 learn a sharp healthy-vs-disease boundary from clean images, free of competition with other labels.
- Grouping SR with other rare or vague labels in S5 gives it more signal than SR alone.
- Per-specialist thresholds can be tuned like the monolithic ones.

The result says at least one assumption fails. Two candidate causes, not yet tested: (a) S1–S4 never see images with several diseases, which are 38% of the data; (b) `train_specialist()` keeps the last phase's model even when an earlier phase was better (see Known Issues).

## Rules for the next run

1. Fix the model-saving bug first, so you are not testing a broken pipeline.
2. Run the duplicate screen and use grouped folds.
3. Tune thresholds on an inner validation split, not the OOF predictions you report.
4. Report per-label F1, macro-F1, micro-F1, Hamming loss and AUC for both pipelines side by side.
5. Run fold 1 of any new design first and compare with the monolithic fold-1 score before spending 25 runs.
