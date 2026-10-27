# Cereal 6 — Results and Findings

Oct 26, 2026 · Anas Alsirafy

The monolithic model with per-label thresholds is the maize result to quote: macro-F1 0.468 over 5 folds. Splitting the problem into five specialists made it worse (0.283), and per-label thresholds were the single most useful change.

## Headline numbers (5-fold OOF)

| Pipeline | Thresholds | Macro-F1 | Micro-F1 | Hamming loss |
| --- | --- | --- | --- | --- |
| Monolithic | Tuned per label | 0.468 | 0.543 | 0.204 |
| Monolithic | 0.5 | 0.432 | 0.492 | 0.264 |
| Specialists (fold 1 only) | — | 0.334 | 0.356 | — |
| Specialists (all folds) | Tuned | 0.283 | — | — |
| Specialists (all folds) | 0.5 | 0.260 | — | — |

Dashes are values not recorded. Tuned scores were chosen and scored on the same OOF predictions, so they are slightly optimistic.

## Findings

- **Threshold tuning matters most.** Moving from 0.5 to per-label thresholds added 3.6 points of macro-F1 and cut Hamming loss by 6 points. Weighted BCE pushes rare labels toward high recall and low precision; the thresholds pull them back.
- **Gray leaf spot is easy, southern rust is not.** Per-label F1 was only recorded approximately: about 0.83 for GLS (1,084 positives) and about 0.14 for SR (39 positives). SR shows some signal but too few examples to learn from. Re-run the evaluation cell for exact per-label values.
- **Specialists lost.** Fold 1 was already 13 points of macro-F1 below the monolithic model, and the full run ended 18.5 points below. Likely causes are listed in the Experiments doc; neither has been tested.
- **Fine-tuning hurt S5.** Its frozen backbone gave the best validation AUC (0.8026); both unfreezing phases were lower.
- **This is a first benchmark.** No earlier deep-learning result exists for this 8-label field dataset, so these numbers set the baseline rather than compete with one.
