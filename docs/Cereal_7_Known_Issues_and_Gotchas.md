# Cereal 7 — Known Issues and Gotchas

Oct 26, 2026 · Anas Alsirafy

One bug in `train_specialist()` likely affects every saved specialist and its OOF predictions; fix it before re-running anything. The rest are evaluation weak spots and errors that were already solved.

## Bugs and evaluation weak spots

| Where | Problem | Effect | Fix |
| --- | --- | --- | --- |
| `train_specialist()` | Saves the model as it stands after phase 2B and then deletes the best-weights checkpoint. Early stopping only restores the best epoch within each phase. | When an earlier phase was better, the saved model and its OOF predictions are the worse one. S5 is a confirmed case (0.8026 frozen vs 0.7919 after 2B). | Track the best validation AUC across all three phases; save and predict with that model; keep the checkpoint file. |
| Specialist training data | S1–S4 train only on images labelled healthy or their one disease. | They never see the 38% of images with several labels, which the test folds contain. | Train one-vs-rest on all images, or include multi-disease images as positives. |
| Threshold tuning | Thresholds chosen on the same OOF predictions that are then scored. | Tuned F1 is optimistic. | Tune on an inner validation split. |
| No duplicate screen | Maize was never run through the MD5 + dHash screen. | Unknown leakage between folds. | Screen and use grouped multilabel folds. |
| Old summary PDF | Listed Other = 136 and UnidentifiedDisease = 201, and per-label F1 values marked approximate. | Wrong counts in circulation. | Use `Database.csv`: 324 and 332. |
| 8-label OOF assembly (section 14) | Images outside a binary specialist's training subset get prediction 0 for that disease, but at inference (section 16) every image goes through every specialist. | The OOF score does not measure the deployed pipeline; a GLS specialist that never saw NCLB-only leaves may fire on them in use. | Score each specialist on every image in its held-out fold, matching inference. |
| SR label | 39 positives in 2,355 images. | Any SR number is noisy (about 8 positives per test fold). | Report SR separately; don't let it drive design decisions alone. |

## Errors already solved

| Error | Cause | Fix |
| --- | --- | --- |
| `AttributeError: Normalization has no .layers` | ConvNeXt-Base inserts a Normalization layer at index 1, so the backbone isn't where code expects | `get_backbone()` scans `model.layers` for the first sub-model with `.layers` |
| `ValueError: array is read-only` in the co-occurrence cell | The NumPy array taken from the co-occurrence table is read-only | `co.to_numpy(dtype=float).copy()` before `np.fill_diagonal()` |
| `ModuleNotFoundError: iterstrat` | Wrong pip package name | `pip install iterative-stratification` |
| Rare labels over-predicted at 0.5 | Weighted BCE favours recall | Per-label threshold tuning |
| Long runs dying mid-way | Session limits / crashes | `specialist_checkpoint.json` + per-fold OOF `.npy` + ModelCheckpoint |
