# Cereal 4 — Configs

Oct 26, 2026 · Anas Alsirafy

Both maize pipelines use a ConvNeXt-Base backbone trained in three phases. They differ in how the 8 labels are split across models. Values not listed here are in each notebook's `CONFIG` cell.

## Monolithic model (`maize_multilabel.ipynb`)

| Setting | Value |
| --- | --- |
| Backbone | ConvNeXt-Base, ImageNet weights |
| Head | Global average pooling → LayerNorm → Dropout 0.4 → Dense 256 ReLU → Dropout 0.3 → Dense 8 sigmoid |
| Pre-processing | CLAHE (L channel, clip 2.0, 8×8) + ImageNet normalisation |
| Augmentation | RandomResizedCrop, horizontal flip, vertical flip, transpose, colour jitter, Gaussian blur |
| Loss | Weighted BCE; pos_weight = (N − positives) / positives, capped at 50 |
| Phase 1 | Backbone frozen, lr 1e-4, 15 epochs |
| Phase 2A | Top 30 backbone layers unfrozen, lr 1e-5, clipnorm 1.0 |
| Phase 2B | Whole network, lr 3e-6 |
| Folds | 5-fold MultilabelStratifiedKFold |
| Thresholds | Per label, grid 0.20–0.80 in steps of 0.05, chosen on OOF predictions |

## Specialist pipeline (`maize_specialist.ipynb`)

Five models run in parallel; their outputs are combined into the 8 labels. There is no gatekeeper model.

| Model | Task | Output | Training images | Folds |
| --- | --- | --- | --- | --- |
| S1 | Healthy vs GLS | 1 sigmoid | NoFoliarSymptoms or GLS | StratifiedKFold, 5 |
| S2 | Healthy vs NCLB | 1 sigmoid | NoFoliarSymptoms or NCLB | StratifiedKFold, 5 |
| S3 | Healthy vs PLS | 1 sigmoid | NoFoliarSymptoms or PLS | StratifiedKFold, 5 |
| S4 | Healthy vs CR | 1 sigmoid | NoFoliarSymptoms or CR | StratifiedKFold, 5 |
| S5 | SR, NoFoliarSymptoms, Other, UnidentifiedDisease | 4 sigmoids | All 2,355 | MultilabelStratifiedKFold, 5 |

Shared training settings (`train_specialist()`):

| Setting | Phases 1 and 2A | Phase 2B |
| --- | --- | --- |
| Monitored metric | Validation AUC (max) | Validation AUC (max) |
| ReduceLROnPlateau | factor 0.3, patience 3, min lr 1e-7 | factor 0.2, patience 2, min lr 1e-8 |
| Early stopping | patience 5, restore best weights | patience 5, restore best weights |
| Optimiser | Adam (2A with clipnorm 1.0) | Adam, clipnorm 1.0 |
| Learning rates | `lr_phase1`, `lr_phase2a` from CONFIG | `lr_phase2b` from CONFIG |

**Latest change:** S5 no longer fine-tunes. Its frozen phase reached validation AUC 0.8026; unfreezing gave 0.7830 (top 30 layers) and 0.7919 (full network).
