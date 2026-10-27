# Cereal 3 — Data

Oct 26, 2026 · Anas Alsirafy

One dataset: 2,355 field photos of maize leaves (Wiesner-Hanks et al., 2018) with 8 binary labels. 891 images (38%) carry more than one label, and the labels are badly imbalanced — southern rust has only 39 positives.

## Source and format

- Images: Kaggle mirror of the Wiesner-Hanks 2018 maize foliar disease set.
- Labels: `Database.csv`, one row per image. Columns: `imgID_id`, `filePath`, then one 0/1 column per label.
- Example row: `1207, 100_CIMG3835_1-135-2.JPG, 1, 0, 0, 0, 0, 0, 0, 0` (gray leaf spot only).

## Labels

| Label | Meaning | Positive images | Share |
| --- | --- | --- | --- |
| GLS | Gray leaf spot | 1,084 | 46.0% |
| NCLB | Northern corn leaf blight | 554 | 23.5% |
| PLS | Phaeosphaeria leaf spot | 493 | 20.9% |
| UnidentifiedDisease | Disease present, not identified | 332 | 14.1% |
| Other | Other damage or condition | 324 | 13.8% |
| CR | Common rust | 300 | 12.7% |
| NoFoliarSymptoms | Healthy leaf | 285 | 12.1% |
| SR | Southern rust | 39 | 1.7% |

Counts come straight from `Database.csv`. An earlier maize summary PDF listed Other as 136 and UnidentifiedDisease as 201; those figures were wrong.

## Co-occurrence

GLS frequently appears together with other diseases on the same image. This matters for the specialist design: a model trained only on "healthy or GLS" images never sees GLS mixed with NCLB or PLS, but the test set is full of them. Recompute the full co-occurrence matrix with the notebook cell before redesigning (copy the array before `np.fill_diagonal`, see Known Issues).

## Pre-processing and splits

- **CLAHE:** LAB colour space, CLAHE on the L channel (clipLimit 2.0, 8×8 tiles), back to RGB; then ImageNet normalisation.
- **Monolithic and S5:** 5-fold MultilabelStratifiedKFold over all 2,355 images.
- **S1–S4:** 5-fold StratifiedKFold over the subset labelled NoFoliarSymptoms or the specialist's disease.
- **Not yet done:** the duplicate screen (MD5 + dHash, grouped folds) used on every other dataset was never run on maize. Run it before the next experiment; see the cross-dataset doc for the method.

## Literature context

No published deep-learning benchmark on this exact 8-label field set was found. Papers reporting 99%+ on maize use PlantVillage's 4-class, single-label, plain-background images, which are not comparable.
