# Cereal: Maize Foliar Disease — Multi-Label Classification

Multi-label disease detection on field photos of maize leaves (Wiesner-Hanks et al., 2018). One photo can show several conditions at once. Two pipelines are compared:

- **Monolithic:** one ConvNeXt-Base with 8 sigmoid outputs. **Best result: macro-F1 0.468.**
- **Specialist:** five parallel models (four binary healthy-vs-disease + one four-label). Macro-F1 0.283 (worse).

No earlier deep-learning benchmark on this 8-label field dataset was found, so these numbers are a first baseline.

## Results (5-fold, out-of-fold predictions)

| Pipeline | Thresholds | Macro-F1 | Micro-F1 | Hamming loss |
| --- | --- | --- | --- | --- |
| **Monolithic** | **tuned per label** | **0.468** | **0.543** | **0.204** |
| Monolithic | 0.5 | 0.432 | 0.492 | 0.264 |
| Specialist | tuned per label | 0.283 | — | — |
| Specialist | 0.5 | 0.260 | — | — |

Tuned scores were chosen and scored on the same out-of-fold predictions, so they are slightly optimistic.

## Repository layout

```
maize-multilabel-foliar-disease/
├── results.csv   # every reported number, one row per metric
├── maize_multilabel.ipynb   # monolithic 8-label model
├── build_maize_nb.py        # builder for maize_multilabel.ipynb (source of truth)
├── maize_specialist.ipynb   # 5-specialist pipeline
├── build_specialist_nb.py   # builder for maize_specialist.ipynb (source of truth)
└── README.md
```

Edit the builder, then run it (`python build_maize_nb.py`, `python build_specialist_nb.py`) to regenerate the notebook. Don't hand-edit the `.ipynb`.

## Dataset

2,355 images; `Database.csv` has one row per image with columns `imgID_id`, `filePath`, then one 0/1 column per label. 891 images (38%) carry more than one label.

| Label | Meaning | Positives | Share |
| --- | --- | --- | --- |
| GLS | Gray leaf spot | 1,084 | 46.0% |
| NCLB | Northern corn leaf blight | 554 | 23.5% |
| PLS | Phaeosphaeria leaf spot | 493 | 20.9% |
| UnidentifiedDisease | Disease present, not identified | 332 | 14.1% |
| Other | Other damage or condition | 324 | 13.8% |
| CR | Common rust | 300 | 12.7% |
| NoFoliarSymptoms | Healthy leaf | 285 | 12.1% |
| SR | Southern rust | 39 | 1.7% |

## Setup

```bash
# Windows users: run inside WSL2
python -m venv .venv && source .venv/bin/activate
pip install tensorflow scikit-learn iterative-stratification albumentations opencv-python pandas numpy matplotlib seaborn tqdm
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

> `iterative-stratification` is imported as `from iterstrat.ml_stratifiers import MultilabelStratifiedKFold`. The package name and the import name differ.

In both notebooks, set the paths in the configuration cell:

```python
CSV_PATH  = '/path/to/Database.csv'
IMAGE_DIR = '/path/to/images'
MODEL_DIR = '/kaggle/working'   # models, plots and checkpoints are written here
```

## Monolithic pipeline — `maize_multilabel.ipynb`

| Section | What it does |
| --- | --- |
| 1–2 | Imports and `CONFIG` |
| 3 | Loads labels, EDA, co-occurrence plot (`cooccurrence.png`) |
| 4 | CLAHE pre-processing (writes processed copies) |
| 5–7 | Augmentation, weighted BCE, model builder |
| 8 | 5-fold `MultilabelStratifiedKFold` training → `fold_{1-5}_maize_convnext.keras` |
| 9–10 | Per-class metrics and plots (`per_class_f1.png`, `multilabel_confusion.png`, `fold_macro_f1.png`) |
| 11 | Per-label threshold tuning (grid 0.20–0.80, step 0.05) → `per_class_f1_tuned.png` |
| 12 | Inference on new images → `inference_results.csv` |

**Model.** ConvNeXt-Base (ImageNet) → GlobalAveragePooling → LayerNorm → Dropout 0.4 → Dense 256 ReLU → Dropout 0.3 → Dense 8 sigmoid.

**Loss.** Weighted binary cross-entropy, `pos_weight = (N − positives) / positives`, capped at 50.

**Training.**

| Phase | Trainable | LR | Epochs |
| --- | --- | --- | --- |
| 1 | Head only | 1e-4 | 15 |
| 2A | Top 30 backbone layers | 1e-5 (clipnorm 1.0) | 10 |
| 2B | Whole network | 3e-6 | 10 |

`CONFIG`: `img_size=384`, `batch_size=16`, `threshold=0.5`. Pre-processing: CLAHE on the LAB L channel (clip 2.0, 8×8), then ImageNet normalisation.

## Specialist pipeline — `maize_specialist.ipynb`

| Model | Task | Output | Trained on | Folds |
| --- | --- | --- | --- | --- |
| S1–S4 | Healthy vs GLS / NCLB / PLS / CR | 1 sigmoid | Images labelled `NoFoliarSymptoms` or that disease | `StratifiedKFold`, 5 |
| S5 | SR, NoFoliarSymptoms, Other, UnidentifiedDisease | 4 sigmoids | All 2,355 images | `MultilabelStratifiedKFold`, 5 |

All five run in parallel at inference, and their outputs are combined into the 8 labels. There is no gatekeeper model.

**`train_specialist(name, train_ds, val_ds, loss_fn, model_path, is_multilabel=False, n_classes=1, finetune=True)`** runs the same three phases as the monolithic model. All callbacks monitor `val_auc`. Set `finetune=False` to stop after phase 1. **S5 is called with `finetune=False`**, because its frozen phase scored best (val AUC 0.8026 vs 0.7830 for top-30 and 0.7919 for full).

**Crash-safe by design.** Every finished (specialist, fold) pair is recorded in `specialist_checkpoint.json` and skipped on re-run. Out-of-fold probabilities are saved after each fold (`oof_arrays/oof_binary_{disease}.npy`, `oof_arrays/oof_final.npy`). Models are saved as `specialist_{disease}_fold{k}.keras` and `specialist_final_fold{k}.keras`.

Sections 12–16 produce the per-specialist reports, threshold tuning, the 8-label comparison (`specialist_f1.png`, `binary_auc.png`) and inference with `predict_specialist(path)` → `specialist_inference_results.csv`.

## Known issues

- **`train_specialist()` keeps the last phase, not the best one.** It saves the model after phase 2B and deletes the best-weights checkpoint, so for S1–S4 a better earlier phase is lost.
- **The 8-label out-of-fold assembly (section 14) doesn't match inference.** Images outside a binary specialist's training subset get prediction 0 for that disease, but at inference every image goes through every specialist.
- **S1–S4 never see multi-disease images**, which are 38% of the data.
- **Thresholds are tuned on the same out-of-fold predictions they are scored on.**
- **The duplicate screen has not been run on maize.**
- **SR has 39 positives** (about 8 per test fold), so its scores are noisy.
- **`epochs_phase1` may differ from the final run.** It is 15 here; the final value used was not recorded.

## Roadmap

- [ ] Keep each specialist's best phase; re-run fold 1 against the monolithic fold 1
- [ ] Retrain S1–S4 one-vs-rest on all images
- [ ] Score specialists on every held-out image, matching inference
- [ ] Duplicate screen + grouped multilabel folds; tune thresholds on an inner split
- [ ] 3–5 seeds; exact per-label precision/recall/F1
- [ ] Write up as the first benchmark on this dataset

## Documentation

Full handover documentation (guide, environment setup, data, configs, experiments, results, known issues, future work) is in [`docs/`](docs/README.md), as Markdown and PDF.

## Contact

Anas Alsirafy, American University in Cairo (anas.samy@aucegypt.edu). Supervisor: Prof. Maki K. Habib.

## Related repositories

- [cassava-leaf-disease-classification](https://github.com/Anas-Alsirafy/cassava-leaf-disease-classification) — cassava classifier, where the training recipe was developed
- [leaf-disease-cross-dataset-study](https://github.com/Anas-Alsirafy/leaf-disease-cross-dataset-study) — Paper 2 five-dataset leakage-controlled comparison
