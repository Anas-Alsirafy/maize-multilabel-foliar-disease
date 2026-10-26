import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": []
}

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src,"id":f"md{len(nb['cells'])}"}

def code(src):
    return {"cell_type":"code","metadata":{},"source":src,"outputs":[],"execution_count":None,"id":f"cc{len(nb['cells'])}"}

# ── TITLE ──────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
# Maize Foliar Disease — Specialist Pipeline
**5 Parallel Binary/Multi-label Specialists · ConvNeXtBase · 5-Fold CV**

| | |
|---|---|
| **Dataset** | 2,355 field images, 8 disease classes |
| **Architecture** | 5 independent specialists, all running in parallel at inference |
| **Specialist 1–4** | Binary: `NoFoliarSymptoms` (Healthy) vs `GLS` / `NCLB` / `PLS` / `CR` |
| **Specialist 5** | Multi-label sigmoid: `SR`, `NoFoliarSymptoms`, `Other`, `UnidentifiedDisease` |
| **Training data** | Specialists 1–4 train on healthy + their disease rows only (clean binary split) |
| **Backbone** | ConvNeXtBase (ImageNet weights), shared architecture across all specialists |
| **Protocol** | 5-fold stratified CV per specialist, per-class threshold tuning |

> **Design rationale:** Each disease specialist sees only clean positive/negative examples
> (healthy leaves vs leaves showing that disease). This avoids the noise of multi-class
> competition and lets each specialist optimise its decision boundary independently.
> Specialist 5 handles ambiguous / rare / non-disease categories as a multi-label problem.
"""))

# ── CELL 1 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 1. Install & Imports"))
nb["cells"].append(code("""\
!pip install -q iterative-stratification albumentations

import os, gc, cv2, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, f1_score,
                             roc_auc_score, confusion_matrix,
                             multilabel_confusion_matrix, hamming_loss)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import albumentations as A
from tqdm import tqdm

warnings.filterwarnings('ignore')
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
"""))

# ── CELL 2 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 2. Configuration"))
nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  PATHS  — update before running
# ─────────────────────────────────────────────
CSV_PATH  = '/path/to/Database.csv'
IMAGE_DIR = '/path/to/images'
MODEL_DIR = '/kaggle/working'

CKPT_PATH    = os.path.join(MODEL_DIR, 'specialist_checkpoint.json')
OOF_DIR      = os.path.join(MODEL_DIR, 'oof_arrays')
os.makedirs(OOF_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
#  CLASS DEFINITIONS
# ─────────────────────────────────────────────
ALL_COLS = ['GLS', 'NCLB', 'PLS', 'CR', 'SR',
            'NoFoliarSymptoms', 'Other', 'UnidentifiedDisease']

# Specialists 1-4: binary Healthy vs Disease
BINARY_DISEASES = ['GLS', 'NCLB', 'PLS', 'CR']
HEALTHY_COL     = 'NoFoliarSymptoms'   # Healthy = NoFoliarSymptoms == 1

# Specialist 5: multi-label over ambiguous/rare/non-specific categories
FINAL_COLS   = ['SR', 'NoFoliarSymptoms', 'Other', 'UnidentifiedDisease']
NUM_FINAL    = len(FINAL_COLS)   # 4

CONFIG = {
    'img_size'      : 384,
    'batch_size'    : 16,
    'lr_phase1'     : 1e-4,
    'lr_phase2a'    : 1e-5,
    'lr_phase2b'    : 3e-6,
    'epochs_phase1' : 15,
    'epochs_phase2a': 10,
    'epochs_phase2b': 10,
    'threshold'     : 0.5,
}

print(f"Binary specialists: {BINARY_DISEASES}")
print(f"Final specialist classes ({NUM_FINAL}): {FINAL_COLS}")
print(f"Healthy negatives defined by: {HEALTHY_COL}=1")
print(f"Checkpoint path: {CKPT_PATH}")
print(f"OOF arrays dir:  {OOF_DIR}")
"""))

# ── CELL 3 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 3. Data Loading & Split Construction"))
nb["cells"].append(code("""\
df = pd.read_csv(CSV_PATH)
df['img_path'] = df['filePath'].apply(lambda x: os.path.join(IMAGE_DIR, x))

print(f"Total images: {len(df)}")
print()

print("=== Full label counts ===")
for col in ALL_COLS:
    pct = df[col].mean() * 100
    print(f"  {col:<22} {df[col].sum():>5}  ({pct:.1f}%)")

print()
print("=== Binary specialist training set sizes ===")
healthy_df = df[df[HEALTHY_COL] == 1]
print(f"  Healthy (NoFoliarSymptoms=1): {len(healthy_df)} images")
for disease in BINARY_DISEASES:
    disease_df = df[df[disease] == 1]
    combined   = len(healthy_df) + len(disease_df)
    print(f"  {disease:<6}: {len(disease_df):>4} disease + {len(healthy_df):>4} healthy = {combined:>4} total")

print()
print("=== Final specialist (Specialist 5) ===")
print(f"  Trains on all {len(df)} images")
print(f"  Classes: {FINAL_COLS}")
for col in FINAL_COLS:
    print(f"    {col:<22} {df[col].sum():>5}  ({df[col].mean()*100:.1f}%)")
"""))

# ── CELL 4 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 4. CLAHE Preprocessing"))
nb["cells"].append(code("""\
CLAHE_DIR = os.path.join(MODEL_DIR, 'clahe_images')
os.makedirs(CLAHE_DIR, exist_ok=True)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_path, out_dir):
    fname    = os.path.basename(img_path)
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path):
        return out_path
    img = cv2.imread(img_path)
    if img is None:
        return img_path
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = _clahe.apply(l)
    lab     = cv2.merge([l, a, b])
    out     = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(out_path, out)
    return out_path

print("Applying CLAHE (skips already-processed files)...")
clahe_paths = [apply_clahe(p, CLAHE_DIR) for p in tqdm(df['img_path'].values)]
df['clahe_path'] = clahe_paths
print("Done ✓")
"""))

# ── CELL 5 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 5. Augmentation & Dataset Builders"))
nb["cells"].append(code("""\
aug_pipeline = A.Compose([
    A.RandomResizedCrop(size=(CONFIG['img_size'], CONFIG['img_size']),
                        scale=(0.85, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Transpose(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
])

def preprocess_convnext(image_path, label):
    img  = tf.io.read_file(image_path)
    img  = tf.image.decode_jpeg(img, channels=3)
    img  = tf.image.resize(img, [CONFIG['img_size'], CONFIG['img_size']])
    img  = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std  = tf.constant([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return img, label

def augment_fn(path, label):
    path_str = path.numpy().decode()
    img = cv2.imread(path_str)
    if img is None:
        img = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = aug_pipeline(image=img)['image'].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    img  = (img - mean) / std
    return img, label

def build_train_ds(image_paths, labels, n_classes):
    \"\"\"Build training dataset. labels shape: (N,) for binary, (N, k) for multi-label.\"\"\"
    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels.astype(np.float32))
    )
    label_shape = [n_classes] if n_classes > 1 else []
    def _aug(path, lbl):
        img, l = tf.py_function(func=augment_fn, inp=[path, lbl],
                                Tout=[tf.float32, tf.float32])
        img.set_shape([CONFIG['img_size'], CONFIG['img_size'], 3])
        if n_classes > 1:
            l.set_shape([n_classes])
        else:
            l.set_shape([])
        return img, l
    ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(512).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    return ds

def build_val_ds(image_paths, labels):
    \"\"\"Validation dataset — no augmentation.\"\"\"
    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels.astype(np.float32))
    )
    ds = ds.map(preprocess_convnext, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    return ds

print("Dataset builders defined ✓")
"""))

# ── CELL 6 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 6. Model Builders"))
nb["cells"].append(code("""\
def build_binary_model(trainable_base=False):
    \"\"\"
    ConvNeXtBase → GAP → LayerNorm → Dropout → Dense(256) → Dropout → Dense(1, sigmoid)
    Binary classifier: output = P(disease present)
    \"\"\"
    inputs = tf.keras.Input(shape=(CONFIG['img_size'], CONFIG['img_size'], 3))
    base   = tf.keras.applications.ConvNeXtBase(
        include_top=False, weights='imagenet', input_tensor=inputs
    )
    base.trainable = trainable_base

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs)

def build_multilabel_model(n_classes, trainable_base=False):
    \"\"\"
    ConvNeXtBase → GAP → LayerNorm → Dropout → Dense(256) → Dropout → Dense(n, sigmoid)
    Multi-label: each output = P(class present), independent.
    Used for Specialist 5 (SR / NoFoliarSymptoms / Other / UnidentifiedDisease).
    \"\"\"
    inputs = tf.keras.Input(shape=(CONFIG['img_size'], CONFIG['img_size'], 3))
    base   = tf.keras.applications.ConvNeXtBase(
        include_top=False, weights='imagenet', input_tensor=inputs
    )
    base.trainable = trainable_base

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='sigmoid', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs)

def get_backbone(model):
    \"\"\"Find the ConvNeXtBase sub-model by scanning for a layer with its own .layers.\"\"\"
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            return layer
    raise ValueError("No backbone sub-model found.")

# Sanity check
m = build_binary_model()
print(f"Binary model output shape:      {m.output_shape}")   # (None, 1)
del m; gc.collect()
m = build_multilabel_model(NUM_FINAL)
print(f"Multi-label model output shape: {m.output_shape}")   # (None, 4)
del m; gc.collect()
"""))

# ── CELL 7 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 7. Loss Helpers"))
nb["cells"].append(code("""\
def compute_pos_weight_binary(labels):
    \"\"\"pos_weight = (N_neg / N_pos), capped at 50.\"\"\"
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    pw  = neg / max(pos, 1.0)
    pw  = min(pw, 50.0)
    return np.float32(pw)

def weighted_bce_binary(pos_weight):
    \"\"\"Weighted BCE for scalar binary label. pos_weight scales positive loss.\"\"\"
    pw = tf.constant(pos_weight, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        bce    = -(pw * y_true * tf.math.log(y_pred)
                   + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)
    return loss

def compute_pos_weights_multilabel(labels_array):
    \"\"\"Per-class pos_weight for multi-label BCE, capped at 50.\"\"\"
    pos = labels_array.sum(axis=0).astype(float)
    N   = float(len(labels_array))
    pw  = (N - pos) / np.maximum(pos, 1.0)
    pw  = np.clip(pw, 1.0, 50.0)
    return pw.astype(np.float32)

def weighted_bce_multilabel(pos_weights):
    \"\"\"Weighted BCE for multi-label output.\"\"\"
    pw = tf.constant(pos_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = -(pw * y_true * tf.math.log(y_pred)
                   + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)
    return loss

print("Loss helpers defined ✓")
"""))

# ── CHECKPOINT HELPERS ─────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 8. Checkpoint Helpers

`specialist_checkpoint.json` tracks completed folds so a crash mid-run resumes
from the next incomplete fold rather than starting over.

```
{
  "GLS":   {"1": {"f1": 0.91, "auc": 0.96},  "2": {...}, ...},
  "NCLB":  {...},
  ...
  "final": {"1": {...}, ...}
}
```

OOF probability arrays are saved as `.npy` files after every fold so they survive
a kernel restart:
- `oof_arrays/oof_binary_{disease}.npy` — shape (N_dataset,)  float32, NaN for rows not in specialist
- `oof_arrays/oof_final.npy`            — shape (2355, 4)     float32
"""))
nb["cells"].append(code("""\
import json as _json

def ckpt_load():
    \"\"\"Load checkpoint dict from disk (returns {} if not found).\"\"\"
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return _json.load(f)
    return {}

def ckpt_save(ckpt):
    \"\"\"Atomically write checkpoint dict to disk.\"\"\"
    tmp = CKPT_PATH + '.tmp'
    with open(tmp, 'w') as f:
        _json.dump(ckpt, f, indent=2)
    os.replace(tmp, CKPT_PATH)

def ckpt_mark(ckpt, specialist_key, fold, metrics_dict):
    \"\"\"Record fold completion. specialist_key = disease name or 'final'.\"\"\"
    if specialist_key not in ckpt:
        ckpt[specialist_key] = {}
    ckpt[specialist_key][str(fold)] = metrics_dict
    ckpt_save(ckpt)

def ckpt_done(ckpt, specialist_key, fold):
    \"\"\"Return True if this fold already completed AND the model file exists.\"\"\"
    model_key = specialist_key if specialist_key != 'final' else 'final'
    if specialist_key == 'final':
        model_path = os.path.join(MODEL_DIR, f'specialist_final_fold{fold}.keras')
    else:
        model_path = os.path.join(MODEL_DIR, f'specialist_{specialist_key}_fold{fold}.keras')
    in_ckpt    = str(fold) in ckpt.get(specialist_key, {})
    on_disk    = os.path.exists(model_path)
    return in_ckpt and on_disk

def oof_save_binary(disease, arr):
    np.save(os.path.join(OOF_DIR, f'oof_binary_{disease}.npy'), arr)

def oof_load_binary(disease, n_rows):
    path = os.path.join(OOF_DIR, f'oof_binary_{disease}.npy')
    if os.path.exists(path):
        return np.load(path)
    return np.full(n_rows, np.nan, dtype=np.float32)

def oof_save_final(arr):
    np.save(os.path.join(OOF_DIR, 'oof_final.npy'), arr)

def oof_load_final(n_rows, n_classes):
    path = os.path.join(OOF_DIR, 'oof_final.npy')
    if os.path.exists(path):
        return np.load(path)
    return np.full((n_rows, n_classes), np.nan, dtype=np.float32)

# ── Inspect existing checkpoint ───────────────────────────────────────────────
ckpt = ckpt_load()
if ckpt:
    print("Existing checkpoint found:")
    for specialist, folds in ckpt.items():
        done_folds = sorted(int(k) for k in folds.keys())
        print(f"  {specialist:<10} completed folds: {done_folds}")
else:
    print("No checkpoint found — starting fresh.")
"""))

# ── CELL 8 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 9. Training Helper — 3-Phase Fine-Tune

Shared training logic used by all 5 specialists.
Uses `tf.keras.callbacks.ModelCheckpoint` internally (best weights only) and
saves the final model after all 3 phases — crash during training loses only
the current phase, not the fold.
"""))
nb["cells"].append(code("""\
def train_specialist(name, train_ds, val_ds, loss_fn, model_path,
                     is_multilabel=False, n_classes=1, finetune=True):
    \"\"\"
    Run 3-phase training (frozen → partial unfreeze → full fine-tune)
    and save the model to model_path.
    finetune=False stops after Phase 1 (frozen backbone, head only).
    Used for Specialist 5: frozen val_auc 0.8026 beat top-30 (0.7830)
    and full fine-tune (0.7919).

    Returns val_probs (numpy array).
    \"\"\"
    tf.keras.backend.clear_session(); gc.collect()

    if is_multilabel:
        model = build_multilabel_model(n_classes, trainable_base=False)
        metric_name = 'auc'
        metrics = [tf.keras.metrics.AUC(multi_label=True, name='auc'),
                   tf.keras.metrics.BinaryAccuracy(threshold=CONFIG['threshold'],
                                                   name='bin_acc')]
    else:
        model = build_binary_model(trainable_base=False)
        metric_name = 'auc'
        metrics = [tf.keras.metrics.AUC(name='auc'),
                   tf.keras.metrics.BinaryAccuracy(threshold=CONFIG['threshold'],
                                                   name='bin_acc')]

    backbone = get_backbone(model)

    # Save best weights to a temp path during training; final model saved after all phases
    best_weights_path = model_path.replace('.keras', '_best_weights.keras')

    callbacks_base = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor=f'val_{metric_name}', mode='max',
            save_best_only=True, save_weights_only=False, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=f'val_{metric_name}', mode='max',
            factor=0.3, patience=3, min_lr=1e-7, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor=f'val_{metric_name}', mode='max',
            patience=5, restore_best_weights=True),
    ]

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['lr_phase1']),
        loss=loss_fn, metrics=metrics
    )
    print(f"  [P1] Frozen...")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=CONFIG['epochs_phase1'],
                   callbacks=callbacks_base, verbose=0)
    best_p1 = max(h1.history[f'val_{metric_name}'])
    print(f"       best val_auc={best_p1:.4f}")

    if not finetune:
        print("  [P2A/P2B] skipped (finetune=False)")
    else:
        # ── Phase 2A: unfreeze top 30 backbone layers ─────────────────────────────
        for layer in backbone.layers[-30:]:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG['lr_phase2a'], clipnorm=1.0),
            loss=loss_fn, metrics=metrics
        )
        print(f"  [P2A] Top-30 unfreeze...")
        h2a = model.fit(train_ds, validation_data=val_ds,
                        epochs=CONFIG['epochs_phase2a'],
                        callbacks=callbacks_base, verbose=0)
        best_p2a = max(h2a.history[f'val_{metric_name}'])
        print(f"        best val_auc={best_p2a:.4f}")

        # ── Phase 2B: full fine-tune ───────────────────────────────────────────────
        for layer in backbone.layers:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG['lr_phase2b'], clipnorm=1.0),
            loss=loss_fn,
            metrics=metrics
        )
        print(f"  [P2B] Full fine-tune...")
        h2b = model.fit(train_ds, validation_data=val_ds,
                        epochs=CONFIG['epochs_phase2b'],
                        callbacks=[
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor=f'val_{metric_name}', mode='max',
                                factor=0.2, patience=2, min_lr=1e-8, verbose=0),
                            tf.keras.callbacks.EarlyStopping(
                                monitor=f'val_{metric_name}', mode='max',
                                patience=5, restore_best_weights=True),
                        ],
                        verbose=0)
        best_p2b = max(h2b.history[f'val_{metric_name}'])
        print(f"        best val_auc={best_p2b:.4f}")

    model.save(model_path)
    print(f"  Saved: {model_path}")

    val_probs = model.predict(val_ds, verbose=0)
    del model, backbone
    tf.keras.backend.clear_session(); gc.collect()

    # Clean up temporary best-weights file
    if os.path.exists(best_weights_path) and best_weights_path != model_path:
        os.remove(best_weights_path)

    return val_probs

print("train_specialist() defined ✓")
"""))

# ── CELL 9 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 10. Train Specialists 1–4 (Binary: Healthy vs Disease)

Each specialist trains only on `NoFoliarSymptoms=1` (healthy) rows
and `[Disease]=1` rows. Labels: 0 = Healthy, 1 = Disease.

**Checkpoint-aware**: completed folds are skipped on re-run; OOF arrays are
reloaded from disk so downstream sections still work correctly.
"""))
nb["cells"].append(code("""\
# ── Load checkpoint + any existing OOF arrays ─────────────────────────────────
ckpt = ckpt_load()

oof_binary = {d: oof_load_binary(d, len(df)) for d in BINARY_DISEASES}
binary_fold_results = {d: [] for d in BINARY_DISEASES}

# Re-populate fold results from checkpoint (for summary at end)
for disease in BINARY_DISEASES:
    for fold_str, metrics in ckpt.get(disease, {}).items():
        binary_fold_results[disease].append(
            {'fold': int(fold_str), **metrics}
        )

healthy_idx = df.index[df[HEALTHY_COL] == 1].tolist()

for disease in BINARY_DISEASES:
    print(f"\\n{'='*60}")
    print(f"SPECIALIST: Healthy vs {disease}")
    print(f"{'='*60}")

    disease_idx = df.index[df[disease] == 1].tolist()
    spec_idx    = np.array(healthy_idx + disease_idx)
    spec_df     = df.loc[spec_idx].reset_index(drop=True)

    # Binary label: 0 = Healthy, 1 = Disease
    spec_labels = spec_df[disease].values.astype(np.float32)

    print(f"  Dataset: {len(spec_df)} images  "
          f"({int(spec_labels.sum())} disease, "
          f"{int((1-spec_labels).sum())} healthy)")

    # Stratified 5-fold — same seed → same splits every run
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(spec_df['clahe_path'].values, spec_labels)
    ):
        fold_num = fold + 1
        model_path = os.path.join(MODEL_DIR,
                                  f'specialist_{disease}_fold{fold_num}.keras')

        # ── CHECKPOINT CHECK ─────────────────────────────────────────────────
        if ckpt_done(ckpt, disease, fold_num):
            saved = ckpt[disease][str(fold_num)]
            print(f"  Fold {fold_num}/5  [SKIPPED — already done] "
                  f"F1={saved['f1']:.4f}  AUC={saved['auc']:.4f}")
            # OOF probs already in oof_binary[disease] from oof_load_binary()
            continue

        print(f"\\n  Fold {fold_num}/5  [TRAINING]")
        train_paths = spec_df['clahe_path'].values[train_idx]
        val_paths   = spec_df['clahe_path'].values[val_idx]
        train_lbl   = spec_labels[train_idx]
        val_lbl     = spec_labels[val_idx]

        pw      = compute_pos_weight_binary(train_lbl)
        loss_fn = weighted_bce_binary(pw)
        print(f"    pos_weight={pw:.2f}  "
              f"(train: {int(train_lbl.sum())} pos / "
              f"{len(train_lbl)-int(train_lbl.sum())} neg)")

        train_ds = build_train_ds(train_paths, train_lbl, n_classes=1)
        val_ds   = build_val_ds(val_paths, val_lbl)

        val_probs = train_specialist(
            name=f'{disease}_fold{fold_num}',
            train_ds=train_ds, val_ds=val_ds,
            loss_fn=loss_fn,
            model_path=model_path,
            is_multilabel=False, n_classes=1
        )

        # val_probs shape (N, 1) → squeeze to (N,)
        val_probs_flat = val_probs.squeeze()
        val_preds      = (val_probs_flat >= CONFIG['threshold']).astype(int)

        f1  = f1_score(val_lbl.astype(int), val_preds, zero_division=0)
        auc = roc_auc_score(val_lbl.astype(int), val_probs_flat)
        print(f"    → val F1={f1:.4f}  AUC={auc:.4f}")

        # Store OOF probs (original df indices)
        orig_val_indices = spec_df.index[val_idx]
        oof_binary[disease][orig_val_indices] = val_probs_flat

        # ── PERSIST ──────────────────────────────────────────────────────────
        oof_save_binary(disease, oof_binary[disease])   # save full array
        binary_fold_results[disease].append(
            {'fold': fold_num, 'f1': f1, 'auc': auc}
        )
        ckpt_mark(ckpt, disease, fold_num,
                  {'f1': round(f1, 4), 'auc': round(auc, 4)})

# ── Binary specialist summary ─────────────────────────────────────────────────
print("\\n" + "="*60)
print("BINARY SPECIALIST SUMMARY")
print("="*60)
for disease in BINARY_DISEASES:
    res = sorted(binary_fold_results[disease], key=lambda r: r['fold'])
    if not res:
        print(f"  {disease:<6}  no results yet")
        continue
    mean_f1  = np.mean([r['f1']  for r in res])
    mean_auc = np.mean([r['auc'] for r in res])
    std_f1   = np.std( [r['f1']  for r in res])
    print(f"  {disease:<6}  Mean F1={mean_f1:.4f} (±{std_f1:.4f})  "
          f"Mean AUC={mean_auc:.4f}  ({len(res)}/5 folds)")
"""))

# ── CELL 10 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 11. Train Specialist 5 (Multi-label: SR / NoFoliarSymptoms / Other / UnidentifiedDisease)

Trains on **all 2,355 images**. Labels: 4-dimensional multi-hot vector.

**Checkpoint-aware**: completed folds are skipped; OOF array reloaded from disk.
"""))
nb["cells"].append(code("""\
final_labels = df[FINAL_COLS].values.astype(np.float32)   # (2355, 4)

# Load existing OOF array (or start fresh)
oof_final = oof_load_final(len(df), NUM_FINAL)
final_fold_results = []

# Re-populate fold results from checkpoint
for fold_str, metrics in ckpt.get('final', {}).items():
    final_fold_results.append({'fold': int(fold_str), **metrics})

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"{'='*60}")
print(f"SPECIALIST 5: {FINAL_COLS}")
print(f"{'='*60}")

for fold, (train_idx, val_idx) in enumerate(
    mskf.split(df['clahe_path'].values, final_labels)
):
    fold_num   = fold + 1
    model_path = os.path.join(MODEL_DIR, f'specialist_final_fold{fold_num}.keras')

    # ── CHECKPOINT CHECK ─────────────────────────────────────────────────────
    if ckpt_done(ckpt, 'final', fold_num):
        saved = ckpt['final'][str(fold_num)]
        print(f"  Fold {fold_num}/5  [SKIPPED — already done] "
              f"Macro F1={saved['macro_f1']:.4f}  Micro F1={saved['micro_f1']:.4f}")
        continue

    print(f"\\nFold {fold_num}/5  [TRAINING]")
    train_paths = df['clahe_path'].values[train_idx]
    val_paths   = df['clahe_path'].values[val_idx]
    train_lbl   = final_labels[train_idx]
    val_lbl     = final_labels[val_idx]

    pw      = compute_pos_weights_multilabel(train_lbl)
    loss_fn = weighted_bce_multilabel(pw)
    print("  pos_weights:")
    for cname, w in zip(FINAL_COLS, pw):
        print(f"    {cname:<22} {w:.2f}")

    train_ds = build_train_ds(train_paths, train_lbl, n_classes=NUM_FINAL)
    val_ds   = build_val_ds(val_paths, val_lbl)

    val_probs = train_specialist(
        name=f'final_fold{fold_num}',
        train_ds=train_ds, val_ds=val_ds,
        loss_fn=loss_fn,
        model_path=model_path,
        is_multilabel=True, n_classes=NUM_FINAL,
        finetune=False   # frozen-only: fine-tuning lowered S5 val AUC
    )

    val_preds = (val_probs >= CONFIG['threshold']).astype(int)
    macro_f1  = f1_score(val_lbl.astype(int), val_preds,
                         average='macro', zero_division=0)
    micro_f1  = f1_score(val_lbl.astype(int), val_preds,
                         average='micro', zero_division=0)
    print(f"  → Macro F1={macro_f1:.4f}  Micro F1={micro_f1:.4f}")

    oof_final[val_idx] = val_probs

    # ── PERSIST ──────────────────────────────────────────────────────────────
    oof_save_final(oof_final)
    final_fold_results.append({'fold': fold_num,
                                'macro_f1': macro_f1, 'micro_f1': micro_f1})
    ckpt_mark(ckpt, 'final', fold_num,
              {'macro_f1': round(macro_f1, 4), 'micro_f1': round(micro_f1, 4)})

res_df = pd.DataFrame(final_fold_results)
if len(res_df):
    print(f"\\nSpecialist 5 — Mean Macro F1: {res_df['macro_f1'].mean():.4f} "
          f"(±{res_df['macro_f1'].std():.4f})  ({len(res_df)}/5 folds)")
"""))

# ── CELL 11 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 12. Per-Class OOF Reports (All Folds Combined)

Using OOF predictions stored during training — no data leakage.
"""))
nb["cells"].append(code("""\
# ── Binary specialists: report on rows where OOF probs are available ──────────
print("=== Binary Specialist OOF Reports ===\\n")

for disease in BINARY_DISEASES:
    probs  = oof_binary[disease]
    # Only rows that have OOF predictions (i.e. were in spec_df for this disease)
    mask   = ~np.isnan(probs)
    y_true = df.loc[mask, disease].values.astype(int)
    y_pred = (probs[mask] >= CONFIG['threshold']).astype(int)

    print(f"--- {disease} specialist ---")
    print(classification_report(y_true, y_pred,
                                 target_names=['Healthy', disease],
                                 digits=3, zero_division=0))

# ── Specialist 5: OOF report ──────────────────────────────────────────────────
print("=== Specialist 5 OOF Report ===\\n")
mask5   = ~np.isnan(oof_final[:, 0])
y5_true = final_labels[mask5].astype(int)
y5_pred = (oof_final[mask5] >= CONFIG['threshold']).astype(int)
print(classification_report(y5_true, y5_pred,
                             target_names=FINAL_COLS, digits=3,
                             zero_division=0))
"""))

# ── CELL 12 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 13. Threshold Tuning (Per Specialist)

Grid search over 0.20–0.80 to maximise F1 per class / per specialist.
"""))
nb["cells"].append(code("""\
thresholds = np.arange(0.20, 0.81, 0.05)

# ── Binary specialists ────────────────────────────────────────────────────────
best_thresh_binary = {}
print("=== Binary Specialist Threshold Tuning ===")

for disease in BINARY_DISEASES:
    probs  = oof_binary[disease]
    mask   = ~np.isnan(probs)
    y_true = df.loc[mask, disease].values.astype(int)

    best_f1 = 0.0; best_t = 0.5
    for t in thresholds:
        preds = (probs[mask] >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    best_thresh_binary[disease] = float(round(best_t, 2))
    print(f"  {disease:<6}  best_thresh={best_t:.2f}  F1={best_f1:.3f}")

print()

# ── Specialist 5 ─────────────────────────────────────────────────────────────
best_thresh_final = []
best_f1s_final    = []
print("=== Specialist 5 Threshold Tuning ===")

mask5   = ~np.isnan(oof_final[:, 0])
y5_true = final_labels[mask5].astype(int)

for c, name in enumerate(FINAL_COLS):
    best_f1 = 0.0; best_t = 0.5
    for t in thresholds:
        preds = (oof_final[mask5, c] >= t).astype(int)
        f1    = f1_score(y5_true[:, c], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    best_thresh_final.append(float(round(best_t, 2)))
    best_f1s_final.append(float(round(best_f1, 3)))
    print(f"  {name:<22}  best_thresh={best_t:.2f}  F1={best_f1:.3f}")

print(f"\\nBinary thresholds: {best_thresh_binary}")
print(f"Final  thresholds: {best_thresh_final}")
"""))

# ── CELL 13 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 14. Tuned OOF Report & Comparison

Apply tuned thresholds, re-compute full 8-class report, compare vs baseline.
"""))
nb["cells"].append(code("""\
# ── Assemble tuned 8-class predictions ───────────────────────────────────────
# We build an (N, 8) prediction matrix using:
#   cols 0-3 : binary specialists (GLS, NCLB, PLS, CR)
#   cols 4-7 : Specialist 5       (SR, NoFoliarSymptoms, Other, UnidentifiedDisease)
# Note: a row may be NaN for binary specialists if that image wasn't in
#       that specialist's dataset — for those rows we set prediction to 0.

pred_8_base  = np.zeros((len(df), 8), dtype=int)
pred_8_tuned = np.zeros((len(df), 8), dtype=int)
true_8       = df[ALL_COLS].values.astype(int)

for i, disease in enumerate(BINARY_DISEASES):
    probs = oof_binary[disease]
    mask  = ~np.isnan(probs)
    pred_8_base[mask, i]  = (probs[mask] >= 0.5).astype(int)
    pred_8_tuned[mask, i] = (probs[mask] >= best_thresh_binary[disease]).astype(int)

# Specialist 5 covers indices 4-7 (SR, NoFoliarSymptoms, Other, UnidentifiedDisease)
mask5 = ~np.isnan(oof_final[:, 0])
thresh_arr = np.array(best_thresh_final, dtype=np.float32)
pred_8_base[mask5,  4:8] = (oof_final[mask5] >= 0.5).astype(int)
pred_8_tuned[mask5, 4:8] = (oof_final[mask5] >= thresh_arr[np.newaxis, :]).astype(int)

# ── Baseline report ───────────────────────────────────────────────────────────
print("=== 8-Class Report — Baseline (threshold=0.5) ===\\n")
print(classification_report(true_8, pred_8_base,
                             target_names=ALL_COLS, digits=3, zero_division=0))

# ── Tuned report ──────────────────────────────────────────────────────────────
print("=== 8-Class Report — Tuned Per-Class Thresholds ===\\n")
print(classification_report(true_8, pred_8_tuned,
                             target_names=ALL_COLS, digits=3, zero_division=0))

# ── Improvement table ─────────────────────────────────────────────────────────
macro_base  = f1_score(true_8, pred_8_base,  average='macro', zero_division=0)
macro_tuned = f1_score(true_8, pred_8_tuned, average='macro', zero_division=0)
micro_base  = f1_score(true_8, pred_8_base,  average='micro', zero_division=0)
micro_tuned = f1_score(true_8, pred_8_tuned, average='micro', zero_division=0)
hl_base     = hamming_loss(true_8, pred_8_base)
hl_tuned    = hamming_loss(true_8, pred_8_tuned)

print("=== Improvement Summary ===")
print(f"{'Metric':<18} {'Baseline':>12} {'Tuned':>10} {'Delta':>10}")
print("-"*54)
print(f"{'Macro F1':<18} {macro_base:>12.4f} {macro_tuned:>10.4f} {macro_tuned-macro_base:>+10.4f}")
print(f"{'Micro F1':<18} {micro_base:>12.4f} {micro_tuned:>10.4f} {micro_tuned-micro_base:>+10.4f}")
print(f"{'Hamming Loss':<18} {hl_base:>12.4f} {hl_tuned:>10.4f} {hl_tuned-hl_base:>+10.4f}")
"""))

# ── CELL 14 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 15. Visualisations"))
nb["cells"].append(code("""\
# ── Per-class F1 comparison: monolithic vs specialist ─────────────────────────
pc_f1_tuned = f1_score(true_8, pred_8_tuned, average=None, zero_division=0)
pc_f1_base  = f1_score(true_8, pred_8_base,  average=None, zero_division=0)

x     = np.arange(8)
width = 0.38
fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x - width/2, pc_f1_base,  width, label='Threshold=0.5', color='#95a5a6', alpha=0.85)
b2 = ax.bar(x + width/2, pc_f1_tuned, width, label='Tuned',         color='#2980b9', alpha=0.85)

# Shade region for Specialist 5 classes
ax.axvspan(3.5, 7.5, alpha=0.06, color='green', label='Specialist 5 region')

ax.axhline(macro_base,  color='grey',     linestyle='--', linewidth=1.0,
           label=f'Macro (base) = {macro_base:.3f}')
ax.axhline(macro_tuned, color='#2980b9', linestyle='--', linewidth=1.2,
           label=f'Macro (tuned) = {macro_tuned:.3f}')
ax.set_xticks(x); ax.set_xticklabels(ALL_COLS, rotation=15, ha='right')
ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
ax.set_title('Specialist Pipeline — Per-Class F1 (OOF, All Folds)', fontsize=13)
ax.legend(loc='upper right', fontsize=8)
for bar, val in zip(b2, pc_f1_tuned):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'specialist_f1.png'), dpi=150)
plt.show()
"""))

nb["cells"].append(code("""\
# ── Binary specialist AUC summary ─────────────────────────────────────────────
auc_means = []
for disease in BINARY_DISEASES:
    mean_auc = np.mean([r['auc'] for r in binary_fold_results[disease]])
    auc_means.append(mean_auc)

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#e74c3c' if a < 0.8 else '#2ecc71' for a in auc_means]
bars = ax.bar(BINARY_DISEASES, auc_means, color=colors, alpha=0.85)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=1.0, label='Random')
ax.axhline(0.9, color='black', linestyle=':', linewidth=1.0, label='Good (0.9)')
ax.set_ylim(0.4, 1.05); ax.set_ylabel('Mean AUC (5-fold)')
ax.set_title('Binary Specialist AUC per Disease', fontsize=12)
ax.legend()
for bar, val in zip(bars, auc_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'binary_auc.png'), dpi=150)
plt.show()
"""))

# ── CELL 15 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 16. Inference on New Images

All 5 specialists run in parallel. Outputs are merged into a single 8-class prediction.
"""))
nb["cells"].append(code("""\
# ── Load all saved models ─────────────────────────────────────────────────────
print("Loading specialist models...")

# Binary specialists: 5 folds each = 20 models
binary_fold_models = {d: [] for d in BINARY_DISEASES}
for disease in BINARY_DISEASES:
    for fold in range(1, 6):
        path  = os.path.join(MODEL_DIR, f'specialist_{disease}_fold{fold}.keras')
        model = tf.keras.models.load_model(path, compile=False)
        binary_fold_models[disease].append(model)
    print(f"  {disease}: 5 folds loaded ✓")

# Final specialist: 5 folds
final_fold_models = []
for fold in range(1, 6):
    path  = os.path.join(MODEL_DIR, f'specialist_final_fold{fold}.keras')
    model = tf.keras.models.load_model(path, compile=False)
    final_fold_models.append(model)
print(f"  Specialist 5: 5 folds loaded ✓")

# ── Thresholds ────────────────────────────────────────────────────────────────
# best_thresh_binary and best_thresh_final must be defined (run Section 12 first).
# Fallback to 0.5 if threshold tuning was skipped.
try:
    BTH = best_thresh_binary
    FTH = np.array(best_thresh_final, dtype=np.float32)
    print(f"\\nUsing tuned thresholds:")
    print(f"  Binary: {BTH}")
    print(f"  Final:  {best_thresh_final}")
except NameError:
    BTH = {d: 0.5 for d in BINARY_DISEASES}
    FTH = np.full(NUM_FINAL, 0.5, dtype=np.float32)
    print("Tuned thresholds not found — falling back to 0.5 for all classes.")

def preprocess_single(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab     = cv2.merge([l, a, b])
    img     = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img     = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size'])).astype(np.float32) / 255.0
    mean    = np.array([0.485, 0.456, 0.406], np.float32)
    std     = np.array([0.229, 0.224, 0.225], np.float32)
    return ((img - mean) / std)[np.newaxis, ...]   # (1, H, W, 3)

def predict_specialist(img_path):
    \"\"\"
    Run all 5 specialists on a single image.

    Returns
    -------
    dict:
      'detected'      — list of active disease/class names
      'probabilities' — dict of ALL_COLS → averaged sigmoid probability
      'thresholds'    — dict of ALL_COLS → threshold used
    \"\"\"
    x = preprocess_single(img_path)

    # Binary specialists (GLS, NCLB, PLS, CR)
    probs_8 = np.zeros(8, dtype=np.float32)
    for i, disease in enumerate(BINARY_DISEASES):
        avg_prob    = np.mean([m.predict(x, verbose=0)[0, 0]
                               for m in binary_fold_models[disease]])
        probs_8[i]  = avg_prob

    # Specialist 5 (SR, NoFoliarSymptoms, Other, UnidentifiedDisease)
    avg_final   = np.mean([m.predict(x, verbose=0)[0]
                            for m in final_fold_models], axis=0)
    probs_8[4:] = avg_final

    # Apply thresholds
    thresh_8 = np.array(
        [BTH[d] for d in BINARY_DISEASES] + list(FTH),
        dtype=np.float32
    )
    preds    = (probs_8 >= thresh_8).astype(int)
    detected = [ALL_COLS[i] for i in range(8) if preds[i] == 1]

    return {
        'image'        : os.path.basename(img_path),
        'detected'     : detected if detected else ['No disease detected'],
        'probabilities': {ALL_COLS[i]: float(round(probs_8[i], 4)) for i in range(8)},
        'thresholds'   : {ALL_COLS[i]: float(thresh_8[i]) for i in range(8)},
    }

# ── Example ───────────────────────────────────────────────────────────────────
# result = predict_specialist('/path/to/leaf.jpg')
# print("Detected:", result['detected'])
# for cls, prob in result['probabilities'].items():
#     t    = result['thresholds'][cls]
#     flag = '✓' if prob >= t else ' '
#     print(f"  {flag} {cls:<22} prob={prob:.4f}  thresh={t:.2f}")

print("\\npredict_specialist() ready ✓")
print(f"Loaded: {len(BINARY_DISEASES)*5} binary models + {len(final_fold_models)} final models")
"""))

nb["cells"].append(code("""\
# ── Batch inference on a folder ───────────────────────────────────────────────
INFER_DIR = '/path/to/new/images'   # <-- change this

image_files = (list(Path(INFER_DIR).glob('*.jpg'))
             + list(Path(INFER_DIR).glob('*.JPG')))
print(f"Found {len(image_files)} images")

results = []
for img_path in tqdm(image_files):
    try:
        r = predict_specialist(str(img_path))
        results.append(r)
    except Exception as e:
        results.append({'image': img_path.name, 'detected': ['ERROR'],
                        'probabilities': {}, 'error': str(e)})

rows = []
for r in results:
    row = {'image': r['image']}
    row.update(r.get('probabilities', {}))
    row['detected'] = ', '.join(r['detected'])
    rows.append(row)

out_df  = pd.DataFrame(rows)
out_csv = os.path.join(MODEL_DIR, 'specialist_inference_results.csv')
out_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(out_df[['image', 'detected']].head(10))
"""))

# ── CELL 16 — NOTES ────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 17. Architecture Notes

### Why this design?

| | Monolithic 8-class | Specialist Pipeline |
|---|---|---|
| **Task** | One sigmoid head, 8 outputs | 5 independent heads |
| **Decision boundary** | All 8 classes compete for weight | Each specialist focuses on one boundary |
| **Healthy negatives** | Diluted across all classes | Binary specialists trained on clean Healthy vs Disease |
| **SR problem** | 39 samples vs 2316 non-SR | SR in Specialist 5 with better calibrated loss |
| **Threshold tuning** | One per class post-hoc | One per specialist post-hoc |
| **Inference cost** | 1 forward pass × 5 folds | 5 forward passes × 5 folds (25 total) |

### Specialist definitions
- **Specialists 1–4**: Binary. Train on `NoFoliarSymptoms=1` (healthy) + `[Disease]=1` rows only.
  Label: 0 = Healthy, 1 = Disease. Uses weighted BCE with `pos_weight = N_neg/N_pos`.
- **Specialist 5**: Multi-label sigmoid over {SR, NoFoliarSymptoms, Other, UnidentifiedDisease}.
  Trains on all 2,355 images. Handles rare and ambiguous categories together.

### Expected per-specialist AUC targets
- GLS, NCLB: AUC > 0.90 (high support, visually distinctive)
- PLS, CR:    AUC > 0.85
- SR:         AUC > 0.70 (39 samples — partially learnable)
- NoFoliarSymptoms / Other / UnidentifiedDisease: AUC > 0.75

### Training subset sizes (estimated)
| Specialist | Healthy | Disease | Total |
|---|---|---|---|
| Healthy vs GLS  | 285 | 1084 | 1369 |
| Healthy vs NCLB | 285 |  554 |  839 |
| Healthy vs PLS  | 285 |  493 |  778 |
| Healthy vs CR   | 285 |  300 |  585 |
| Specialist 5    | all 2355 (multi-label) | — | 2355 |
"""))

with open('maize_specialist.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook written ✓")
