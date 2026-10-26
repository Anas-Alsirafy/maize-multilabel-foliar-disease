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
# Maize Foliar Disease — Multi-Label Classification
**Pipeline B (adapted) · ConvNeXtBase · Sigmoid + Weighted BCE · 5-Fold CV**

| | |
|---|---|
| **Dataset** | 2,355 field images, 8 disease classes |
| **Task** | Multi-label (one image can have multiple diseases simultaneously) |
| **Classes** | GLS, NCLB, PLS, CR, SR, NoFoliarSymptoms, Other, UnidentifiedDisease |
| **Backbone** | ConvNeXtBase (ImageNet weights) |
| **Loss** | Per-class weighted Binary Cross-Entropy |
| **Protocol** | 5-fold stratified CV (stratified on dominant label) |

> **Key difference from Cassava:** Output layer uses **sigmoid** (not softmax).
> Each class is an independent binary prediction — multiple classes can be active simultaneously.
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
CSV_PATH  = '/path/to/Database.csv'        # the CSV file provided
IMAGE_DIR = '/path/to/images'              # folder containing the .JPG files
MODEL_DIR = '/kaggle/working'              # where .keras files are saved

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
#  MODEL CONFIG
# ─────────────────────────────────────────────
DISEASE_COLS = ['GLS', 'NCLB', 'PLS', 'CR', 'SR',
                'NoFoliarSymptoms', 'Other', 'UnidentifiedDisease']
NUM_CLASSES  = len(DISEASE_COLS)   # 8

CONFIG = {
    'img_size'      : 384,
    'batch_size'    : 16,
    'num_classes'   : NUM_CLASSES,
    'lr_phase1'     : 1e-4,
    'lr_phase2a'    : 1e-5,
    'lr_phase2b'    : 3e-6,
    'epochs_phase1' : 15,
    'epochs_phase2a': 10,
    'epochs_phase2b': 10,
    'threshold'     : 0.5,    # sigmoid threshold for positive prediction
}

print(f"Classes ({NUM_CLASSES}): {DISEASE_COLS}")
print(f"Image size: {CONFIG['img_size']}x{CONFIG['img_size']}")
"""))

# ── CELL 3 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 3. Data Loading & EDA"))
nb["cells"].append(code("""\
df = pd.read_csv(CSV_PATH)
df['img_path'] = df['filePath'].apply(lambda x: os.path.join(IMAGE_DIR, x))

print(f"Total images: {len(df)}")
print()

# Label distribution
print("=== Label counts (images per disease) ===")
for col in DISEASE_COLS:
    pct = df[col].mean() * 100
    print(f"  {col:<22} {df[col].sum():>5}  ({pct:.1f}%)")

print()
df['n_labels'] = df[DISEASE_COLS].sum(axis=1)
print("=== Multi-label distribution ===")
print(df['n_labels'].value_counts().sort_index().to_string())
print()
print(f"Single-label:  {(df['n_labels']==1).sum()} ({(df['n_labels']==1).mean()*100:.1f}%)")
print(f"Multi-label:   {(df['n_labels']>=2).sum()} ({(df['n_labels']>=2).mean()*100:.1f}%)")
"""))

nb["cells"].append(code("""\
# ── Co-occurrence heatmap ──────────────────────────────────────────────────────
co = df[DISEASE_COLS].T.dot(df[DISEASE_COLS])
co_arr = co.to_numpy(dtype=float).copy()   # writable array — fill_diagonal requires it
np.fill_diagonal(co_arr, 0)               # zero out self-co-occurrence diagonal
co_plot = pd.DataFrame(co_arr, index=DISEASE_COLS, columns=DISEASE_COLS)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(co_plot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            xticklabels=DISEASE_COLS, yticklabels=DISEASE_COLS)
ax.set_title('Disease Co-occurrence Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'cooccurrence.png'), dpi=150)
plt.show()
print("Note: SR=39 samples — co-occurrences will be low by definition")
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
    \"\"\"ConvNeXt preprocessing: resize → /255 → ImageNet normalise\"\"\"
    img  = tf.io.read_file(image_path)
    img  = tf.image.decode_jpeg(img, channels=3)
    img  = tf.image.resize(img, [CONFIG['img_size'], CONFIG['img_size']])
    img  = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std  = tf.constant([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return img, label

def augment_fn(path, label):
    \"\"\"Albumentations augmentation wrapped for tf.py_function\"\"\"
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

def build_train_ds(image_paths, labels):
    \"\"\"Training dataset with augmentation. Labels are float32 multi-hot vectors.\"\"\"
    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels.astype(np.float32))
    )
    def _aug(path, lbl):
        img, l = tf.py_function(func=augment_fn, inp=[path, lbl],
                                Tout=[tf.float32, tf.float32])
        img.set_shape([CONFIG['img_size'], CONFIG['img_size'], 3])
        l.set_shape([NUM_CLASSES])
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
nb["cells"].append(md("## 6. Class Weights & Loss"))
nb["cells"].append(code("""\
def compute_pos_weights(labels_array):
    \"\"\"
    pos_weight[c] = (N - pos[c]) / pos[c]
    Penalises false negatives proportionally to class rarity.
    SR (39 samples) will get a very high weight — this is intentional.
    \"\"\"
    pos = labels_array.sum(axis=0).astype(float)
    N   = float(len(labels_array))
    pw  = (N - pos) / np.maximum(pos, 1.0)
    # Cap at 50 to prevent SR weight from destabilising training
    pw  = np.clip(pw, 1.0, 50.0)
    print("Positive weights per class:")
    for name, w in zip(DISEASE_COLS, pw):
        print(f"  {name:<22} {w:.2f}")
    return pw.astype(np.float32)

def weighted_bce_loss(pos_weights):
    \"\"\"
    Weighted Binary Cross-Entropy for multi-label.
    pos_weight[c] scales the loss for positive examples of class c.
    \"\"\"
    pw = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Clip for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = -(pw * y_true * tf.math.log(y_pred)
                   + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)

    return loss

print("Loss function defined ✓")
"""))

# ── CELL 7 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 7. Model Builder"))
nb["cells"].append(code("""\
def build_model(trainable_base=False):
    \"\"\"
    ConvNeXtBase → GAP → LayerNorm → Dropout → Dense(256) → Dropout → Dense(8, sigmoid)
    Sigmoid output: each class is an independent binary prediction.
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
    # sigmoid — NOT softmax — for multi-label
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid',
                                    dtype='float32')(x)

    return tf.keras.Model(inputs, outputs)

def get_backbone(model):
    \"\"\"
    Find the ConvNeXtBase sub-model by scanning model.layers for the first
    layer that has its own .layers attribute (i.e. is itself a sub-model).
    Avoids brittle index-based access: ConvNeXtBase inserts an internal
    Normalization layer so model.layers[1] is NOT the backbone.
    \"\"\"
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            return layer
    raise ValueError("No backbone sub-model found — check model architecture.")

# Quick sanity check
m  = build_model()
bb = get_backbone(m)
print(f"Output shape:    {m.output_shape}")        # (None, 8)
print(f"Total params:    {m.count_params():,}")
print(f"Backbone name:   {bb.name}")
print(f"Backbone layers: {len(bb.layers)}")
del m; gc.collect()
"""))

# ── CELL 8 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 8. Training — 5-Fold Multi-Label Stratified CV

Uses `MultilabelStratifiedKFold` from `iterative-stratification` to ensure
each fold has representative proportions of **all 8 classes**, including SR.
"""))
nb["cells"].append(code("""\
# ── Prepare labels matrix ─────────────────────────────────────────────────────
labels_matrix = df[DISEASE_COLS].values.astype(np.float32)   # (2355, 8)

# Dominant label for fallback stratification (used in fold summary only)
df['dominant_label'] = df[DISEASE_COLS].idxmax(axis=1)

# ── Stratified folds ──────────────────────────────────────────────────────────
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_results = []
all_true_global = []
all_pred_global = []

for fold, (train_idx, val_idx) in enumerate(
    mskf.split(df['clahe_path'].values, labels_matrix)
):
    print(f"\\n{'='*50}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*50}")

    train_df    = df.iloc[train_idx]
    val_df      = df.iloc[val_idx]
    train_lbls  = labels_matrix[train_idx]
    val_lbls    = labels_matrix[val_idx]

    # SR in val fold
    sr_val = int(val_lbls[:, DISEASE_COLS.index('SR')].sum())
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | SR in val: {sr_val}")

    # Compute pos weights on TRAIN split only (prevent leakage)
    pos_weights = compute_pos_weights(train_lbls)

    train_ds = build_train_ds(train_df['clahe_path'].values, train_lbls)
    val_ds   = build_val_ds(val_df['clahe_path'].values,   val_lbls)

    tf.keras.backend.clear_session(); gc.collect()
    model    = build_model(trainable_base=False)
    backbone = get_backbone(model)   # robust — avoids brittle index (model.layers[1])

    # ── PHASE 1: frozen backbone ───────────────────────────────────────────────
    print("\\n[Phase 1] Frozen backbone")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['lr_phase1']),
        loss=weighted_bce_loss(pos_weights),
        metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc'),
                 tf.keras.metrics.BinaryAccuracy(threshold=CONFIG['threshold'],
                                                 name='bin_acc')]
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=CONFIG['epochs_phase1'],
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max', factor=0.3,
                patience=3, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max', patience=5,
                restore_best_weights=True)
        ],
        verbose=1
    )

    # ── PHASE 2A: unfreeze top 30 backbone layers ─────────────────────────────
    print("\\n[Phase 2A] Partial unfreeze (top 30 backbone layers)")
    for layer in backbone.layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['lr_phase2a'], clipnorm=1.0),
        loss=weighted_bce_loss(pos_weights),
        metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc'),
                 tf.keras.metrics.BinaryAccuracy(threshold=CONFIG['threshold'],
                                                 name='bin_acc')]
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=CONFIG['epochs_phase2a'],
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max', factor=0.3,
                patience=3, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max', patience=5,
                restore_best_weights=True)
        ],
        verbose=1
    )

    # ── PHASE 2B: full fine-tune ───────────────────────────────────────────────
    print("\\n[Phase 2B] Full fine-tune")
    for layer in backbone.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['lr_phase2b'], clipnorm=1.0),
        loss=weighted_bce_loss(pos_weights),
        metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc'),
                 tf.keras.metrics.BinaryAccuracy(threshold=CONFIG['threshold'],
                                                 name='bin_acc')]
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=CONFIG['epochs_phase2b'],
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', mode='max', factor=0.2,
                patience=2, min_lr=1e-8, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max', patience=5,
                restore_best_weights=True)
        ],
        verbose=1
    )

    # ── Save & evaluate ───────────────────────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, f'fold_{fold+1}_maize_convnext.keras')
    model.save(save_path)
    print(f"Saved: {save_path}")

    # Predict on val set
    val_probs  = model.predict(val_ds, verbose=0)          # (N, 8) float
    val_preds  = (val_probs >= CONFIG['threshold']).astype(int)

    # Metrics
    macro_f1 = f1_score(val_lbls.astype(int), val_preds, average='macro',
                        zero_division=0)
    micro_f1 = f1_score(val_lbls.astype(int), val_preds, average='micro',
                        zero_division=0)
    h_loss   = hamming_loss(val_lbls.astype(int), val_preds)

    fold_results.append({
        'fold'    : fold + 1,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'hamming' : h_loss,
        'sr_val'  : sr_val,
    })
    all_true_global.extend(val_lbls.astype(int).tolist())
    all_pred_global.extend(val_preds.tolist())

    print(f"\\nFold {fold+1} Results:")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  Micro F1:     {micro_f1:.4f}")
    print(f"  Hamming Loss: {h_loss:.4f}  (lower = better)")

    del model, backbone
    tf.keras.backend.clear_session(); gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(fold_results)
print("\\n" + "="*50)
print("FINAL RESULTS — 5-Fold CV")
print("="*50)
print(results_df.to_string(index=False))
print()
print(f"Mean Macro F1:  {results_df['macro_f1'].mean():.4f} (+/- {results_df['macro_f1'].std():.4f})")
print(f"Mean Micro F1:  {results_df['micro_f1'].mean():.4f} (+/- {results_df['micro_f1'].std():.4f})")
print(f"Mean Hamming:   {results_df['hamming'].mean():.4f} (+/- {results_df['hamming'].std():.4f})")
"""))

# ── CELL 9 ─────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 9. Per-Class Metrics (All Folds Combined)"))
nb["cells"].append(code("""\
all_true_np = np.array(all_true_global)   # (2355, 8)
all_pred_np = np.array(all_pred_global)   # (2355, 8)

print("=== Per-Class Classification Report (All Folds Combined) ===\\n")
print(classification_report(all_true_np, all_pred_np,
                             target_names=DISEASE_COLS, digits=3,
                             zero_division=0))

# SR specific callout
sr_idx   = DISEASE_COLS.index('SR')
sr_true  = all_true_np[:, sr_idx]
sr_pred  = all_pred_np[:, sr_idx]
sr_tp    = int((sr_true * sr_pred).sum())
sr_total = int(sr_true.sum())
print(f"SR note: {sr_tp}/{sr_total} SR positives correctly predicted")
print(f"  → If SR F1 ≈ 0: visually not distinct enough with only 39 samples")
print(f"  → If SR F1 > 0.5: SR is learnable despite low support — report it")
"""))

# ── CELL 10 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("## 10. Visualisations"))
nb["cells"].append(code("""\
# ── Per-class F1 bar chart ─────────────────────────────────────────────────────
per_class_f1 = f1_score(all_true_np, all_pred_np, average=None, zero_division=0)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if c == 'SR' else '#2980b9' for c in DISEASE_COLS]
bars = ax.bar(DISEASE_COLS, per_class_f1, color=colors, alpha=0.85)
ax.axhline(per_class_f1.mean(), color='black', linestyle='--', linewidth=1.2,
           label=f'Macro mean = {per_class_f1.mean():.3f}')
ax.set_xlabel('Disease Class'); ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 — All Folds Combined', fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend()
for bar, val in zip(bars, per_class_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'per_class_f1.png'), dpi=150)
plt.show()
"""))

nb["cells"].append(code("""\
# ── Per-fold macro F1 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(results_df['fold'], results_df['macro_f1'],
       color='steelblue', alpha=0.8)
ax.axhline(results_df['macro_f1'].mean(), color='red', linestyle='--',
           label=f"Mean = {results_df['macro_f1'].mean():.4f}")
ax.set_xlabel('Fold'); ax.set_ylabel('Macro F1')
ax.set_title('Per-Fold Macro F1 — Maize Multi-Label')
ax.set_ylim(0, 1.0); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'fold_macro_f1.png'), dpi=150)
plt.show()
"""))

nb["cells"].append(code("""\
# ── Multilabel confusion matrices ─────────────────────────────────────────────
mcm = multilabel_confusion_matrix(all_true_np, all_pred_np)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (ax, name) in enumerate(zip(axes.flat, DISEASE_COLS)):
    sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    ax.set_title(f'{name}  (F1={per_class_f1[i]:.2f})', fontsize=10)
plt.suptitle('Per-Class Confusion Matrices — All Folds', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'multilabel_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── CELL 11 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 11. Threshold Tuning (Optional)

The default threshold of 0.5 may not be optimal for every class.
This cell searches for the best per-class threshold using the validation predictions.
"""))
nb["cells"].append(code("""\
# ── Reload all fold val probs for threshold search ────────────────────────────
# NOTE: only run this AFTER training is complete and all .keras files are saved

print("Loading saved models for threshold tuning...")

mskf      = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
all_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(
    mskf.split(df['clahe_path'].values, labels_matrix)
):
    model_path = os.path.join(MODEL_DIR, f'fold_{fold+1}_maize_convnext.keras')
    model = tf.keras.models.load_model(model_path, compile=False)

    val_df  = df.iloc[val_idx]
    val_lbl = labels_matrix[val_idx]
    val_ds  = build_val_ds(val_df['clahe_path'].values, val_lbl)

    probs = model.predict(val_ds, verbose=0)
    all_probs[val_idx] = probs

    del model; tf.keras.backend.clear_session(); gc.collect()
    print(f"  Fold {fold+1} done")

# Search best threshold per class
thresholds = np.arange(0.20, 0.81, 0.05)
best_thresh = []
best_f1s    = []

for c, name in enumerate(DISEASE_COLS):
    best_f1 = 0.0; best_t = 0.5
    for t in thresholds:
        preds = (all_probs[:, c] >= t).astype(int)
        f1    = f1_score(labels_matrix[:, c].astype(int), preds,
                         zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    best_thresh.append(float(round(best_t, 2)))
    best_f1s.append(float(round(best_f1, 3)))
    print(f"  {name:<22} best_thresh={best_t:.2f}  F1={best_f1:.3f}")

print(f"\\nBest thresholds: {best_thresh}")
print(f"Tuned macro F1 estimate: {np.mean(best_f1s):.4f}")
print("\\nbest_thresh is now available for use in Section 12 (Inference).")
"""))

# ── NEW CELL: tuned classification report ─────────────────────────────────────
nb["cells"].append(md("""\
### 11b. Final Classification Report — Tuned Thresholds

Apply per-class thresholds found above, re-run the full report, and compare
against the baseline 0.5 report from Section 9.
"""))
nb["cells"].append(code("""\
# ── Apply per-class thresholds ────────────────────────────────────────────────
# best_thresh must be defined (run Section 11 first)
thresh_arr = np.array(best_thresh, dtype=np.float32)   # shape (8,)

# all_probs is the OOF probability matrix from Section 11 (shape: 2355 x 8)
all_pred_tuned = (all_probs >= thresh_arr[np.newaxis, :]).astype(int)

# ── Full classification report ────────────────────────────────────────────────
print("=== Per-Class Report — Tuned Per-Class Thresholds ===\\n")
print(classification_report(labels_matrix.astype(int), all_pred_tuned,
                             target_names=DISEASE_COLS, digits=3,
                             zero_division=0))

# ── Summary metrics ───────────────────────────────────────────────────────────
macro_tuned = f1_score(labels_matrix.astype(int), all_pred_tuned,
                       average='macro', zero_division=0)
micro_tuned = f1_score(labels_matrix.astype(int), all_pred_tuned,
                       average='micro', zero_division=0)
hloss_tuned = hamming_loss(labels_matrix.astype(int), all_pred_tuned)

# Baseline (threshold=0.5, from Section 9)
all_pred_base = (all_probs >= 0.5).astype(int)
macro_base = f1_score(labels_matrix.astype(int), all_pred_base,
                      average='macro', zero_division=0)
micro_base = f1_score(labels_matrix.astype(int), all_pred_base,
                      average='micro', zero_division=0)
hloss_base = hamming_loss(labels_matrix.astype(int), all_pred_base)

print("\\n=== Improvement from Threshold Tuning ===")
print(f"{'Metric':<18} {'Baseline (0.5)':>16} {'Tuned':>10} {'Delta':>10}")
print("-" * 58)
print(f"{'Macro F1':<18} {macro_base:>16.4f} {macro_tuned:>10.4f} {macro_tuned-macro_base:>+10.4f}")
print(f"{'Micro F1':<18} {micro_base:>16.4f} {micro_tuned:>10.4f} {micro_tuned-micro_base:>+10.4f}")
print(f"{'Hamming Loss':<18} {hloss_base:>16.4f} {hloss_tuned:>10.4f} {hloss_tuned-hloss_base:>+10.4f}")

# ── Per-class tuned F1 bar chart ──────────────────────────────────────────────
per_class_f1_tuned = f1_score(labels_matrix.astype(int), all_pred_tuned,
                               average=None, zero_division=0)
per_class_f1_base  = f1_score(labels_matrix.astype(int), all_pred_base,
                               average=None, zero_division=0)

x     = np.arange(NUM_CLASSES)
width = 0.38
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width/2, per_class_f1_base,  width, label='Threshold=0.5',     color='#95a5a6', alpha=0.85)
b2 = ax.bar(x + width/2, per_class_f1_tuned, width, label='Tuned thresholds',  color='#2980b9', alpha=0.85)
ax.axhline(macro_base,  color='grey',     linestyle='--', linewidth=1.0,
           label=f'Macro (base) = {macro_base:.3f}')
ax.axhline(macro_tuned, color='#2980b9', linestyle='--', linewidth=1.2,
           label=f'Macro (tuned) = {macro_tuned:.3f}')
ax.set_xticks(x); ax.set_xticklabels(DISEASE_COLS, rotation=15, ha='right')
ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
ax.set_title('Per-Class F1: Baseline vs. Tuned Thresholds', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
# Annotate tuned bars
for bar, val, t in zip(b2, per_class_f1_tuned, best_thresh):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}\\n(t={t:.2f})', ha='center', va='bottom', fontsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'per_class_f1_tuned.png'), dpi=150)
plt.show()

print(f"\\nFinal Macro F1 (tuned): {macro_tuned:.4f}")
print(f"SR F1 (tuned): {per_class_f1_tuned[DISEASE_COLS.index('SR')]:.3f}  "
      f"(threshold={best_thresh[DISEASE_COLS.index('SR')]:.2f})")
"""))

# ── CELL 12 ────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 12. Inference on New Images

Uses the **per-class tuned thresholds** found in Section 11.
If you haven't run Section 11, the fallback is the global `CONFIG['threshold']` (0.5).
"""))
nb["cells"].append(code("""\
# ── Load all 5 fold models ────────────────────────────────────────────────────
print("Loading fold models for inference...")
fold_models = []
for fold in range(1, 6):
    path  = os.path.join(MODEL_DIR, f'fold_{fold}_maize_convnext.keras')
    model = tf.keras.models.load_model(path, compile=False)
    fold_models.append(model)
    print(f"  Fold {fold} loaded ✓")

# ── Per-class thresholds ──────────────────────────────────────────────────────
# best_thresh is set by Section 11 (threshold tuning).
# If you skipped Section 11, a global fallback of 0.5 is used for all classes.
try:
    THRESH_PER_CLASS = np.array(best_thresh, dtype=np.float32)   # (8,) from Sec 11
    print(f"Using tuned per-class thresholds: {best_thresh}")
except NameError:
    THRESH_PER_CLASS = np.full(NUM_CLASSES, CONFIG['threshold'], dtype=np.float32)
    print(f"best_thresh not found — falling back to global threshold={CONFIG['threshold']}")
    print("Run Section 11 first for better per-class thresholds.")

def preprocess_single(img_path):
    \"\"\"CLAHE + ConvNeXt normalisation for a single image file.\"\"\"
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
    img     = (img - mean) / std
    return img[np.newaxis, ...]   # (1, H, W, 3)

def predict_single(img_path, thresh_per_class=None):
    \"\"\"
    Predict diseases in a single leaf image.

    Parameters
    ----------
    img_path : str  — path to the leaf image (.jpg / .JPG)
    thresh_per_class : array-like of shape (8,), optional
        Per-class decision thresholds.  Defaults to THRESH_PER_CLASS
        (tuned thresholds from Section 11, or 0.5 if not available).

    Returns
    -------
    dict with keys:
        'image'         — filename
        'detected'      — list of positive disease names
        'probabilities' — dict {class_name: float probability}
        'thresholds'    — dict {class_name: float threshold used}
    \"\"\"
    if thresh_per_class is None:
        thresh_per_class = THRESH_PER_CLASS
    thresh_per_class = np.array(thresh_per_class, dtype=np.float32)

    x         = preprocess_single(img_path)
    probs_avg = np.mean([m.predict(x, verbose=0) for m in fold_models], axis=0)[0]
    preds     = (probs_avg >= thresh_per_class).astype(int)
    diseases  = [DISEASE_COLS[i] for i in range(NUM_CLASSES) if preds[i] == 1]

    return {
        'image'        : os.path.basename(img_path),
        'detected'     : diseases if diseases else ['No disease detected'],
        'probabilities': {DISEASE_COLS[i]: float(round(probs_avg[i], 4))
                          for i in range(NUM_CLASSES)},
        'thresholds'   : {DISEASE_COLS[i]: float(thresh_per_class[i])
                          for i in range(NUM_CLASSES)},
    }

# ── Example ───────────────────────────────────────────────────────────────────
# result = predict_single('/path/to/leaf.jpg')
# print("Detected:", result['detected'])
# for cls, prob in result['probabilities'].items():
#     t = result['thresholds'][cls]
#     flag = '✓' if prob >= t else ' '
#     print(f"  {flag} {cls:<22} prob={prob:.4f}  thresh={t:.2f}")

print("\\npredict_single() ready ✓")
print(f"Active thresholds: { {d: float(THRESH_PER_CLASS[i]) for i, d in enumerate(DISEASE_COLS)} }")
print("Usage: result = predict_single('/path/to/leaf.jpg')")
"""))

nb["cells"].append(code("""\
# ── Batch inference on a folder ───────────────────────────────────────────────
INFER_DIR = '/path/to/new/images'   # <-- change this

image_files = list(Path(INFER_DIR).glob('*.jpg')) + list(Path(INFER_DIR).glob('*.JPG'))
print(f"Found {len(image_files)} images")

results = []
for img_path in tqdm(image_files):
    try:
        r = predict_single(str(img_path))
        results.append(r)
    except Exception as e:
        results.append({'image': img_path.name, 'detected': ['ERROR'], 'error': str(e)})

# Expand to one column per disease for easy CSV export
rows = []
for r in results:
    row = {'image': r['image']}
    row.update(r.get('probabilities', {}))
    row['detected'] = ', '.join(r['detected'])
    rows.append(row)

out_df  = pd.DataFrame(rows)
out_csv = os.path.join(MODEL_DIR, 'inference_results.csv')
out_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(out_df[['image', 'detected']].head(10))
"""))

# ── CELL 13 — NOTES ────────────────────────────────────────────────────────────
nb["cells"].append(md("""\
## 13. Notes & Expected Results

### Key differences from Cassava pipeline
| | Cassava (Exp 11) | Maize (this notebook) |
|---|---|---|
| Task | Single-label (5 classes) | Multi-label (8 classes) |
| Output activation | Softmax | **Sigmoid** |
| Loss | Focal loss (per-class gamma) | **Weighted BCE** |
| Ensemble | EB0 generalist + ConvNeXt specialist | ConvNeXt only (5-fold avg) |
| Stratification | StratifiedKFold | **MultilabelStratifiedKFold** |
| Primary metric | Accuracy | **Macro F1 + AUC** |

### SR class (39 samples)
- pos_weight will be ~59× (capped at 50) — the model is heavily penalised for missing SR
- If SR F1 > 0 after training, it's genuinely learnable; report it
- If SR F1 = 0, it means 39 samples is below the learning threshold for this backbone — consider merging into Other for the paper

### Metrics to report
- **Macro F1** (primary — treats all 8 classes equally, highlights SR performance)
- **Micro F1** (secondary — weighted by support, reflects overall system accuracy)
- **Hamming Loss** (fraction of label-image pairs predicted incorrectly — lower is better)
- **Per-class F1** table (required for paper)
- **AUC** (threshold-independent, good complement to F1)

### Threshold
- Default 0.5 is a reasonable starting point
- Run Section 11 after training to find optimal per-class thresholds
- Expect GLS/NCLB thresholds to stay near 0.5; SR threshold may need to drop to 0.3-0.4
"""))

with open('maize_multilabel.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook written ✓")
