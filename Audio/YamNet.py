

import os
import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURATION & HYPERPARAMETERS
class Config:

    # Audio
    SAMPLE_RATE          = 16000
    CLIP_DURATION        = 10

    # Dataset
    DATA_DIR             = r"D:\PythonYouTube\Audio\data"
    LABEL_MAP            = {"safe": 0, "unsafe": 1}
    VAL_SPLIT            = 0.20
    RANDOM_SEED          = 42

    # YAMNet (frozen)
    YAMNET_URL           = "https://tfhub.dev/google/yamnet/1"
    EMBEDDING_DIM        = 1024

    # New classification layers
    DENSE_1_UNITS        = 256
    DENSE_2_UNITS        = 128
    DROPOUT_RATE         = 0.4
    L2_REG               = 1e-4

    # Training
    BATCH_SIZE           = 32
    MAX_EPOCHS           = 100
    LEARNING_RATE        = 1e-3
    LR_PATIENCE          = 5
    LR_FACTOR            = 0.5
    MIN_LR               = 1e-6
    EARLY_STOP_PATIENCE  = 15


    MIN_SAFE_ACC         = 0.70   # minimum acceptable safe accuracy

    # Output paths
    RESULTS_DIR          = r"D:\PythonYouTube\Audio\results"
    MODEL_PATH           = r"D:\PythonYouTube\Audio\models\yamnet_transfer.keras"

cfg = Config()




def load_audio(file_path: str) -> np.ndarray:
    """
    Load one audio file and prepare it for YAMNet.
    Resamples to 16kHz mono, pads/trims to fixed length, normalises to [-1, 1].
    """
    waveform, _ = librosa.load(file_path, sr=cfg.SAMPLE_RATE, mono=True)
    waveform = waveform.astype(np.float32)

    target = cfg.SAMPLE_RATE * cfg.CLIP_DURATION
    if len(waveform) < target:
        waveform = np.pad(waveform, (0, target - len(waveform)))
    else:
        waveform = waveform[:target]

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak

    return waveform

# DATASET
def build_dataset() -> pd.DataFrame:
    """Scan data/safe/ and data/unsafe/, return labeled DataFrame."""
    records   = []
    data_path = Path(cfg.DATA_DIR)
    exts      = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    for class_name, label in cfg.LABEL_MAP.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Folder not found: {class_dir}\n"
                f"Expected:\n"
                f"  {cfg.DATA_DIR}\\safe\\\n"
                f"  {cfg.DATA_DIR}\\unsafe\\"
            )
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in exts]
        if not files:
            raise ValueError(f"[ERROR] No audio files in {class_dir}")

        for f in files:
            records.append({"file_path": str(f), "label": label})
        print(f"  {class_name:8s} (label={label}): {len(files):4d} files")

    df = pd.DataFrame(records).sample(frac=1, random_state=cfg.RANDOM_SEED).reset_index(drop=True)
    print(f"\n  Total : {len(df)} samples  "
          f"(safe={( df['label']==0).sum()}, unsafe={(df['label']==1).sum()})\n")
    return df


def split_dataset(df: pd.DataFrame):
    """Stratified 80/20 train / val split."""
    train_df, val_df = train_test_split(
        df, test_size=cfg.VAL_SPLIT,
        stratify=df["label"], random_state=cfg.RANDOM_SEED
    )
    print(f"  Train : {len(train_df)}  |  Val : {len(val_df)}")
    print(f"  Train — safe: {(train_df['label']==0).sum()}, "
          f"unsafe: {(train_df['label']==1).sum()}")
    print(f"  Val   — safe: {(val_df['label']==0).sum()}, "
          f"unsafe: {(val_df['label']==1).sum()}\n")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_waveforms(df: pd.DataFrame, split_name: str):
    """Load all audio files into a NumPy array of waveforms."""
    print(f"  Loading [{split_name}] waveforms ...")
    waveforms, labels, skipped = [], [], 0

    for _, row in df.iterrows():
        try:
            waveforms.append(load_audio(row["file_path"]))
            labels.append(row["label"])
        except Exception as e:
            print(f"    [SKIP] {Path(row['file_path']).name}: {e}")
            skipped += 1

    if skipped:
        print(f"    {skipped} file(s) skipped.")

    X = np.array(waveforms, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)
    print(f"  Done  — X: {X.shape}, y: {y.shape}\n")
    return X, y


# YAMNET FEATURE EXTRACTION  (frozen — runs once before training)

def extract_embeddings(yamnet_model, X: np.ndarray, split_name: str) -> np.ndarray:
    """
    Pass every waveform through frozen YAMNet.
    Returns mean-pooled embeddings shape (N, 1024).
    Done once before training — much faster than running inside the training loop.
    """
    print(f"  Extracting embeddings [{split_name}] ...")
    embeddings = []

    for i, waveform in enumerate(X):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(X)}")
        _, emb, _ = yamnet_model(waveform)
        embeddings.append(tf.reduce_mean(emb, axis=0).numpy())

    E = np.array(embeddings, dtype=np.float32)
    print(f"  Done  — embeddings: {E.shape}\n")
    return E



# MODEL
def build_model() -> tf.keras.Model:
    """
    New classification layers trained on top of frozen YAMNet embeddings.

    Input  (1024,)
      Dense(256, ReLU) + BatchNorm + Dropout(0.4)
      Dense(128, ReLU) + BatchNorm + Dropout(0.2)
      Dense(1, Sigmoid)  ->  P(unsafe)
    """
    reg = tf.keras.regularizers.l2(cfg.L2_REG)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.EMBEDDING_DIM,), name="yamnet_embedding"),

        tf.keras.layers.Dense(cfg.DENSE_1_UNITS, activation="relu",
                               kernel_regularizer=reg, name="dense_1"),
        tf.keras.layers.BatchNormalization(name="bn_1"),
        tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="drop_1"),

        tf.keras.layers.Dense(cfg.DENSE_2_UNITS, activation="relu",
                               kernel_regularizer=reg, name="dense_2"),
        tf.keras.layers.BatchNormalization(name="bn_2"),
        tf.keras.layers.Dropout(cfg.DROPOUT_RATE / 2, name="drop_2"),

        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ], name="transfer_head")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model

# CALLBACKS
def build_callbacks() -> list:
    """
    EarlyStopping     — stops if val_auc does not improve, restores best weights
    ModelCheckpoint   — saves best model to disk
    ReduceLROnPlateau — halves LR when val_loss plateaus
    CSVLogger         — logs every epoch to CSV
    """
    os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=cfg.EARLY_STOP_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg.MODEL_PATH,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.LR_FACTOR,
            patience=cfg.LR_PATIENCE,
            min_lr=cfg.MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(cfg.RESULTS_DIR, "training_log.csv"),
            append=False
        ),
    ]


# TRAINING
def train(model, E_train, y_train, E_val, y_val):
    """Train the classification head with class balancing."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    print(f"  Class weights: safe={class_weight[0]:.3f}, "
          f"unsafe={class_weight[1]:.3f}\n")

    print(f"  Training up to {cfg.MAX_EPOCHS} epochs  "
          f"(early stop patience={cfg.EARLY_STOP_PATIENCE})...\n")

    history = model.fit(
        E_train, y_train,
        validation_data=(E_val, y_val),
        epochs=cfg.MAX_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        class_weight=class_weight,
        callbacks=build_callbacks(),
        verbose=1
    )
    return history



# SAVE HISTORY

def save_history(history):
    """Save training history as JSON and plot Loss / AUC / Accuracy curves."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(cfg.RESULTS_DIR, "training_history.json")
    with open(json_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"  History  → {json_path}")

    epochs = range(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, train_key, val_key, title, ylabel in [
        (axes[0], "loss",     "val_loss",     "Loss",     "Binary Cross-entropy"),
        (axes[1], "auc",      "val_auc",      "AUC",      "AUC"),
        (axes[2], "accuracy", "val_accuracy", "Accuracy", "Accuracy"),
    ]:
        ax.plot(epochs, history.history[train_key], label="Train")
        ax.plot(epochs, history.history[val_key],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("YAMNet Transfer Learning — Training History", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(cfg.RESULTS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Curves   → {plot_path}")


# THRESHOLD TUNING  (finds best decision boundary for unsafe detection)

def tune_threshold(y_val: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Sweep thresholds from 0.10 to 0.90 and find the one that gives
    the best F1-unsafe while keeping safe accuracy >= MIN_SAFE_ACC.

    For child safety, missing an unsafe clip (false negative) is worse
    than a false alarm on a safe clip, so we prioritise unsafe recall.

    Returns the recommended threshold.
    """
    print("\n" + "=" * 65)
    print(f"  {'Threshold':>9} | {'Accuracy':>8} | {'Safe Acc':>8} | "
          f"{'Unsafe Acc':>10} | {'F1-Unsafe':>9}")
    print("=" * 65)

    thresholds   = np.arange(0.10, 0.91, 0.05)
    rows         = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        safe_mask   = (y_val == 0)
        unsafe_mask = (y_val == 1)

        overall_acc = float(np.mean(y_pred == y_val))
        safe_acc    = float(np.mean(y_pred[safe_mask]   == 0)) if safe_mask.sum()   > 0 else 0.0
        unsafe_acc  = float(np.mean(y_pred[unsafe_mask] == 1)) if unsafe_mask.sum() > 0 else 0.0

        tp = int(((y_pred == 1) & (y_val == 1)).sum())
        fp = int(((y_pred == 1) & (y_val == 0)).sum())
        fn = int(((y_pred == 0) & (y_val == 1)).sum())
        prec_u  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_u   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_u    = (2 * prec_u * rec_u / (prec_u + rec_u)
                   if (prec_u + rec_u) > 0 else 0.0)

        print(f"  {t:>9.2f} | {overall_acc:>8.4f} | {safe_acc:>8.4f} | "
              f"{unsafe_acc:>10.4f} | {f1_u:>9.4f}")

        rows.append({
            "threshold":  round(float(t),           2),
            "accuracy":   round(overall_acc,         4),
            "safe_acc":   round(safe_acc,            4),
            "unsafe_acc": round(unsafe_acc,          4),
            "f1_unsafe":  round(f1_u,                4),
        })

    print("=" * 65)

    # Best threshold = highest F1-unsafe where safe_acc >= MIN_SAFE_ACC
    candidates = [r for r in rows if r["safe_acc"] >= cfg.MIN_SAFE_ACC]
    if candidates:
        best = max(candidates, key=lambda r: r["f1_unsafe"])
    else:
        # Fallback: just best F1-unsafe regardless of safe accuracy
        best = max(rows, key=lambda r: r["f1_unsafe"])

    print(f"\n  Recommended threshold : {best['threshold']}")
    print(f"  Safe accuracy         : {best['safe_acc']:.4f}")
    print(f"  Unsafe accuracy       : {best['unsafe_acc']:.4f}")
    print(f"  F1-unsafe             : {best['f1_unsafe']:.4f}\n")

    return best["threshold"], rows



# EVALUATION

def evaluate(model, E_val, y_val):
    """
    Full evaluation:
        1. Default threshold (0.5) — baseline
        2. Threshold tuning sweep
        3. Final report at recommended threshold
        4. Per-class accuracy for safe and unsafe
        5. Confusion matrix, PR curve, threshold plot — all saved
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    y_prob = model.predict(E_val, verbose=0).flatten()

    # ── Default threshold ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EVALUATION AT DEFAULT THRESHOLD (0.5)")
    print("=" * 65)
    _print_per_class(y_val, (y_prob >= 0.5).astype(int), label="0.5")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    print("\nTuning threshold for best unsafe recall...")
    recommended, sweep_rows = tune_threshold(y_val, y_prob)

    # ── Final results at recommended threshold ────────────────────────────────
    y_pred_final = (y_prob >= recommended).astype(int)

    print("=" * 65)
    print(f"  FINAL RESULTS AT RECOMMENDED THRESHOLD ({recommended})")
    print("=" * 65)
    safe_acc, unsafe_acc = _print_per_class(
        y_val, y_pred_final, label=str(recommended)
    )

    # ── Save text report ──────────────────────────────────────────────────────
    report_path = os.path.join(cfg.RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("YAMNet Transfer Learning — Evaluation Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Recommended threshold : {recommended}\n\n")
        f.write(f"SAFE   accuracy : {safe_acc:.4f}  "
                f"({int(safe_acc*(y_val==0).sum())}/{(y_val==0).sum()})\n")
        f.write(f"UNSAFE accuracy : {unsafe_acc:.4f}  "
                f"({int(unsafe_acc*(y_val==1).sum())}/{(y_val==1).sum()})\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred_final,
                                       target_names=["SAFE", "UNSAFE"], digits=4))
        f.write("\nThreshold Sweep:\n")
        for r in sweep_rows:
            f.write(f"  {r}\n")
    print(f"\n  Report   → {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_evaluation(y_val, y_prob, y_pred_final, sweep_rows, recommended)


def _print_per_class(y_val, y_pred, label=""):
    """Print per-class accuracy and classification report. Returns (safe_acc, unsafe_acc)."""
    safe_mask   = (y_val == 0)
    unsafe_mask = (y_val == 1)
    safe_acc    = float(np.mean(y_pred[safe_mask]   == 0)) if safe_mask.sum()   > 0 else 0.0
    unsafe_acc  = float(np.mean(y_pred[unsafe_mask] == 1)) if unsafe_mask.sum() > 0 else 0.0

    print(f"\n  Per-class accuracy (threshold={label}):")
    print(f"    SAFE   : {safe_acc:.4f}  "
          f"({int(safe_acc * safe_mask.sum())}/{safe_mask.sum()} correct)")
    print(f"    UNSAFE : {unsafe_acc:.4f}  "
          f"({int(unsafe_acc * unsafe_mask.sum())}/{unsafe_mask.sum()} correct)")
    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred,
                                 target_names=["SAFE", "UNSAFE"], digits=4))
    return safe_acc, unsafe_acc


def _plot_evaluation(y_val, y_prob, y_pred_final, sweep_rows, recommended):
    """Save threshold sweep, PR curve, and confusion matrix as one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Threshold sweep
    thrs    = [r["threshold"]  for r in sweep_rows]
    s_accs  = [r["safe_acc"]   for r in sweep_rows]
    u_accs  = [r["unsafe_acc"] for r in sweep_rows]
    f1s     = [r["f1_unsafe"]  for r in sweep_rows]

    axes[0].plot(thrs, s_accs, "b-o", label="Safe accuracy",   markersize=5)
    axes[0].plot(thrs, u_accs, "r-o", label="Unsafe accuracy", markersize=5)
    axes[0].plot(thrs, f1s,    "g--", label="F1 unsafe",       markersize=5)
    axes[0].axvline(recommended, color="orange", linestyle="--",
                    label=f"Recommended ({recommended})")
    axes[0].axvline(0.5,         color="gray",   linestyle=":",
                    label="Default (0.5)")
    axes[0].set_title("Threshold Sweep")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # 2. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, "b-", linewidth=2)
    axes[1].set_title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    axes[1].set_xlabel("Recall (Unsafe)")
    axes[1].set_ylabel("Precision (Unsafe)")
    axes[1].grid(True, alpha=0.3)

    # 3. Confusion matrix at recommended threshold
    cm = confusion_matrix(y_val, y_pred_final)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["SAFE", "UNSAFE"],
                yticklabels=["SAFE", "UNSAFE"],
                annot_kws={"size": 13})
    axes[2].set_title(f"Confusion Matrix (threshold={recommended})")
    axes[2].set_ylabel("True Label")
    axes[2].set_xlabel("Predicted Label")

    plt.suptitle("YAMNet — Evaluation & Threshold Tuning", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(cfg.RESULTS_DIR, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plots    → {plot_path}\n")


# MAIN
def main():
    print("\n" + "=" * 65)
    print("  YAMNet Transfer Learning — Child Safety Audio")
    print("=" * 65 + "\n")

    # Step 1: Dataset
    print("[1/5] Building dataset...")
    df = build_dataset()

    print("[2/5] Splitting into train / val...")
    train_df, val_df = split_dataset(df)

    # Step 2: Load audio
    print("[3/5] Loading audio files...")
    X_train, y_train = load_waveforms(train_df, "train")
    X_val,   y_val   = load_waveforms(val_df,   "val")

    # Step 3: Extract frozen YAMNet embeddings (once)
    print("[4/5] Extracting frozen YAMNet embeddings...")
    print("  Loading YAMNet from TensorFlow Hub...")
    yamnet = hub.load(cfg.YAMNET_URL)
    print("  YAMNet loaded and FROZEN — weights will not change.\n")

    E_train = extract_embeddings(yamnet, X_train, "train")
    E_val   = extract_embeddings(yamnet, X_val,   "val")

    # Save embeddings so you can re-run without re-extracting
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(cfg.RESULTS_DIR, "E_train.npy"), E_train)
    np.save(os.path.join(cfg.RESULTS_DIR, "E_val.npy"),   E_val)
    np.save(os.path.join(cfg.RESULTS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(cfg.RESULTS_DIR, "y_val.npy"),   y_val)
    print(f"  Embeddings saved to {cfg.RESULTS_DIR}\n")

    # Step 4: Train
    print("[5/5] Building model and training...")
    model = build_model()
    model.summary()
    print()

    history = train(model, E_train, y_train, E_val, y_val)

    print("\nSaving training history...")
    save_history(history)

    # Step 5: Evaluate + threshold tuning (all in one)
    print("\nEvaluating and tuning threshold...")
    evaluate(model, E_val, y_val)

    print("=" * 65)
    print(f"  All done.")
    print(f"  Model    → {cfg.MODEL_PATH}")
    print(f"  Results  → {cfg.RESULTS_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()