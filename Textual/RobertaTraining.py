
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_DIR     = r"D:\PythonYouTube\Textual"
OUTPUT_DIR   = r"D:\PythonYouTube\Textual"
TEXT_COL     = "text"
LABEL_COL    = "label"

# Hyperparameters — tuned for 4GB VRAM RTX 3050
MAX_LEN      = 128    # covers 95%+ of your texts (median was 36 tokens)
BATCH_SIZE   = 16     # safe for 4GB VRAM
EPOCHS       = 5      # early stopping will kick in before if needed
LEARNING_RATE= 2e-5   # standard for RoBERTa fine-tuning
WARMUP_RATIO = 0.1    # 10% of total steps used for warmup
WEIGHT_DECAY = 0.01   # L2 regularisation
GRAD_CLIP    = 1.0    # gradient clipping threshold
PATIENCE     = 3      # early stopping patience (epochs)
DROPOUT      = 0.1    # classification head dropout
SEED         = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# REPRODUCIBILITY
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# GPU CHECK

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f" GPU detected  : {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version  : {torch.version.cuda}")
    print(f"   Total VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("️  No GPU detected — running on CPU (will be slow)")

print(f"   Device in use : {device}")

#  LOAD DATASETS
print("\n" + "=" * 60)
print("SECTION 1: LOADING DATASETS")
print("=" * 60)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_split.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val_split.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))

# Drop any nulls
train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL])
val_df   = val_df.dropna(subset=[TEXT_COL, LABEL_COL])
test_df  = test_df.dropna(subset=[TEXT_COL, LABEL_COL])

# Ensure labels are integers
train_df[LABEL_COL] = train_df[LABEL_COL].astype(int)
val_df[LABEL_COL]   = val_df[LABEL_COL].astype(int)
test_df[LABEL_COL]  = test_df[LABEL_COL].astype(int)

print(f" Datasets loaded")
print(f"   Train : {len(train_df):,} rows  |  Safe: {(train_df[LABEL_COL]==0).sum():,}  Harmful: {(train_df[LABEL_COL]==1).sum():,}")
print(f"   Val   : {len(val_df):,}  rows  |  Safe: {(val_df[LABEL_COL]==0).sum():,}   Harmful: {(val_df[LABEL_COL]==1).sum():,}")
print(f"   Test  : {len(test_df):,}  rows  |  Safe: {(test_df[LABEL_COL]==0).sum():,}   Harmful: {(test_df[LABEL_COL]==1).sum():,}")


#  TOKENIZER
print("\n" + "=" * 60)
print("SECTION 2: LOADING TOKENIZER")
print("=" * 60)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print(f" RobertaTokenizer loaded")
print(f"   Vocabulary size : {tokenizer.vocab_size:,}")
print(f"   Max length      : {MAX_LEN} tokens")

# ─────────────────────────────────────────────
# SECTION 3: DATASET CLASS
# ─────────────────────────────────────────────
class HarmfulContentDataset(Dataset):
    """
    PyTorch Dataset for harmful content classification.
    Tokenizes text on-the-fly and returns input_ids,
    attention_mask, and labels.
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts     = dataframe[TEXT_COL].astype(str).tolist()
        self.labels    = dataframe[LABEL_COL].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create datasets
train_dataset = HarmfulContentDataset(train_df, tokenizer, MAX_LEN)
val_dataset   = HarmfulContentDataset(val_df,   tokenizer, MAX_LEN)
test_dataset  = HarmfulContentDataset(test_df,  tokenizer, MAX_LEN)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\n Datasets and DataLoaders created")
print(f"   Train batches : {len(train_loader):,}")
print(f"   Val batches   : {len(val_loader):,}")
print(f"   Test batches  : {len(test_loader):,}")

#  MODEL
print("\n" + "=" * 60)
print("SECTION 4: LOADING ROBERTA MODEL")
print("=" * 60)

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT
)
model = model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ roberta-base loaded and moved to {device}")
print(f"   Total parameters     : {total_params:,}")
print(f"   Trainable parameters : {trainable_params:,}")

# SECTION 5: OPTIMIZER & SCHEDULER
print("\n" + "=" * 60)
print("SECTION 5: OPTIMIZER & SCHEDULER")
print("=" * 60)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

# AdamW — best optimizer for transformer fine-tuning
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8
)

# Linear warmup then linear decay scheduler
# Warmup prevents catastrophic forgetting of pre-trained weights
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f" AdamW optimizer configured")
print(f"   Learning rate  : {LEARNING_RATE}")
print(f"   Weight decay   : {WEIGHT_DECAY}")
print(f"   Total steps    : {total_steps:,}")
print(f"   Warmup steps   : {warmup_steps:,}  ({WARMUP_RATIO*100:.0f}% of total)")
print(f"   Gradient clip  : {GRAD_CLIP}")

# SECTION 6: EVALUATION FUNCTION
def evaluate(model, dataloader, device):
    """
    Runs model on a dataloader and returns
    loss, accuracy, per-class precision/recall/F1.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    eval_bar = tqdm(
        dataloader,
        desc="  Evaluating",
        unit="batch",
        ncols=100,
        colour="green",
        leave=False
    )

    with torch.no_grad():
        for batch in eval_bar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            eval_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

    avg_loss  = total_loss / len(dataloader)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    return {
        "loss"       : avg_loss,
        "accuracy"   : accuracy,
        "precision"  : precision,
        "recall"     : recall,
        "f1"         : f1,
        "per_class"  : {
            "safe"    : {"precision": p_per_class[0], "recall": r_per_class[0], "f1": f1_per_class[0], "support": support[0]},
            "harmful" : {"precision": p_per_class[1], "recall": r_per_class[1], "f1": f1_per_class[1], "support": support[1]}
        },
        "preds"      : all_preds,
        "labels"     : all_labels
    }

# ─────────────────────────────────────────────
# SECTION 7: TRAINING LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7: TRAINING")
print("=" * 60)
print(f"   Epochs         : {EPOCHS}")
print(f"   Batch size     : {BATCH_SIZE}")
print(f"   Max token len  : {MAX_LEN}")
print(f"   Early stopping : patience = {PATIENCE} epochs (monitors val F1)")
print()

# History tracking
history = {
    "train_loss" : [], "train_accuracy" : [], "train_f1" : [],
    "val_loss"   : [], "val_accuracy"   : [], "val_f1"   : []
}

# Early stopping state
best_val_f1      = 0.0
patience_counter = 0
best_epoch       = 0

for epoch in range(1, EPOCHS + 1):
    print(f"── Epoch {epoch}/{EPOCHS} {'─'*45}")

    # ── Training phase ──
    model.train()
    train_loss  = 0
    train_preds = []
    train_labels= []

    train_bar = tqdm(
        train_loader,
        desc=f"  Epoch {epoch}/{EPOCHS} [Train]",
        unit="batch",
        ncols=100,
        colour="blue"
    )

    for batch in train_bar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        # Update progress bar with live loss and LR
        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr"  : f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # Training metrics for epoch
    avg_train_loss = train_loss / len(train_loader)
    train_acc      = accuracy_score(train_labels, train_preds)
    _, _, train_f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average="weighted"
    )

    # ── Validation phase ──
    val_results = evaluate(model, val_loader, device)

    # Save to history
    history["train_loss"].append(avg_train_loss)
    history["train_accuracy"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["val_loss"].append(val_results["loss"])
    history["val_accuracy"].append(val_results["accuracy"])
    history["val_f1"].append(val_results["f1"])

    print(f"\n   TRAIN  →  Loss: {avg_train_loss:.4f}  |  Acc: {train_acc:.4f}  |  F1: {train_f1:.4f}")
    print(f"   VAL    →  Loss: {val_results['loss']:.4f}  |  Acc: {val_results['accuracy']:.4f}  |  F1: {val_results['f1']:.4f}")
    print(f"\n   Val Per-Class:")
    print(f"   ├── Safe    →  P: {val_results['per_class']['safe']['precision']:.4f}  R: {val_results['per_class']['safe']['recall']:.4f}  F1: {val_results['per_class']['safe']['f1']:.4f}")
    print(f"   └── Harmful →  P: {val_results['per_class']['harmful']['precision']:.4f}  R: {val_results['per_class']['harmful']['recall']:.4f}  F1: {val_results['per_class']['harmful']['f1']:.4f}")

    # ── Early Stopping Check ──
    if val_results["f1"] > best_val_f1:
        best_val_f1      = val_results["f1"]
        best_epoch       = epoch
        patience_counter = 0

        # Save best model checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, "best_roberta_model")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"\n    New best model saved! Val F1: {best_val_f1:.4f}  →  {checkpoint_path}")
    else:
        patience_counter += 1
        print(f"\n     No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch}.")
            print(f"   Best model was at epoch {best_epoch} with Val F1: {best_val_f1:.4f}")
            break

    print()

# SECTION 8: SAVE TRAINING HISTORY

print("\n" + "=" * 60)
print("SECTION 8: SAVING TRAINING HISTORY")
print("=" * 60)

history_path = os.path.join(OUTPUT_DIR, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=4)
print(f" Training history saved → {history_path}")

#  PLOT TRAINING GRAPHS
print("\n" + "=" * 60)
print("SECTION 9: SAVING TRAINING GRAPHS")
print("=" * 60)

epochs_ran = list(range(1, len(history["train_loss"]) + 1))
fig, axes  = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("RoBERTa Fine-Tuning — Training History", fontsize=14, fontweight="bold")

# Loss
axes[0].plot(epochs_ran, history["train_loss"], "b-o", label="Train Loss")
axes[0].plot(epochs_ran, history["val_loss"],   "r-o", label="Val Loss")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs_ran, history["train_accuracy"], "b-o", label="Train Accuracy")
axes[1].plot(epochs_ran, history["val_accuracy"],   "r-o", label="Val Accuracy")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# F1 Score
axes[2].plot(epochs_ran, history["train_f1"], "b-o", label="Train F1")
axes[2].plot(epochs_ran, history["val_f1"],   "r-o", label="Val F1")
axes[2].set_title("F1 Score per Epoch")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("F1 Score")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
graph_path = os.path.join(OUTPUT_DIR, "training_history_graphs.png")
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Training graphs saved → {graph_path}")

# ─────────────────────────────────────────────
# SECTION 10: FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 10: FINAL TEST SET EVALUATION")
print("=" * 60)
print("   Loading best model checkpoint...")

best_model = RobertaForSequenceClassification.from_pretrained(
    os.path.join(OUTPUT_DIR, "best_roberta_model")
)
best_model = best_model.to(device)

test_results = evaluate(best_model, test_loader, device)

print(f"\n{'─'*50}")
print(f"  FINAL TEST RESULTS")
print(f"{'─'*50}")
print(f"  Accuracy          : {test_results['accuracy']:.4f}  ({test_results['accuracy']*100:.2f}%)")
print(f"  Weighted Precision: {test_results['precision']:.4f}")
print(f"  Weighted Recall   : {test_results['recall']:.4f}")
print(f"  Weighted F1 Score : {test_results['f1']:.4f}")
print(f"{'─'*50}")
print(f"  PER-CLASS RESULTS")
print(f"{'─'*50}")
print(f"  Safe (0):")
print(f"    Precision : {test_results['per_class']['safe']['precision']:.4f}")
print(f"    Recall    : {test_results['per_class']['safe']['recall']:.4f}")
print(f"    F1 Score  : {test_results['per_class']['safe']['f1']:.4f}")
print(f"    Support   : {test_results['per_class']['safe']['support']:,}")
print(f"\n  Harmful (1):")
print(f"    Precision : {test_results['per_class']['harmful']['precision']:.4f}")
print(f"    Recall    : {test_results['per_class']['harmful']['recall']:.4f}")
print(f"    F1 Score  : {test_results['per_class']['harmful']['f1']:.4f}")
print(f"    Support   : {test_results['per_class']['harmful']['support']:,}")
print(f"{'─'*50}")

# Full classification report
print(f"\n  Full Classification Report:")
print(classification_report(
    test_results["labels"],
    test_results["preds"],
    target_names=["Safe (0)", "Harmful (1)"]
))

# Save test results to JSON
test_results_save = {
    "accuracy"          : test_results["accuracy"],
    "weighted_precision": test_results["precision"],
    "weighted_recall"   : test_results["recall"],
    "weighted_f1"       : test_results["f1"],
    "per_class"         : {
        "safe"   : {k: float(v) for k, v in test_results["per_class"]["safe"].items()},
        "harmful": {k: float(v) for k, v in test_results["per_class"]["harmful"].items()}
    }
}
results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(test_results_save, f, indent=4)
print(f" Test results saved → {results_path}")

# ─────────────────────────────────────────────
# SECTION 11: CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 11: CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(test_results["labels"], test_results["preds"])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted Safe", "Predicted Harmful"],
    yticklabels=["Actual Safe",    "Actual Harmful"],
    ax=ax
)
ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual Label")
ax.set_xlabel("Predicted Label")

# Annotate cells
tn, fp, fn, tp = cm.ravel()
print(f"   True Negatives  (Safe correctly identified)     : {tn:,}")
print(f"   False Positives (Safe wrongly flagged harmful)  : {fp:,}")
print(f"   False Negatives (Harmful missed)                : {fn:,}")
print(f"   True Positives  (Harmful correctly identified)  : {tp:,}")

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f" Confusion matrix saved → {cm_path}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Best epoch            : {best_epoch}")
print(f"  Best val F1           : {best_val_f1:.4f}")
print(f"  Test accuracy         : {test_results['accuracy']*100:.2f}%")
print(f"  Test F1 (weighted)    : {test_results['f1']:.4f}")
print(f"\n  Files saved to: {OUTPUT_DIR}")
print(f"  ├── best_roberta_model/       ← model weights & tokenizer")
print(f"  ├── training_history.json     ← loss, accuracy, F1 per epoch")
print(f"  ├── training_history_graphs.png")
print(f"  ├── test_results.json         ← final test metrics")
print(f"  └── confusion_matrix.png")
print("=" * 60)