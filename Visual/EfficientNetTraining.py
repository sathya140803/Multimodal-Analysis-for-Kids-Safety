import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, classification_report)
from timm import create_model
from tqdm import tqdm

# CONFIG
DATA_DIR        = r"D:\CollectedImagesNew"
SAVE_DIR        = r"D:\CollectedImagesNew\model_output"
IMG_SIZE        = 260
BATCH_SIZE      = 32
EPOCHS          = 40
LR              = 0.0001
PATIENCE        = 5           # Early stopping patience
UNFREEZE_LAYERS = 20          # Last N layers of EfficientNetB2 to unfreeze
SEED            = 42
NUM_WORKERS     = 0   # Must be 0 on Windows to avoid multiprocessing pickle errors

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)

# GPU SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed(SEED)
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU found — running on CPU")

# CUSTOM TRANSFORM — scale tensor back to 0-255

class ScaleTo255:
    def __call__(self, x):
        return x * 255.0

# TRANSFORMS
# Images already saved as 260x260 on disk
# EfficientNetB2 expects raw 0-255 pixels
# Augmentation on train only
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # force 260x260 — handles any size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),   # converts to [0,1] float tensor
    ScaleTo255(),            # back to 0-255 for EfficientNetB2
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # force 260x260 — handles any size
    transforms.ToTensor(),
    ScaleTo255(),            # 0-255 for EfficientNetB2
])


# DATASETS — loaded from separate train/val folders
# No shuffling of val — no data leakage possible
# because train and val videos are completely separate

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == "cuda" else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == "cuda" else False
)

CLASS_NAMES = train_dataset.classes  # ['safe', 'unsafe']
print(f"\n  Classes      : {train_dataset.class_to_idx}")
print(f"  Train images : {len(train_dataset)}")
print(f"  Val images   : {len(val_dataset)}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches  : {len(val_loader)}\n")

# BUILD MODEL — EfficientNetB2 via timm

print("=" * 55)
print("   BUILDING MODEL")
print("=" * 55)

# Load pretrained EfficientNetB2
model = create_model(
    'efficientnet_b2',
    pretrained=True,
    num_classes=0,          # Remove original classifier head
    global_pool='avg'       # GlobalAveragePooling
)

# Get feature size from EfficientNetB2
feature_size = model.num_features  # 1408 for B2

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last N layers
all_layers = list(model.named_parameters())
for name, param in all_layers[-UNFREEZE_LAYERS:]:
    param.requires_grad = True

frozen   = sum(1 for p in model.parameters() if not p.requires_grad)
unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
print(f"  🔒 Frozen params  : {frozen}")
print(f"  🔓 Unfrozen params: {unfrozen}")

# FULL MODEL — EfficientNetB2 base + custom head

class CartoonClassifier(nn.Module):
    def __init__(self, base_model, feature_size):
        super().__init__()
        self.base  = base_model
        self.head  = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),      # Binary output
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base(x)
        return self.head(features).squeeze(1)

full_model = CartoonClassifier(model, feature_size).to(device)

# Count total parameters
total_params    = sum(p.numel() for p in full_model.parameters())
trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
print(f"  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")


# LOSS, OPTIMIZER, SCHEDULER

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, full_model.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=True
)

# EVALUATE — get predictions and metrics

def evaluate(loader):
    full_model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    val_bar = tqdm(
        loader,
        desc="  Validating",
        ncols=100,
        leave=False
    )

    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = full_model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    precision = precision_score(all_labels, all_preds,
                                average=None, zero_division=0)
    recall    = recall_score(all_labels, all_preds,
                             average=None, zero_division=0)
    f1        = f1_score(all_labels, all_preds,
                         average=None, zero_division=0)
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels



# TRAINING LOOP

print("\n" + "=" * 55)
print("   TRAINING STARTING")
print(f"   Max epochs    : {EPOCHS}")
print(f"   Learning rate : {LR}")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"   Early stopping: patience={PATIENCE}")
print(f"   Device        : {device}")
print("=" * 55 + "\n")

best_val_acc    = 0.0
patience_counter = 0
epoch_metrics   = []
history         = {
    "train_loss": [], "train_accuracy": [],
    "val_loss":   [], "val_accuracy":   []
}

for epoch in range(EPOCHS):
    # ── TRAIN ──
    full_model.train()
    train_loss = 0.0
    train_preds  = []
    train_labels = []

    train_bar = tqdm(
        train_loader,
        desc=f"  Epoch {epoch+1}/{EPOCHS} [Train]",
        ncols=100,
        leave=True
    )

    for batch_idx, (images, labels) in enumerate(train_bar):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = full_model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (outputs > 0.5).long().cpu().detach().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy().astype(int))

        # Update progress bar with live loss
        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc" : f"{np.mean(np.array(train_preds) == np.array(train_labels))*100:.1f}%"
        })

    avg_train_loss = train_loss / len(train_loader)
    train_acc      = np.mean(np.array(train_preds) == np.array(train_labels))

    # ── VALIDATE ──
    val_loss, val_acc, precision, recall, f1, _, _ = evaluate(val_loader)

    # Update learning rate scheduler
    scheduler.step(val_loss)

    # Print epoch summary
    print(f"\n  Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss  : {val_loss:.4f}       | Val Acc  : {val_acc*100:.2f}%")
    print(f"  Metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"     [{class_name}] "
              f"Precision: {precision[i]:.4f} | "
              f"Recall: {recall[i]:.4f} | "
              f"F1: {f1[i]:.4f}")

    # Save history
    history["train_loss"].append(avg_train_loss)
    history["train_accuracy"].append(float(train_acc))
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(float(val_acc))

    # Save per epoch metrics
    row = {
        "epoch"          : epoch + 1,
        "train_loss"     : round(avg_train_loss, 4),
        "train_accuracy" : round(float(train_acc), 4),
        "val_loss"       : round(val_loss, 4),
        "val_accuracy"   : round(float(val_acc), 4),
    }
    for i, class_name in enumerate(CLASS_NAMES):
        row[f"precision_{class_name}"] = round(float(precision[i]), 4)
        row[f"recall_{class_name}"]    = round(float(recall[i]), 4)
        row[f"f1_{class_name}"]        = round(float(f1[i]), 4)
    epoch_metrics.append(row)

    # Save metrics after every epoch
    pd.DataFrame(epoch_metrics).to_csv(
        os.path.join(SAVE_DIR, "metrics_per_epoch.csv"), index=False
    )
    with open(os.path.join(SAVE_DIR, "metrics_per_epoch.json"), "w") as f:
        json.dump(epoch_metrics, f, indent=4)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(full_model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"\n  Best model saved — Val Acc: {val_acc*100:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"\n   No improvement — patience: {patience_counter}/{PATIENCE}")

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n Early stopping triggered at epoch {epoch+1}")
        break

    print(f"  {'='*50}")



# SAVE FINAL MODEL AND HISTORY

torch.save(full_model.state_dict(),
           os.path.join(SAVE_DIR, "final_model.pth"))

with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=4)

# FINAL METRICS
print("\n" + "=" * 55)
print("   FINAL METRICS")
print("=" * 55)

# Load best model for final evaluation
full_model.load_state_dict(
    torch.load(os.path.join(SAVE_DIR, "best_model.pth"),
               map_location=device)
)

_, _, precision, recall, f1, all_preds, all_labels = evaluate(val_loader)

report = classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES,
    output_dict=True
)
print("\n" + classification_report(
    all_labels, all_preds, target_names=CLASS_NAMES
))

# Build final metrics dict
final_metrics = {}
for class_name in CLASS_NAMES:
    final_metrics[class_name] = {
        "precision" : round(report[class_name]["precision"], 4),
        "recall"    : round(report[class_name]["recall"], 4),
        "f1_score"  : round(report[class_name]["f1-score"], 4),
        "support"   : int(report[class_name]["support"])
    }
final_metrics["overall"] = {
    "accuracy"        : round(report["accuracy"], 4),
    "macro_avg_f1"    : round(report["macro avg"]["f1-score"], 4),
    "weighted_avg_f1" : round(report["weighted avg"]["f1-score"], 4),
}

# Save final metrics
with open(os.path.join(SAVE_DIR, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

final_rows = []
for class_name in CLASS_NAMES:
    final_rows.append({
        "class"     : class_name,
        "precision" : final_metrics[class_name]["precision"],
        "recall"    : final_metrics[class_name]["recall"],
        "f1_score"  : final_metrics[class_name]["f1_score"],
        "support"   : final_metrics[class_name]["support"],
    })
pd.DataFrame(final_rows).to_csv(
    os.path.join(SAVE_DIR, "final_metrics.csv"), index=False
)

print(f"\n Saved to: {SAVE_DIR}")
print(f"   best_model.pth          ← best val_accuracy checkpoint")
print(f"   final_model.pth         ← final epoch model")
print(f"   history.json            ← for plotting graphs")
print(f"   metrics_per_epoch.csv   ← precision, recall, f1 per epoch")
print(f"   metrics_per_epoch.json  ← same in JSON")
print(f"   final_metrics.csv       ← final scores per class")
print(f"   final_metrics.json      ← same in JSON")
print(f"\n  Best Val Accuracy: {best_val_acc*100:.2f}%")
print("\n Training complete!")