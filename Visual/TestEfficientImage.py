import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from timm import create_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────
MODEL_PATH  = r"D:\CollectedImagesNew\model_output\best_model.pth"
IMG_SIZE    = 260
THRESHOLD   = 0.5    # above = unsafe, below = safe

# ─────────────────────────────────────────────────
# WHAT TO PREDICT ON
# Option 1 — single image:  set IMAGE_PATH, leave FOLDER_PATH = None
# Option 2 — folder:        set FOLDER_PATH, leave IMAGE_PATH = None
# Option 3 — both:          set both
# ─────────────────────────────────────────────────
IMAGE_PATH  = r"D:\CollectedImagesNew\val\safe\peppa_pig_000333.jpg"   # ← single image path
FOLDER_PATH = r""      # ← folder of images

# ─────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ─────────────────────────────────────────────────
# CUSTOM TRANSFORM — same as training
# ─────────────────────────────────────────────────
class ScaleTo255:
    def __call__(self, x):
        return x * 255.0

predict_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    ScaleTo255(),
])

# ─────────────────────────────────────────────────
# REBUILD MODEL ARCHITECTURE — must match train.py
# ─────────────────────────────────────────────────
class CartoonClassifier(nn.Module):
    def __init__(self, base_model, feature_size):
        super().__init__()
        self.base = base_model
        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base(x)
        return self.head(features).squeeze(1)


# ─────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────
def load_model():
    print("Loading model...")
    base = create_model(
        'efficientnet_b2',
        pretrained=False,      # no need to download weights
        num_classes=0,
        global_pool='avg'
    )
    feature_size = base.num_features
    model = CartoonClassifier(base, feature_size).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print("✅ Model loaded\n")
    return model


# ─────────────────────────────────────────────────
# PREDICT SINGLE IMAGE
# Returns label, confidence, raw probability
# ─────────────────────────────────────────────────
def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"  ❌ File not found: {image_path}")
        return None, None, None

    # Load and transform image
    img = Image.open(image_path).convert("RGB")
    tensor = predict_transforms(img).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        prob = model(tensor).item()  # probability 0-1

    label      = "UNSAFE" if prob > THRESHOLD else "SAFE"
    confidence = prob if prob > THRESHOLD else 1 - prob
    confidence = round(confidence * 100, 2)

    return label, confidence, prob, img


# ─────────────────────────────────────────────────
# SHOW SINGLE IMAGE WITH PREDICTION
# ─────────────────────────────────────────────────
def show_single(image_path, model):
    label, confidence, prob, img = predict_image(model, image_path)
    if label is None:
        return

    # Color — green for safe, red for unsafe
    color = "#e74c3c" if label == "UNSAFE" else "#2ecc71"

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    ax.imshow(img)
    ax.axis("off")

    # Title with prediction
    ax.set_title(
        f"{label}  —  {confidence}% confidence\n"
        f"(Raw probability: {prob:.4f})",
        fontsize=14,
        fontweight='bold',
        color=color,
        pad=15
    )

    # Colored border around image
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4)

    plt.tight_layout()
    plt.show()

    print(f"  Image    : {os.path.basename(image_path)}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence}%")
    print(f"  Raw prob  : {prob:.4f}")


# ─────────────────────────────────────────────────
# SHOW FOLDER — grid of images with predictions
# ─────────────────────────────────────────────────
def show_folder(folder_path, model, max_display=16):
    if not os.path.exists(folder_path):
        print(f"  ❌ Folder not found: {folder_path}")
        return

    # Get all images
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(valid_ext)]

    if not images:
        print(f"  ⚠️  No images found in: {folder_path}")
        return

    print(f"  Found {len(images)} images in folder")
    print(f"  Running predictions...\n")

    results = []
    safe_count   = 0
    unsafe_count = 0

    for filename in images:
        path = os.path.join(folder_path, filename)
        label, confidence, prob, img = predict_image(model, path)
        if label is None:
            continue
        results.append((filename, label, confidence, prob, img))
        if label == "SAFE":
            safe_count += 1
        else:
            unsafe_count += 1
        print(f"  {filename:<30} → {label:<8} ({confidence}%)")

    # ── Summary ──
    print(f"\n  ========== FOLDER SUMMARY ==========")
    print(f"  Total    : {len(results)}")
    print(f"  Safe     : {safe_count}")
    print(f"  Unsafe   : {unsafe_count}")
    print(f"  =====================================\n")

    # ── Display grid (up to max_display images) ──
    display_results = results[:max_display]
    n     = len(display_results)
    cols  = 4
    rows  = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    axes = np.array(axes).flatten()

    for i, (filename, label, confidence, prob, img) in enumerate(display_results):
        ax    = axes[i]
        color = "#e74c3c" if label == "UNSAFE" else "#2ecc71"
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{label}\n{confidence}%",
            fontsize=10,
            fontweight='bold',
            color=color
        )
        # Colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Folder Predictions — Safe: {safe_count} | Unsafe: {unsafe_count}",
        fontsize=14,
        fontweight='bold',
        y=1.01
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import math

    print("=" * 55)
    print("   CARTOON SAFETY PREDICTOR")
    print("=" * 55)

    # Load model once
    model = load_model()

    # ── Single image ──
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        print("── SINGLE IMAGE PREDICTION ──")
        show_single(IMAGE_PATH, model)

    # ── Folder ──
    if FOLDER_PATH and os.path.exists(FOLDER_PATH):
        print("\n── FOLDER PREDICTION ──")
        show_folder(FOLDER_PATH, model)

    print("\n✅ Done!")