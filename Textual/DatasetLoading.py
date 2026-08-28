import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
import re
import html
import contractions
import warnings
import torch
warnings.filterwarnings("ignore")

# GPU CHECK
print("=" * 60)
print("GPU CHECK")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f" GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version       : {torch.version.cuda}")
    print(f"   Total VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   Available VRAM     : {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("  No GPU detected — falling back to CPU")
    print("   Make sure your PyTorch is installed with CUDA support")

print(f"   Device in use      : {device}")

DATASET_PATH = r"D:\PythonYouTube\UniqueCombinedDataset.csv"
OUTPUT_DIR   = r"D:\PythonYouTube\Textual"
TEXT_COL     = "text"
LABEL_COL    = "label"
MAX_TOKENS   = 512

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    if not text:
        return ""
    try:
        text = contractions.fix(str(text))
    except IndexError:
        text = str(text)
    text = html.unescape(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#  Load Dataset
print("=" * 60)
print("SECTION 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded successfully")
print(f"   Shape             : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Columns           : {list(df.columns)}")

# Label counts immediately after loading
label_counts = df[LABEL_COL].value_counts().sort_index()
label_pcts   = df[LABEL_COL].value_counts(normalize=True).sort_index() * 100
print(f"\n   Label Distribution:")
print(f"   ├── 0 (safe)    : {label_counts[0]:,} rows  ({label_pcts[0]:.1f}%)")
print(f"   └── 1 (harmful) : {label_counts[1]:,} rows  ({label_pcts[1]:.1f}%)")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

#  Column & Type Checks
print("\n" + "=" * 60)
print("SECTION 2: COLUMN & DATA TYPE CHECKS")
print("=" * 60)

# Check required columns exist
for col in [TEXT_COL, LABEL_COL]:
    if col in df.columns:
        print(f" Column '{col}' found — dtype: {df[col].dtype}")
    else:
        print(f" Column '{col}' NOT FOUND — please rename your column!")

# Ensure label is integer
if df[LABEL_COL].dtype != int:
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    print(f"⚠️  Label column converted to int")
else:
    print(f"✅ Label column is already integer type")

# Unique label values
unique_labels = sorted(df[LABEL_COL].unique().tolist())
print(f"   Unique labels : {unique_labels}")
if unique_labels == [0, 1]:
    print(f" Binary labels confirmed (0 = safe, 1 = harmful)")
else:
    print(f" Unexpected labels — expected [0, 1]")

# SECTION 3: Null / Missing Value Check
print("\n" + "=" * 60)
print("SECTION 3: NULL / MISSING VALUE CHECK")
print("=" * 60)

null_text  = df[TEXT_COL].isnull().sum()
null_label = df[LABEL_COL].isnull().sum()
empty_text = (df[TEXT_COL].astype(str).str.strip() == "").sum()

print(f"   Null text values   : {null_text:,}")
print(f"   Empty text strings : {empty_text:,}")
print(f"   Null label values  : {null_label:,}")

if null_text + empty_text + null_label == 0:
    print(" No missing values detected")
else:
    print("  Missing values found — dropping them now...")
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    print(f"   Rows after cleaning: {len(df):,}")

#  Apply clean_text
print("\n" + "=" * 60)
print("SECTION 4: APPLYING clean_text() — INSPECTION ONLY")
print("=" * 60)

working_df = df.copy()

print("   Applying clean_text() to working copy... (may take a moment)")
working_df[TEXT_COL] = working_df[TEXT_COL].apply(clean_text)

# Check how many rows would become empty after cleaning
before  = len(working_df)
would_drop = (working_df[TEXT_COL].str.strip() == "").sum()

print(f"✅ clean_text() applied to working copy (original dataset untouched)")
print(f"   Rows that would become empty after cleaning: {would_drop:,}")
print(f"   Rows that would remain                     : {before - would_drop:,}")
print(f"   ℹ️  These counts are for inspection only — original file not changed")

# SECTION 5: Class Balance Check

print("\n" + "=" * 60)
print("SECTION 5: CLASS BALANCE CHECK")
print("=" * 60)

counts = df[LABEL_COL].value_counts().sort_index()
percents = df[LABEL_COL].value_counts(normalize=True).sort_index() * 100

print(f"   Label 0 (safe)   : {counts[0]:,}  ({percents[0]:.1f}%)")
print(f"   Label 1 (harmful): {counts[1]:,}  ({percents[1]:.1f}%)")

ratio = max(percents) / min(percents)
if ratio <= 1.5:
    print(" Dataset is well balanced — no resampling needed")
elif ratio <= 2.5:
    print(" Mild imbalance — consider using class weights during training")
else:
    print(" Severe imbalance — strongly recommend oversampling or class weights")

# Plot class distribution
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Safe (0)", "Harmful (1)"], [counts[0], counts[1]],
              color=["#4CAF50", "#F44336"], edgecolor="black", width=0.5)
for bar, pct in zip(bars, [percents[0], percents[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{pct:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_xlabel("Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("    Saved: class_distribution.png")

# ─────────────────────────────────────────────
# SECTION 6: Token Length Check
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6: TOKEN LENGTH CHECK (RoBERTa Tokenizer)")
print("=" * 60)

print("   Loading RobertaTokenizer... (downloads if first time)")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print(f"   Tokenizer loaded on: {device}")

print("   Tokenizing all texts to count token lengths...")
print("   (This may take 1-2 minutes for large datasets)")

# Sample up to 20k for speed if dataset is huge
sample_df = working_df.sample(min(20000, len(working_df)), random_state=42)

# Tokenize in batches on GPU for speed
def batch_tokenize_lengths(texts, tokenizer, batch_size=512):
    lengths = []
    texts = texts.tolist()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_tensors=None
        )
        lengths.extend([len(ids) for ids in encoded["input_ids"]])
    return lengths

token_lengths = pd.Series(
    batch_tokenize_lengths(sample_df[TEXT_COL], tokenizer)
)

print(f"\n   Token Length Statistics:")
print(f"   Min    : {token_lengths.min()}")
print(f"   Max    : {token_lengths.max()}")
print(f"   Mean   : {token_lengths.mean():.1f}")
print(f"   Median : {token_lengths.median():.1f}")
print(f"   95th % : {token_lengths.quantile(0.95):.1f}")
print(f"   99th % : {token_lengths.quantile(0.99):.1f}")

over_512 = (token_lengths > MAX_TOKENS).sum()
pct_over = over_512 / len(token_lengths) * 100
print(f"\n   Texts exceeding 512 tokens: {over_512:,} ({pct_over:.1f}%)")

if pct_over < 5:
    print(" Less than 5% exceed 512 — simple truncation is fine")
elif pct_over < 20:
    print("️  5-20% exceed 512 — truncation acceptable, mention in dissertation")
else:
    print(" Over 20% exceed 512 — consider chunking strategy")

# Plot token length distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(token_lengths, bins=50, color="#2196F3", edgecolor="black", alpha=0.8)
ax.axvline(x=512, color="red", linestyle="--", linewidth=2, label="512 token limit")
ax.set_title("Token Length Distribution (RoBERTa)", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Tokens")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "token_length_distribution.png"), dpi=150)
plt.close()
print("   📊 Saved: token_length_distribution.png")

# ─────────────────────────────────────────────
# SECTION 7: Train / Val / Test Split
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7: TRAIN / VAL / TEST SPLIT")
print("=" * 60)

train_df, temp_df = train_test_split(
    working_df, test_size=0.2, random_state=42, stratify=working_df[LABEL_COL]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df[LABEL_COL]
)

print(f"   Train : {len(train_df):,} rows ({len(train_df)/len(working_df)*100:.1f}%)")
print(f"   Val   : {len(val_df):,}  rows ({len(val_df)/len(working_df)*100:.1f}%)")
print(f"   Test  : {len(test_df):,}  rows ({len(test_df)/len(working_df)*100:.1f}%)")

# Confirm stratification worked
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    pct1 = split_df[LABEL_COL].mean() * 100
    print(f"   {split_name} — harmful%: {pct1:.1f}%  safe%: {100-pct1:.1f}%")

print(" Stratified split complete — class ratios preserved across splits")

# Save splits as separate files — original dataset is NEVER touched
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR,   "val_split.csv"),   index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR,  "test_split.csv"),  index=False)
print("    Saved: train_split.csv, val_split.csv, test_split.csv")
print(f"    Location: {OUTPUT_DIR}")
print("     Original UniqueCombinedDataset.csv was NOT modified")

# FINAL SUMMARY

print("\n" + "=" * 60)
print("FINAL SUMMARY — READY FOR ROBERTA TRAINING?")
print("=" * 60)
print(f"  Total samples        : {len(working_df):,}")
print(f"  Train / Val / Test   : {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"  Class balance        : {percents[0]:.1f}% safe / {percents[1]:.1f}% harmful")
print(f"  Texts over 512 tokens: {pct_over:.1f}% (will be truncated)")
print(f"  Tokenizer            : roberta-base")
print(f"  Max token length     : {MAX_TOKENS}")
print("\n✅ All checks complete — proceed to RoBERTa fine-tuning!")
print("=" * 60)