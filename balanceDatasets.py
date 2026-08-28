import pandas as pd
import re
import html
import contractions

# ---------- Text cleaning ----------
def clean_text(text):
    if not text:
        return ""
    text = contractions.fix(str(text))
    text = html.unescape(text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Paths ----------
base_path = "D:/PythonYouTube/"

files = [
    "NewToxigen.csv",
    "NewCyberHateXplain.csv",
    "NewJigsaw.csv",
    "NewMultilingualToxicityDetection.csv",
    "NewHateOffensiveBinarySafeUnsafe.csv",
    "Newcyberbullying_tweets.csv",
    "NewCyberBullyingNayan90.csv"
]

dfs = []

# ---------- Load & merge ----------
for file in files:
    path = base_path + file
    print("Loading:", path)
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "text" in df.columns and "label" in df.columns:
        df["text"] = df["text"].apply(clean_text)
        dfs.append(df[["text", "label"]])

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.dropna(inplace=True)

print("\nTotal rows:", len(combined_df))
print("\nLabel distribution:")
print(combined_df["label"].value_counts())

# ---------- Save ----------
output_path = base_path + "combined_dataset.csv"
combined_df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)


