import pandas as pd

base_path = "D:/PythonYouTube/"
df = pd.read_csv(base_path + "combined_dataset.csv")

print("Before:", len(df))

# Remove duplicate text rows
df_clean = df.drop_duplicates(subset=["text"])

print("After:", len(df_clean))
print(df_clean["label"].value_counts())

# Save cleaned dataset
df_clean.to_csv(base_path + "UniqueCombinedDataset.csv", index=False)

print("Saved: UniqueCombinedDataset.csv")
