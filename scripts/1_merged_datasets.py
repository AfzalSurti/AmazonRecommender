# ------------------------------------------
# Script: 1_merge_datasets.py
# Purpose: merge all Amazon category CSVs
# Output: data/processed/products_master.csv
# ------------------------------------------

import pandas as pd
import glob, os

path = "data/raw/"
files = glob.glob(os.path.join(path, "*_products.csv"))

frames = []
for f in files:
    df = pd.read_csv(f)
    if "Category" not in df.columns:
        # fallback: extract from filename if missing
        cat = os.path.basename(f).split("_")[0].capitalize()
        df["Category"] = cat
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
merged = merged.drop_duplicates(subset="Title", keep="first")
merged["Description"] = merged["Description"].fillna("")
os.makedirs("data/processed", exist_ok=True)
merged.to_csv("data/processed/products_master.csv", index=False)

print(f"✅ Merged {len(files)} files → {len(merged)} total products.")
print("   Saved to data/processed/products_master.csv")