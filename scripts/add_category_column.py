# ------------------------------------------
# Script: add_category_column.py
# Goal  : Automatically add 'Category' column
#         to each _products.csv file based on the filename
# ------------------------------------------

import pandas as pd
import glob, os

# path to all category CSVs
path = "data/raw/"
files = glob.glob(os.path.join(path, "*_products.csv"))

print(f"Found {len(files)} files to update:\n")

# loop through all CSV files
for file in files:
    filename = os.path.basename(file)
    # extract category name before "_products.csv"
    category = filename.replace("_products.csv", "").strip().capitalize()
    print(f"Processing: {filename}  →  Category = {category}")

    # read file
    df = pd.read_csv(file)

    # check if 'Category' column already exists
    if "Category" not in df.columns:
        df["Category"] = category
    else:
        df["Category"] = df["Category"].fillna(category)

    # overwrite same file or save to processed folder
    output_path = os.path.join("data/raw/", filename)
    df.to_csv(output_path, index=False, encoding="utf-8")

print("\n✅ Category column added successfully to all files.")
