# ------------------------------------------
# Script: 2_train_model_semantic.py
# Purpose: Train a semantic (embedding-based) recommender
# ------------------------------------------

import pandas as pd, numpy as np, os, pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/products_master.csv")
df["Description"] = df["Description"].fillna("")
df["text"] = df["Title"].astype(str) + " " + df["Description"].astype(str) + " " + df["Category"].astype(str)

# 2️ Load SentenceTransformer model
# all-MiniLM-L6-v2 is lightweight and great balance of speed & accuracy
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️ Compute embeddings
print("Encoding all product texts... this may take a few minutes on large datasets.")
embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True)
print(f"✅ Generated embeddings: {embeddings.shape}")

# 4️ Save artifacts
os.makedirs("models", exist_ok=True)
np.save("models/embeddings.npy", embeddings)
df.to_csv("models/products_master.csv", index=False)
model.save("models/sentence_model")
print("✅ Semantic model saved (embeddings + dataset).")
