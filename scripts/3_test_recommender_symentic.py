# ------------------------------------------
# Script: 3_test_recommender_semantic.py
# ------------------------------------------

import pandas as pd, numpy as np, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# Load artifacts
print("Loading model...")
df = pd.read_csv("models/products_master.csv")
embeddings = np.load("models/embeddings.npy")
model = SentenceTransformer("models/sentence_model")

# Pre-compute product titles for fuzzy matching
titles = df["Title"].astype(str).tolist()

def correct_spelling(query):
    match, score, _ = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if score >= 70:      # accept if reasonably close
        print(f"🔍 Corrected '{query}' → '{match}' (score={score})")
        return match
    return query

def get_recommendations(query, num=5):
    # correct typos
    query_corr = correct_spelling(query)

    # encode search text
    query_vec = model.encode([query_corr], normalize_embeddings=True)

    # compute similarity
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sims.argsort()[-num-1:][::-1][1:]
    return df.iloc[top_indices][["Title","Price (INR)","Image URL","Product Link"]]

query = input("Enter product name: ").strip()
recs = get_recommendations(query, 5)
print(f"\nRecommendations for '{query}':\n")
print(recs.to_string(index=False))
