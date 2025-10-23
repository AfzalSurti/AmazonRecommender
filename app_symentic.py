from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

app = Flask(__name__)

df = pd.read_csv("models/products_master.csv")
embeddings = np.load("models/embeddings.npy")
model = SentenceTransformer("models/sentence_model")
titles = df["Title"].astype(str).tolist()

def correct_spelling(query):
    match, score, _ = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if score >= 70:
        return match
    return query

# Main recommend function, returns full list of best matches sorted by similarity
def get_recommendations(query, top_n=300):
    query = correct_spelling(query)
    # Category exact match: all items in that category
    if query.capitalize() in df["Category"].unique():
        results = df[df["Category"] == query.capitalize()].copy()
    else:
        query_vec = model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(query_vec, embeddings)[0]
        top_indices = sims.argsort()[::-1]  # all sorted by similarity
        results = df.iloc[top_indices][["Title","Price (INR)","Image URL","Product Link","Category","Description"]].copy()
    results["idx"] = results.index
    # Only top_n if desired, otherwise return all
    return results.head(top_n)

@app.route("/", methods=["GET","POST"])
def home():
    query = request.form.get("product_name", "") if request.method == "POST" else request.args.get("product_name", "")
    page = int(request.args.get("page", 1))
    per_page = 25
    recommendations = None
    total = 0
    if query:
        all_matches = get_recommendations(query, top_n=1000)
        total = len(all_matches)
        # page slicing
        start = (page - 1) * per_page
        end = start + per_page
        recommendations = all_matches.iloc[start:end]
    # Generate paginator info
    total_pages = max(1, (total + per_page - 1) // per_page)
    prev_page = page - 1 if page > 1 else None
    next_page = page + 1 if page < total_pages else None
    return render_template("index.html", query=query, recommendations=recommendations, page=page,
                          total_pages=total_pages, prev_page=prev_page, next_page=next_page, total=total)

@app.route("/product/<int:idx>", methods=["GET"])
def product_detail(idx):
    if idx < 0 or idx >= len(df):
        return redirect(url_for('home'))
    item = df.iloc[idx]
    return render_template("product.html", product=item)

if __name__ == "__main__":
    app.run(debug=True)
