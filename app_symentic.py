import os

from flask import Flask, jsonify, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "").strip() or "llama-3.1-8b-instant"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

df = pd.read_csv("models/products_master.csv")
embeddings = np.load("models/embeddings.npy")
model = SentenceTransformer("models/sentence_model")
titles = df["Title"].astype(str).tolist()


def call_groq_chat(system_prompt, user_prompt, max_tokens=220, temperature=0.2):
    if not GROQ_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def optimize_query_with_groq(raw_query):
    if not raw_query:
        return raw_query

    optimized = call_groq_chat(
        system_prompt=(
            "You optimize e-commerce product search queries. "
            "Return only one concise query string with no extra text."
        ),
        user_prompt=(
            "Rewrite this search into a clean Amazon-style product query while preserving intent. "
            f"Query: {raw_query}"
        ),
        max_tokens=40,
        temperature=0,
    )

    if not optimized:
        return raw_query
    return optimized.split("\n")[0].strip(' "') or raw_query

def correct_spelling(query):
    match, score, _ = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if score >= 70:
        return match
    return query


def format_product_context(item):
    title = str(item.get("Title", ""))
    category = str(item.get("Category", ""))
    price = str(item.get("Price (INR)", ""))
    desc = str(item.get("Description", ""))
    link = str(item.get("Product Link", ""))
    return (
        f"Title: {title}\n"
        f"Category: {category}\n"
        f"Price (INR): {price}\n"
        f"Description: {desc}\n"
        f"Product Link: {link}"
    )


def format_results_context(query, results_df, max_items=12):
    lines = [f"Original user query: {query}"]
    for rank, (_, row) in enumerate(results_df.head(max_items).iterrows(), start=1):
        lines.append(
            f"{rank}. {row.get('Title', '')} | Category: {row.get('Category', '')} | "
            f"Price (INR): {row.get('Price (INR)', '')}"
        )
    return "\n".join(lines)


def analyze_product_text(item):
    product_context = format_product_context(item)
    analysis = call_groq_chat(
        system_prompt=(
            "You are an expert shopping assistant. Analyze a product and provide concise, practical guidance. "
            "Structure answer in bullets with sections: Summary, Pros, Considerations, Best For, Buying Tip."
        ),
        user_prompt=product_context,
        max_tokens=350,
        temperature=0.3,
    )

    if analysis:
        return analysis

    return (
        "Summary:\n"
        f"- {item.get('Title', 'Product')} in {item.get('Category', 'N/A')} priced at ₹{item.get('Price (INR)', 'N/A')}.\n"
        "Pros:\n- Good match for the searched category and price filtering.\n"
        "Considerations:\n- Review ratings, warranty, and seller details on Amazon before purchase.\n"
        "Best For:\n- Shoppers comparing similar products in this category.\n"
        "Buying Tip:\n- Open the Amazon link to verify latest price and offers."
    )

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
    optimized_query = ""
    page = int(request.args.get("page", 1))
    per_page = 25
    recommendations = None
    all_matches = None
    total = 0
    if query:
        optimized_query = optimize_query_with_groq(query)
        all_matches = get_recommendations(optimized_query, top_n=1000)
        total = len(all_matches)
        # page slicing
        start = (page - 1) * per_page
        end = start + per_page
        recommendations = all_matches.iloc[start:end]
    # Generate paginator info
    total_pages = max(1, (total + per_page - 1) // per_page)
    prev_page = page - 1 if page > 1 else None
    next_page = page + 1 if page < total_pages else None
    chat_context = ""
    if query and all_matches is not None and not all_matches.empty:
        chat_context = format_results_context(query=optimized_query or query, results_df=all_matches)

    return render_template(
        "index.html",
        query=query,
        optimized_query=optimized_query,
        recommendations=recommendations,
        page=page,
        total_pages=total_pages,
        prev_page=prev_page,
        next_page=next_page,
        total=total,
        chat_context=chat_context,
    )

@app.route("/product/<int:idx>", methods=["GET"])
def product_detail(idx):
    if idx < 0 or idx >= len(df):
        return redirect(url_for('home'))
    item = df.iloc[idx]
    chat_context = format_product_context(item)
    return render_template("product.html", product=item, idx=idx, chat_context=chat_context)


@app.route("/api/analyze-product/<int:idx>", methods=["GET"])
def analyze_product(idx):
    if idx < 0 or idx >= len(df):
        return jsonify({"error": "Invalid product index"}), 400

    item = df.iloc[idx]
    analysis = analyze_product_text(item)
    return jsonify({"analysis": analysis})


@app.route("/api/chat", methods=["POST"])
def chat_with_context():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    context_text = (payload.get("context") or "").strip()

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    system_prompt = (
        "You are a product shopping assistant. "
        "Only answer using the provided context. "
        "If the answer is not in context, clearly say that and suggest what to check. "
        "Be concise and helpful."
    )
    user_prompt = (
        f"Context:\n{context_text if context_text else 'No product context provided.'}\n\n"
        f"User question:\n{user_message}"
    )

    answer = call_groq_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=320,
        temperature=0.25,
    )

    if not answer:
        answer = "I could not reach the AI service right now. Please try again in a moment."

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)