import os
import time

from flask import Flask, jsonify, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import requests


def load_env_file_fallback(path=".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    load_env_file_fallback()

LAST_GROQ_ERROR = ""

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "").strip() or "llama-3.1-8b-instant"
MODEL_FALLBACK_NAME = os.getenv("MODEL_FALLBACK_NAME", "").strip() or "llama-3.1-70b-versatile"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

df = pd.read_csv("models/products_master.csv")
embeddings = np.load("models/embeddings.npy")
model = SentenceTransformer("models/sentence_model")
titles = df["Title"].astype(str).tolist()


def call_groq_chat(system_prompt, user_prompt, max_tokens=220, temperature=0.2):
    global LAST_GROQ_ERROR

    if not GROQ_API_KEY:
        LAST_GROQ_ERROR = "Missing GROQ_API_KEY."
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    base_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    def extract_answer(data):
        choices = data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                elif isinstance(part, str) and part.strip():
                    parts.append(part.strip())
            return "\n".join(parts).strip()

        text_fallback = choices[0].get("text")
        if isinstance(text_fallback, str):
            return text_fallback.strip()

        return ""

    try:
        candidate_models = []
        for model_name in [MODEL_NAME, MODEL_FALLBACK_NAME]:
            if model_name and model_name not in candidate_models:
                candidate_models.append(model_name)

        for model_name in candidate_models:
            for attempt in range(2):
                current_payload = dict(base_payload)
                current_payload["model"] = model_name
                if attempt == 1:
                    current_payload["temperature"] = 0
                    current_payload["max_tokens"] = max(max_tokens, 96)

                response = requests.post(GROQ_CHAT_URL, headers=headers, json=current_payload, timeout=25)
                response.raise_for_status()
                data = response.json()
                answer = extract_answer(data)

                if answer:
                    LAST_GROQ_ERROR = ""
                    return answer

                if attempt == 0:
                    time.sleep(0.6)

        LAST_GROQ_ERROR = "AI returned an empty reply from all configured models."
        return None
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        body_preview = ""
        if exc.response is not None and exc.response.text:
            body_preview = exc.response.text[:180]
        LAST_GROQ_ERROR = f"Groq HTTP {status_code}. {body_preview}".strip()
        return None
    except requests.exceptions.RequestException as exc:
        LAST_GROQ_ERROR = f"Groq network error: {exc}"
        return None
    except Exception as exc:
        LAST_GROQ_ERROR = f"Unexpected AI error: {exc}"
        return None


def optimize_query_with_groq(raw_query):
    if not raw_query:
        return raw_query

    optimized = call_groq_chat(
        system_prompt=(
            "You optimize e-commerce product search queries. "
            "Return only one concise query string with corrected spelling and key product attributes. "
            "Do not add explanation, markdown, or quotes."
        ),
        user_prompt=(
            "Rewrite this search into a clean Amazon-style product query while preserving intent. "
            "Keep it under 12 words. "
            f"Query: {raw_query}"
        ),
        max_tokens=120,
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


@app.get("/health")
def health():
    return jsonify({"status": "ok"})

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

    return_query = (request.args.get("product_name") or "").strip()
    return_page_raw = (request.args.get("page") or "1").strip()
    try:
        return_page = max(1, int(return_page_raw))
    except ValueError:
        return_page = 1

    item = df.iloc[idx]
    chat_context = format_product_context(item)
    return render_template(
        "product.html",
        product=item,
        idx=idx,
        chat_context=chat_context,
        return_query=return_query,
        return_page=return_page,
    )


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
        answer = (
            "AI reply is temporarily unavailable. "
            "Please try once again in a few seconds."
        )

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        use_reloader=False,
    )