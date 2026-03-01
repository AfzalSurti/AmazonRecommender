from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from threading import Lock
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

app = Flask(__name__)

df = pd.read_csv("models/products_master.csv")
embeddings = np.load("models/embeddings.npy")
model = SentenceTransformer("models/sentence_model")
titles = df["Title"].astype(str).tolist()
MEMORY_MAX_TURNS = 30
MEMORY_SIM_THRESHOLD = 0.35
MEMORY_STORE = defaultdict(lambda: deque(maxlen=MEMORY_MAX_TURNS))
MEMORY_LOCK = Lock()


def correct_spelling(query: str) -> str:
    match = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if not match:
        return query
    value, score, _ = match
    if score >= 70:
        return value
    return query


def embed_text(text: str) -> np.ndarray:
    return model.encode([text], normalize_embeddings=True)[0]


def summarize_results(results: pd.DataFrame, limit: int = 3) -> str:
    if results.empty:
        return "No results found."
    top_rows = results.head(limit)
    titles_text = ", ".join(top_rows["Title"].astype(str).tolist())
    categories_text = ", ".join(sorted(set(top_rows["Category"].astype(str).tolist())))
    return f"Top products: {titles_text}. Categories: {categories_text}."


def get_memory_hits(session_id: str, query_vec: np.ndarray, top_k: int = 3) -> list:
    if not session_id:
        return []

    with MEMORY_LOCK:
        turns = list(MEMORY_STORE.get(session_id, []))

    if not turns:
        return []

    scored = []
    for turn in turns:
        mem_vec = np.array(turn["embedding"], dtype=np.float32)
        sim = float(np.dot(query_vec, mem_vec))
        if sim >= MEMORY_SIM_THRESHOLD:
            scored.append((sim, turn))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits = []
    for sim, turn in scored[:top_k]:
        hits.append(
            {
                "similarity": round(sim, 4),
                "question": turn["question"],
                "answer_summary": turn["answer_summary"],
            }
        )
    return hits


def build_contextual_query(query: str, memory_hits: list, max_context_items: int = 2) -> str:
    if not memory_hits:
        return query

    context_bits = []
    for hit in memory_hits[:max_context_items]:
        context_bits.append(hit["question"])
        context_bits.append(hit["answer_summary"])
    context_text = " ".join(context_bits)
    return f"{query}. Previous context: {context_text}"


def store_memory_turn(session_id: str, question: str, answer_summary: str, query_vec: np.ndarray) -> None:
    if not session_id:
        return

    turn = {
        "question": question,
        "answer_summary": answer_summary,
        "embedding": query_vec.tolist(),
    }
    with MEMORY_LOCK:
        MEMORY_STORE[session_id].append(turn)


def get_recommendations(query: str, top_n: int = 300) -> pd.DataFrame:
    query = correct_spelling(query)
    if query.capitalize() in df["Category"].unique():
        results = df[df["Category"] == query.capitalize()].copy()
    else:
        query_vec = model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(query_vec, embeddings)[0]
        top_indices = sims.argsort()[::-1]
        results = df.iloc[top_indices][["Title", "Price (INR)", "Image URL", "Product Link", "Category", "Description"]].copy()
    results["idx"] = results.index
    return results.head(top_n)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    try:
        top_n = int(payload.get("top_n", 300))
    except (TypeError, ValueError):
        top_n = 300
    session_id = str(payload.get("session_id", "")).strip()
    try:
        memory_k = int(payload.get("memory_k", 3))
    except (TypeError, ValueError):
        memory_k = 3

    if not query:
        return jsonify({"results": []})

    top_n = max(1, min(top_n, 2000))
    memory_k = max(1, min(memory_k, 10))

    query_vec = embed_text(query)
    memory_hits = get_memory_hits(session_id, query_vec, top_k=memory_k)
    contextual_query = build_contextual_query(query, memory_hits)
    results = get_recommendations(contextual_query, top_n=top_n)

    answer_summary = summarize_results(results)
    store_memory_turn(session_id, query, answer_summary, query_vec)

    return jsonify(
        {
            "results": results.to_dict(orient="records"),
            "session_id": session_id,
            "memory_hits": memory_hits,
            "contextual_query": contextual_query,
        }
    )


@app.get("/memory/<session_id>")
def get_memory(session_id: str):
    with MEMORY_LOCK:
        turns = list(MEMORY_STORE.get(session_id, []))

    public_turns = [
        {"question": t["question"], "answer_summary": t["answer_summary"]} for t in turns
    ]
    return jsonify({"session_id": session_id, "turns": public_turns})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
