# 🛍️ Amazon-Like Semantic E‑Commerce Recommender System

## 📘 Project Overview
A full‑stack **AI‑powered e‑commerce search and recommendation engine** inspired by Amazon 🧠.

It scrapes real product data using Selenium & BeautifulSoup, builds a **semantic understanding model** (using Sentence‑BERT), and serves intelligent product recommendations through a Flask backend and a Bootstrap‑based frontend that visually resembles Amazon.

---

## 🚀 Features
- **Web Scraping:** Automated product extraction from Amazon using Selenium + BeautifulSoup.
- **Data Engineering:** Cleaned & merged all category‑wise CSVs into a single master dataset.
- **Semantic Search:** Deep learning via SentenceTransformer for intelligent text–product matching.
- **Smart Ranking:** Cosine similarity to find contextually closest products.
- **Frontend Design:** HTML + CSS + Bootstrap for a clean, Amazon‑style interface.
- **Flask Backend:** REST‑based architecture to serve real‑time search results.
- **Modular Design:** Two‑script architecture for ease of debugging and updates.

---

## 🧠 Model & Training

### Model Used
`SentenceTransformer` → **all‑MiniLM‑L6‑v2**

### Why This Model?
- Compact and fast transformer model by HuggingFace.
- Learns semantic meaning — “phones under 50000” ≈ “mobiles below 50k.”
- Built for sentence similarity and recommendation tasks.

### Algorithm Used
**Cosine Similarity** for ranking embeddings:
\[
\text{similarity}(A,B) = \frac{A \cdot B}{||A|| \times ||B||}
\]

### Model Workflow
1. Each product title → vector embedding (1024‑dimensional numeric representation).
2. User query → encoded using the same model.
3. Cosine similarity calculated between query and all products.
4. Products sorted by relevance score.
5. Flask backend returns top K matching products to the frontend.

---

## 💻 Tech Stack

| Layer | Tools Used |
|--------|-------------|
| Web Scraping | Selenium, BeautifulSoup |
| Data Cleaning | pandas, NumPy |
| ML/NLP | SentenceTransformers, Scikit‑learn, RapidFuzz |
| Backend | Flask |
| Frontend | HTML, CSS, Bootstrap |
| Deployment | Render / PythonAnywhere (Free Tier) |

---

## ⚙️ How To Run the Project

### 1️⃣ Clone the Repository
git clone https://github.com/AfzalSurti/AmazonRecommender
cd Amazon-Web-Scrapper

### 2️⃣ Install Dependencies
pip install -r requirements.txt

project/
|──data/
├── models/
│ ├── products_master.csv
│ ├── embeddings.npy
│ └── sentence_model/ 
├── templates/
│ ├── index.html
│ └── product.html
├── app_symentic.py
├── requirements.txt
└── README.md


### 4️⃣ Run Flask App
python app_symentic.py


Your app will now run on:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔧 Important Notes
- Install **Flask** and **transformers** only — other dependencies will auto‑install.
- Run those **two core scripts** separately:  
  - The data/model preparation script  
  - The main Flask file `app_symentic.py`
- Avoid running the merger file directly (due to GitHub file size restrictions).
- After setup, only one command is needed for execution: python app_symentic.py

---

## 🏗️ Future Enhancements
- Add sub‑categories like:
- Mobiles → iPhones, Samsung, Realme
- Clothing → Men‑T‑Shirts, Women‑Tops
- Integrate Flipkart, Meesho, and Myntra scrapers for cross‑platform results.
- Build advanced query understanding with NER or fine‑tuned T5.
- Add online deployment with persistent DB.

---

## 📄 Dependencies
Read  `requirements.txt` 

---

## 🧠 What You’ll Learn
- How modern e‑commerce recommendation systems understand intent.
- How to apply transformer models for semantic understanding.
- Building scalable full‑stack AI products with Flask.

---

## ✨ Author
Developed by **Afzal Nasirbhai Surti**

LinkedIn:  https://www.linkedin.com/in/afzal-surti-9904b2287/
GitHub: https://github.com/AfzalSurti

---

🛒 *A real‑world AI + Web Development project made to demonstrate hands‑on NLP for product recommendation.*

