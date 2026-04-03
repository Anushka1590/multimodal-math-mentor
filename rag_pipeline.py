import os
import json
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from knowledge_base.math_docs import docs

load_dotenv()

# ── Config ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # lightweight, runs fine on your laptop
DATA_DIR        = "data"
INDEX_PATH      = os.path.join(DATA_DIR, "faiss_index.bin")
DOCS_PATH       = os.path.join(DATA_DIR, "docs_store.pkl")

# ── Build index ─────────────────────────────────────────
def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding documents...")
    texts = [f"{d['title']}\n{d['content']}" for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Index built with {index.ntotal} documents.")
    print(f"Saved to: {INDEX_PATH}")

# ── Retrieve ─────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        stored_docs = pickle.load(f)

    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        doc = stored_docs[idx].copy()
        doc["score"] = float(distances[0][i])
        results.append(doc)
    return results
