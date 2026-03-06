import os
import json
import pickle
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

MEMORY_DIR        = "data/memory"
MEMORY_JSON_PATH  = os.path.join(MEMORY_DIR, "memory_store.json")
MEMORY_INDEX_PATH = os.path.join(MEMORY_DIR, "memory_index.bin")
MEMORY_VECS_PATH  = os.path.join(MEMORY_DIR, "memory_vectors.pkl")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

os.makedirs(MEMORY_DIR, exist_ok=True)


# ── Load embedding model once ────────────────────────────────────────────────
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── Load / Save memory store ─────────────────────────────────────────────────
def load_memory() -> list:
    if os.path.exists(MEMORY_JSON_PATH):
        with open(MEMORY_JSON_PATH, "r") as f:
            return json.load(f)
    return []


def save_memory(memories: list):
    with open(MEMORY_JSON_PATH, "w") as f:
        json.dump(memories, f, indent=2)


# ── Rebuild FAISS index from all memories ────────────────────────────────────
def rebuild_index(memories: list):
    if not memories:
        return

    embedder = get_embedder()
    texts = [m["problem_text"] for m in memories]
    vectors = embedder.encode(texts).astype("float32")

    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, MEMORY_INDEX_PATH)
    with open(MEMORY_VECS_PATH, "wb") as f:
        pickle.dump(vectors, f)


# ── Add a new memory entry ────────────────────────────────────────────────────
def add_memory(
    problem_text:    str,
    parsed_problem:  dict,
    solution:        str,
    explanation:     str,
    verification:    dict,
    feedback:        str  = "pending",
    input_type:      str  = "text"
):
    memories = load_memory()

    entry = {
        "id":             len(memories) + 1,
        "timestamp":      datetime.now().isoformat(),
        "input_type":     input_type,
        "problem_text":   problem_text,
        "topic":          parsed_problem.get("topic", "general"),
        "parsed_problem": parsed_problem,
        "solution":       solution,
        "explanation":    explanation,
        "is_correct":     verification.get("is_correct", True),
        "confidence":     verification.get("confidence", 0.0),
        "feedback":       feedback        # "correct" | "incorrect" | "pending"
    }

    memories.append(entry)
    save_memory(memories)
    rebuild_index(memories)

    print(f"[Memory] Saved problem #{entry['id']}: {problem_text[:60]}...")
    return entry["id"]


# ── Retrieve similar past problems ───────────────────────────────────────────
def retrieve_similar(query: str, top_k: int = 2) -> list:
    memories = load_memory()
    if not memories:
        return []

    if not os.path.exists(MEMORY_INDEX_PATH):
        rebuild_index(memories)

    embedder  = get_embedder()
    index     = faiss.read_index(MEMORY_INDEX_PATH)
    query_vec = embedder.encode([query]).astype("float32")

    # Only search up to number of stored memories
    k = min(top_k, len(memories))
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(memories):
            entry = memories[idx].copy()
            entry["similarity_score"] = float(distances[0][i])
            # Only return memories with positive feedback or verified correct
            if entry["feedback"] in ("correct", "pending") and entry["is_correct"]:
                results.append(entry)

    return results


# ── Update feedback on a memory entry ────────────────────────────────────────
def update_feedback(memory_id: int, feedback: str):
    """
    feedback: 'correct' or 'incorrect'
    """
    memories = load_memory()
    for m in memories:
        if m["id"] == memory_id:
            m["feedback"] = feedback
            break
    save_memory(memories)
    print(f"[Memory] Updated feedback for problem #{memory_id}: {feedback}")


# ── Get memory summary ────────────────────────────────────────────────────────
def get_memory_summary() -> dict:
    memories = load_memory()
    if not memories:
        return {"total": 0, "correct": 0, "incorrect": 0, "pending": 0}

    return {
        "total":     len(memories),
        "correct":   sum(1 for m in memories if m["feedback"] == "correct"),
        "incorrect": sum(1 for m in memories if m["feedback"] == "incorrect"),
        "pending":   sum(1 for m in memories if m["feedback"] == "pending"),
        "topics":    list(set(m["topic"] for m in memories))
    }


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing memory system...\n")

    # Add a test memory
    mem_id = add_memory(
        problem_text   = "If alpha and beta are roots of x^2 - 5x + 6 = 0, find alpha^2 + beta^2",
        parsed_problem = {"topic": "algebra", "variables": ["alpha", "beta"], "constraints": []},
        solution       = "Using Vieta's formulas: alpha+beta=5, alpha*beta=6. alpha^2+beta^2 = (alpha+beta)^2 - 2*alpha*beta = 25 - 12 = 13",
        explanation    = "Step-by-step using Vieta's formulas. Final answer: 13",
        verification   = {"is_correct": True, "confidence": 0.95},
        feedback       = "correct",
        input_type     = "text"
    )
    print(f"Saved with ID: {mem_id}")

    # Retrieve similar
    print("\nRetrieving similar problems to: 'roots of quadratic equation sum of squares'")
    similar = retrieve_similar("roots of quadratic equation sum of squares", top_k=2)
    for s in similar:
        print(f"  Found: {s['problem_text'][:60]}  (score={s['similarity_score']:.2f})")

    # Summary
    print("\nMemory Summary:")
    print(json.dumps(get_memory_summary(), indent=2))