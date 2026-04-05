import os
import pickle
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from knowledge_base.math_docs import docs

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_DIR        = "data"
INDEX_PATH      = os.path.join(DATA_DIR, "langchain_faiss_index")

# ── Load embedding model once ─────────────────────────────────────────────────
print("[RAG] Loading embedding model via LangChain...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("[RAG] Embedding model ready.")

# ── Convert docs to LangChain Document format ─────────────────────────────────
def get_langchain_docs() -> list[Document]:
    return [
        Document(
            page_content=f"{d['title']}\n{d['content']}",
            metadata={
                "title": d["title"],
                "topic": d["topic"],
                "content": d["content"]
            }
        )
        for d in docs
    ]

# ── Build index ───────────────────────────────────────────────────────────────
def build_index():
    print("[RAG] Building FAISS index using LangChain...")
    langchain_docs = get_langchain_docs()

    vectorstore = FAISS.from_documents(langchain_docs, embeddings)

    os.makedirs(DATA_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)

    print(f"[RAG] Index built with {len(langchain_docs)} documents.")
    print(f"[RAG] Saved to: {INDEX_PATH}")

# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    results = retriever.invoke(query)

    return [
        {
            "title":   doc.metadata.get("title", "Unknown"),
            "topic":   doc.metadata.get("topic", "general"),
            "content": doc.metadata.get("content", doc.page_content),
            "score":   0.0
        }
        for doc in results
    ]

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_index()
    print("\n--- Test Retrieval ---")
    results = retrieve("how to find roots of quadratic equation", top_k=3)
    for r in results:
        print(f"\n[{r['topic'].upper()}] {r['title']}")
        print(r['content'][:150], "...")