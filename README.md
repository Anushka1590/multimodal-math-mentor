# 🧮 Math Mentor
An end-to-end AI application that solves JEE-level math problems using RAG + Multi-Agent System + Memory.

## Architecture
```mermaid
graph TD
    A[User Input: Text/Image/Audio] --> B[Input Handler]
    B --> C[Parser Agent]
    C --> D{Needs Clarification?}
    D -->|Yes| E[HITL Review]
    D -->|No| F[Intent Router Agent]
    E --> F
    F --> G[Solver Agent]
    G --> H[RAG Pipeline]
    G --> I[Python + Sympy Calculator]
    H --> G
    I --> G
    G --> J[Verifier Agent]
    J --> K{Confident?}
    K -->|No| E
    K -->|Yes| L[Explainer Agent]
    L --> M[Final Answer + Explanation]
    M --> N[Memory Storage]
    O[User Feedback] --> N
```

## Setup
1. Clone the repository
2. Create virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and add your key:
```
   GROQ_API_KEY=your_groq_api_key_here
```
5. Build knowledge base index:
```bash
   python rag_pipeline.py
```
6. Run the app:
```bash
   streamlit run app.py
```

## Features
- Text, Image and Audio input
- 5-agent pipeline: Parser, Router, Solver, Verifier, Explainer
- RAG with FAISS vector store
- Exact math via Python + Sympy code execution
- Human-in-the-loop for low confidence answers
- Memory system with feedback learning

## Tech Stack
- LLM: Groq (LLaMA 3.3 70B)
- RAG: FAISS + sentence-transformers
- Symbolic Math: Sympy
- UI: Streamlit