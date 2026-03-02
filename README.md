# 🧠 DocuMind-AI

A production-ready **Retrieval-Augmented Generation (RAG)** system — a Mini-ChatGPT that reads your documents, retrieves relevant context, and generates accurate answers.

## Features

- 📄 **Document Upload** — PDF, DOCX, TXT
- ✂️ **Smart Chunking** — configurable size + overlap
- 🔢 **Embedding Generation** — sentence-transformers (all-MiniLM-L6-v2)
- 🔍 **FAISS Vector Search** — millisecond-level similarity retrieval
- 🤖 **LLM Integration** — Groq API *or* local Ollama
- ⚡ **FastAPI Backend** — async, production-ready REST API
- 💬 **Chat UI** — elegant dark-mode interface with drag-and-drop upload

## Quick Start

```bash
# 1. Clone & enter
cd DocuMind-AI

# 2. Create virtual environment
python -m venv rag-env
rag-env\Scripts\activate   # Windows
# source rag-env/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Configure
copy .env.example .env
# Edit .env with your API keys

# 5. Run
uvicorn backend.main:app --reload

# 6. Open frontend
# Open frontend/index.html in your browser
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `groq` or `ollama` | `groq` |
| `GROQ_API_KEY` | Your Groq API key | — |
| `GROQ_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | Sentence-transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Characters per chunk | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `TOP_K` | Number of chunks to retrieve | `5` |

## Architecture

```
Upload → Extract Text → Chunk → Embed → FAISS Index
                                              ↓
Question → Embed Query → FAISS Search → Top-K Chunks → Prompt + LLM → Answer
```

## License

MIT
