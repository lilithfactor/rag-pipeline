# Local RAG Pipeline

Chat with your own documents — **private, on-device, no API keys.** A retrieval-augmented
generation (RAG) pipeline that ingests your files, indexes them in a vector store, retrieves
the most relevant passages for a query, and generates a grounded answer using a **local
[llama.cpp](https://github.com/ggml-org/llama.cpp) model** instead of a cloud LLM.

## How it works

```
PDFs & notes  →  Chunk + Embed  →  Vector store  →  Retrieve  →  Local LLM  →  Grounded answer
                (sentence-transformers)  (FAISS / Chroma)        (llama.cpp, OpenAI-compatible)
```

1. **Ingest** — load PDF, TXT, CSV, Excel, Word, and JSON files ([src/data_loader.py](src/data_loader.py)).
2. **Embed** — split documents into overlapping chunks and embed them with
   `all-MiniLM-L6-v2` ([src/embedding.py](src/embedding.py)).
3. **Store** — persist embeddings in a FAISS index ([src/vectorstore.py](src/vectorstore.py));
   the notebook variant uses ChromaDB.
4. **Retrieve + answer** — embed the query, pull the top-k chunks by similarity, and summarize
   them with the local LLM ([src/search.py](src/search.py)).

## Tech stack

`llama.cpp` · `FAISS` · `ChromaDB` · `sentence-transformers` · `LangChain`

## Prerequisites

- Python **3.14+**
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local inference:
  ```bash
  brew install llama.cpp
  ```

## Setup

```bash
# install dependencies
uv sync

# (optional) configure the inference endpoint — defaults work out of the box
cp .env.example .env
```

## Run the local LLM server

The pipeline talks to llama.cpp over its OpenAI-compatible API. Start the server first —
the model (~1 GB) downloads automatically on the first launch:

```bash
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8080 -ngl 99 -c 8192 --parallel 4
```

No API key is required. To point at a different host/port, set `LLAMA_BASE_URL` (see
[.env.example](.env.example)).

## Usage

Drop your documents into `data/`, then:

**End-to-end module** (FAISS — builds the index on first run, reuses it after):

```bash
uv run python -m src.search
```

**Notebook** (ChromaDB — full walkthrough from ingestion to answer):

```bash
uv run jupyter lab notebook/pdf_loader.ipynb
```

**Example script:**

```bash
uv run python app.py
```

```python
from src.search import RAGSearch

rag = RAGSearch()
print(rag.search_and_summarize("What is the attention mechanism?", top_k=3))
```

## Project structure

```
src/
  data_loader.py   # load PDF/TXT/CSV/Excel/Word/JSON → LangChain documents
  embedding.py     # chunking + sentence-transformer embeddings
  vectorstore.py   # FAISS index: build, persist, query
  search.py        # RAGSearch: retrieve context + local-LLM summarization
notebook/
  pdf_loader.ipynb # end-to-end RAG walkthrough (ChromaDB)
data/              # your source documents (gitignored vector stores rebuild from here)
```

## Notes

- Generated vector stores (`faiss_store/`, `data/vector_store/`) are gitignored — they rebuild
  from `data/`.
- Run modules as packages from the project root (`python -m src.search`), not as scripts
  (`python src/search.py`), so the `src.` imports resolve.
