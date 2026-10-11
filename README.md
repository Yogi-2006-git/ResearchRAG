# ResearchRAG

Local AI research paper assistant. Upload PDFs, search them with embeddings + FAISS, and generate answers with a local Ollama model. Built for resume and placement projects: free, open-source, and fully offline after the first model download.

## What it does

- Extracts text from research PDFs with PyMuPDF and keeps **filename + page number** on every chunk.
- Splits pages into overlapping word chunks.
- Embeds chunks with `all-MiniLM-L6-v2` and stores normalized vectors in a FAISS index.
- Retrieves the most similar passages for a question.
- Asks **Ollama `qwen2.5:3b`** to answer **only from retrieved evidence**.
- Shows the answer plus the source filename, page number, and passage text.

This project does **not** verify academic citations, DOIs, or bibliography entries. Source labels are retrieval metadata from the PDFs you uploaded.

## Architecture

```
PDF upload (Streamlit)
        |
        v
 papers/*.pdf
        |
        v
 document_processor.py     extract text, skip empty/scanned pages, chunk with overlap
        |
        v
 rag_engine.py             SentenceTransformer embeddings -> FAISS IndexFlatIP
        |                  persist to data/papers.index + data/chunks.json
        v
 retrieve top-k passages
        |
        v
 Ollama qwen2.5:3b         grounded answer from evidence only
        |
        v
 app.py                    Streamlit UI: question, answer, evidence
```

| File | Role |
| --- | --- |
| `app.py` | Streamlit interface |
| `document_processor.py` | PDF text extraction and chunking |
| `rag_engine.py` | Embeddings, FAISS, Ollama generation |
| `papers/` | Uploaded PDFs |
| `data/` | Saved FAISS index and chunk metadata |

## Requirements

- Windows
- Python 3.11
- [Ollama](https://ollama.com/download) installed and running
- Enough disk space for `all-MiniLM-L6-v2` (first embedding run) and `qwen2.5:3b`

## Setup (Windows PowerShell)

Run these commands yourself from the project folder. Do not skip Ollama.

```powershell
cd C:\Users\yoges\OneDrive\Desktop\ResearchRAG

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull qwen2.5:3b

streamlit run app.py
```

If PowerShell blocks the virtual environment script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

If a `.venv` already exists, skip `python -m venv .venv` and only activate it.

Confirm Ollama is running before asking questions:

```powershell
ollama list
```

You should see `qwen2.5:3b`. If Ollama is not running, start the Ollama app from the Start menu, then retry.

## How to use

1. Open the Streamlit app in the browser (usually `http://localhost:8501`).
2. Upload one or more research PDFs in the sidebar. They are saved under `papers/`.
3. Click **Process PDFs** to extract text, embed chunks, and build the FAISS index.
4. Ask a question. The app retrieves passages, then generates an answer with Ollama.
5. Expand **Retrieved evidence** to see filename, page number, similarity score, and passage text.

## Error handling

| Situation | What you will see |
| --- | --- |
| No PDFs in `papers/` | Processing notes say no PDFs were found |
| Scanned / image-only PDF | Warning that a page has little extractable text; OCR is not included |
| Empty document | No chunks created; processing stops with a clear message |
| Ollama not running | Connection error with the local URL and a start hint |
| Missing `qwen2.5:3b` | Message to run `ollama pull qwen2.5:3b` |

## Design notes

- Embeddings are L2-normalized. FAISS uses inner product, which is cosine similarity for unit vectors.
- Chunks never cross page boundaries, so page numbers stay correct.
- Generation is instructed to refuse when the retrieved text is not enough.
- No cloud LLM APIs and no hardcoded secrets.

## Project status

Suitable as a portfolio RAG demo. It is a local retrieval pipeline, not a citation checker or production literature-review tool.
