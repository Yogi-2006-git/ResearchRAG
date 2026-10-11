"""
ResearchRAG — local Streamlit UI for asking questions over research PDFs.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from document_processor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    process_pdf_folder,
    save_uploaded_pdfs,
)
from rag_engine import DEFAULT_LLM_MODEL, DEFAULT_TOP_K, RAGEngine

ROOT = Path(__file__).resolve().parent
PAPERS_DIR = ROOT / "papers"
DATA_DIR = ROOT / "data"


def main() -> None:
    st.set_page_config(
        page_title="ResearchRAG",
        page_icon="📄",
        layout="wide",
    )

    _init_state()
    engine: RAGEngine = st.session_state.engine

    st.title("ResearchRAG")
    st.caption("Local AI research paper assistant. No paid APIs and no API keys.")

    with st.sidebar:
        _render_sidebar(engine)

    ollama_status = engine.check_ollama()
    if ollama_status["ok"]:
        st.success(ollama_status["message"])
    elif ollama_status["connected"]:
        st.warning(ollama_status["message"])
    else:
        st.error(ollama_status["message"])

    if engine.is_ready:
        st.info(
            f"Index ready: {len(engine.chunks)} chunks from "
            f"{_unique_filenames(engine)} PDF source(s)."
        )
    else:
        st.warning("No papers indexed yet. Upload PDFs in the sidebar, then click Process PDFs.")

    st.subheader("Ask a question")
    question = st.text_area(
        "Question",
        placeholder="Example: What evaluation datasets does this paper use?",
        label_visibility="collapsed",
        height=100,
    )

    col_ask, col_clear = st.columns([1, 1])
    with col_ask:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear answer", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

    if ask_clicked:
        _handle_question(engine, question, ollama_status)

    result = st.session_state.last_result
    if result:
        _render_result(result)

    st.divider()
    st.markdown(
        "Retrieved sources show the **PDF filename and page number** of matching "
        "chunks. ResearchRAG does **not** verify bibliographic citations, DOIs, "
        "or whether a paper is correctly attributed beyond that retrieval metadata."
    )


def _init_state() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if "engine" not in st.session_state:
        engine = RAGEngine(data_dir=DATA_DIR, llm_model=DEFAULT_LLM_MODEL)
        engine.load()
        st.session_state.engine = engine

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "process_warnings" not in st.session_state:
        st.session_state.process_warnings = []


def _render_sidebar(engine: RAGEngine) -> None:
    st.header("Library")
    uploaded = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are saved into the papers/ folder.",
    )

    if uploaded:
        saved = save_uploaded_pdfs(uploaded, PAPERS_DIR)
        if saved:
            st.success(f"Saved {len(saved)} PDF(s) to papers/.")

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if pdfs:
        st.write("PDFs in papers/:")
        for pdf in pdfs:
            st.caption(f"• {pdf.name}")
    else:
        st.caption("No PDFs in papers/ yet.")

    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (words)", 150, 800, DEFAULT_CHUNK_SIZE, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 20, 200, DEFAULT_CHUNK_OVERLAP, 10)

    st.subheader("Retrieval")
    top_k = st.slider("Passages to retrieve", 1, 10, DEFAULT_TOP_K)
    st.session_state.top_k = top_k

    if st.button("Process PDFs", type="primary", use_container_width=True):
        _process_library(engine, chunk_size, chunk_overlap)

    if st.session_state.process_warnings:
        with st.expander("Processing notes", expanded=True):
            for warning in st.session_state.process_warnings:
                st.write(f"- {warning}")

    st.subheader("Local models")
    st.code("Embeddings: all-MiniLM-L6-v2\nLLM: qwen2.5:3b via Ollama", language="text")


def _process_library(engine: RAGEngine, chunk_size: int, chunk_overlap: int) -> None:
    with st.spinner("Extracting text, creating chunks, and building the FAISS index..."):
        try:
            chunks, warnings = process_pdf_folder(
                PAPERS_DIR,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.process_warnings = warnings
            if not chunks:
                st.error("No searchable text was created. Check the processing notes.")
                return
            engine.build_index(chunks)
            st.session_state.last_result = None
            st.success(f"Indexed {len(chunks)} chunks.")
        except Exception as exc:
            st.error(f"Failed to process PDFs: {exc}")


def _handle_question(engine: RAGEngine, question: str, ollama_status: dict) -> None:
    if not question.strip():
        st.warning("Enter a question first.")
        return
    if not engine.is_ready:
        st.warning("Process at least one PDF before asking questions.")
        return
    if not ollama_status["ok"]:
        st.session_state.last_result = {
            "question": question,
            "answer": "",
            "retrieved": [],
            "error": ollama_status["message"],
        }
        return

    with st.spinner("Retrieving passages and generating an answer..."):
        try:
            retrieved = engine.retrieve(question, top_k=st.session_state.get("top_k", DEFAULT_TOP_K))
            generation = engine.generate_answer(question, retrieved)
            st.session_state.last_result = {
                "question": question,
                "answer": generation["answer"],
                "retrieved": retrieved,
                "error": generation["error"],
            }
        except Exception as exc:
            st.session_state.last_result = {
                "question": question,
                "answer": "",
                "retrieved": [],
                "error": str(exc),
            }


def _render_result(result: dict) -> None:
    st.subheader("Answer")
    if result.get("error"):
        st.error(result["error"])
        return

    st.write(result.get("answer") or "No answer was generated.")

    retrieved = result.get("retrieved") or []
    st.subheader("Retrieved evidence")
    if not retrieved:
        st.write("No matching passages were retrieved.")
        return

    for i, item in enumerate(retrieved, start=1):
        title = f"[{i}] {item['filename']} — page {item['page_number']} (similarity {item['score']:.3f})"
        with st.expander(title, expanded=(i == 1)):
            st.write(item["text"])


def _unique_filenames(engine: RAGEngine) -> int:
    return len({chunk.filename for chunk in engine.chunks})


if __name__ == "__main__":
    main()
