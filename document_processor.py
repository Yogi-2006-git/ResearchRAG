"""
PDF extraction and chunking for ResearchRAG.

Each chunk keeps the source filename and page number so retrieval
can show where a passage came from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

# Pages with fewer characters than this are treated as empty or scanned.
MIN_PAGE_CHARS = 40

DEFAULT_CHUNK_SIZE = 450
DEFAULT_CHUNK_OVERLAP = 80


@dataclass
class TextChunk:
    """One overlapping text window from a single PDF page."""

    chunk_id: str
    filename: str
    page_number: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TextChunk":
        return cls(
            chunk_id=data["chunk_id"],
            filename=data["filename"],
            page_number=int(data["page_number"]),
            text=data["text"],
        )


def extract_pages(pdf_path: str | Path) -> tuple[list[dict], list[str]]:
    """
    Extract text from each page of a PDF.

    Returns:
        pages: list of {filename, page_number, text}
        warnings: human-readable issues (missing file, scanned page, etc.)
    """
    path = Path(pdf_path)
    warnings: list[str] = []

    if not path.exists():
        warnings.append(f"File not found: {path.name}")
        return [], warnings

    if path.suffix.lower() != ".pdf":
        warnings.append(f"Skipped non-PDF file: {path.name}")
        return [], warnings

    try:
        document = fitz.open(path)
    except Exception as exc:
        warnings.append(f"Could not open {path.name}: {exc}")
        return [], warnings

    pages: list[dict] = []
    try:
        if document.page_count == 0:
            warnings.append(f"{path.name} has no pages.")
            return [], warnings

        extracted_chars = 0
        for index in range(document.page_count):
            page = document.load_page(index)
            text = page.get_text("text") or ""
            text = _normalize_whitespace(text)
            page_number = index + 1  # 1-based page numbers for display

            if len(text) < MIN_PAGE_CHARS:
                warnings.append(
                    f"{path.name} page {page_number} has little or no extractable "
                    "text. It may be a scanned image without OCR."
                )
                continue

            extracted_chars += len(text)
            pages.append(
                {
                    "filename": path.name,
                    "page_number": page_number,
                    "text": text,
                }
            )

        if extracted_chars == 0:
            warnings.append(
                f"{path.name} produced no extractable text. "
                "Scanned PDFs need OCR before they can be used."
            )
    finally:
        document.close()

    return pages, warnings


def chunk_pages(
    pages: Iterable[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split each page into overlapping word windows.

    Chunks stay on a single page so page numbers stay accurate.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        part = 0
        while start < len(words):
            window = words[start : start + chunk_size]
            text = " ".join(window).strip()
            if text:
                chunk_id = f"{page['filename']}::p{page['page_number']}::{part}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        filename=page["filename"],
                        page_number=page["page_number"],
                        text=text,
                    )
                )
                part += 1

            if start + chunk_size >= len(words):
                break
            start += step

    return chunks


def process_pdf_folder(
    folder: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[TextChunk], list[str]]:
    """Process every PDF in a folder and return chunks plus warnings."""
    folder_path = Path(folder)
    warnings: list[str] = []

    if not folder_path.exists():
        warnings.append(f"Folder not found: {folder_path}")
        return [], warnings

    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        warnings.append(f"No PDF files found in {folder_path}.")
        return [], warnings

    all_chunks: list[TextChunk] = []
    for pdf_file in pdf_files:
        pages, page_warnings = extract_pages(pdf_file)
        warnings.extend(page_warnings)
        all_chunks.extend(chunk_pages(pages, chunk_size, chunk_overlap))

    if not all_chunks:
        warnings.append("No text chunks were created from the uploaded PDFs.")

    return all_chunks, warnings


def save_uploaded_pdfs(uploaded_files, destination: str | Path) -> list[str]:
    """Save Streamlit uploaded files into papers/ and return saved filenames."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for uploaded in uploaded_files:
        filename = Path(uploaded.name).name
        if not filename.lower().endswith(".pdf"):
            continue
        target = dest / filename
        target.write_bytes(uploaded.getbuffer())
        saved.append(filename)

    return saved


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())
