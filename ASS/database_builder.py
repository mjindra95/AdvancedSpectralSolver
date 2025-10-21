# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:00:34 2025

@author: Martin Jindra 

Build a local RAG database from PDFs with Tkinter dialogs.
- Extracts text with PyMuPDF
- Chunks text with overlap
- Embeds with Sentence-Transformers
- Stores vectors in FAISS + metadata in JSONL
"""

import os, re, json, sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss

# --- NEW: Tkinter imports ---
import tkinter as tk
from tkinter import filedialog, messagebox


def normalize_whitespace(s: str) -> str:
    s = s.replace("-\n", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    pages = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            txt = normalize_whitespace(txt)
            if txt and len(txt.split()) > 10:
                pages.append((i + 1, txt))
    finally:
        doc.close()
    return pages


def chunk_by_words(text: str, chunk_size: int = 150, overlap: int = 30, min_words: int = 30) -> List[str]:
    words = text.split()
    if len(words) < min_words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def batch_embed(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i+batch_size]
        v = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        vecs.append(v.astype(np.float32))
    if not vecs:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(vecs)


def build_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.size == 0:
        raise ValueError("No vectors to index.")
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, path: Path):
    faiss.write_index(index, str(path))


def main():
    # --- Tkinter setup ---
    root = tk.Tk()
    root.withdraw()  # hide main window

    # Ask for PDF folder
    pdf_dir = filedialog.askdirectory(title="Select folder with PDFs")
    if not pdf_dir:
        messagebox.showerror("Error", "No input folder selected.")
        return
    pdf_dir = Path(pdf_dir)

    # Ask for output DB folder
    db_dir = filedialog.askdirectory(title="Select folder to save database")
    if not db_dir:
        messagebox.showerror("Error", "No output folder selected.")
        return
    db_dir = Path(db_dir)

    db_dir.mkdir(parents=True, exist_ok=True)
    meta_path = db_dir / "meta.jsonl"
    index_path = db_dir / "index.faiss"
    stats_path = db_dir / "stats.json"

    # Overwrite warning
    if any(db_dir.iterdir()):
        if not messagebox.askyesno("Warning", f"Folder {db_dir} is not empty.\nOverwrite existing files?"):
            return

    # Load model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    messagebox.showinfo("Info", f"Loading model {model_name} on {device}…")
    model = SentenceTransformer(model_name, device=device)

    pdf_paths = sorted([p for p in pdf_dir.rglob("*.pdf")])
    if not pdf_paths:
        messagebox.showerror("Error", "No PDFs found in the selected folder.")
        return

    all_chunks, all_meta = [], []

    for pdf_path in tqdm(pdf_paths, desc="Reading PDFs", unit="pdf"):
        try:
            pages = extract_pdf_text(pdf_path)
            for page_num, text in pages:
                chunks = chunk_by_words(text)
                for i, ch in enumerate(chunks):
                    meta = {
                        "source": str(pdf_path.resolve()),
                        "page": page_num,
                        "chunk_idx": i,
                        "text": ch
                    }
                    all_meta.append(meta)
                    all_chunks.append(ch)
        except Exception as e:
            print(f"[WARN] Failed {pdf_path}: {e}", file=sys.stderr)

    if not all_chunks:
        messagebox.showerror("Error", "No text extracted. Are these scanned PDFs?")
        return

    vectors = batch_embed(model, all_chunks, batch_size=64)
    index = build_index(vectors)

    save_index(index, index_path)
    with meta_path.open("w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {
        "created": datetime.utcnow().isoformat() + "Z",
        "pdf_dir": str(pdf_dir.resolve()),
        "num_pdfs": len(pdf_paths),
        "num_chunks": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]),
        "model": model_name,
        "device": device,
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    messagebox.showinfo("Done", f"Database built!\n\nSaved to:\n{db_dir}")


if __name__ == "__main__":
    main()
