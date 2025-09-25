# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:06:17 2025

@author: marti
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:11:58 2025
Adapted for top-level chat window with fixed RAG folder

@author: marti
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinter.font as tkfont
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama   # local LLM (install Ollama and a model, e.g., `ollama pull llama3`)


# ---------- RAG utils ----------
def load_rag_db(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    db = Path(__file__).parent / "RAG"
    index_file = db / "index.faiss"
    meta_file = db / "meta.jsonl"

    if not db.exists() or not index_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"RAG folder or files not found in {db}")

    index = faiss.read_index(str(index_file))
    with meta_file.open("r", encoding="utf-8") as f:
        meta = [json.loads(ln) for ln in f]
    model = SentenceTransformer(model_name)
    return index, meta, model


def retrieve(index, meta, model, query: str, k: int = 5):
    q = model.encode([query], convert_to_numpy=True,
                     normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        m = meta[int(idx)]
        results.append({"score": float(score), **m})
    return results


def answer_with_rag(question: str, index, meta, model, llm_model="llama3"):
    docs = retrieve(index, meta, model, question, k=3)
    context = "\n\n".join(
        [f"Source: {d['source']} (p.{d['page']})\n{d['text']}" for d in docs]
    )
    prompt = f"""You are a helpful assistant.
Answer the following question using the provided documents.

Context:
{context}

Question:
{question}

Answer:"""
    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# # ---------- Chatbot Window ----------
# class ChatbotApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Advanced Spectral Solver Chatbot")

#         # Chat history
#         self.chat_area = scrolledtext.ScrolledText(
#             self.master, wrap=tk.WORD, state="disabled", width=80, height=25
#         )
#         self.chat_area.pack(padx=10, pady=10, fill="both", expand=True)

#         # Bottom input frame
#         input_frame = tk.Frame(self.master)
#         input_frame.pack(fill="x", padx=10, pady=(0, 10))

#         self.entry = tk.Entry(input_frame)
#         self.entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 10))
#         self.entry.bind("<Return>", self.send_message)

#         self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
#         self.send_button.pack(side=tk.RIGHT)

#         # Try to load DB
#         try:
#             self.index, self.meta, self.model = load_rag_db()
#             self.display_message("System", f"RAG DB loaded. {len(self.meta)} chunks available.")
#         except Exception as e:
#             self.index = self.meta = self.model = None
#             self.display_message("Error", f"Failed to load RAG DB:\n{e}")

#     def display_message(self, sender, message):
#         self.chat_area.configure(state="normal")
#         self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
#         self.chat_area.configure(state="disabled")
#         self.chat_area.see(tk.END)

#     def send_message(self, event=None):
#         user_text = self.entry.get().strip()
#         if not user_text:
#             return
#         self.display_message("You", user_text)
#         self.entry.delete(0, tk.END)

#         if not self.index:
#             self.display_message("System", "RAG DB not loaded. Cannot answer.")
#             return

#         try:
#             answer = answer_with_rag(user_text, self.index, self.meta, self.model)
#             self.display_message("Assistant", answer)
#         except Exception as e:
#             self.display_message("Error", str(e))

class ChatbotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Spectral Solver Chatbot")
        
        # define a font once
        self.chat_font = tkfont.Font(family="Consolas", size=11)

        # # --- Header ---
        # self.info_label = ttk.Label(self.master, text="Advanced Spectral Solver Chatbot", anchor="center")
        # self.info_label.pack(fill=tk.X, pady=(10, 2))

        # ttk.Separator(self.master, orient="horizontal").pack(fill=tk.X, pady=(5, 5))

        # --- Chat history ---
        # self.chat_area = scrolledtext.ScrolledText(
        #     self.master, wrap=tk.WORD, state="disabled", height=25
        # )
        self.chat_area = scrolledtext.ScrolledText(
            self.master, wrap=tk.WORD, state="disabled", 
            height=25, font=self.chat_font)   # <- main chat font here)
        self.chat_area.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Separator(self.master, orient="horizontal").pack(fill=tk.X, pady=(5, 5))

        # --- Input frame ---
        input_frame = ttk.Frame(self.master)
        input_frame.pack(fill="x", padx=10, pady=10)

        # self.entry = ttk.Entry(input_frame)
        self.entry = ttk.Entry(input_frame, font=self.chat_font)  # <- entry font
        self.entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        # Load DB
        try:
            self.index, self.meta, self.model = load_rag_db()
            self.display_message("System", f"RAG DB loaded. {len(self.meta)} chunks available.")
        except Exception as e:
            self.index = self.meta = self.model = None
            self.display_message("Error", f"Failed to load RAG DB:\n{e}")

    def display_message(self, sender, message):
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def send_message(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.display_message("You", user_text)
        self.entry.delete(0, tk.END)

        if not self.index:
            self.display_message("System", "RAG DB not loaded. Cannot answer.")
            return

        try:
            answer = answer_with_rag(user_text, self.index, self.meta, self.model)
            self.display_message("Assistant", answer)
        except Exception as e:
            self.display_message("Error", str(e))


# # ---------- Function to open from main GUI ----------
# def open_chat():
#     chat_window = tk.Toplevel()
#     ChatbotApp(chat_window)
#     chat_window.grab_set()  # focus on this window
