import json
import os
import sys
import re
from typing import List, Union, Dict

# === RAG Retrieval Dependencies ===
from sentence_transformers import SentenceTransformer, util

# === LLM Dependencies ===
import ollama  # ollama client for local LLM calls

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-r1:1.5b"
TOP_K = 5
SCORE_THRESHOLD1 = 0.7
SCORE_THRESHOLD2 = 0.3
BOOK_JSON_PATH = "cfa2025.json"

# -------------------------------------------------------------------------
# Load book data
# -------------------------------------------------------------------------
with open(BOOK_JSON_PATH, "r") as f:
    book_json = json.load(f)

# Flatten structure for retrieval
flat_sections = [f"{book} -> {chap}" 
                 for book, chaps in book_json.items() 
                 for chap in chaps]
flat_book = [f"{book} -> {chap} -> {book_json[book][chap]}" 
             for book, chaps in book_json.items() 
             for chap in chaps]

# -------------------------------------------------------------------------
# Retrieval function
# -------------------------------------------------------------------------
def find_relevant_sections(
    query: str,
    top_k: int = TOP_K,
    score_threshold1: float = SCORE_THRESHOLD1,
    score_threshold2: float = SCORE_THRESHOLD2,
    model_name: str = MODEL_NAME
) -> List[str]:
    """
    Return a list of the most relevant 'Book -> Chapter' strings for the given query.
    """
    model = SentenceTransformer(model_name)
    query_emb = model.encode(query, convert_to_tensor=True)

    # 1) Title-level matching
    sec_embs = model.encode(flat_sections, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, sec_embs)[0]
    hits = sorted(
        [(i, s.item()) for i, s in enumerate(scores) if s.item() >= score_threshold1],
        key=lambda x: x[1], reverse=True
    )[:top_k]

    if hits:
        return [flat_sections[i] for i, _ in hits]

    # 2) Fallback: book-level matching
    book_embs = model.encode(flat_book, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, book_embs)[0]
    hits = sorted(
        [(i, s.item()) for i, s in enumerate(scores) if s.item() >= score_threshold2],
        key=lambda x: x[1], reverse=True
    )[: top_k + 2]
    # Normalize to "Book -> Chapter"
    section_keys = []
    for idx, _ in hits:
        book, chap, _ = flat_book[idx].split(" -> ", 2)
        section_keys.append(f"{book} -> {chap}")
    return list(dict.fromkeys(section_keys))  # Remove duplicates

# -------------------------------------------------------------------------
# Local LLM wrapper
# -------------------------------------------------------------------------
class LocalLLM:
    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature}
        )
        return response["message"]["content"].strip()

# -------------------------------------------------------------------------
# Summarization
# -------------------------------------------------------------------------
def summarize_chapter(title: str, content: str, llm: LocalLLM) -> str:
    """
    Generate a markdown-formatted summary for a single chapter.
    """
    prompt = (
        f"You are an expert summarizer. Please provide a complete markdown-formatted "
        f"summary of the chapter '{title}'. Use headings, bullet points, and ensure clarity.\n\n"
        f"Chapter Content:\n{content}"
    )
    return llm.invoke(prompt)

# -------------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------------
def main():
    # 1) Get user query
    query = input("Enter your query: ")

    # 2) Retrieve relevant chapters
    sections = find_relevant_sections(query)
    if not sections:
        print("No relevant chapters found. Please try rephrasing your query.")
        sys.exit(1)

    # 3) Summarize each chapter
    llm = LocalLLM()
    summaries: List[str] = []
    for sec in sections:
        book, chap = sec.split(" -> ", 1)
        text = book_json[book][chap]
        summary_md = summarize_chapter(sec, text, llm)
        summaries.append(f"## {sec}\n\n{summary_md}\n")

    # 4) Combine and save to markdown file
    output = "\n".join(summaries)
    out_path = os.path.join(os.getcwd(), "summary.md")
    with open(out_path, "w") as f:
        f.write(output)

    print(f"Markdown summary created: {out_path}")

if __name__ == "__main__":
    main()
