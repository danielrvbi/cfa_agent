#!/usr/bin/env python3
"""Simple CFA retrieval agent.

This script provides a minimal command line interface to query CFA content.
It expects a ``cfa2025.json`` file in the same directory which contains a
mapping of ``{"Book": {"Chapter": "text"}}``.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Annotated, TypedDict

from sentence_transformers import SentenceTransformer, util
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import ollama


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_book(path: str) -> Dict[str, Dict[str, str]]:
    """Load the CFA book JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        # JSON may not be strictly formatted; ``eval`` used in notebook
        content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return eval(content)


def build_flat_views(book_json: Dict[str, Dict[str, str]]):
    """Return lists used for retrieval."""
    flat_sections = [
        f"{book} -> {chapter}"
        for book, chapters in book_json.items()
        for chapter in chapters
    ]
    flat_book = [
        f"{book} -> {chapter} -> {book_json[book][chapter]}"
        for book, chapters in book_json.items()
        for chapter in chapters
    ]
    return flat_sections, flat_book


# ---------------------------------------------------------------------------
# Retrieval utilities
# ---------------------------------------------------------------------------


class SectionRetriever:
    """Embeds sections and returns the most relevant ones."""

    def __init__(self, book: Dict[str, Dict[str, str]], model_name: str = "all-MiniLM-L6-v2"):
        self.book = book
        self.flat_sections, self.flat_book = build_flat_views(book)
        self.model = SentenceTransformer(model_name)
        # Pre-compute embeddings
        self.section_embs = self.model.encode(self.flat_sections, convert_to_tensor=True)
        self.book_embs = self.model.encode(self.flat_book, convert_to_tensor=True)

    def find(self, query: str, top_k: int = 5, score_threshold1: float = 0.5, score_threshold2: float = 0.3,
             return_content: bool = False) -> Dict[str, str] | List[str]:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.section_embs)[0]
        hits = sorted(
            [(i, s.item()) for i, s in enumerate(scores) if s.item() >= score_threshold1],
            key=lambda x: x[1], reverse=True
        )[:top_k]
        if not hits:
            scores = util.cos_sim(query_emb, self.book_embs)[0]
            hits = sorted(
                [(i, s.item()) for i, s in enumerate(scores) if s.item() >= score_threshold2],
                key=lambda x: x[1], reverse=True
            )[: top_k + 2]
            section_keys = []
            for idx, _ in hits:
                book, chap, _ = self.flat_book[idx].split(" -> ", 2)
                section_keys.append(f"{book} -> {chap}")
        else:
            section_keys = [self.flat_sections[i] for i, _ in hits]

        if not return_content:
            return section_keys

        return {
            sec: self.book[sec.split(" -> ", 1)[0]][sec.split(" -> ", 1)[1]]
            for sec in section_keys
        }


# ---------------------------------------------------------------------------
# Local LLM wrapper
# ---------------------------------------------------------------------------


class LocalLLM:
    def __init__(self, model: str = "deepseek-r1:1.5b", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]


# ---------------------------------------------------------------------------
# LangGraph nodes and graph setup
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    context: Optional[str]
    response: Optional[str]
    messages: Annotated[List[Dict[str, str]], add_messages]


def make_graph(retriever: SectionRetriever) -> StateGraph:
    def retrieval_node(state: GraphState) -> dict:
        query = state["messages"][-1]["content"]
        content = retriever.find(query, return_content=True)
        context = "\n\n".join(
            f"Book & Chapter: {sec}\n{text}" for sec, text in content.items()
        )
        return {"context": context}

    def response_node(state: GraphState) -> dict:
        query = state["messages"][-1]["content"]
        prompt = (
            "Answer the following question using only the provided context. "
            f"Context:\n{state['context']}\n\nQuestion:\n{query}\n"
            "Cite your sources based on the context. If no information is found, say so."
        )
        llm = LocalLLM()
        answer = llm.invoke(prompt)
        new_msg = {"role": "assistant", "content": answer}
        return {"response": answer, "messages": state["messages"] + [new_msg]}

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("respond", response_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "respond")
    builder.add_edge("respond", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), "cfa2025.json")
    if not os.path.exists(data_path):
        print("cfa2025.json not found. Please place it next to this script.")
        sys.exit(1)

    book_json = load_book(data_path)
    retriever = SectionRetriever(book_json)
    graph = make_graph(retriever)

    print("Type 'exit' to quit.")
    state: GraphState = {"messages": []}
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        state["messages"].append({"role": "user", "content": user_input})
        state = graph.invoke(state)
        print("Assistant:", state["response"])


if __name__ == "__main__":
    main()
