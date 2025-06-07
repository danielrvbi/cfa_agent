# === Standard Library ===
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union, TypedDict, Annotated

# === Data Handling ===
import pandas as pd

# === NLP Models ===
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

# === LangGraph Core ===
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# === LangChain & LLM ===
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import ollama  # used directly in custom LLM calls

# === Validation ===
from pydantic import BaseModel

# -------------------------------------------------------------------------
# 0. RAG Retrieval
# -------------------------------------------------------------------------

path = "/Users/danielrubibreton/Desktop/PythonStuff/hface/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)

with open("cfa2025.json", "r") as file:
    book_json = eval(file.read())
flat_sections = [f"{book} -> {chapter}" for book, chapters in book_json.items() for chapter in chapters]

flat_book = [f"{book} -> {chapter} -> {book_json[book][chapter] }" for book, chapters in book_json.items() for chapter in chapters]

def find_relevant_sections(
    query: str,
    top_k: int = 5,
    score_threshold1: float = 0.5,
    score_threshold2: float = 0.3,
    model_name: str = 'all-MiniLM-L6-v2',
    return_content: bool = False
) -> Union[List[str], Dict[str, str]]:
    """
    If return_content=False: return top_k section titles (e.g. ["Genesis -> 1", ...]).
    If return_content=True: return a dict mapping each section title to its full text.
    """
    model = SentenceTransformer(model_name, device='mps')
    query_emb = model.encode(query, convert_to_tensor=True)

    # 1) Title‐level matching
    sec_embs = model.encode(flat_sections, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, sec_embs)[0]
    hits = sorted(
        [(i, s.item()) for i, s in enumerate(scores) if s.item() >= score_threshold1],
        key=lambda x: x[1], reverse=True
    )[:top_k]

    if not hits:
        # 2) Fallback: book‐level matching
        book_embs = model.encode(flat_book, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, book_embs)[0]
        hits = sorted(
            [(i, s.item()) for i, s in enumerate(scores) if s.item() >= score_threshold2],
            key=lambda x: x[1], reverse=True
        )[: top_k + 2]
        # normalize to just "Book -> Chapter"
        section_keys = []
        for idx, _ in hits:
            book, chap, _ = flat_book[idx].split(" -> ", 2)
            section_keys.append(f"{book} -> {chap}")
    else:
        section_keys = [flat_sections[i] for i, _ in hits]

    if not return_content:
        return section_keys

    # if return_content=True, build the full dict
    return {
        sec: book_json[sec.split(" -> ", 1)[0]][sec.split(" -> ", 1)[1]]
        for sec in section_keys
    }
    

# -------------------------------------------------------------------------
# 1. LocalLLM
# -------------------------------------------------------------------------
class LocalLLM:
    def __init__(
        self,
        model: str = "deepseek-r1:1.5b",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_thread": 10,
                "low_vram": False,
            }
        )
        return response["message"]["content"].split("</think>")[-1].strip()

llm = LocalLLM()



# -------------------------------------------------------------------------
# 2. GraphState definition
# -------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    retrieved_sections: Optional[Any]       # Will hold either List[str] or str
    response: Optional[str]
    messages: Annotated[List[Dict[str, str]], add_messages]
    goto: Optional[str]          # Used by the confirm node to drive conditional edges
    context: str


# -------------------------------------------------------------------------
# 3. Retrieval node (unchanged)
# -------------------------------------------------------------------------

def user_query_node(state: GraphState) -> GraphState:
    query = state["query"]
    # find_relevant_sections(query) returns List[str]

    
    return {
        "messages": [HumanMessage(content=query)]
    }



def retrieval_node(state: GraphState) -> GraphState:
    query = state["query"]
    # find_relevant_sections(query) returns List[str]
    section_list = find_relevant_sections(query)

    
    return {
        "messages": [SystemMessage(content=section_list)],
        "retrieved_sections": section_list
    }


# -------------------------------------------------------------------------
# 4. Confirm node using interrupt()
# -------------------------------------------------------------------------

def confirm_node(state: GraphState) -> Dict[str, Any]:
    sections = state["retrieved_sections"]

    # 1) Exactly one → skip confirmation
    if len(sections) == 1:
        return {
            "messages": [
                SystemMessage(content=f"Only one section found: {sections[0]}. Skipping confirmation.")
            ],
            "retrieved_sections": sections,
            "goto": "full_retrieval"
        }

    # 2) None → ask user to rephrase query
    if not sections:
        new_q = interrupt({
            "prompt": "I couldn't find any sections matching your question. Please rephrase or clarify your query."
        })
        return {
            "messages": [SystemMessage(content=f"User provided new query: {new_q}")],
            "query": new_q,
            "goto": "retrieve"
        }

    # 3) Multiple → present numbered options
    opts = "\n".join(f"{i+1}. {sec}" for i, sec in enumerate(sections))
    choice = interrupt({
        "prompt": (
            "I found multiple relevant sections. Please select one by number:\n\n"
            f"{opts}\n\nReply with 1, 2, 3, etc."
        )
    })

    # validate
    try:
        idx = int(choice.strip()) - 1
        if idx < 0 or idx >= len(sections):
            raise ValueError()
    except ValueError:
        return {
            "messages": [SystemMessage(content=f"Invalid choice '{choice}'. Let me try again.")],
            "goto": "confirm"
        }

    # valid selection
    picked = sections[idx]
    return {
        "messages": [
            SystemMessage(content=f"User selected option {idx+1}: {picked}")
        ],
        "retrieved_sections": [picked],
        "goto": "full_retrieval"
    }

# -------------------------------------------------------------------------
# 5. Full retrieval node (unchanged)
# -------------------------------------------------------------------------
def full_retrieval_node(state: GraphState) -> GraphState:
    section_list = state["retrieved_sections"]  # This is still List[str]
    all_text = []
    for sec in section_list:
        book, chap = sec.split(" -> ", 1)
        text = book_json[book][chap]
        all_text.append(f"Book & Chapter: {sec}\n{text}")
        
    concatenated = "\n".join(all_text)

    
    return {
        "context": concatenated,
        "messages": [SystemMessage(content="Fetched full text for all candidate sections.")]
    }


# -------------------------------------------------------------------------
# 6. Response node (unchanged)
# -------------------------------------------------------------------------
def response_node(state: GraphState) -> GraphState:
    context_text = state["context"]  # Now a big string
    query = state["query"]
    prompt = (
        "Below are the relevant CFA curriculum sections:\n\n"
        f"{context_text}\n\n"
        f"Question: {query}\n\n"
        "Please answer in a concise, yet complete manner, citing the relevant sections as needed."
    )
    llm_answer = llm.invoke(prompt)
    ai_message = AIMessage(content=llm_answer)

    ai_message.pretty_print()

    return {
        "response": llm_answer,
        "messages": [ai_message]
    }


# -------------------------------------------------------------------------
# 7. Build and compile the graph with MemorySaver checkpointer
# -------------------------------------------------------------------------
graph = (
    StateGraph(GraphState)
    .add_node("retrieve", retrieval_node)
    .add_node("confirm", confirm_node)
    .add_node("full_retrieval", full_retrieval_node)
    .add_node("respond", response_node)

    # Start → retrieve
    .add_edge(START, "retrieve")

    # retrieve → confirm (always ask user to confirm first)
    .add_edge("retrieve", "confirm")

    # Conditional after confirm: use state["goto"] to pick next
    .add_conditional_edges(
        "confirm",
        lambda out: out.get("goto"),
        {
            "full_retrieval": "full_retrieval",
            "retrieve": "retrieve",
            "confirm": "confirm"
        },
    )

    # If user confirmed, we go to full_retrieval; otherwise loop back to retrieve
    .add_edge("full_retrieval", "respond")
    .add_edge("respond", END)

    # Compile with a checkpointer so that interrupt() can save state
    .compile(checkpointer=MemorySaver())
)


# -------------------------------------------------------------------------
# 8. Run the agent
# -------------------------------------------------------------------------


config = {"configurable": {"thread_id": "cfa_chat"}}

def interactive_query(user_query: str):
    # Initialize the state for this run
    state = {
        "query": user_query,
        "retrieved_data": None,
        "context": None,
        "response": None,
        "messages": [],
        "goto": None,
    }

    # We'll use stream_mode="updates" so we only see each node's new outputs
    stream_iter = graph.stream(state, config=config, stream_mode="updates")

    for chunk in stream_iter:
        # 1) Handle any interrupt from confirm_node
        if "__interrupt__" in chunk:
            interrupt_obj = chunk["__interrupt__"][0]
            payload = interrupt_obj.value

            # Always show the prompt
            print(payload["prompt"])

            # If there are options, list them
            if "sections" in payload:
                for idx, sec in enumerate(payload["sections"], start=1):
                    print(f"  {idx}. {sec}")

            # Get the user's reply
            user_input = input("Your answer: ")

            # Resume the graph with that reply
            state = graph.invoke(Command(resume=user_input), config=config)
            # Restart streaming from the resumed state
            stream_iter = graph.stream(state, config=config, stream_mode="updates")
            continue

        # 2) No interrupt → unpack node outputs
        for node_out in chunk.values():
            # If the response node has fired, it will set "response"
            if "response" in node_out and node_out["response"] is not None:
                print("\n📄 Final Answer:")
                print(node_out["response"])
                return
            # Otherwise, if a node emitted messages, show them
            if "messages" in node_out and node_out["messages"]:
                last_msg = node_out["messages"][-1]
                # SystemMessage or assistant message
                last_msg.pretty_print()

def main_loop():
    print("=== CFA-Chat Agent ===")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        user_query = input("You: ")
        if user_query.strip().lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        interactive_query(user_query)
        print("\n===============================================================\n")

if __name__ == "__main__":
    main_loop()








