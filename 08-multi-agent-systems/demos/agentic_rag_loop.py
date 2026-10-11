"""
Demo 1: Agentic RAG with an evidence check and a retry limit
=============================================================

Scenario: a college FAQ assistant.

Traditional RAG searches once and answers. This agent works in a loop:

    search -> check the evidence -> (if not enough) write a better query -> search again

It stops when the evidence is enough, when the retry limit is reached, or when
it would only repeat a query it already tried. If the evidence is still not
enough, it says so instead of guessing.

Run:
    pip install -r requirements.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python agentic_rag_loop.py
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from college_faq import COLLEGE_FAQ

MAX_ATTEMPTS = 3   # retry limit: the loop can never run forever
TOP_K = 2          # chunks returned per search

# 1. Knowledge base: COLLEGE_FAQ is a small, made-up college FAQ in college_faq.py
#    (shared with Demo 3).


# ---------------------------------------------------------------------------
# 2. Structured output: the evidence check must return this exact shape
# ---------------------------------------------------------------------------
class EvidenceCheck(BaseModel):
    decision: Literal["SUFFICIENT", "INSUFFICIENT"] = Field(
        description="SUFFICIENT only if the evidence answers every part of the question."
    )
    reason: str = Field(description="One short sentence explaining the decision.")
    next_query: str = Field(
        description="If INSUFFICIENT, a better search query for the missing part. Otherwise an empty string."
    )


# ---------------------------------------------------------------------------
# 3. Tools the agent can use
# ---------------------------------------------------------------------------
def build_store(embeddings) -> InMemoryVectorStore:
    """Embed the FAQ and keep the vectors in memory."""
    return InMemoryVectorStore.from_documents(COLLEGE_FAQ, embeddings)


def search(store: InMemoryVectorStore, query: str, k: int = TOP_K) -> list[Document]:
    """Retrieval tool: return the k most similar chunks."""
    return store.similarity_search(query, k=k)


def format_evidence(evidence: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in evidence)


def check_evidence(llm, question: str, evidence: list[Document]) -> EvidenceCheck:
    """Ask the LLM to judge the evidence. It proposes; our code decides what happens next."""
    checker = llm.with_structured_output(EvidenceCheck)
    return checker.invoke(
        [
            ("system", "You check whether retrieved evidence answers a question. "
                       "Use only the evidence given. Never assume information that is not written there."),
            ("human", f"Question:\n{question}\n\nEvidence:\n{format_evidence(evidence)}"),
        ]
    )


# ---------------------------------------------------------------------------
# 4. The agent loop (the controller)
# ---------------------------------------------------------------------------
def run_agentic_rag(question: str, store, llm, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Search, check, refine and repeat, within a fixed retry limit."""
    if not question.strip():
        raise ValueError("The question cannot be empty.")

    state = {"question": question, "queries": [], "evidence": [], "attempts": 0,
             "decision": "INSUFFICIENT", "trace": []}
    query = question

    while state["attempts"] < max_attempts:
        state["attempts"] += 1
        state["queries"].append(query)

        # Act: use the retrieval tool, keeping only chunks we have not seen yet
        seen = {d.page_content for d in state["evidence"]}
        new_chunks = [d for d in search(store, query) if d.page_content not in seen]
        state["evidence"].extend(new_chunks)

        # Observe: judge the evidence collected so far
        check = check_evidence(llm, question, state["evidence"])
        state["decision"] = check.decision
        state["trace"].append({"attempt": state["attempts"], "query": query,
                               "new_chunks": len(new_chunks), "decision": check.decision,
                               "reason": check.reason})

        # Decide: stop, or try a better query
        if check.decision == "SUFFICIENT":
            break
        next_query = check.next_query.strip()
        if not next_query or next_query.lower() in (q.lower() for q in state["queries"]):
            state["trace"][-1]["reason"] += " (No new query to try, so stopping early.)"
            break
        query = next_query

    return state


# ---------------------------------------------------------------------------
# 5. Final answer: grounded, or an honest "not found"
# ---------------------------------------------------------------------------
def generate_answer(llm, state: dict) -> str:
    if state["decision"] != "SUFFICIENT":
        # No LLM call here, so nothing can be invented.
        return (f"I could not find this in the college FAQ after {state['attempts']} search(es). "
                "Please contact the college office for this information.")

    response = llm.invoke(
        [
            ("system", "Answer only from the evidence. Mention the source in square brackets "
                       "for each fact, for example [hostel]."),
            ("human", f"Question:\n{state['question']}\n\nEvidence:\n{format_evidence(state['evidence'])}"),
        ]
    )
    return response.content


def run_traditional_rag(question: str, store, llm) -> str:
    """Baseline for comparison: one search, then answer. No checking, no retry."""
    evidence = search(store, question)
    response = llm.invoke(
        [
            ("system", "Answer only from the evidence. If the answer is not there, say so."),
            ("human", f"Question:\n{question}\n\nEvidence:\n{format_evidence(evidence)}"),
        ]
    )
    return response.content


def print_run(state: dict, answer: str) -> None:
    print(f"\nQUESTION: {state['question']}")
    for step in state["trace"]:
        print(f"  attempt {step['attempt']}: query='{step['query']}' | new chunks={step['new_chunks']} "
              f"| {step['decision']} | {step['reason']}")
    print(f"ANSWER: {answer}")


def main() -> None:
    load_dotenv()  # reads OPENAI_API_KEY from your .env file
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    store = build_store(OpenAIEmbeddings(model="text-embedding-3-small"))

    questions = [
        # Answer is in two different FAQ entries
        "How do I apply for B.Tech, and can I get a merit scholarship?",
        # Answer is NOT in the FAQ: the agent should say so, not invent it
        "What is the average placement salary for B.Tech students?",
    ]
    for q in questions:
        state = run_agentic_rag(q, store, llm)
        print_run(state, generate_answer(llm, state))

    print("\n--- Traditional RAG on the first question (one search, no check) ---")
    print(run_traditional_rag(questions[0], store, llm))


if __name__ == "__main__":
    main()
