"""
Demo 3: Agentic RAG with tool calling (the LLM decides when to search)
======================================================================

Demo 1 used a controller: our code checked the evidence and decided whether to search again.
Here the LLM itself decides. We give it a search tool, and in each round it either:

    asks to call the tool (with a query it chose)  ->  our code runs the search and returns the result
    or replies with no tool call                   ->  that reply is the final answer

The loop has a maximum number of rounds, so it always ends.

Run:
    pip install -r requirements.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python tool_calling_agent.py
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from college_faq import COLLEGE_FAQ

MAX_ROUNDS = 4  # the agent can ask for the tool at most this many rounds

SYSTEM_PROMPT = """You are a college FAQ assistant.

Rules:
1. Use the search_college_faq tool for any question about the college.
2. If the search result is not enough, search again with a better query.
3. Answer only from the search results. Never invent dates, fees or rules.
4. If the FAQ does not contain the answer, say so clearly.
5. If the question is not about the college, politely say you only answer college questions.
   Do not answer it from your own knowledge."""

_store = None  # set in main(); the tool searches this store


@tool
def search_college_faq(query: str) -> str:
    """Search the college FAQ. Use it for questions about admissions, hostel,
    scholarships, fee refunds and the library."""
    docs = _store.similarity_search(query, k=2)
    if not docs:
        return "No relevant information found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


TOOLS = {"search_college_faq": search_college_faq}


def run_tool_agent(question: str, llm, max_rounds: int = MAX_ROUNDS, verbose: bool = True) -> dict:
    """The tool calling loop. Returns the answer and a trace of tool calls."""
    if not question.strip():
        raise ValueError("The question cannot be empty.")

    llm_with_tools = llm.bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
    trace = []

    for round_no in range(1, max_rounds + 1):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool call means the model is ready to answer
        if not response.tool_calls:
            return {"answer": response.content, "rounds": round_no, "trace": trace}

        # Run every tool call the model asked for, and send each result back
        for call in response.tool_calls:
            if call["name"] in TOOLS:
                result = TOOLS[call["name"]].invoke(call["args"])
            else:
                # Every tool call needs a reply, even when the tool does not exist
                result = f"Error: there is no tool named {call['name']}."
            trace.append({"round": round_no, "tool": call["name"], "args": call["args"]})
            if verbose:
                print(f"  round {round_no}: {call['name']}({call['args']})")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return {"answer": f"Stopped after {max_rounds} rounds without a final answer.",
            "rounds": max_rounds, "trace": trace}


def main() -> None:
    global _store
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    _store = InMemoryVectorStore.from_documents(COLLEGE_FAQ, OpenAIEmbeddings(model="text-embedding-3-small"))
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    questions = [
        "When do B.Tech applications close?",                                # simple: one search
        "If I withdraw before classes start, do I get my fee back? "
        "And how many library books can I borrow?",                          # two parts: maybe two searches
        "What is the average placement salary?",                             # not in the FAQ
        "Who won the cricket world cup in 2011?",                            # off-topic: should refuse
    ]
    for q in questions:
        print(f"\nQUESTION: {q}")
        result = run_tool_agent(q, llm)
        print(f"ANSWER ({result['rounds']} round(s), {len(result['trace'])} tool call(s)): {result['answer']}")


if __name__ == "__main__":
    main()
