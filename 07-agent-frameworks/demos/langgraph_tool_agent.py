"""
LangGraph tool-calling agent, built by hand
===========================================

The same loop that create_agent() runs for you, written as a graph so you can see every part:

    START -> llm -> (tool calls?) --yes--> tools -> llm -> ... --no--> END

Scenario: an order help desk with two tools (order status and a safe calculator).

Run:
    pip install -r requirements.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python langgraph_tool_agent.py
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from safe_calculator import calculate

MAX_LLM_CALLS = 6   # safety limit for the loop

ORDERS = {"A101": "Shipped, arriving tomorrow", "A102": "Packed, ships today", "A103": "Delivered"}


@tool
def order_status(order_id: str) -> str:
    """Get the delivery status of an order, for example 'A101'."""
    return ORDERS.get(order_id.strip().upper(), f"No order found with id {order_id}.")


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression, for example '450 * 2 - 100'."""
    return calculate(expression)


TOOLS = {t.name: t for t in [order_status, calculator]}


class AgentState(TypedDict):
    # add_messages is the reducer for chat: it APPENDS new messages instead of replacing the list
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


def build_graph(llm):
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))
    system = SystemMessage("You are an order help desk. Use tools for order status and arithmetic.")

    def llm_node(state: AgentState):
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

    def tool_node(state: AgentState):
        results = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] in TOOLS:
                output = TOOLS[call["name"]].invoke(call["args"])
            else:
                output = f"Error: unknown tool {call['name']}"   # every tool call needs a reply
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    def route(state: AgentState) -> Literal["tools", "__end__"]:
        if state["messages"][-1].tool_calls and state.get("llm_calls", 0) < MAX_LLM_CALLS:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", route)
    builder.add_edge("tools", "llm")   # the cycle: after tools, think again
    return builder.compile()


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    from langchain_openai import ChatOpenAI
    graph = build_graph(ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0))

    question = "Where is order A102? Also, what is 450 * 2 - 100?"
    result = graph.invoke({"messages": [{"role": "user", "content": question}], "llm_calls": 0})
    for message in result["messages"]:
        message.pretty_print()
    print(f"\nLLM calls used: {result['llm_calls']}")


if __name__ == "__main__":
    main()
