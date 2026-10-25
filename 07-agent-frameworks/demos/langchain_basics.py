"""
LangChain basics: from one LLM call to an agent with memory
===========================================================

Scenario: a small online stationery shop.

    1. Messages          system + human messages to a chat model
    2. Prompt template   reusable prompt with {placeholders}
    3. LCEL chain        prompt | model | parser
    4. Structured output the model returns a Pydantic object, not free text
    5. batch()           run the same chain on several inputs
    6. Tools             @tool turns a Python function into a tool
    7. Tool calling      bind_tools: the model ASKS for a tool (it does not run it)
    8. Agent + memory    create_agent runs the loop; a checkpointer + thread_id remembers the chat

Run:
    pip install -r requirements.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python langchain_basics.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from safe_calculator import calculate

PRICES = {"notebook": 60, "pen": 15, "geometry box": 120}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_price(item: str) -> str:
    """Get the price in rupees of one stationery item, for example 'pen' or 'notebook'."""
    price = PRICES.get(item.strip().lower())
    return f"{item}: ₹{price}" if price is not None else f"No price found for '{item}'."


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression, for example '3 * 60 + 2 * 15'."""
    return calculate(expression)


TOOLS = [get_price, calculator]


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class ProductIdea(BaseModel):
    name: str = Field(description="Short product name")
    audience: str = Field(description="Who it is for")
    selling_point: str = Field(description="One sentence on why people would buy it")


def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly shop assistant. Answer in two sentences."),
        ("human", "Explain why a student needs a {item}."),
    ])
    return prompt | llm | StrOutputParser()   # LCEL: each step feeds the next


def build_agent(llm, checkpointer=None):
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=(
            "You are a stationery shop assistant. Use get_price for prices and "
            "calculator for any arithmetic. Never guess prices."
        ),
        checkpointer=checkpointer,
    )


def ask(agent, text: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}   # thread_id = which conversation
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
    return result["messages"][-1].content


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    print("\n1. Messages")
    reply = llm.invoke([SystemMessage("You answer in one sentence."), HumanMessage("What is a gel pen?")])
    print(reply.content)
    print("tokens used:", reply.usage_metadata)

    print("\n2 and 3. Prompt template + LCEL chain")
    chain = build_chain(llm)
    print(chain.invoke({"item": "geometry box"}))

    print("\n4. Structured output")
    structured_llm = llm.with_structured_output(ProductIdea)
    idea = structured_llm.invoke("Suggest one new stationery product for engineering students.")
    print(type(idea).__name__, "->", idea.model_dump())

    print("\n5. batch()")
    for answer in chain.batch([{"item": "pen"}, {"item": "notebook"}]):
        print("-", answer)

    print("\n6 and 7. Tool calling: the model only ASKS for the tool")
    response = llm.bind_tools(TOOLS).invoke("What is the price of a pen?")
    print("tool calls requested:", response.tool_calls)

    print("\n8. Agent with memory (same thread_id)")
    agent = build_agent(llm, checkpointer=InMemorySaver())
    print(ask(agent, "My name is Ravi. I want 3 notebooks and 2 pens. What is the total?", "ravi-1"))
    print(ask(agent, "What is my name, and what did I want to buy?", "ravi-1"))
    print("New thread (no memory):", ask(agent, "What is my name?", "someone-else"))


if __name__ == "__main__":
    main()
