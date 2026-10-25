"""
LangGraph basics: 7 small graphs, NO API key needed
===================================================

Every concept here runs with plain Python functions, so it is free to run.

    1. Linear graph          START -> node -> END
    2. Conditional routing   choose the next node from state
    3. Cycle (loop)          go back to an earlier node until a condition is met
    4. Parallel + reducer    three nodes write to the same list at the same time
    5. Checkpoint            save state per thread_id and inspect it
    6. Human in the loop     pause with interrupt(), resume with Command(resume=...)
    7. Retry policy          retry a node that fails with a temporary error
    8. Subgraph              a whole graph used as one node inside another graph

Run:
    pip install -r requirements.txt
    python langgraph_basics.py
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt


def title(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Linear graph
# ---------------------------------------------------------------------------
class OrderState(TypedDict):
    item: str
    message: str


def confirm_order(state: OrderState):
    # A node returns ONLY the keys it wants to update
    return {"message": f"Order received for {state['item']}."}


def linear_graph():
    builder = StateGraph(OrderState)
    builder.add_node("confirm", confirm_order)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 2. Conditional routing
# ---------------------------------------------------------------------------
class RefundState(TypedDict):
    amount: int
    decision: str


def check_request(state: RefundState):
    return {}  # nothing to update; it only exists so we can route after it


def auto_refund(state: RefundState):
    return {"decision": "Refunded automatically"}


def send_to_manager(state: RefundState):
    return {"decision": "Sent to manager for review"}


def route_refund(state: RefundState) -> Literal["auto_refund", "send_to_manager"]:
    return "auto_refund" if state["amount"] <= 1000 else "send_to_manager"


def conditional_graph():
    builder = StateGraph(RefundState)
    builder.add_node("check", check_request)
    builder.add_node("auto_refund", auto_refund)
    builder.add_node("send_to_manager", send_to_manager)
    builder.add_edge(START, "check")
    builder.add_conditional_edges("check", route_refund)
    builder.add_edge("auto_refund", END)
    builder.add_edge("send_to_manager", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 3. Cycle: improve a draft until it is long enough (with a safety limit)
# ---------------------------------------------------------------------------
class DraftState(TypedDict):
    draft: str
    rounds: int


def improve(state: DraftState):
    return {"draft": state["draft"] + " More detail added.", "rounds": state["rounds"] + 1}


def good_enough(state: DraftState) -> Literal["improve", "__end__"]:
    if len(state["draft"]) >= 60 or state["rounds"] >= 5:   # condition OR safety limit
        return END
    return "improve"


def cycle_graph():
    builder = StateGraph(DraftState)
    builder.add_node("improve", improve)
    builder.add_edge(START, "improve")
    builder.add_conditional_edges("improve", good_enough)
    return builder.compile()


# ---------------------------------------------------------------------------
# 4. Parallel nodes + reducer
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    # operator.add is the REDUCER: updates to this key are joined, not overwritten
    findings: Annotated[list[str], operator.add]


def prices(state: ResearchState):
    return {"findings": ["Competitor prices"]}


def reviews(state: ResearchState):
    return {"findings": ["Customer reviews"]}


def trends(state: ResearchState):
    return {"findings": ["Market trends"]}


def parallel_graph():
    builder = StateGraph(ResearchState)
    for name, fn in [("prices", prices), ("reviews", reviews), ("trends", trends)]:
        builder.add_node(name, fn)
        builder.add_edge(START, name)   # all three start together
        builder.add_edge(name, END)     # the graph ends when all three finish
    return builder.compile()


# ---------------------------------------------------------------------------
# 5. Checkpoint: saved state per thread
# ---------------------------------------------------------------------------
class CounterState(TypedDict):
    count: int


def add_one(state: CounterState):
    return {"count": state["count"] + 1}


def checkpoint_graph():
    builder = StateGraph(CounterState)
    builder.add_node("step_one", add_one)
    builder.add_node("step_two", add_one)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 6. Human in the loop: pause for approval
# ---------------------------------------------------------------------------
class ApprovalState(TypedDict):
    amount: int
    approved: bool


def ask_human(state: ApprovalState):
    # Execution pauses here. The value passed to Command(resume=...) comes back as `decision`.
    decision = interrupt({"question": "Approve this refund?", "amount": state["amount"]})
    return {"approved": bool(decision)}


def approval_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("ask_human", ask_human)
    builder.add_edge(START, "ask_human")
    builder.add_edge("ask_human", END)
    # interrupt() NEEDS a checkpointer, otherwise the graph cannot resume
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 7. Retry policy
# ---------------------------------------------------------------------------
class SearchState(TypedDict):
    result: str


def make_flaky_search(fail_times: int, error_type: type[Exception]):
    calls = {"n": 0}

    def flaky_search(state: SearchState):
        calls["n"] += 1
        print(f"  attempt {calls['n']}")
        if calls["n"] <= fail_times:
            raise error_type("temporary search failure")
        return {"result": "Search completed"}

    return flaky_search


def retry_graph(node_fn):
    builder = StateGraph(SearchState)
    # By default LangGraph retries ConnectionError and HTTP 5xx errors,
    # but NOT ValueError, RuntimeError, TypeError and similar.
    builder.add_node("search", node_fn, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1))
    builder.add_edge(START, "search")
    builder.add_edge("search", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 8. Subgraph
# ---------------------------------------------------------------------------
class CheckState(TypedDict):
    text: str
    valid: bool


def validate(state: CheckState):
    return {"valid": len(state["text"]) >= 10}


def checker_subgraph():
    builder = StateGraph(CheckState)
    builder.add_node("validate", validate)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", END)
    return builder.compile()


class ParentState(TypedDict):
    text: str
    valid: bool


def parent_graph():
    builder = StateGraph(ParentState)
    builder.add_node("checker", checker_subgraph())   # a compiled graph used as a node
    builder.add_edge(START, "checker")
    builder.add_edge("checker", END)
    return builder.compile()


def main() -> None:
    title("1. Linear graph")
    print(linear_graph().invoke({"item": "blue notebook", "message": ""}))

    title("2. Conditional routing")
    g = conditional_graph()
    print(g.invoke({"amount": 500, "decision": ""}))
    print(g.invoke({"amount": 5000, "decision": ""}))

    title("3. Cycle with a safety limit")
    print(cycle_graph().invoke({"draft": "Short draft.", "rounds": 0}))

    title("4. Parallel nodes + reducer")
    print(parallel_graph().invoke({"findings": []}))

    title("5. Checkpoint per thread")
    g = checkpoint_graph()
    config = {"configurable": {"thread_id": "user-1"}}
    print(g.invoke({"count": 0}, config=config))
    snapshot = g.get_state(config)
    print("saved values:", snapshot.values, "| next nodes:", snapshot.next)

    title("6. Human in the loop")
    g = approval_graph()
    cfg_a = {"configurable": {"thread_id": "refund-A"}}
    paused = g.invoke({"amount": 4000, "approved": False}, config=cfg_a)
    print("paused with:", paused["__interrupt__"][0].value)
    print("after approve:", g.invoke(Command(resume=True), config=cfg_a))
    cfg_b = {"configurable": {"thread_id": "refund-B"}}   # new thread for a new request
    g.invoke({"amount": 9000, "approved": False}, config=cfg_b)
    print("after reject:", g.invoke(Command(resume=False), config=cfg_b))

    title("7a. Retry: ConnectionError is retried")
    print(retry_graph(make_flaky_search(2, ConnectionError)).invoke({"result": ""}))

    title("7b. Retry: RuntimeError is NOT retried by default")
    try:
        retry_graph(make_flaky_search(2, RuntimeError)).invoke({"result": ""})
    except RuntimeError as exc:
        print(f"  stopped after the first attempt: {exc}")

    title("8. Subgraph inside a parent graph")
    print(parent_graph().invoke({"text": "Hello LangGraph", "valid": False}))


if __name__ == "__main__":
    main()
