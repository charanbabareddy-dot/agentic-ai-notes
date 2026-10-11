"""
Demo 2: A supervisor with specialist agents
===========================================

Scenario: a small neighbourhood bookstore wants customers to order books on WhatsApp.

    Market Researcher  -> who are the customers, what do they need?
    Tech Planner       -> how do we build it simply?
    Risk Reviewer      -> what could go wrong?
    Supervisor         -> approves or sends the plan back once, then writes the final proposal

Each "agent" here is the same LLM with a different role, input and output schema.
Python controls the order, the revision limit and the call count.

Run:
    pip install -r requirements.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python supervisor_team.py
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

MAX_REVISIONS = 1  # the supervisor can send work back at most this many times


# ---------------------------------------------------------------------------
# 1. Output contracts: every agent must return a fixed shape
# ---------------------------------------------------------------------------
class SpecialistReport(BaseModel):
    summary: str = Field(description="Two or three sentences.")
    key_points: list[str] = Field(description="Three to five concrete points.")
    assumptions: list[str] = Field(description="Things you assumed but did not verify.")


class SupervisorReview(BaseModel):
    approved: bool = Field(description="True only if the plan is complete, consistent and addresses the risks.")
    issues: list[str] = Field(description="Specific problems to fix. Empty if approved.")


# ---------------------------------------------------------------------------
# 2. Roles: one clear responsibility each
# ---------------------------------------------------------------------------
ROLES = {
    "Market Researcher": "You study customers and their needs. Do not plan technology.",
    "Tech Planner": "You design a simple, low cost technical plan. Do not discuss marketing.",
    "Risk Reviewer": "You find risks: privacy, payments, wrong orders, cost. Suggest a safeguard for each.",
}

RULES = ("Stay within your role. Mark guesses as assumptions. "
         "Do not claim you did real research or looked anything up.")


class Team:
    """Holds the LLM and counts calls, so cost is visible."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0

    def specialist(self, role: str, task: str, context: str) -> SpecialistReport:
        self.calls += 1
        agent = self.llm.with_structured_output(SpecialistReport)
        return agent.invoke(
            [
                ("system", f"You are the {role}. {ROLES[role]} {RULES}"),
                ("human", f"Task:\n{task}\n\nContext:\n{context}"),
            ]
        )

    def review(self, problem: str, reports: dict) -> SupervisorReview:
        self.calls += 1
        reviewer = self.llm.with_structured_output(SupervisorReview)
        return reviewer.invoke(
            [
                ("system", "You are the Supervisor. Check that the plan solves the problem and that every "
                           "risk raised has a safeguard in the plan. Be strict but fair."),
                ("human", f"Problem:\n{problem}\n\n{as_text(reports)}"),
            ]
        )

    def synthesize(self, problem: str, reports: dict, open_issues: list[str]) -> str:
        self.calls += 1
        response = self.llm.invoke(
            [
                ("system", "You are the Supervisor. Combine the reports into one short proposal with sections: "
                           "Problem, Customers, Plan, Risks and safeguards, Assumptions, Open issues. "
                           "Do not present assumptions as facts."),
                ("human", f"Problem:\n{problem}\n\n{as_text(reports)}\n\nOpen issues:\n{open_issues or 'None'}"),
            ]
        )
        return response.content


def as_text(reports: dict) -> str:
    """Turn the reports into plain text so the next agent can read them (shared state)."""
    parts = []
    for role, r in reports.items():
        parts.append(f"## {role}\n{r.summary}\nPoints: {r.key_points}\nAssumptions: {r.assumptions}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 3. The workflow (orchestration lives in code, not in the prompt)
# ---------------------------------------------------------------------------
def run_team(problem: str, llm, max_revisions: int = MAX_REVISIONS) -> dict:
    team = Team(llm)
    log = []
    start = time.perf_counter()

    reports = {}
    reports["Market Researcher"] = team.specialist(
        "Market Researcher", "Describe the target customers and what they need from WhatsApp ordering.", problem)
    log.append("Market Researcher done")

    plan_task = "Create a simple technical plan for WhatsApp ordering."
    reports["Tech Planner"] = team.specialist("Tech Planner", plan_task, f"{problem}\n\n{as_text(reports)}")
    log.append("Tech Planner done")

    review = SupervisorReview(approved=False, issues=[])
    for round_no in range(max_revisions + 1):
        reports["Risk Reviewer"] = team.specialist(
            "Risk Reviewer", "Review the plan for risks.", f"{problem}\n\n{as_text(reports)}")
        review = team.review(problem, reports)
        log.append(f"Review round {round_no + 1}: approved={review.approved}, issues={len(review.issues)}")

        if review.approved or round_no == max_revisions:
            break

        # Send the plan back once with the supervisor's issues (bounded revision)
        reports["Tech Planner"] = team.specialist(
            "Tech Planner", f"{plan_task}\nFix these issues: {review.issues}", f"{problem}\n\n{as_text(reports)}")
        log.append("Tech Planner revised the plan")

    open_issues = [] if review.approved else review.issues
    proposal = team.synthesize(problem, reports, open_issues)
    log.append("Supervisor wrote the final proposal")

    return {"proposal": proposal, "approved": review.approved, "log": log,
            "llm_calls": team.calls, "seconds": round(time.perf_counter() - start, 1)}


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    problem = ("A small neighbourhood bookstore in Hyderabad wants customers to order books on WhatsApp "
               "and pay online. The owner has a small budget and no developer on staff.")

    result = run_team(problem, llm)
    print("STEPS:")
    for line in result["log"]:
        print(" -", line)
    print(f"\nApproved by supervisor: {result['approved']}")
    print(f"LLM calls: {result['llm_calls']} | Time: {result['seconds']} s")
    print("\nFINAL PROPOSAL:\n")
    print(result["proposal"])


if __name__ == "__main__":
    main()
