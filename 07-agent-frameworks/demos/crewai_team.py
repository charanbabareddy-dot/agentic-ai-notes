"""
CrewAI: a small marketing team for a home bakery's festival offer
=================================================================

    Part 1  Sequential crew   Researcher -> Copywriter -> Reviewer
                               with task context, a custom tool, a guardrail and structured output
    Part 2  Hierarchical crew  a Manager agent decides who does what
    Part 3  Parallel tasks     two independent tasks run at the same time, then one summary

Run:
    pip install -r requirements-crewai.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python crewai_team.py            (runs Part 1)
    python crewai_team.py all        (runs all three parts; more LLM calls)
"""

import os
import sys
import time

from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.tools import tool
from pydantic import BaseModel, Field

from safe_calculator import calculate

load_dotenv()   # load .env first, so OPENAI_MODEL is read correctly
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_NAME = f"openai/{MODEL}"   # CrewAI reads "provider/model"


# ---------------------------------------------------------------------------
# A custom tool: only the agent that needs it gets it (least privilege)
# ---------------------------------------------------------------------------
@tool("Price Calculator")
def price_calculator(expression: str) -> str:
    """Calculate prices and discounts, for example '40 * 450 * 0.85'. Only + - * / ( ) are allowed."""
    return calculate(expression)


# ---------------------------------------------------------------------------
# Structured output for the final task
# ---------------------------------------------------------------------------
class OfferPost(BaseModel):
    headline: str = Field(description="Catchy headline, at most 10 words")
    body: str = Field(description="Instagram caption, at most 80 words")
    offer_price: str = Field(description="The discounted price with the rupee sign")
    hashtags: list[str] = Field(description="3 to 5 hashtags")


# ---------------------------------------------------------------------------
# A guardrail: checked BEFORE the output moves on. Failing sends feedback to the agent to retry.
# ---------------------------------------------------------------------------
def caption_guardrail(output: TaskOutput) -> tuple[bool, str]:
    words = len(output.raw.split())
    if words > 150:
        return False, f"The caption is {words} words. Keep it under 150 words."
    if "₹" not in output.raw:
        return False, "Include the offer price with the ₹ sign."
    return True, output.raw


def build_agents():
    researcher = Agent(
        role="Local Market Researcher",
        goal="Find what festival buyers in a small city look for in cakes and sweets",
        backstory="You studied small food businesses for years and give practical, honest points.",
        llm=LLM_NAME,
        max_iter=5,            # stop the agent from thinking in circles
    )
    copywriter = Agent(
        role="Instagram Copywriter",
        goal="Write a short, warm festival offer post that makes people order",
        backstory="You write simple, friendly captions for home businesses.",
        tools=[price_calculator],   # only the copywriter can calculate prices
        llm=LLM_NAME,
        max_iter=5,
    )
    reviewer = Agent(
        role="Brand Reviewer",
        goal="Check the post is accurate, friendly and has no false claims",
        backstory="You protect small brands from misleading or pushy marketing.",
        llm=LLM_NAME,
        max_iter=5,
    )
    return researcher, copywriter, reviewer


def build_tasks(researcher, copywriter, reviewer):
    research = Task(
        description="List what customers in {city} care about when buying festival cakes for {festival}.",
        expected_output="5 short bullet points.",
        agent=researcher,
    )
    write = Task(
        description=(
            "Write an Instagram caption for a home bakery's {festival} offer: 15% off a 1 kg cake that "
            "normally costs ₹{price}. Use the Price Calculator to work out the offer price."
        ),
        expected_output="A caption under 150 words that includes the offer price with the ₹ sign.",
        agent=copywriter,
        context=[research],          # receives the research output
        guardrail=caption_guardrail,
        guardrail_max_retries=2,
    )
    review = Task(
        description="Review the caption and return the final, corrected post.",
        expected_output="The final post as structured fields.",
        agent=reviewer,
        context=[write],
        output_pydantic=OfferPost,   # the final answer must match this shape
    )
    return [research, write, review]


INPUTS = {"city": "Warangal", "festival": "Diwali", "price": "650"}   # fills {placeholders}


def run_sequential():
    agents = build_agents()
    crew = Crew(agents=list(agents), tasks=build_tasks(*agents), process=Process.sequential, verbose=True)
    start = time.perf_counter()
    result = crew.kickoff(inputs=INPUTS)
    seconds = time.perf_counter() - start

    print("\n=== FINAL POST (structured) ===")
    print(result.pydantic.model_dump() if result.pydantic else result.raw)
    print("\n=== EACH TASK ===")
    for task_output in result.tasks_output:
        print(f"- {task_output.agent}: {len(task_output.raw)} characters")
    print(f"\nTokens: {result.token_usage.total_tokens} | Requests: "
          f"{result.token_usage.successful_requests} | Time: {seconds:.1f} s")


def run_hierarchical():
    researcher, copywriter, reviewer = build_agents()
    manager = Agent(
        role="Marketing Manager",
        goal="Plan the work, give each task to the right specialist and check the final post",
        backstory="You manage a small marketing team and keep work simple and accurate.",
        llm=LLM_NAME,
        allow_delegation=True,
    )
    tasks = build_tasks(researcher, copywriter, reviewer)
    for t in tasks:
        t.agent = None   # in hierarchical mode the manager assigns the work
    crew = Crew(agents=[researcher, copywriter, reviewer], tasks=tasks,
                process=Process.hierarchical, manager_agent=manager, verbose=True)
    result = crew.kickoff(inputs=INPUTS)
    print("\n=== HIERARCHICAL RESULT ===")
    print(result.raw[:800])
    print(f"Tokens: {result.token_usage.total_tokens}")


def run_parallel():
    researcher, copywriter, reviewer = build_agents()
    sweets = Task(
        description="Give 3 popular {festival} sweet trends for home bakers.",
        expected_output="3 bullet points.",
        agent=researcher,
        async_execution=True,         # does not wait for other tasks
    )
    gifting = Task(
        description="Give 3 {festival} gift box ideas a home bakery could sell.",
        expected_output="3 bullet points.",
        agent=copywriter,
        async_execution=True,
    )
    summary = Task(
        description="Combine both lists into one short plan for the bakery owner.",
        expected_output="A plan of at most 6 bullet points.",
        agent=reviewer,
        context=[sweets, gifting],    # waits for both
    )
    crew = Crew(agents=[researcher, copywriter, reviewer], tasks=[sweets, gifting, summary],
                process=Process.sequential, verbose=True)
    print("\n=== PARALLEL RESULT ===")
    print(crew.kickoff(inputs=INPUTS).raw)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")
    run_sequential()
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_hierarchical()
        run_parallel()


if __name__ == "__main__":
    main()
