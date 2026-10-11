"""
Demo B: Build a small RAG system and evaluate it with Ragas
===========================================================

Steps:
    1. Build RAG on a made-up college FAQ (embed, store, retrieve, answer)
    2. Answer three test questions
    3. Score each answer with Ragas: faithfulness, answer relevancy,
       context precision and context recall
    4. One reference answer is WRONG on purpose, to show how a bad answer key
       makes a correct system look bad

Run:
    pip install -r requirements.txt
    copy .env.example to .env and add your own OPENAI_API_KEY
    python rag_evaluation.py

Note: Ragas changes quickly. This file is written for the versions in requirements.txt.
Your run may print deprecation warnings from Ragas; they are warnings, not errors.
"""

import os
import warnings

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from college_faq import COLLEGE_FAQ

TOP_K = 2

# Test set: question + the correct answer written by a human (the "reference" or ground truth)
TEST_SET = [
    {"question": "When do B.Tech applications close?",
     "reference": "B.Tech applications close on 30 June."},
    {"question": "How many books can a student borrow from the library?",
     "reference": "Students can borrow up to 4 books for 14 days."},
    # WRONG reference on purpose: the FAQ says 90%, not 85%
    {"question": "What marks do I need for a merit scholarship?",
     "reference": "Students need 85% or above in Class 12."},
]

PROMPT = ChatPromptTemplate.from_template(
    "Answer only from the context. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def build_rag(embeddings, llm):
    """Return a function that answers a question and also returns the chunks it used."""
    store = InMemoryVectorStore.from_documents(COLLEGE_FAQ, embeddings)
    chain = PROMPT | llm  # LCEL: the filled prompt flows into the LLM

    def answer(question: str) -> dict:
        docs = store.similarity_search(question, k=TOP_K)
        contexts = [d.page_content for d in docs]
        response = chain.invoke({"context": "\n\n".join(contexts), "question": question})
        return {"response": response.content, "contexts": contexts}

    return answer


def make_samples(answer_fn, test_set: list[dict]) -> list[dict]:
    """Run the RAG system on every test question and collect what Ragas needs."""
    rows = []
    for t in test_set:
        out = answer_fn(t["question"])
        rows.append({
            "user_input": t["question"],          # the question
            "response": out["response"],          # what our system answered
            "retrieved_contexts": out["contexts"],  # the chunks it used
            "reference": t["reference"],          # the human answer key
        })
    return rows


def evaluate_with_ragas(rows: list[dict], llm, embeddings):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    dataset = EvaluationDataset(samples=[SingleTurnSample(**r) for r in rows])
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    # Ragas uses an LLM as the judge, and embeddings for answer relevancy
    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    rows = make_samples(build_rag(embeddings, llm), TEST_SET)
    for r in rows:
        print(f"\nQ: {r['user_input']}\nA: {r['response']}\nReference: {r['reference']}")

    results = evaluate_with_ragas(rows, llm, embeddings)
    table = results.to_pandas()
    cols = ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\nSCORES (0 = bad, 1 = good):")
    print(table[[c for c in cols if c in table.columns]].round(2).to_string(index=False))
    print("\nLook at question 3: the answer can be faithful to the FAQ, but the context scores drop,")
    print("because the reference (85%) is wrong. Always check the answer key first.")


if __name__ == "__main__":
    main()
