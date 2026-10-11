"""
Demo A: Retrieval metrics by hand (Precision@k, Recall@k, MRR)
==============================================================

No API key needed. No cost. Pure Python.

We pretend a retriever already ran for three test questions. For each question we know:
    retrieved = chunk ids the retriever returned, in rank order (best first)
    relevant  = chunk ids that truly answer the question (our answer key)

Then we score the retriever with three metrics.

Run:
    python retrieval_metrics.py
"""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the top k chunks we fetched, what share were relevant?"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for c in top_k if c in relevant) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant chunks that exist, what share did we find in the top k?"""
    if not relevant:
        return 0.0
    return sum(1 for c in retrieved[:k] if c in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant chunk. 0 if none was found."""
    for position, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            return 1 / position
    return 0.0


def mean_reciprocal_rank(test_set: list[dict]) -> float:
    """Average reciprocal rank over all test questions."""
    if not test_set:
        return 0.0
    return sum(reciprocal_rank(t["retrieved"], t["relevant"]) for t in test_set) / len(test_set)


# A tiny, made-up test set for a college FAQ
TEST_SET = [
    {
        "question": "When do B.Tech applications close?",
        "retrieved": ["admissions", "scholarships", "hostel"],
        "relevant": {"admissions"},
    },
    {
        "question": "Can I get a scholarship and a fee refund?",
        "retrieved": ["library", "hostel", "scholarships"],
        "relevant": {"scholarships", "fee_refund"},
    },
    {
        "question": "What are the library timings?",
        "retrieved": ["hostel", "admissions", "fee_refund"],
        "relevant": {"library"},
    },
]

K = 3


def main() -> None:
    print(f"{'Question':45} {'P@3':>6} {'R@3':>6} {'RR':>6}")
    print("-" * 66)
    for t in TEST_SET:
        p = precision_at_k(t["retrieved"], t["relevant"], K)
        r = recall_at_k(t["retrieved"], t["relevant"], K)
        rr = reciprocal_rank(t["retrieved"], t["relevant"])
        print(f"{t['question'][:45]:45} {p:6.2f} {r:6.2f} {rr:6.2f}")
    print("-" * 66)
    print(f"MRR (mean reciprocal rank) = {mean_reciprocal_rank(TEST_SET):.2f}")
    print("\nHow to read it:")
    print("  Question 1: the right chunk is first (recall 1.00, RR 1.00), but precision is only 0.33")
    print("              because the other 2 chunks are noise. A smaller k would help here.")
    print("  Question 2: found 1 of 2 relevant chunks (recall 0.50), and only at position 3 (RR 0.33).")
    print("  Question 3: the library chunk was never retrieved. Every score is 0: fix retrieval first.")


if __name__ == "__main__":
    main()
