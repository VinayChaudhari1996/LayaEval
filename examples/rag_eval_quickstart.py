"""Quickstart example for evaluating RAG pipelines with LayaEval."""

import layaeval
from layaeval import Groundedness, HallucinationRisk, ResponseRelevance


def main():
    print("Evaluating RAG response with LayaEval...")

    # Define RAG inputs
    query = "What is the capital and population of France?"
    retrieved_context = (
        "Paris is the capital and most populous city of France, with an "
        "estimated population of 2.1 million residents within city limits."
    )
    response = "Paris is the capital of France, home to approximately 2.1 million people."

    # Option 1: Ultra-simple 1-liner with metric names or instances!
    results = layaeval.evaluate(
        [
            Groundedness(threshold=3.0),
            HallucinationRisk(threshold=0.25),
            ResponseRelevance(threshold=3.0),
        ],
        query=query,
        retrieved_context=retrieved_context,
        response=response,
    )

    for res in results:
        status = "PASSED" if res.passed else "FAILED"
        val = f"Score: {res.score}" if res.score is not None else f"Risk: {res.noul:.2f}"
        print(f"[{status}] {res.metric_name} ({res.metric_id}): {val} (Conf: {res.confidence})")


if __name__ == "__main__":
    main()
