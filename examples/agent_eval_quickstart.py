"""Quickstart example for evaluating autonomous agent execution with LayaEval."""

import layaeval
from layaeval import TaskCompletion


def main():
    print("Evaluating Agent execution trace with LayaEval...")

    task = "Find customer #402 and retrieve their recent invoice."
    agent_trace = (
        "Step 1: Called crm.search_customer(id=402) -> Found Jane Doe\n"
        "Step 2: Called billing.get_invoices(customer_id=402, limit=1) -> Found INV-9901 for $120.00"
    )
    final_result = "Customer Jane Doe (ID #402) has a recent invoice INV-9901 for $120.00."

    # Direct 1-liner evaluation
    res = layaeval.evaluate(
        TaskCompletion(threshold=0.8),
        task=task,
        agent_trace=agent_trace,
        final_result=final_result,
    )

    status = "PASSED" if res.passed else "FAILED"
    print(f"[{status}] Task Completion: P(true)={res.noul:.2f} (Confidence: {res.confidence})")


if __name__ == "__main__":
    main()
