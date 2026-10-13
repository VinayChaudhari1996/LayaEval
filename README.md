<p align="center">
  <img alt="LayaEval Logo" src="https://raw.githubusercontent.com/VinayChaudhari1996/LayaEval/main/assets/logo.svg" width="130" height="130">
</p>

<h1 align="center">LayaEval</h1>

<p align="center">
  <strong>Ultra-Fast Neural Evaluation Framework for Agent and RAG Systems</strong><br>
  <em>42 production-ready metrics &bull; Sub-50ms latency &bull; Zero external LLM judge costs &bull; Calibrated confidence</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/layaeval/"><img src="https://img.shields.io/pypi/v/layaeval.svg?style=flat-square&color=8A6AFF" alt="PyPI version"></a>
  <a href="https://pypi.org/project/layaeval/"><img src="https://img.shields.io/pypi/pyversions/layaeval.svg?style=flat-square&color=3776AB" alt="Python Versions"></a>
  <a href="https://github.com/VinayChaudhari1996/LayaEval/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/NandhaKishorM/laya"><img src="https://img.shields.io/badge/powered%20by-Laya%20MCP-8A6AFF?style=flat-square" alt="Powered by Laya MCP"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square" alt="uv"></a>
</p>

<p align="center">
  <a href="#key-highlights">Highlights</a> &bull;
  <a href="#why-layaeval">Why LayaEval</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#interactive-notebooks">Notebooks</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#supported-metrics-catalog">Metric Catalog</a> &bull;
  <a href="#cli-usage">CLI Reference</a> &bull;
  <a href="#acknowledgements--special-thanks">Acknowledgements</a> &bull;
  <a href="#citation">Citation</a> &bull;
  <a href="#license">License</a>
</p>

---

**LayaEval** is an ultra-fast, developer-first Python evaluation SDK for **Agentic workflows and RAG pipelines**, powered by the **Laya MCP Server**.

LayaEval brings **42 production-grade evaluation metrics** (20 RAG metrics + 22 Agent metrics) directly into your CI/CD and production verification loops—executed as calibrated, non-autoregressive neural decisions with **zero external LLM API cost** and **sub-50ms latency**.

---

## Key Highlights

- ⚡ **Sub-50ms Latency**: Non-autoregressive neural verification replaces slow, multi-second LLM-as-a-judge roundtrips.
- 💰 **Zero External Token Cost**: No OpenAI or Anthropic API bills for running quality guardrails and regression suites.
- 🎯 **42 Production-Ready Metrics**: Comprehensive coverage spanning retrieval quality, groundedness, hallucination risk, multi-step agent traces, tool accuracy, policy compliance, and prompt injections.
- 🚀 **Single Forward Pass Batching**: Evaluates multiple metrics for an interaction in a unified forward pass for optimal throughput.
- 📊 **Calibrated Confidence**: Outputs statistical probabilities ($P(\text{true})$), standard confidence intervals, and configurable pass/fail thresholds.
- 🔌 **Native MCP Integration**: Direct connection via stdio JSON-RPC or embedded runtime to the Laya MCP ecosystem.
- 🪶 **Ultra Lightweight**: Zero heavy dependencies required at install time (`numpy` only).

---

## Why LayaEval?

| Feature | Traditional LLM-as-a-Judge | LayaEval |
| :--- | :--- | :--- |
| **Latency per Metric** | 1,500ms – 5,000ms | **< 50ms** |
| **API Cost** | ~$0.01 – $0.05 per eval call | **$0.00 (Zero external API fees)** |
| **Determinism & Calibration** | Brittle prompt parsing & hallucinations | **Calibrated neural probabilities & confidence** |
| **Multi-Metric Evaluation** | Sequential calls or bloated judge prompts | **Single forward-pass batching** |
| **CI/CD Regression Suitability** | Slow, flaky, expensive | **Fast, offline-capable, cost-free** |
| **Dependencies** | Bulky orchestration frameworks | **Ultra-lightweight (`numpy` only)** |

---

## Installation

Install via [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`:

```bash
# Core package (ultra-lightweight, connects to existing laya-mcp-server)
uv add layaeval

# Or with full local Laya runtime included:
uv add "layaeval[laya]"
```

Using standard `pip`:

```bash
pip install layaeval
# Or with local Laya runtime:
pip install "layaeval[laya]"
```

---

## Quickstart

### 1. Instant Multi-Metric RAG Evaluation

Evaluate multiple metrics in a single neural forward pass with zero boilerplate:

```python
import layaeval

results = layaeval.evaluate(
    ["groundedness", "hallucination", "response_relevance"],
    query="What is the capital and population of France?",
    retrieved_context="Paris is the capital and most populous city of France, with 2.1M residents.",
    response="Paris is the capital of France, home to about 2.1 million people.",
)

for r in results:
    print(
        f"[{r.metric_name:<20}] Passed: {r.passed:<5} | Score: {r.score or r.noul:.2f} | Confidence: {r.confidence:.2f}"
    )
```

### 2. Autonomous Agent Execution & Trace Evaluation

Audit multi-step tool calls, goal achievement, and task completion using metric classes or identifiers:

```python
import layaeval
from layaeval import TaskCompletion, ToolCallAccuracy

task = "Find user #104 and check subscription status."
trace = "Step 1: db.query(user_id=104) -> Plan: Pro\nStep 2: stripe.check(sub_id=sub_104) -> Active"
result = "User #104 has an active Pro subscription."

res = layaeval.evaluate(
    TaskCompletion(threshold=0.75),
    task=task,
    agent_trace=trace,
    final_result=result,
)

print(f"Task Completed: {res.passed} (P(true) = {res.noul:.2f}, confidence = {res.confidence:.2f})")
```

### 3. Direct Callable Metric Objects

Metric objects can be instantiated with custom thresholds and called directly like standard functions:

```python
from layaeval import Groundedness

groundedness = Groundedness(threshold=3.0)

res = groundedness(
    retrieved_context="Paris is the capital of France.",
    response="Paris is the capital.",
)

print(f"Passed: {res.passed} (Score: {res.score}/4.0 | Reason: {res.reasoning})")
```

### 4. Managed Evaluator Context

For dedicated server processes, custom models, or high-throughput batch sessions:

```python
from layaeval import Evaluator, Groundedness, HallucinationRisk

with Evaluator(model="auto") as ev:
    results = ev.evaluate(
        [Groundedness(threshold=3.0), HallucinationRisk(threshold=0.2)],
        retrieved_context="Revenue grew by 24% year-over-year.",
        response="The company posted 24% YoY revenue growth.",
    )
    for r in results:
        print(f"{r.metric_name}: {r.passed} (Score/Risk: {r.score or r.noul})")
```

### 5. Dataset Evaluation & Reporting

Run evaluations against entire benchmark datasets and generate structured summary reports:

```python
from layaeval import EvalDataset, evaluate

dataset = EvalDataset.from_jsonl("eval_benchmark.jsonl")

report = evaluate(
    dataset,
    metrics=["groundedness", "hallucination", "response_relevance"],
    name="v1.2-rag-baseline",
)

print(f"Pass Rate: {report.pass_rate * 100:.1f}%")
print(f"Total Evaluations: {report.total_evaluations}")
report.save("eval_report.json")
```

---

## Interactive Notebooks

Hands-on Jupyter notebooks with end-to-end code, real-world examples, and step-by-step walkthroughs are available in the [`examples/notebook/`](https://github.com/VinayChaudhari1996/LayaEval/tree/main/examples/notebook) folder:

| Evaluation Domain | Notebook File Path | Workflows & Metrics Covered | Run Command |
| :--- | :--- | :--- | :--- |
| 🔍 **RAG Evaluation** | [`examples/notebook/rag_metrics_evaluation.ipynb`](https://github.com/VinayChaudhari1996/LayaEval/blob/main/examples/notebook/rag_metrics_evaluation.ipynb) | End-to-end evaluation of retrieval relevance, document relevance, context sufficiency & coverage, groundedness, hallucination detection, citation correctness, and contradiction detection. | `uv run --with jupyter jupyter notebook examples/notebook/rag_metrics_evaluation.ipynb` |
| 🤖 **Autonomous Agents** | [`examples/notebook/agent_metrics_evaluation.ipynb`](https://github.com/VinayChaudhari1996/LayaEval/blob/main/examples/notebook/agent_metrics_evaluation.ipynb) | Multi-step agent trace auditing, task completion, tool call & parameter validation, workflow navigation efficiency, agent groundedness, policy compliance, and prompt injection detection. | `uv run --with jupyter jupyter notebook examples/notebook/agent_metrics_evaluation.ipynb` |

To launch the notebooks with all dependencies:

```bash
# Launch Jupyter Lab directly via uv
uv run --with jupyter jupyter lab examples/notebook/
```

---

## How It Works

<p align="center">
  <img alt="LayaEval Architecture" src="https://raw.githubusercontent.com/VinayChaudhari1996/LayaEval/main/assets/architecture.svg" width="100%">
</p>

<details>
<summary><b>View Mermaid Source</b></summary>

```mermaid
flowchart LR
    subgraph Client["Your Application / Pipeline"]
        A["RAG Context & Response"] --> C["LayaEval SDK"]
        B["Agent Execution Trace"] --> C
    end

    subgraph Batcher["LayaEval Orchestrator"]
        C --> D["Pack Questions & States"]
        D --> E["Single JSON-RPC / MCP Request"]
    end

    subgraph MCP["Laya MCP Server"]
        E --> F["Non-Autoregressive Neural Model"]
        F --> G["Multi-Question Forward Pass"]
    end

    G --> H["Calibrated Verdicts\nScore (0-4) • P(true) • Confidence"]
    H --> C
```

</details>

1. **State Aggregation**: You supply standard state keys (`query`, `retrieved_context`, `response`, `agent_trace`, etc.).
2. **Batch Construction**: LayaEval compiles the requested metrics into typed neural decision prompts.
3. **Sub-50ms Inference**: The Laya MCP Server runs a non-autoregressive forward pass, returning calibrated probabilities rather than generating arbitrary text tokens.
4. **Normalized Results**: Every output is mapped to a typed `EvalResult` with normalized scores, confidence intervals, and pass/fail statuses against your specified thresholds.

---

## Supported Metrics Catalog

LayaEval uses three calibrated output types:
- **Score (0.0 – 4.0)**: Graded rubric (0: poor, 1: weak, 2: partial, 3: mostly good, 4: perfect). Default pass threshold is **$\ge 3.0$**.
- **Probability $P(\text{true})$ (0.0 – 1.0)**: Calibrated likelihood for positive qualities. Default pass threshold is **$\ge 0.50$** (or **$\ge 0.70$**).
- **Risk Probability $P(\text{true}) \le \text{threshold}$**: Calibrated likelihood of negative events (hallucination, contradiction, security injection). Safe pass threshold is **$\le 0.20$ or $\le 0.30$**.

---

### 1. RAG Metrics Guide (20 Metrics)

> 📓 **Interactive Notebook**: Open [`examples/notebook/rag_metrics_evaluation.ipynb`](https://github.com/VinayChaudhari1996/LayaEval/blob/main/examples/notebook/rag_metrics_evaluation.ipynb) for runnable code examples of each RAG metric.

| Metric & ID | Definition | Type / Scale | Good Score (Pass) | Bad Score (Fail) | Required Inputs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`RetrievalRelevance`**<br>`RAG-001` &bull; `retrieval_relevance` | Measures how relevant the retrieved context chunks are to the user's question. | Score (0–4) | **$\ge 3.0$**<br>Context directly addresses question | **$< 2.0$**<br>Context is noisy or off-topic | `query`, `retrieved_context` |
| **`DocumentRelevance`**<br>`RAG-002` &bull; `document_relevance` | Evaluates whether a single specific document chunk contains pertinent information. | Score (0–4) | **$\ge 3.0$**<br>Document is strongly relevant | **$< 2.0$**<br>Document contains useless data | `query`, `document` |
| **`ContextSufficiency`**<br>`RAG-003` &bull; `context_sufficient` | Checks if the retrieved context has all needed facts to answer without guessing. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Sufficient facts to answer | **$P < 0.50$**<br>Critical information missing | `query`, `retrieved_context` |
| **`ContextCoverage`**<br>`RAG-004` &bull; `context_coverage` | Quantifies what fraction of the required information is present in the context. | Score (0–4) | **$\ge 3.0$**<br>Covers almost all required details | **$< 2.0$**<br>Major knowledge gaps | `query`, `retrieved_context` |
| **`RetrievalCompleteness`**<br>`RAG-005` &bull; `retrieval_completeness` | Evaluates whether retrieved evidence is complete or missing major pieces. | Score (0–4) | **$\ge 3.0$**<br>Complete evidence package | **$< 2.0$**<br>Fragmented / incomplete evidence | `query`, `retrieved_context` |
| **`Groundedness`**<br>`RAG-006` &bull; `groundedness` | Measures how well the response statements are supported by facts in the context. | Score (0–4) | **$\ge 3.0$**<br>Claims solidly backed by context | **$< 2.0$**<br>Unverified claims made | `retrieved_context`, `response` |
| **`StrictGroundedness`**<br>`RAG-007` &bull; `strictly_grounded` | Strict binary verification that every material claim in the response is proven. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>100% of claims verified | **$P < 0.50$**<br>Contains unverified claim(s) | `retrieved_context`, `response` |
| **`HallucinationRisk`**<br>`RAG-008` &bull; `hallucination` | Detects whether the response fabricates facts unsupported by the retrieved context. | Risk $P \le \text{max}$ | **$P \le 0.30$**<br>Zero/minimal hallucination risk | **$P > 0.30$**<br>Fabricated facts detected | `retrieved_context`, `response` |
| **`ResponseRelevance`**<br>`RAG-009` &bull; `response_relevance` | Evaluates how directly and appropriately the response answers the user's query. | Score (0–4) | **$\ge 3.0$**<br>Directly answers user question | **$< 2.0$**<br>Rambling, evasive, or off-topic | `query`, `response` |
| **`ResponseCompleteness`**<br>`RAG-010` &bull; `response_completeness` | Checks whether the response answers all sub-questions and key parts of the query. | Score (0–4) | **$\ge 3.0$**<br>All questions addressed | **$< 2.0$**<br>Key questions left unanswered | `query`, `response` |
| **`FactualCorrectness`**<br>`RAG-011` &bull; `factual_correctness` | Compares the generated response against a reference ground-truth answer. | Score (0–4) | **$\ge 3.0$**<br>Accurately matches ground truth | **$< 2.0$**<br>Factually conflicts with ground truth | `query`, `response`, `ground_truth` |
| **`Answerability`**<br>`RAG-012` &bull; `answerable` | Checks if the question can realistically be answered given the retrieved context. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Question is answerable | **$P < 0.50$**<br>Context cannot answer question | `query`, `retrieved_context` |
| **`AbstentionCorrectness`**<br>`RAG-013` &bull; `abstention_correct` | Verifies if the model appropriately refused to answer when context was missing. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Correct decision to answer/abstain | **$P < 0.50$**<br>Answered blindly without facts | `query`, `retrieved_context`, `response` |
| **`ContextUtilization`**<br>`RAG-014` &bull; `context_utilization` | Checks how effectively the response incorporated available facts from context. | Score (0–4) | **$\ge 3.0$**<br>Appropriately uses provided facts | **$< 2.0$**<br>Ignores relevant facts in context | `retrieved_context`, `response` |
| **`CitationCorrectness`**<br>`RAG-015` &bull; `citation_correctness` | Verifies whether inline citations accurately support the claim they are attached to. | Score (0–4) | **$\ge 3.0$**<br>Citations accurately back claim | **$< 2.0$**<br>Citations point to wrong text | `query`, `response`, `citations`, `cited_source_text` |
| **`CitationCompleteness`**<br>`RAG-016` &bull; `citation_completeness` | Measures whether all important factual claims include appropriate citations. | Score (0–4) | **$\ge 3.0$**<br>All major claims properly cited | **$< 2.0$**<br>Claims made without citations | `response`, `citations` |
| **`CitationRelevance`**<br>`RAG-017` &bull; `citation_relevance` | Evaluates if the cited source document is directly relevant to the specific claim. | Score (0–4) | **$\ge 3.0$**<br>Source is pertinent to claim | **$< 2.0$**<br>Cited source is irrelevant | `claim`, `citation`, `source_text` |
| **`ClaimSupport`**<br>`RAG-018` &bull; `claim_supported` | Tests whether a single isolated claim is directly corroborated by given evidence. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Claim is directly supported | **$P < 0.50$**<br>Evidence does not back claim | `claim`, `evidence` |
| **`ContradictionDetection`**<br>`RAG-019` &bull; `contradiction` | Detects if the response directly contradicts or refutes facts stated in the context. | Risk $P \le \text{max}$ | **$P \le 0.20$**<br>No contradictions found | **$P > 0.20$**<br>Contradicts retrieved context | `context`, `response` |
| **`ContextConflictDetection`**<br>`RAG-020` &bull; `context_conflict` | Flags when two or more retrieved source documents provide conflicting information. | Risk $P \le \text{max}$ | **$P \le 0.30$**<br>Sources are in agreement | **$P > 0.30$**<br>Sources present conflicting facts | `query`, `retrieved_context` |

---

### 2. Agent Metrics Guide (22 Metrics)

> 📓 **Interactive Notebook**: Open [`examples/notebook/agent_metrics_evaluation.ipynb`](https://github.com/VinayChaudhari1996/LayaEval/blob/main/examples/notebook/agent_metrics_evaluation.ipynb) for runnable code examples of each Agent metric.

| Metric & ID | Definition | Type / Scale | Good Score (Pass) | Bad Score (Fail) | Required Inputs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`IntentResolution`**<br>`AG-001` &bull; `intent` | Classifies the user's primary intent (information, action, troubleshooting, etc.). | Choice (Category) | **Confidence $\ge 0.70$**<br>Correct intent identified | **Low confidence**<br>Misclassified or ambiguous intent | `query` |
| **`TaskCompletion`**<br>`AG-002` &bull; `task_completed` | Verifies whether the agent fully finished the task and produced the expected result. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Task completed successfully | **$P < 0.50$**<br>Task aborted or left incomplete | `task`, `agent_trace`, `final_result` |
| **`TaskAdherence`**<br>`AG-003` &bull; `task_adherence` | Measures how strictly the agent adhered to instructions, constraints, and guidelines. | Score (0–4) | **$\ge 3.0$**<br>Followed instructions fully | **$< 2.0$**<br>Ignored or broke constraints | `instructions`, `query`, `agent_trace`, `response` |
| **`GoalAchievement`**<br>`AG-004` &bull; `goal_achieved` | Checks whether the agent's actions achieved the true underlying objective. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>End goal accomplished | **$P < 0.50$**<br>Underlying goal not achieved | `goal`, `agent_trace`, `final_result` |
| **`ToolSelection`**<br>`AG-005` &bull; `tool_selection` | Identifies which tool is the optimal choice for the current step in the workflow. | Choice (Category) | **Confidence $\ge 0.70$**<br>Optimal tool selected | **Low confidence**<br>Wrong tool selected | `task`, `available_tools`, `selected_tool` |
| **`ToolSelectionCorrectness`**<br>`AG-006` &bull; `tool_selection_correct` | Verifies whether the tool chosen by the agent was appropriate for the task step. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Correct tool selected | **$P < 0.50$**<br>Inappropriate tool chosen | `task`, `tool_definition`, `selected_tool` |
| **`ToolCallAccuracy`**<br>`AG-007` &bull; `tool_call_accuracy` | Evaluates whether the generated tool call was structured and executed correctly. | Score (0–4) | **$\ge 3.0$**<br>Accurate, valid tool invocation | **$< 2.0$**<br>Malformed or invalid tool call | `task`, `tool_definition`, `tool_call` |
| **`ToolInputAccuracy`**<br>`AG-008` &bull; `tool_input_accuracy` | Checks whether tool input arguments are correct, schema-compliant, and accurate. | Score (0–4) | **$\ge 3.0$**<br>Valid and correct arguments | **$< 2.0$**<br>Invalid, missing, or fake params | `task`, `tool_schema`, `tool_input` |
| **`ToolOutputUtilization`**<br>`AG-009` &bull; `tool_output_utilization` | Measures how well the agent leveraged tool return values in next steps or answers. | Score (0–4) | **$\ge 3.0$**<br>Correctly used tool output | **$< 2.0$**<br>Ignored or misinterpreted output | `task`, `tool_output`, `subsequent_actions`, `final_response` |
| **`ToolNecessity`**<br>`AG-010` &bull; `tool_needed` | Evaluates whether invoking this tool was truly necessary or could be avoided. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Tool was genuinely necessary | **$P < 0.50$**<br>Unnecessary or extraneous call | `task`, `available_context`, `tool_call` |
| **`ToolRedundancy`**<br>`AG-011` &bull; `tool_call_redundant` | Flags whether the agent repeated a call for data it already had in prior turns. | Risk $P \le \text{max}$ | **$P \le 0.30$**<br>Non-redundant, clean execution | **$P > 0.30$**<br>Wasted, redundant tool call | `task`, `prior_tool_calls`, `current_tool_call` |
| **`ToolSequenceCorrectness`**<br>`AG-012` &bull; `tool_sequence` | Evaluates whether the multi-step ordering of tool calls was logical and optimal. | Score (0–4) | **$\ge 3.0$**<br>Optimal, logical sequence | **$< 2.0$**<br>Disordered or illogical steps | `task`, `executed_sequence`, `expected_sequence` |
| **`NavigationEfficiency`**<br>`AG-013` &bull; `navigation_efficiency` | Evaluates how directly the agent reached the solution without useless steps or loops. | Score (0–4) | **$\ge 3.0$**<br>Direct path with few extra steps | **$< 2.0$**<br>Inefficient path with retry loops | `task`, `agent_trace`, `expected_workflow` |
| **`AgentGroundedness`**<br>`AG-014` &bull; `agent_groundedness` | Measures if the agent's final answer is supported by observed tool outputs. | Score (0–4) | **$\ge 3.0$**<br>Backed by tool observations | **$< 2.0$**<br>Claims unsupported by tool data | `tool_outputs`, `context`, `response` |
| **`AgentHallucinationRisk`**<br>`AG-015` &bull; `agent_hallucination` | Detects if the agent fabricated action outcomes not present in execution evidence. | Risk $P \le \text{max}$ | **$P \le 0.30$**<br>No fabricated results | **$P > 0.30$**<br>Fictitious action results claimed | `tool_outputs`, `context`, `response` |
| **`AgentResponseRelevance`**<br>`AG-016` &bull; `agent_response_relevance` | Evaluates how directly the agent's final message responds to the user's prompt. | Score (0–4) | **$\ge 3.0$**<br>Directly answers user request | **$< 2.0$**<br>Off-topic or misses the request | `query`, `response` |
| **`AgentAnswerCompleteness`**<br>`AG-017` &bull; `agent_answer_completeness` | Checks whether the agent's final response covers all parts of the assigned task. | Score (0–4) | **$\ge 3.0$**<br>Thorough, complete resolution | **$< 2.0$**<br>Incomplete, leaves items pending | `task`, `response` |
| **`AgentCorrectness`**<br>`AG-018` &bull; `agent_correct` | Compares the agent's final outcome against expected ground-truth answers. | Score (0–4) | **$\ge 3.0$**<br>Matches ground-truth outcome | **$< 2.0$**<br>Incorrect or conflicting outcome | `task`, `final_result`, `ground_truth` |
| **`AgentAbstention`**<br>`AG-019` &bull; `should_abstain` | Verifies whether the agent properly abstains from unauthorized or impossible actions. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Safely refuses out-of-scope tasks | **$P < 0.50$**<br>Recklessly executes unsafe action | `task`, `context`, `response`, `available_tools` |
| **`PolicyCompliance`**<br>`AG-020` &bull; `policy_compliant` | Checks whether the agent's actions and responses adhered to policies & guardrails. | $P(\text{true})$ (0–1) | **$P \ge 0.50$**<br>Fully compliant with policy | **$P < 0.50$**<br>Policy violation detected | `policy`, `task`, `agent_trace`, `response` |
| **`PromptInjectionDetection`**<br>`AG-021` &bull; `prompt_injection` | Detects direct user jailbreak attempts or instruction overrides in user prompts. | Risk $P \le \text{max}$ | **$P \le 0.20$**<br>Safe, benign prompt | **$P > 0.20$**<br>Direct injection attack detected | `user_input`, `system_instructions` |
| **`IndirectPromptInjectionDetection`**<br>`AG-022` &bull; `indirect_prompt_injection` | Detects hidden adversarial commands inside retrieved third-party text or tool data. | Risk $P \le \text{max}$ | **$P \le 0.20$**<br>Safe third-party content | **$P > 0.20$**<br>Indirect injection payload found | `retrieved_or_tool_content`, `system_instructions` |

---

## CLI Usage

LayaEval includes a command-line interface for testing, batch benchmarking, and regression comparison:

```bash
# List all 42 available metrics with descriptions
layaeval list-metrics

# Evaluate a JSONL dataset with selected metrics
layaeval run dataset.jsonl --metrics groundedness hallucination task_completed --output report.json

# Compare candidate run against a baseline to check for regressions
layaeval compare baseline_report.json candidate_report.json
```

---

## Development & Testing

Run unit tests and lint checks using `uv`:

```bash
# Install dev dependencies
uv sync --all-extras

# Run test suite
uv run pytest

# Check code formatting and linting
uv run ruff check .
```

---

## Acknowledgements & Special Thanks

Special thanks and gratitude to the **[Laya](https://github.com/NandhaKishorM/laya)** open-source contributors and community. LayaEval is powered by the foundational non-autoregressive neural architectures, latency breakthroughs, and high-performance Model Context Protocol (MCP) server engineered by the Laya project.

- **Laya GitHub Repository**: [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
- **Laya MCP Ecosystem & Docs**: [github.com/NandhaKishorM/laya/blob/main/docs](https://github.com/NandhaKishorM/laya/blob/main/docs)

---

## Citation

If you use LayaEval or its metric suite in your research, pipelines, or benchmarks, please cite:

```bibtex
@software{layaeval2026,
  author = {Chaudhari, Vinay},
  title = {LayaEval: Ultra-Fast Neural Evaluation Framework for Agent and RAG Systems},
  year = {2026},
  url = {https://github.com/VinayChaudhari1996/LayaEval},
  publisher = {GitHub}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/VinayChaudhari1996/LayaEval/blob/main/LICENSE) file for details.
