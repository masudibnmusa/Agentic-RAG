# Agentic RAG

**An LLM-controlled retrieval loop that decides when to retrieve, rewrites queries, retrieves iteratively, and verifies its own answers.**

Standard RAG follows a fixed "retrieve once, then generate" pipeline, and it fails quietly when the first retrieval is bad. Agentic RAG turns the LLM into a controller that notices weak evidence and corrects it, or honestly says *"I couldn't find this in the sources."*

---

## Core Idea

Retrieval is a **tool the model calls**, not a fixed pipeline step.

The agent plans, calls search tools (possibly several, with different queries and sources), grades the results, and either answers or tries again. A verification pass checks that every claim is supported by retrieved text. If the corpus doesn't contain the answer, the agent abstains instead of guessing.

## Features

- **Routing**: decides whether retrieval is needed at all, and which source to use (docs, SQL, web, code)
- **Query rewriting**: resolves pronouns from chat history and produces standalone, keyword-rich queries
- **Decomposition**: splits complex questions into sub-questions ("Compare X and Y" becomes one query for each)
- **Multiple retrieval strategies**: vector, keyword (BM25), hybrid with rank fusion, and metadata filters
- **Relevance and sufficiency grading**: LLM grader or cross-encoder reranker scores
- **Iterative retrieval**: rewrite, widen or narrow filters, switch source, or decompose further, all within a step budget
- **Grounded answers**: generated only from retrieved evidence, with citations
- **Self-verification**: claim-level groundedness checks against cited chunks
- **Revision or abstention**: fix unsupported claims, retrieve more, or say the answer isn't in the corpus
- **Full tracing**: every query, grade, retry, and decision is logged and viewable
- **Built-in evaluation**: baseline RAG comparison, benchmark runner, and ablations

## How It Works

| Stage | What happens |
|-------|--------------|
| **1. Route** | Decide whether the question needs retrieval, and which source fits |
| **2. Rewrite / Decompose** | Turn the raw question into better search queries or sub-questions |
| **3. Retrieve** | The agent calls `search_docs(query, filters)` using vector, keyword, hybrid, or other tools |
| **4. Grade** | Judge whether the retrieved chunks are relevant and sufficient |
| **5. Iterate** | If evidence is weak: rewrite, change filters, switch source, or break the question down further |
| **6. Draft** | Generate an answer using only retrieved evidence, with citations |
| **7. Verify** | Extract atomic claims and check each against its cited chunks |
| **8. Revise or Abstain** | Fix unsupported claims, retrieve more, or explicitly abstain |
| **9. Trace** | Log every decision for debugging and analysis |

## Architecture

```
User question (+ chat history)
        ↓
router.py ── no retrieval needed? ──► answer directly
        ↓ retrieval needed
rewriter.py / decomposer.py → search queries
        ↓
┌──────────────── agent_loop.py (repeats, within budget) ─────────────────┐
│  tools/ (vector / keyword / hybrid / SQL / web)                         │
│        ↓                                                                │
│  reranker.py → evidence_pool.py                                         │
│        ↓                                                                │
│  relevance_grader.py + sufficiency_grader.py                            │
│        ↓                                                                │
│  sufficient? ── no ──► rewrite query / change source / decompose more   │
│        ↓ yes                                                            │
└─────────────────────────────────────────────────────────────────────────┘
        ↓
answer_generator.py (grounded, cited)
        ↓
claim_extractor.py → groundedness_checker.py
        ↓
all supported? ── no ──► answer_reviser.py (fix, or retrieve more)
        ↓ yes
Final answer with citations (or abstention) + trace
```

## Project Structure

```
agentic-rag/
├── app/
│   ├── main.py                  # Entry point (API / CLI / Streamlit)
│   ├── config.py                # Models, step budget, thresholds, source list
│   ├── agent/                   # Controller: loop, state, router, planner, stopping criteria, LLM client
│   ├── query/                   # Rewriter, decomposer, HyDE, query expander
│   ├── tools/                   # Tools exposed to the LLM (vector, keyword, hybrid, metadata, SQL, web, fetch)
│   ├── retrieval/               # Vector store wrapper, reranker, evidence pool
│   ├── grading/                 # Relevance and sufficiency graders
│   ├── generation/              # Grounded answer generator and abstention
│   ├── verification/            # Claim extraction, groundedness, citation validation, answer revision
│   ├── ingestion/               # Loader, chunker, indexer (embeddings + BM25)
│   └── observability/           # Trace logger/viewer and cost tracker
├── evaluation/
│   ├── datasets/                # Single-hop, multi-hop, unanswerable, chat-follow-up questions
│   ├── baseline_rag.py          # Fixed retrieve-then-generate for comparison
│   ├── benchmark_runner.py      # Agentic vs. baseline on the same set
│   ├── metrics.py               # Correctness, faithfulness, retrieval recall, abstention accuracy
│   └── ablation_runner.py       # Toggle rewriting, grading, verification
├── data/
│   ├── docs/                    # Source documents
│   ├── vector_db/               # Vector index
│   └── traces/                  # Saved run traces
├── tests/
├── .env.example
├── requirements.txt
├── run.sh
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- An Anthropic API key (Claude with native tool use)
- A vector store: Chroma (default), Qdrant, or pgvector

### Installation

```bash
git clone https://github.com/masudibnmusa/Agentic-RAG.git
cd agentic-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=your_key_here
# Optional
COHERE_API_KEY=your_key_here     # if using Cohere rerank
TAVILY_API_KEY=your_key_here     # if enabling web search fallback
```

### Ingest Your Documents

Place files in `data/docs/`, then build the index:

```bash
python -m app.ingestion.indexer --source data/docs --reset
```

## Configuration

Settings live in `app/config.py`:

| Setting | Description | Example |
|---------|-------------|---------|
| `PLANNER_MODEL` | Stronger model for planning and answer drafting | `claude-sonnet-4-6` |
| `GRADER_MODEL` | Smaller, cheaper model for grading and claim extraction | `claude-haiku-4-5` |
| `MAX_STEPS` | Maximum agent loop iterations | `5` |
| `MAX_TOKENS_BUDGET` | Per-question token cap | `50000` |
| `RELEVANCE_THRESHOLD` | Minimum grader/reranker score to keep a chunk | `0.5` |
| `TOP_K` | Chunks retrieved per search call | `8` |
| `ENABLED_SOURCES` | Sources the router may choose from | `["docs"]` |
| `ENABLE_VERIFICATION` | Toggle the groundedness pass | `true` |

> Tip: use a cheaper model for high-volume calls (grading, claim extraction) and a stronger one for planning and drafting. Model names above are placeholders; set whichever models you have access to.

## Usage

### CLI

```bash
python -m app.main --query "How does the refund policy differ for annual vs. monthly plans?"
```

### API

```bash
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What changed in v2.0?", "history": []}'
```

### Streamlit (optional)

```bash
streamlit run app/main.py
```

### View a Trace

```bash
python -m app.observability.trace_viewer data/traces/<run_id>.json
```

## Example Trace

> Replace this with a real trace from your own corpus. A run where the first retrieval fails and the agent recovers is the most convincing demo.

```
Question: "Compare the rate limits for the Pro and Team plans."

[1] ROUTE        → retrieval needed (source: docs)
[2] DECOMPOSE    → ["Pro plan rate limits", "Team plan rate limits"]
[3] RETRIEVE     → search_docs("Pro plan rate limits")       → 8 chunks
[4] GRADE        → 2/8 relevant, insufficient
[5] REWRITE      → "Pro tier requests per minute quota"
[6] RETRIEVE     → search_docs(hybrid, rewritten query)      → 8 chunks
[7] GRADE        → 5/8 relevant, sufficient for sub-question 1
[8] RETRIEVE     → search_docs("Team plan rate limits")      → 8 chunks
[9] GRADE        → 6/8 relevant, sufficient for sub-question 2
[10] DRAFT       → answer with 4 citations
[11] VERIFY      → 5 claims extracted, 4 supported, 1 unsupported
[12] REVISE      → removed unsupported claim
[13] FINAL       → answer + citations | steps: 13 | tokens: 14,200 | latency: 9.1s
```

## Evaluation

The `evaluation/` module compares the agent against a fixed retrieve-then-generate baseline on the same questions.

**Dataset categories**

- Single-hop factual questions
- Multi-hop questions requiring several retrievals
- Unanswerable questions (to test abstention)
- Chat follow-ups with pronouns and implicit context

**Metrics**

- Answer correctness
- Faithfulness (groundedness)
- Retrieval recall
- Abstention accuracy
- Cost (tokens) and latency per question
- Grader accuracy (are the graders themselves reliable?)

**Run the benchmark**

```bash
python -m evaluation.benchmark_runner --dataset evaluation/datasets/multi_hop.json
```

**Run ablations** (toggle rewriting, grading, and verification to measure each component's contribution)

```bash
python -m evaluation.ablation_runner --disable rewriting
python -m evaluation.ablation_runner --disable verification
```

### Results

> Fill in after running your benchmarks.

| Metric | Baseline RAG | Agentic RAG |
|--------|--------------|-------------|
| Answer correctness | – | – |
| Faithfulness | – | – |
| Retrieval recall | – | – |
| Abstention accuracy | – | – |
| Avg. tokens / question | – | – |
| Avg. latency | – | – |

## Testing

```bash
pytest tests/
```

Key tests cover the rewriter, relevance grader, groundedness checker, stopping criteria, and the full agent loop.

## Security Notes

- **SQL tool**: keep it strictly **read-only**, run it with a restricted database role, and limit accessible tables.
- **Prompt injection**: treat all retrieved document and web content as **untrusted data**, never as instructions. Keep retrieved text clearly delimited in prompts.
- **Budgets**: enforce step, token, and time limits so a confused agent can't loop or spend indefinitely.
- **Secrets**: never commit `.env`; keep API keys out of traces and logs.

## Contributing

Contributions are welcome. Please open an issue to discuss major changes first, and include tests for new behavior.

## License

MIT 