# RiskGraph — Agentic Portfolio Risk Synthesizer

An agentic RAG system that analyzes investment portfolios for concentration risk, sector imbalance, and regulatory red flags, built with LangGraph and LangChain.

## What It Does

Given a set of portfolio holdings (fund name, weight, sector), the system:

1. Retrieves relevant fund documents and regulatory text using RAG (TF-IDF retrieval over embedded fund prospectuses and mock SEBI-style guidance)
2. Runs deterministic checks for single-fund concentration and sector concentration
3. Cross-references retrieved regulatory text to flag scheme-category compliance issues
4. Routes high-severity findings through an escalation step
5. Synthesizes a structured risk report, grounded in the retrieved evidence

## Architecture

A LangGraph state machine with six nodes and one conditional edge:

```
parse_portfolio -> retrieve_context -> run_risk_checks -> regulatory_check
                                                                |
                                              [HIGH severity?] -+- yes -> escalate_node -> synthesize_report
                                                                +- no  -----------------> synthesize_report
```

- **Retrieval** uses a custom `SimpleVectorStore` (TF-IDF + cosine similarity) built on LangChain `Document` objects. It follows a minimal vector-store interface, so it can be swapped for a neural embedding store (Chroma, FAISS with HuggingFace or OpenAI embeddings) without changing any graph node.
- **Risk arithmetic** (concentration thresholds, sector totals) is computed in plain Python, not delegated to an LLM, so numeric checks are never subject to hallucination.
- **Regulatory flags** are traced back to specific retrieved documents, not invented by the model.
- **Report synthesis** calls Claude via the Anthropic API if `ANTHROPIC_API_KEY` is set. If no key is present, it falls back to a deterministic template built from the same structured flags, so the pipeline always produces output.

## Setup

```bash
pip install langgraph langchain-core anthropic scikit-learn numpy
export ANTHROPIC_API_KEY=sk-...   # optional, enables LLM-written narrative report
```

## Usage

```bash
python portfolio_risk_synthesizer.py             # run the demo on a sample portfolio
python portfolio_risk_synthesizer.py --evaluate   # run the evaluation suite
```

## Evaluation

The project includes a test suite of four synthetic portfolios with known expected risk flags, used to measure precision, recall, and escalation-routing accuracy rather than relying on manual spot-checks.

Results from the current version:

| Metric | Score |
|---|---|
| Average precision | 0.88 |
| Average recall | 1.00 |
| Escalation routing accuracy | 1.00 |

One known false positive: a debt fund held at 45% of NAV is flagged by the same 40% concentration threshold used for equity funds. Debt and equity schemes carry different regulatory concentration limits in practice, so the fix is to parameterize the threshold by fund category rather than applying a single global cutoff. This is a documented limitation, not a hidden one.

## Design Choices Worth Knowing

**TF-IDF retrieval instead of neural embeddings.** Fund documents use precise, repeated vocabulary (fund names, sector labels, regulation clause language), where lexical retrieval is fast, requires no model download, and is easy to test. Neural embeddings can be substituted behind the same `SimpleVectorStore` interface if higher semantic recall is needed.

**Mock regulatory text.** The SEBI-style guidance embedded in the project is illustrative and simplified to mirror the shape of real concentration and categorization norms. It is not a reproduction of any actual circular and should not be used for real compliance decisions.

**Graceful LLM fallback.** The synthesis step never blocks the pipeline on API availability. This mirrors a real production pattern: an optional enrichment step failing should not take down the system.

## File Structure

Everything lives in a single file, `portfolio_risk_synthesizer.py`, organized into seven sections: knowledge base, vector store, state and data models, graph nodes, graph construction, evaluation suite, and CLI entry point.
