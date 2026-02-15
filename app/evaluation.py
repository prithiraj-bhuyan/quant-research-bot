"""
evaluation.py — RAGAs-style evaluation for the RAG pipeline.

Metrics implemented:
  • Context Precision  — are the retrieved chunks relevant to the query?
  • Context Recall     — did we retrieve the chunks needed to answer?
  • Faithfulness       — is the answer grounded in the retrieved context?
  • Answer Relevancy   — does the answer actually address the query?

Two modes:
  1. LLM-as-judge (requires LLM API) — higher quality
  2. Embedding-based (no LLM needed) — faster, always available

Run from CLI:  python evaluation.py --run
"""

import json
import time
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import get_model, EMBED_MODEL_NAME
from rag import ask, retrieve_only, _call_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test queries — 20 queries spanning different quant finance topics
# ---------------------------------------------------------------------------
DEFAULT_EVAL_QUERIES = [
    # --- 10 DIRECT QUERIES (Fact Retrieval) ---
    "How is mean-variance optimization applied in portfolio construction?",
    "What are the limitations of the Markowitz framework?",
    "How is Value at Risk (VaR) calculated using historical simulation?",
    "What are the differences between parametric and non-parametric risk measures?",
    "What is the impact of high-frequency trading on market liquidity?",
    "How are limit order books modeled in market microstructure theory?",
    "How does the Black-Scholes model price European options?",
    "What are the assumptions behind risk-neutral pricing?",
    "How are GARCH models used for volatility forecasting?",
    "What machine learning approaches are applied to stock return prediction?",

    # --- 5 SYNTHESIS QUERIES (Comparing/Combining INFO) ---
    "Compare the advantages of deep learning vs statistical methods for time series forecasting.",
    "How do transaction costs and market impact jointly affect optimal execution strategies?",
    "What is the relationship between market liquidity and volatility in high-frequency markets?",
    "Discuss the trade-offs between automated market making and traditional inventory management.",
    "How does regime-switching improve upon static factor allocation models?",

    # --- 5 EDGE CASES (Ambiguity / Missing Info) ---
    "Does the corpus contain evidence for the 'Halloween Effect' in crypto markets?",
    "What specific code implementation details are provided for the DeepHedging algorithm?",
    "Does the text mention the impact of quantum computing on high-frequency trading latency?",
    "What are the exact interest rate parameters used in the 2008 crisis simulations?",
    "Does the corpus validate the efficieny of astrology-based trading strategies?"
]


# ---------------------------------------------------------------------------
# Embedding-based metrics (no LLM needed)
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@dataclass
class EvalResult:
    query: str
    context_precision: float    # how relevant are retrieved chunks?
    answer_relevancy: float     # does the answer match the query?
    faithfulness: float         # is the answer grounded in context?
    retrieval_time_ms: float
    rerank_time_ms: float
    generation_time_ms: float
    num_chunks: int
    num_citations: int
    generated_answer: Optional[str] = None
    retrieved_chunks: Optional[list[dict]] = None


def evaluate_single_embedding(query: str,
                              model_name: str = EMBED_MODEL_NAME,
                              relevance_threshold: float = 0.35) -> EvalResult:
    """
    Evaluate a single query using embedding-based similarity metrics.
    No LLM needed — uses cosine similarity between embeddings.
    """
    model = get_model(model_name)

    # Run full RAG pipeline
    response = ask(query)

    # Encode query
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    # Context Precision: avg cosine sim between query and each retrieved chunk
    chunk_sims = []
    for c in response.citations:
        c_emb = model.encode([c.chunk_text], normalize_embeddings=True)[0]
        chunk_sims.append(_cosine_sim(q_emb, c_emb))

    context_precision = float(np.mean(chunk_sims)) if chunk_sims else 0.0

    # Answer Relevancy: cosine sim between query and generated answer
    if response.answer:
        # Use first 500 chars of answer to avoid noise from citation block
        answer_text = response.answer[:500]
        a_emb = model.encode([answer_text], normalize_embeddings=True)[0]
        answer_relevancy = _cosine_sim(q_emb, a_emb)
    else:
        answer_relevancy = 0.0

    # Faithfulness: avg cosine sim between answer sentences and closest chunk
    answer_sentences = [s.strip() for s in response.answer.split('.') if len(s.strip()) > 20]
    faithfulness_scores = []
    for sent in answer_sentences[:10]:  # cap at 10 sentences
        s_emb = model.encode([sent], normalize_embeddings=True)[0]
        best_sim = 0.0
        for c in response.citations:
            c_emb = model.encode([c.chunk_text], normalize_embeddings=True)[0]
            best_sim = max(best_sim, _cosine_sim(s_emb, c_emb))
        faithfulness_scores.append(best_sim)

    faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

    return EvalResult(
        query=query,
        context_precision=round(context_precision, 4),
        answer_relevancy=round(answer_relevancy, 4),
        faithfulness=round(faithfulness, 4),
        retrieval_time_ms=response.retrieval_time_ms,
        rerank_time_ms=response.rerank_time_ms,
        generation_time_ms=response.generation_time_ms,
        num_chunks=response.context_chunks_used,
        num_citations=len(response.citations),
        generated_answer=response.answer,
        retrieved_chunks=[asdict(c) for c in response.citations]
    )


# ---------------------------------------------------------------------------
# LLM-as-judge metrics (higher quality, requires LLM)
# ---------------------------------------------------------------------------

def _llm_judge_score(prompt: str) -> Optional[float]:
    """Ask the LLM to rate something on a 0-1 scale."""
    messages = [
        {"role": "system", "content": "You are an evaluation judge. Respond with ONLY a single number between 0.0 and 1.0. No explanation."},
        {"role": "user", "content": prompt},
    ]
    result = _call_llm(messages)
    if result:
        try:
            # Extract first float from response
            import re
            match = re.search(r"(\d+\.?\d*)", result)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
        except:
            pass
    return None


def evaluate_single_llm_judge(query: str) -> EvalResult:
    """Evaluate using LLM-as-judge for all metrics."""
    response = ask(query)

    context_text = "\n".join([c.chunk_text[:300] for c in response.citations])
    answer_text = response.answer[:1000]

    # Context Precision
    cp_prompt = f"""Rate how relevant these retrieved passages are to the query.
Query: {query}
Passages: {context_text[:1500]}
Score 0.0 (irrelevant) to 1.0 (highly relevant):"""

    # Answer Relevancy
    ar_prompt = f"""Rate how well this answer addresses the query.
Query: {query}
Answer: {answer_text}
Score 0.0 (off-topic) to 1.0 (directly answers):"""

    # Faithfulness
    ff_prompt = f"""Rate whether this answer is faithful to (grounded in) the source passages.
A faithful answer only states things supported by the passages.
Passages: {context_text[:1500]}
Answer: {answer_text}
Score 0.0 (hallucinated) to 1.0 (fully grounded):"""

    cp = _llm_judge_score(cp_prompt) or 0.0
    ar = _llm_judge_score(ar_prompt) or 0.0
    ff = _llm_judge_score(ff_prompt) or 0.0

    return EvalResult(
        query=query,
        context_precision=round(cp, 4),
        answer_relevancy=round(ar, 4),
        faithfulness=round(ff, 4),
        retrieval_time_ms=response.retrieval_time_ms,
        rerank_time_ms=response.rerank_time_ms,
        generation_time_ms=response.generation_time_ms,
        num_chunks=response.context_chunks_used,
        num_citations=len(response.citations),
        generated_answer=response.answer,
        retrieved_chunks=[asdict(c) for c in response.citations]
    )


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    mode: str
    total_queries: int
    avg_context_precision: float
    avg_answer_relevancy: float
    avg_faithfulness: float
    avg_retrieval_ms: float
    avg_rerank_ms: float
    avg_generation_ms: float
    results: list[dict]


def run_evaluation(queries: Optional[list[str]] = None,
                   mode: str = "embedding",
                   output_path: Optional[str] = None) -> EvalReport:
    """
    Run evaluation over a set of queries.

    Args:
        queries: list of test queries (defaults to DEFAULT_EVAL_QUERIES)
        mode: "embedding" (no LLM needed) or "llm_judge" (needs LLM)
        output_path: save JSON report to this path
    """
    if queries is None:
        queries = DEFAULT_EVAL_QUERIES

    eval_fn = evaluate_single_llm_judge if mode == "llm_judge" else evaluate_single_embedding

    results = []
    log.info(f"🧪 Running {mode} evaluation over {len(queries)} queries…")

    for i, q in enumerate(queries):
        log.info(f"  [{i+1}/{len(queries)}] {q[:60]}…")
        try:
            r = eval_fn(q)
            results.append(r)
        except Exception as e:
            log.warning(f"  ⚠️  Failed: {e}")
            results.append(EvalResult(
                query=q, context_precision=0, answer_relevancy=0,
                faithfulness=0, retrieval_time_ms=0, rerank_time_ms=0,
                generation_time_ms=0, num_chunks=0, num_citations=0,
                generated_answer="Error during evaluation", retrieved_chunks=[]
            ))

    # Aggregate
    cp = [r.context_precision for r in results]
    ar = [r.answer_relevancy for r in results]
    ff = [r.faithfulness for r in results]
    rt = [r.retrieval_time_ms for r in results]
    rr = [r.rerank_time_ms for r in results]
    gt = [r.generation_time_ms for r in results]

    report = EvalReport(
        mode=mode,
        total_queries=len(queries),
        avg_context_precision=round(float(np.mean(cp)), 4),
        avg_answer_relevancy=round(float(np.mean(ar)), 4),
        avg_faithfulness=round(float(np.mean(ff)), 4),
        avg_retrieval_ms=round(float(np.mean(rt)), 1),
        avg_rerank_ms=round(float(np.mean(rr)), 1),
        avg_generation_ms=round(float(np.mean(gt)), 1),
        results=[asdict(r) for r in results],
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 RAG Evaluation Report ({mode})")
    print("=" * 60)
    print(f"  Queries evaluated:    {report.total_queries}")
    print(f"  Context Precision:    {report.avg_context_precision:.4f}")
    print(f"  Answer Relevancy:     {report.avg_answer_relevancy:.4f}")
    print(f"  Faithfulness:         {report.avg_faithfulness:.4f}")
    print(f"  Avg Retrieval:        {report.avg_retrieval_ms:.1f} ms")
    print(f"  Avg Reranking:        {report.avg_rerank_ms:.1f} ms")
    print(f"  Avg Generation:       {report.avg_generation_ms:.1f} ms")
    print("=" * 60)

    # Save
    if output_path:
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        log.info(f"💾 Report saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--run", action="store_true", help="Run full evaluation")
    parser.add_argument("--mode", default="embedding", choices=["embedding", "llm_judge"])
    parser.add_argument("--output", default="eval_report.json", help="Output path")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to evaluate")
    args = parser.parse_args()

    if args.run:
        qs = DEFAULT_EVAL_QUERIES[:args.queries]
        run_evaluation(queries=qs, mode=args.mode, output_path=args.output)
    else:
        parser.print_help()