"""
ragas_runner.py
===============
WHAT THIS FILE DOES (Feynman explanation)
-----------------------------------------
Think of this as an automated exam for your RAG system.

You prepare 20 questions you already know the answers to (test_dataset.json).
This script:
  1. Asks each question to your RAG system
  2. Collects: the answer given + the chunks retrieved
  3. Hands everything to RAGAS, which grades on 4 dimensions
  4. Saves results to evaluation/results/ with a timestamp
  5. Prints a pass/fail report

The 4 dimensions RAGAS grades:
  - Faithfulness:          Did the answer use only information from the retrieved chunks?
                           High score = no hallucination.
  - Response Relevancy:    Did the answer actually address the question asked?
                           High score = on-topic, useful answers.
  - Context Precision:     Of all chunks retrieved, how many were actually useful?
                           High score = retrieval is not pulling junk.
  - Context Recall:        Of all information needed, how much did retrieval find?
                           High score = retrieval is not missing important chunks.

HOW TO RUN
----------
    python evaluation/ragas_runner.py

WHAT TO DO WITH THE RESULTS
----------------------------
1. Save the scores — this is your baseline (Phase 1 result).
2. After Phase 4 (hybrid search), run this again.
3. Compare scores. The improvement is your engineering story.
4. Screenshot the before/after for your portfolio and interviews.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# ── Path setup ────────────────────────────────────────────────────────────────
# We need to import from src/, which is in the project root.
# This adds the project root to Python's module search path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "test_dataset.json"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Pass/fail thresholds — used to flag which metrics need improvement.
# These are industry baselines; adjust based on your use case.
THRESHOLDS: Dict[str, float] = {
    "faithfulness":             0.85,   # below 0.85 = hallucination risk
    "response_relevancy":       0.80,   # below 0.80 = off-topic answers
    "llm_context_precision":    0.75,   # below 0.75 = noisy retrieval
    "llm_context_recall":       0.80,   # below 0.80 = missing key chunks
}

# ── Step 1: Load test dataset ─────────────────────────────────────────────────

def load_test_dataset() -> List[Dict[str, str]]:
    """
    Load the ground-truth Q&A pairs from test_dataset.json.

    Each item must have:
        "question":     the question to ask the RAG system
        "ground_truth": the correct answer a human would give
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found: {DATASET_PATH}"
            "\n   Create evaluation/test_dataset.json before running."
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Skip any placeholder entries the user has not filled in yet
    valid = [
        d for d in data
        if "UPDATE THIS" not in d.get("ground_truth", "")
    ]
    skipped = len(data) - len(valid)

    print(f"✅ Dataset loaded: {len(valid)} samples ready, {skipped} placeholders skipped")
    if not valid:
        raise ValueError(
            "No valid samples found. Fill in ground_truth values in test_dataset.json."
        )
    return valid


# ── Step 2: Run each question through the RAG chain ──────────────────────────

def run_rag_on_dataset(test_data: List[Dict]) -> List[Dict[str, Any]]:
    """
    Ask each question to the RAG system and collect:
      - the answer it gave
      - the chunks it retrieved (as plain text strings)

    We use create_rag_chain_with_sources() because we need both the
    answer AND the retrieved documents. The simple chain only returns text.
    """
    from src.rag_chain import create_rag_chain_with_sources

    print("\n⏳ Running RAG chain on test dataset...")
    print("   (Each question hits the LLM — this takes ~1-2 min)\n")

    rag_chain = create_rag_chain_with_sources(retriever_type="multi_query")

    results = []
    for i, sample in enumerate(test_data, 1):
        question = sample["question"]
        print(f"  [{i:02d}/{len(test_data):02d}] {question[:65]}...")

        try:
            output = rag_chain.invoke({"question": question})
            results.append({
                "user_input":         question,
                "reference":          sample["ground_truth"],
                "response":           output["answer"],
                # RAGAS expects retrieved_contexts as a list of strings,
                # not LangChain Document objects
                "retrieved_contexts": [
                    doc.page_content for doc in output["sources"]
                ],
            })
        except Exception as e:
            print(f"     ⚠️  Skipped (error): {e}")

    print(f"\n✅ Collected {len(results)} responses")
    return results


# ── Step 3: Build the RAGAS dataset ──────────────────────────────────────────

def build_ragas_dataset(results: List[Dict]):
    """
    Convert our list of dicts into a RAGAS EvaluationDataset.

    RAGAS uses its own data schema (SingleTurnSample) so we map our
    field names to its field names here.
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],           # the question
            reference=r["reference"],             # ground truth answer
            response=r["response"],               # what our RAG said
            retrieved_contexts=r["retrieved_contexts"],  # chunks as strings
        )
        for r in results
    ]
    return EvaluationDataset(samples=samples)


# ── Step 4: Run RAGAS evaluation ──────────────────────────────────────────────

def run_evaluation(ragas_dataset) -> Dict[str, float]:
    """
    Run all 4 RAGAS metrics against the dataset.

    IMPORTANT: RAGAS itself uses an LLM to grade answers (LLM-as-judge).
    We inject our own Groq LLM so it doesn't try to use OpenAI by default.
    Same for embeddings (needed by ResponseRelevancy).
    """
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecision,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.llm import get_llm
    from src.embedding import get_embeddings

    print("\n📊 Running RAGAS evaluation...")
    print("   (RAGAS uses the LLM to grade each answer — takes 3-5 minutes)\n")

    # Wrap our LLM and embeddings in RAGAS-compatible wrappers
    evaluator_llm        = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # Each metric gets injected with our LLM/embeddings
    # Without this, RAGAS tries to use OpenAI and throws an auth error
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecision(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    result = evaluate(dataset=ragas_dataset, metrics=metrics)

    # Return as a clean {metric_name: score} dict
    return {str(k): float(v) for k, v in result.items()}


# ── Step 5: Save and print results ────────────────────────────────────────────

def save_and_print_results(scores: Dict[str, float], raw_results: List[Dict]) -> None:
    """
    Save a timestamped JSON file to evaluation/results/
    and print a formatted summary table.

    The timestamp in the filename lets you compare runs over time:
      eval_20250615_143022.json  ← Phase 1 baseline
      eval_20250630_091544.json  ← Phase 4 after hybrid search
    """
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp":   timestamp,
                "num_samples": len(raw_results),
                "thresholds":  THRESHOLDS,
                "scores":      scores,
                "samples":     raw_results,
            },
            f, indent=2, ensure_ascii=False,
        )

    # ── Print summary ──────────────────────────────────────────────────────────
    width = 60
    print("\n" + "=" * width)
    print("  RAGAS EVALUATION RESULTS")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * width)

    all_passed = True
    for metric, score in scores.items():
        threshold = THRESHOLDS.get(metric, 0.75)
        passed    = score >= threshold
        status    = "✅ PASS" if passed else "❌ NEEDS WORK"
        bar       = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        if not passed:
            all_passed = False
        print(f"  {metric:<32} {score:.3f}  |{bar}|  {status}")

    print("─" * width)
    print(f"  Samples evaluated : {len(raw_results)}")
    print(f"  Results saved to  : {output_path.name}")
    print("=" * width)

    if all_passed:
        print("\n  🎉 All metrics above threshold.")
        print("  This is your Phase 1 baseline. Record these scores.")
        print("  Re-run after Phase 4 (hybrid search) to measure improvement.\n")
    else:
        failing = [m for m, s in scores.items() if s < THRESHOLDS.get(m, 0.75)]
        print(f"\n  ⚠️  Metrics needing improvement: {', '.join(failing)}")
        print("  Open LangSmith → look at traces for failing questions.")
        print("  Low context_recall = retrieval missing chunks → fix in Phase 4.")
        print("  Low faithfulness = LLM hallucinating → improve prompt in Phase 1.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  RAG EVALUATION  —  RAGAS + LangSmith")
    print("=" * 60 + "\n")

    test_data    = load_test_dataset()
    raw_results  = run_rag_on_dataset(test_data)
    ragas_data   = build_ragas_dataset(raw_results)
    scores       = run_evaluation(ragas_data)
    save_and_print_results(scores, raw_results)


if __name__ == "__main__":
    main()