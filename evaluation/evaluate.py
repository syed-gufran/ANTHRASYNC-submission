"""Lightweight, automated evaluation harness.

Run (after building the index):

    python -m evaluation.evaluate

It runs every case in ``test_cases.json`` through the RAG pipeline and reports:

  * Answer accuracy  — do answerable questions contain the expected fact(s)?
  * Citation accuracy — is the correct source document cited?
  * Hallucination avoidance — are out-of-scope questions correctly refused?

This is deliberately keyword/source based rather than LLM-graded: it's fast,
free, deterministic, and good enough to catch regressions. The README explains
how it would extend to an LLM-judge (faithfulness/relevance) for richer scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.rag_graph import NO_ANSWER, answer_question

CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"


def _looks_refused(result: dict) -> bool:
    """An answer counts as 'refused' if it cites nothing and flags unavailability."""
    answer = result["answer"].lower()
    refusal_markers = ("could not find", "not available", "no information", "don't have")
    return not result["sources"] and any(m in answer for m in refusal_markers)


def _keywords_present(answer: str, keywords: list[str]) -> bool:
    low = answer.lower()
    return all(kw.lower() in low for kw in keywords)


def _source_cited(result: dict, expected_source: str) -> bool:
    return any(s["document"] == expected_source for s in result["sources"])


def run_evaluation() -> None:
    cases = json.loads(CASES_PATH.read_text())

    answerable = [c for c in cases if c["should_answer"]]
    unanswerable = [c for c in cases if not c["should_answer"]]

    answer_hits = 0
    citation_hits = 0
    refusal_hits = 0

    print(f"\nRunning {len(cases)} test cases "
          f"({len(answerable)} answerable, {len(unanswerable)} out-of-scope)\n")
    print(f"{'Q':<55} {'ANSWER':<8} {'CITATION':<10} {'CONF'}")
    print("-" * 85)

    for case in cases:
        result = answer_question(case["question"])
        q = (case["question"][:52] + "...") if len(case["question"]) > 52 else case["question"]

        if case["should_answer"]:
            ok_answer = _keywords_present(result["answer"], case["expected_keywords"])
            ok_citation = _source_cited(result, case["expected_source"])
            answer_hits += ok_answer
            citation_hits += ok_citation
            print(f"{q:<55} {'✅' if ok_answer else '❌':<8} "
                  f"{'✅' if ok_citation else '❌':<10} {result['confidence']:.2f}")
        else:
            ok_refusal = _looks_refused(result)
            refusal_hits += ok_refusal
            print(f"{q:<55} {'(refuse)':<8} "
                  f"{'✅' if ok_refusal else '❌':<10} {result['confidence']:.2f}")

    # --- Summary -------------------------------------------------------------
    def pct(num: int, den: int) -> str:
        return f"{(num / den * 100):.1f}%" if den else "n/a"

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Answer accuracy        : {pct(answer_hits, len(answerable))} "
          f"({answer_hits}/{len(answerable)})")
    print(f"Citation accuracy      : {pct(citation_hits, len(answerable))} "
          f"({citation_hits}/{len(answerable)})")
    print(f"Hallucination avoidance: {pct(refusal_hits, len(unanswerable))} "
          f"({refusal_hits}/{len(unanswerable)})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_evaluation()
