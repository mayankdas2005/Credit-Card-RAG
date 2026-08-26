"""
Evaluation harness for the credit-card RAG advisor.

Runs each case in eval/cases.py through the real compiled LangGraph agent (agent.rag_agent),
applies deterministic checks against the final response text and agent state, and prints a
pass/fail report plus a JSON results file under eval/results/.

Usage:
    python eval/run_eval.py
"""
import os
import sys
import json
import uuid
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from agent.rag_agent import rag_agent
from langchain_core.messages import HumanMessage
from eval.cases import CASES


def evaluate_check(check: dict, answer_lower: str, state: dict):
    check_type = check["type"]

    if check_type == "contains_any":
        phrases = [p.lower() for p in check["phrases"]]
        ok = any(p in answer_lower for p in phrases)
        return ok, "ok" if ok else f"expected any of {check['phrases']}"

    if check_type == "contains_all":
        phrases = [p.lower() for p in check["phrases"]]
        missing = [p for p in phrases if p not in answer_lower]
        return (len(missing) == 0), "ok" if not missing else f"missing: {missing}"

    if check_type == "not_contains":
        phrases = [p.lower() for p in check["phrases"]]
        found = [p for p in phrases if p in answer_lower]
        return (len(found) == 0), "ok" if not found else f"unexpectedly found: {found}"

    if check_type == "intent_equals":
        actual = state.get("intent")
        ok = actual == check["value"]
        return ok, "ok" if ok else f"expected intent='{check['value']}', got '{actual}'"

    return False, f"unknown check type '{check_type}'"


def run_case(case: dict) -> dict:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    for turn in case["turns"]:
        final_state = rag_agent.invoke({"messages": [HumanMessage(content=turn)]}, config=config)

    answer = final_state["messages"][-1].content
    answer_lower = answer.lower()

    check_results = []
    case_passed = True
    for check in case["checks"]:
        ok, detail = evaluate_check(check, answer_lower, final_state)
        check_results.append({"type": check["type"], "passed": ok, "detail": detail})
        if not ok:
            case_passed = False

    return {
        "id": case["id"],
        "category": case["category"],
        "turns": case["turns"],
        "passed": case_passed,
        "intent": final_state.get("intent"),
        "retrieval_grounded": final_state.get("retrieval_grounded"),
        "answer": answer,
        "checks": check_results,
    }


def main():
    print(f"Running {len(CASES)} eval cases against the live agent...\n")
    results = []
    for case in CASES:
        print(f"  [{case['category']:22s}] {case['id']:45s} ", end="", flush=True)
        try:
            result = run_case(case)
        except Exception as e:
            result = {
                "id": case["id"], "category": case["category"], "turns": case["turns"],
                "passed": False, "intent": None, "retrieval_grounded": None,
                "answer": None, "checks": [], "error": str(e),
            }
        results.append(result)
        print("PASS" if result["passed"] else "FAIL")
        if not result["passed"]:
            for check in result.get("checks", []):
                if not check["passed"]:
                    print(f"        - {check['type']}: {check['detail']}")
            if result.get("error"):
                print(f"        - ERROR: {result['error']}")

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print(f"\n{'=' * 60}")
    print(f"Overall: {passed}/{total} passed ({100 * passed / total:.0f}%)")

    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, [0, 0])
        by_category[cat][1] += 1
        if r["passed"]:
            by_category[cat][0] += 1

    print("\nBy category:")
    for cat, (cat_passed, cat_total) in sorted(by_category.items()):
        print(f"  {cat:25s} {cat_passed}/{cat_total}")

    os.makedirs(os.path.join(ROOT_DIR, "eval", "results"), exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(ROOT_DIR, "eval", "results", f"eval_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total": total,
            "passed": passed,
            "by_category": {k: {"passed": v[0], "total": v[1]} for k, v in by_category.items()},
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull report written to {report_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
