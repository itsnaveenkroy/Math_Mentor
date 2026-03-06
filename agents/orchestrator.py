"""
Orchestrator - runs the full pipeline: parse -> route -> solve -> verify -> explain.
Also handles memory storage and provides trace info for the UI.
"""

import time
import traceback

from agents import parser_agent, router_agent, solver_agent, verifier_agent, explainer_agent
from memory.memory_store import store_problem, find_similar_problems


def run_pipeline(raw_text, input_source="text"):
    """
    Full end-to-end pipeline. Takes raw text, returns everything
    the UI needs to display.

    Returns a dict with:
    - answer, explanation, confidence
    - agent_trace (what each agent did)
    - context_used (KB chunks)
    - needs_hitl (whether human review is recommended)
    - timing info
    """
    trace = []
    start_time = time.time()

    result = {
        "answer": "",
        "explanation": "",
        "confidence": 0.0,
        "agent_trace": trace,
        "context_used": [],
        "needs_hitl": False,
        "similar_problems": [],
        "key_concepts": [],
        "common_mistakes": [],
        "related_topics": [],
        "difficulty": "medium",
        "error": None,
        "timing": {},
    }

    try:
        # step 1: parse
        t0 = time.time()
        parsed = parser_agent.parse(raw_text, input_source=input_source)
        trace.append({
            "agent": "Parser",
            "input": raw_text[:200],
            "output": parsed,
            "time": round(time.time() - t0, 2),
        })

        if parsed.get("error") and not parsed.get("problem"):
            result["error"] = f"Could not parse input: {parsed['error']}"
            result["needs_hitl"] = True
            return result

        # if parser flags ambiguity, mark for human review
        if parsed.get("needs_clarification"):
            result["needs_hitl"] = True
            result["clarification_reason"] = parsed.get(
                "clarification_reason", "The problem seems ambiguous or incomplete."
            )

        problem_text = parsed.get("problem", raw_text)

        # check memory for similar past problems
        similar = find_similar_problems(problem_text, k=3)
        result["similar_problems"] = similar

        # step 2: route
        t0 = time.time()
        route_info = router_agent.route(parsed)
        trace.append({
            "agent": "Router",
            "input": problem_text[:200],
            "output": {
                "topic": route_info.get("topic"),
                "strategy": route_info.get("strategy"),
                "difficulty": route_info.get("difficulty"),
                "tools": route_info.get("tools_needed"),
            },
            "time": round(time.time() - t0, 2),
        })

        result["context_used"] = route_info.get("context", [])
        result["difficulty"] = route_info.get("difficulty", "medium")

        # step 3: solve
        t0 = time.time()
        solver_result = solver_agent.solve(parsed, route_info, similar_problems=similar)
        trace.append({
            "agent": "Solver",
            "input": f"{problem_text[:100]}... (strategy: {route_info.get('strategy', '')})",
            "output": {
                "solution": solver_result.get("solution", "")[:200],
                "method": solver_result.get("method"),
                "steps_count": len(solver_result.get("steps", [])),
                "has_computation": solver_result.get("computation") is not None,
            },
            "time": round(time.time() - t0, 2),
        })

        if solver_result.get("error") and not solver_result.get("solution"):
            result["error"] = f"Solver failed: {solver_result['error']}"
            result["needs_hitl"] = True
            return result

        # step 4: verify
        t0 = time.time()
        verification = verifier_agent.verify(parsed, solver_result)
        trace.append({
            "agent": "Verifier",
            "input": f"solution: {solver_result.get('solution', '')[:100]}",
            "output": {
                "is_correct": verification.get("is_correct"),
                "confidence": verification.get("confidence"),
                "issues": verification.get("issues", []),
            },
            "time": round(time.time() - t0, 2),
        })

        # step 5: explain
        t0 = time.time()
        explanation = explainer_agent.explain(parsed, solver_result, verification)
        trace.append({
            "agent": "Explainer",
            "input": f"verified solution for: {problem_text[:100]}",
            "output": {
                "has_explanation": bool(explanation.get("explanation")),
                "concepts": explanation.get("key_concepts", []),
                "difficulty": explanation.get("difficulty_rating"),
            },
            "time": round(time.time() - t0, 2),
        })

        # assemble final result
        result["answer"] = verification.get(
            "verified_solution", solver_result.get("solution", "")
        )
        result["explanation"] = explanation.get("explanation", "")
        result["confidence"] = verification.get("confidence", solver_result.get("confidence", 0.5))
        result["key_concepts"] = explanation.get("key_concepts", [])
        result["common_mistakes"] = explanation.get("common_mistakes", [])
        result["related_topics"] = explanation.get("related_topics", [])
        result["steps"] = solver_result.get("steps", [])
        result["method"] = solver_result.get("method", "")
        result["verification_issues"] = verification.get("issues", [])

        # flag for human review if confidence is low or verifier found problems
        if result["confidence"] < 0.6 or not verification.get("is_correct", True):
            result["needs_hitl"] = True

        # save to memory for future reference
        try:
            store_problem(
                problem_text,
                result["answer"],
                topic=route_info.get("topic", "general"),
                confidence=result["confidence"],
                context_used=route_info.get("context", []),
                verifier_outcome=verification,
            )
        except Exception:
            pass  # memory failure shouldn't break the pipeline

    except Exception as e:
        result["error"] = f"Pipeline error: {str(e)}"
        result["needs_hitl"] = True
        trace.append({
            "agent": "Orchestrator",
            "input": "error handling",
            "output": {"error": str(e), "traceback": traceback.format_exc()[-500:]},
            "time": 0,
        })

    result["timing"]["total"] = round(time.time() - start_time, 2)
    return result
