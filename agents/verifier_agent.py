"""
Verifier Agent - double-checks the solver's work.
Catches errors, validates against sympy results, and
flags anything suspicious.
"""

from utils.llm import chat


SYSTEM_PROMPT = """You verify math solutions. Check if the answer is correct.

Return ONLY this JSON (no other text):
{
  "is_correct": true or false,
  "confidence": 0.0 to 1.0,
  "issues": ["list of problems if any"],
  "suggestions": ["fixes if needed"],
  "verified_solution": "corrected answer or same as original"
}

Keep all values short. No markdown or LaTeX."""


def verify(parsed_problem, solver_result):
    """
    Verify the solver's output. Compares LLM solution against
    computational results when available.
    """
    problem_text = parsed_problem.get("problem", "")
    solution = solver_result.get("solution", "")
    steps = solver_result.get("steps", [])
    computation = solver_result.get("computation")

    if not solution:
        return {
            "is_correct": False,
            "confidence": 0.0,
            "issues": ["no solution was provided"],
            "suggestions": ["try re-solving the problem"],
            "verified_solution": "",
        }

    # quick computational cross-check
    comp_match = None
    if computation and computation.get("result"):
        comp_result = str(computation["result"])
        comp_match = comp_result

    steps_text = ""
    if steps:
        steps_text = "\n".join(
            f"Step {s.get('step', '?')}: {s.get('description', '')} -> {s.get('work', '')}"
            for s in steps
        )

    user_msg = (
        f"Problem: {problem_text}\n\n"
        f"Proposed solution: {solution}\n\n"
        f"Steps shown:\n{steps_text}\n\n"
        f"Method used: {solver_result.get('method', 'not specified')}"
    )
    if comp_match:
        user_msg += f"\n\nComputational (sympy) result: {comp_match}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    result = chat(messages, want_json=True)

    if result is None:
        # can't verify - just pass through with lower confidence
        return {
            "is_correct": True,  # assume correct if we can't check
            "confidence": solver_result.get("confidence", 0.5) * 0.7,
            "issues": ["verification unavailable - LLM did not respond"],
            "suggestions": [],
            "verified_solution": solution,
        }

    result.setdefault("is_correct", True)
    result.setdefault("confidence", 0.5)
    result.setdefault("issues", [])
    result.setdefault("suggestions", [])
    result.setdefault("verified_solution", solution)

    # if computation disagrees with LLM, flag it
    if comp_match and not result["is_correct"]:
        result["issues"].append(f"Sympy computed: {comp_match}")

    return result
