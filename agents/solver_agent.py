"""
Solver Agent - does the actual math. Uses a combination of
LLM reasoning + sympy computation for reliability.
"""

from utils.llm import chat
from utils import math_tools


SYSTEM_PROMPT = """You are a math solver. Solve the problem step by step.

Return ONLY this JSON (no other text):
{
  "solution": "final answer (short)",
  "steps": [
    {"step": 1, "description": "brief description", "work": "math expression"}
  ],
  "method": "method name",
  "confidence": 0.0 to 1.0
}

Rules:
- Keep each step's "work" under 80 chars
- Max 5 steps
- No markdown in JSON values
- No LaTeX, use plain text like x^2, sqrt(x)"""


def solve(parsed_problem, route_info, similar_problems=None):
    """
    Solve the math problem. Uses LLM for reasoning + sympy for computation.
    Also considers similar past problems from memory for pattern reuse.
    """
    problem_text = parsed_problem.get("problem", "")
    if not problem_text:
        return {
            "solution": "",
            "steps": [],
            "method": "",
            "confidence": 0.0,
            "computation": None,
            "error": "no problem to solve",
        }

    tools_needed = route_info.get("tools_needed", [])
    strategy = route_info.get("strategy", "")
    context_chunks = route_info.get("context", [])

    # try sympy computation first if applicable
    comp_result = _try_computation(problem_text, tools_needed, parsed_problem)

    # build context from KB
    kb_context = ""
    if context_chunks:
        kb_pieces = [c["text"] for c in context_chunks[:3]]
        kb_context = f"\n\nRelevant reference material:\n" + "\n---\n".join(kb_pieces)

    comp_context = ""
    if comp_result and comp_result.get("result"):
        comp_context = f"\n\nComputational verification: {comp_result['result']}"

    # include similar past problems for pattern reuse
    similar_context = ""
    if similar_problems:
        past = []
        for sp in similar_problems[:2]:
            past.append(sp.get("text", "")[:200])
        if past:
            similar_context = "\n\nSimilar previously solved problems:\n" + "\n---\n".join(past)

    user_msg = (
        f"Problem: {problem_text}\n"
        f"Strategy hint: {strategy}\n"
        f"Variables: {parsed_problem.get('variables', [])}\n"
        f"Given: {parsed_problem.get('given', [])}\n"
        f"Find: {parsed_problem.get('find', '')}"
        f"{kb_context}{comp_context}{similar_context}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    result = chat(messages, want_json=True)

    if result is None:
        # if LLM fails but we have computation, use that
        if comp_result and comp_result.get("result"):
            return {
                "solution": str(comp_result["result"]),
                "steps": [{"step": 1, "description": "computed via sympy", "work": str(comp_result["result"])}],
                "method": comp_result.get("tool", "computation"),
                "confidence": 0.7,
                "computation": comp_result,
                "error": None,
            }
        return {
            "solution": "",
            "steps": [],
            "method": "",
            "confidence": 0.0,
            "computation": None,
            "error": "LLM solver failed and no computation available",
        }

    result.setdefault("solution", "")
    result.setdefault("steps", [])
    result.setdefault("method", "")
    result.setdefault("confidence", 0.5)
    result["computation"] = comp_result

    return result


def _try_computation(problem_text, tools_needed, parsed_problem):
    """
    Try to solve (or partially solve) using sympy.
    This gives us a ground truth to compare the LLM against.
    """
    given = parsed_problem.get("given", [])

    # try different tools based on what the router suggested
    for tool_name in tools_needed:
        try:
            if tool_name == "equation_solver" and given:
                # try solving the first equation
                for eq in given:
                    result = math_tools.solve_equation(eq)
                    if result.get("solutions"):
                        return {"tool": "equation_solver", "result": result["solutions"], "raw": result}

            elif tool_name == "derivative":
                result = math_tools.compute_derivative(problem_text)
                if result.get("result"):
                    return {"tool": "derivative", "result": result["result"], "raw": result}

            elif tool_name == "integral":
                result = math_tools.compute_integral(problem_text)
                if result.get("result"):
                    return {"tool": "integral", "result": result["result"], "raw": result}

            elif tool_name == "limit":
                result = math_tools.compute_limit(problem_text)
                if result.get("result"):
                    return {"tool": "limit", "result": result["result"], "raw": result}

            elif tool_name == "matrix_op":
                result = math_tools.compute_matrix_op(problem_text)
                if result.get("result"):
                    return {"tool": "matrix_op", "result": result["result"], "raw": result}

            elif tool_name == "probability":
                result = math_tools.compute_probability(problem_text)
                if result.get("result") is not None:
                    return {"tool": "probability", "result": result["result"], "raw": result}

        except Exception:
            # not a big deal, we'll rely on the LLM
            continue

    # also try a generic eval on the given conditions
    for eq in given:
        try:
            result = math_tools.safe_eval(eq)
            if result.get("result") is not None:
                return {"tool": "eval", "result": result["result"], "raw": result}
        except Exception:
            pass

    return None
