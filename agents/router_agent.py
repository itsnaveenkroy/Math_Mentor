"""
Router Agent - decides which solving strategy + topic area to use,
and pulls relevant context from the knowledge base.
"""

from utils.llm import chat
from rag.retriever import retrieve_context


SYSTEM_PROMPT = """You are a math problem router. Given a parsed math problem, decide:
1. What topic area it falls under
2. What solving strategy to use
3. Whether we need any special tools (sympy, matrix ops, etc.)

Return JSON:
{
  "topic": "one of: algebra, calculus, probability, linear_algebra, trigonometry, geometry, number_theory, other",
  "strategy": "brief description of how to approach this problem",
  "tools_needed": ["list of tools like: equation_solver, derivative, integral, matrix_op, probability, limit, none"],
  "difficulty": "easy / medium / hard",
  "subtopics": ["more specific tags, e.g. quadratic, optimization, bayes_theorem"]
}"""


def route(parsed_problem):
    """
    Figure out the best approach for a parsed problem.
    Also grabs relevant knowledge base context.
    """
    problem_text = parsed_problem.get("problem", "")
    type_hint = parsed_problem.get("type_hint", "other")

    if not problem_text:
        return {
            "topic": "other",
            "strategy": "no problem to route",
            "tools_needed": [],
            "difficulty": "easy",
            "subtopics": [],
            "context": [],
        }

    # get relevant KB context while we're at it
    context_chunks = retrieve_context(problem_text, k=5)

    user_msg = (
        f"Problem: {problem_text}\n"
        f"Parser's type hint: {type_hint}\n"
        f"Variables: {parsed_problem.get('variables', [])}\n"
        f"Find: {parsed_problem.get('find', '')}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    result = chat(messages, want_json=True)

    if result is None:
        # fallback - use the parser's hint
        result = {
            "topic": type_hint,
            "strategy": "direct solving",
            "tools_needed": [],
            "difficulty": "medium",
            "subtopics": [],
        }

    result.setdefault("topic", type_hint)
    result.setdefault("strategy", "direct solving")
    result.setdefault("tools_needed", [])
    result.setdefault("difficulty", "medium")
    result.setdefault("subtopics", [])

    # attach the retrieved context
    result["context"] = context_chunks

    return result
