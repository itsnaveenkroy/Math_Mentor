"""
Explainer Agent - takes a verified solution and generates a
student-friendly explanation with clear step-by-step reasoning.
"""

from utils.llm import chat
from memory.memory_store import get_topic_stats


SYSTEM_PROMPT = """You are a math tutor. Explain the solution clearly but briefly.

Return ONLY this JSON (no other text):
{
  "explanation": "step-by-step explanation (max 300 words, plain text)",
  "key_concepts": ["concept1", "concept2"],
  "common_mistakes": ["mistake1"],
  "difficulty_rating": "easy / medium / hard",
  "related_topics": ["topic1"],
  "summary": "one sentence answer"
}

Rules:
- Keep explanation under 300 words
- No LaTeX, use plain text like x^2, sqrt(x)
- No markdown formatting in values"""


def explain(parsed_problem, solver_result, verification):
    """
    Generate a student-friendly explanation of the solution.
    Takes into account the user's history if available.
    """
    problem_text = parsed_problem.get("problem", "")
    solution = verification.get("verified_solution", solver_result.get("solution", ""))
    steps = solver_result.get("steps", [])
    issues = verification.get("issues", [])

    # check if user has struggled with this topic before
    topic_context = ""
    try:
        stats = get_topic_stats()
        topic = parsed_problem.get("type_hint", "other")
        if topic in stats:
            s = stats[topic]
            if s["avg_confidence"] < 0.6:
                topic_context = (
                    f"\nNote: this student has asked about {topic} before and seemed "
                    f"to struggle (avg confidence: {s['avg_confidence']:.0%}). "
                    "Be extra clear and thorough."
                )
    except Exception:
        pass

    steps_text = ""
    if steps:
        steps_text = "\n".join(
            f"Step {s.get('step', '?')}: {s.get('description', '')} | {s.get('work', '')}"
            for s in steps
        )

    issues_text = ""
    if issues:
        issues_text = f"\nVerification notes: {', '.join(issues)}"

    user_msg = (
        f"Problem: {problem_text}\n"
        f"Solution: {solution}\n"
        f"Method: {solver_result.get('method', 'not specified')}\n"
        f"Steps:\n{steps_text}"
        f"{issues_text}"
        f"{topic_context}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    result = chat(messages, want_json=True)

    if result is None:
        # fallback: construct basic explanation from what we have
        basic_explanation = f"**Solution:** {solution}\n\n"
        if steps:
            basic_explanation += "**Steps:**\n"
            for s in steps:
                basic_explanation += f"- {s.get('description', '')}: {s.get('work', '')}\n"

        return {
            "explanation": basic_explanation,
            "key_concepts": [],
            "common_mistakes": [],
            "difficulty_rating": "medium",
            "related_topics": [],
            "summary": solution,
        }

    result.setdefault("explanation", solution)
    result.setdefault("key_concepts", [])
    result.setdefault("common_mistakes", [])
    result.setdefault("difficulty_rating", "medium")
    result.setdefault("related_topics", [])
    result.setdefault("summary", solution)

    return result
