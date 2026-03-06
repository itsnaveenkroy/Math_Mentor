"""
Parser Agent - takes raw input text (possibly noisy from OCR/audio)
and extracts a clean, structured math problem from it.
"""

from utils.llm import chat


SYSTEM_PROMPT = """You are a math problem parser. Your job is to take raw input 
(which might be messy OCR text or a voice transcription) and extract a clean, 
well-formatted math problem from it.

Return JSON with these fields:
- "problem_text": the cleaned-up math problem statement
- "problem": same as problem_text (alias)
- "topic": your best guess at the problem type. Pick from:
  algebra, calculus, probability, linear_algebra, trigonometry, geometry, number_theory, other
- "variables": list of variables involved (e.g. ["x", "y"])
- "constraints": list of constraints or conditions (e.g. ["x > 0"])
- "given": any given values or conditions (as a list of strings)
- "find": what needs to be found/solved (as a string)
- "needs_clarification": true if the input is ambiguous, incomplete, or unclear; false otherwise
- "clarification_reason": if needs_clarification is true, explain what is unclear

If the input is too garbled to understand, set "problem" to empty string, 
set "needs_clarification" to true, and add "error": "description of what went wrong".

Example output:
{
  "problem_text": "Solve x^2 - 5x + 6 = 0",
  "problem": "Solve x^2 - 5x + 6 = 0",
  "topic": "algebra",
  "variables": ["x"],
  "constraints": ["x > 0"],
  "given": ["x^2 - 5x + 6 = 0"],
  "find": "values of x",
  "needs_clarification": false
}"""


def parse(raw_text, input_source="text"):
    """
    Parse raw input into a structured problem.
    input_source: 'text', 'image', or 'audio' - gives context about
    what kind of noise to expect.
    """
    if not raw_text or not raw_text.strip():
        return {
            "problem_text": "",
            "problem": "",
            "topic": "other",
            "variables": [],
            "constraints": [],
            "given": [],
            "find": "",
            "type_hint": "other",
            "needs_clarification": True,
            "clarification_reason": "empty input",
            "error": "empty input",
        }

    source_note = ""
    if input_source == "image":
        source_note = (
            "Note: this came from OCR so there might be garbled symbols. "
            "Common OCR mistakes: y2 means y^2, V means sqrt, F might mean =, "
            "~ might mean minus. Try to figure out the intended math."
        )
    elif input_source == "audio":
        source_note = (
            "Note: this came from speech-to-text so math notation might be "
            "written out in words (e.g. 'x squared' means x^2)."
        )

    user_msg = f"{source_note}\n\nRaw input:\n{raw_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    result = chat(messages, want_json=True)

    if result is None:
        return {
            "problem_text": raw_text,
            "problem": raw_text,  # fall back to raw text
            "topic": "other",
            "variables": [],
            "constraints": [],
            "given": [],
            "find": "unknown",
            "type_hint": "other",
            "needs_clarification": False,
            "error": "LLM failed to parse - using raw text as fallback",
        }

    # make sure all expected fields exist
    result.setdefault("problem", raw_text)
    result.setdefault("problem_text", result["problem"])
    result.setdefault("topic", result.get("type_hint", "other"))
    result.setdefault("variables", [])
    result.setdefault("constraints", [])
    result.setdefault("given", [])
    result.setdefault("find", "")
    result.setdefault("type_hint", result.get("topic", "other"))
    result.setdefault("needs_clarification", False)

    return result
