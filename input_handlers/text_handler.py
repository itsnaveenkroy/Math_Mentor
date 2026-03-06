"""
Text input handler. Not much to do here honestly -
just basic validation and cleanup.
"""

import re


def process_text_input(text):
    """
    Take raw text input from the user, clean it up,
    and check if it actually looks like a math problem.
    """
    if not text or not text.strip():
        return {
            "text": "",
            "confidence": 0.0,
            "needs_hitl": False,
            "error": "Empty input",
            "message": "You didn't type anything.",
        }

    cleaned = text.strip()
    # collapse weird whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # quick sanity check - does it have any math-ish content?
    has_math_chars = bool(re.search(r'[\d+\-*/=^()√∫∑πxyz]', cleaned, re.IGNORECASE))
    has_math_words = bool(re.search(
        r'(solve|find|calculate|compute|evaluate|simplify|prove|'
        r'integral|derivative|limit|matrix|equation|factor|root|'
        r'probability|area|volume|sum|product|average|mean|'
        r'triangle|circle|angle|graph|function|domain|range)',
        cleaned, re.IGNORECASE
    ))

    is_math = has_math_chars or has_math_words

    if not is_math and len(cleaned) < 10:
        return {
            "text": cleaned,
            "confidence": 0.3,
            "needs_hitl": True,
            "error": None,
            "message": "This doesn't look like a math problem. Did you mean something else?",
        }

    return {
        "text": cleaned,
        "confidence": 1.0,  # user typed it, so we trust them
        "needs_hitl": False,
        "error": None,
        "message": None,
    }
