"""
Handles LLM calls with Groq as primary and Ollama as fallback.
Uses the openai client lib since both providers speak the same protocol.
"""

import json
import re
from openai import OpenAI
from utils.config import Config

# keep max_tokens reasonable so free models don't time out or
# produce truncated output that breaks json parsing
DEFAULT_MAX_TOKENS = 1024


def _get_groq_client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=Config.GROQ_API_KEY,
        timeout=60,
    )


def _get_ollama_client():
    return OpenAI(
        base_url=Config.OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=60,
    )


def _repair_truncated_json(text):
    """
    Try to fix JSON that got cut off mid-stream (common with free models
    that hit token limits). We close open strings, arrays, and braces.
    """
    # strip trailing incomplete escape sequences
    text = re.sub(r'\\$', '', text.rstrip())

    # if we're inside an unclosed string, close it
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        text += '"'

    # close any open brackets/braces
    opens = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ('{', '['):
            opens.append(ch)
        elif ch == '}' and opens and opens[-1] == '{':
            opens.pop()
        elif ch == ']' and opens and opens[-1] == '[':
            opens.pop()

    for bracket in reversed(opens):
        text += ']' if bracket == '[' else '}'

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json(text):
    """
    Pull a JSON object out of model output. Handles markdown fences,
    extra text, and truncated output from token-limited responses.
    """
    text = text.strip()

    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # strip markdown fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', text)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # find the first { ... } block (greedy)
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # find opening { and try to repair from there (truncated response)
    idx = cleaned.find('{')
    if idx >= 0:
        fragment = cleaned[idx:]
        repaired = _repair_truncated_json(fragment)
        if repaired is not None:
            return repaired

    raise ValueError(f"couldn't parse JSON from model output: {text[:300]}")


def chat(messages, temperature=0.2, want_json=False, max_tokens=None):
    """
    Send a chat completion request. Tries Groq first, falls back to Ollama.
    If want_json=True we parse the result into a dict.
    max_tokens caps reply length (default 1024) to avoid truncation on free models.
    """
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS

    # nudge the model to keep it json-only
    if want_json:
        if messages and messages[0]["role"] == "system":
            if "JSON" not in messages[0]["content"]:
                messages[0]["content"] += (
                    "\n\nIMPORTANT: Reply ONLY with valid JSON. Keep strings short. "
                    "No markdown fences. No extra text."
                )

    kwargs = dict(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # --- try groq ---
    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(model=Config.GROQ_MODEL, **kwargs)
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("groq returned empty content")
        content = content.strip()

        if want_json:
            return _extract_json(content)
        return content

    except Exception as e:
        print(f"[llm] groq failed: {e}, trying ollama...")

    # --- fallback to ollama ---
    try:
        client = _get_ollama_client()
        # Use higher token limit for the local model to avoid truncation.
        ollama_max = max(max_tokens, 2048)
        resp = client.chat.completions.create(
            model=Config.OLLAMA_MODEL, messages=messages,
            temperature=temperature, max_tokens=ollama_max,
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("ollama returned empty content")
        content = content.strip()

        if want_json:
            return _extract_json(content)
        return content

    except Exception as e:
        print(f"[llm] ollama also failed: {e}")
        if want_json:
            return None  # let caller handle gracefully
        raise RuntimeError(f"both groq and ollama failed. last error: {e}")
