"""
OCR-based image input. Uses easyocr then cleans up
common math symbol misreads (sqrt, exponents, etc).
"""

import io
import re
import numpy as np
from PIL import Image


_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


# common OCR mistakes for math symbols
# important: patterns must NOT break things like tan2x, cos2x, etc.
OCR_FIXES = [
    # square root variants
    (r'[Vv][\s]?\(', 'sqrt('),       # V( -> sqrt(
    (r'\bV\s*(\d)', r'sqrt(\1'),      # V3 -> sqrt(3
    (r'[JjIl]\/', 'sqrt'),            # J/ or l/ -> sqrt
    # equals variants
    (r'(?<=\d)\s*F\s*(?=\d)', ' = '), # 3 F 4 -> 3 = 4
    # plus/minus
    (r'~', '-'),                      # tilde often means minus in ocr
    # arrows
    (r'7-0', '-> 0'),                 # "7-0" is OCR for arrow -> 0
    (r'7\s*-\s*>', '->'),             # another arrow variant
    # limit notation often mangled
    (r'\blim-\b', 'lim'),            # "lim-" -> "lim"
    (r'\b1_\s*lim', 'lim'),          # garbled prefix
    # fractions cleanup
    (r'(\d)\s*/\s*(\d)', r'\1/\2'),   # clean up fraction spacing
    # pi
    (r'\bTT\b', 'pi'),
    # misc cleanup
    (r'\?(?=x)', 'n'),               # ?x often means nx in OCR
    (r'\s{2,}', ' '),                 # collapse multiple spaces
]


def _clean_ocr_math(text):
    """
    Post-process OCR text to fix common math symbol mistakes.
    This is the main fix for sqrt / exponent detection issues.
    """
    result = text
    for pattern, replacement in OCR_FIXES:
        result = re.sub(pattern, replacement, result)

    return result.strip()


def _apply_learned_corrections(text):
    """Apply corrections from memory (learned from past user edits)."""
    try:
        from memory.memory_store import get_corrections
        corrections = get_corrections(source="image", n=50)
        for c in corrections:
            original = c.get("original", "")
            corrected = c.get("corrected", "")
            if original and corrected and original in text:
                text = text.replace(original, corrected)
    except Exception:
        pass  # memory not available, skip
    return text


def extract_text_from_image(image_bytes):
    """
    Run OCR on image bytes, return extracted text + confidence info.
    The text goes through math-specific cleanup and learned corrections afterward.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)

        reader = _get_reader()
        results = reader.readtext(img_np)

        if not results:
            return {
                "text": "",
                "confidence": 0.0,
                "details": [],
                "needs_hitl": True,
                "error": None,
                "message": "No text found in the image. Try a clearer photo or just type it out.",
            }

        texts = []
        confidences = []
        details = []

        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)
            details.append({"text": text, "confidence": round(conf, 3)})

        raw_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences)

        # apply math-specific fixes
        cleaned_text = _clean_ocr_math(raw_text)

        # apply learned corrections from past user edits
        cleaned_text = _apply_learned_corrections(cleaned_text)

        needs_hitl = avg_conf < 0.85 or any(c < 0.5 for c in confidences)
        # math images almost always need human review because OCR
        # can't reliably handle superscripts, fractions, greek letters
        needs_hitl = True
        message = (
            f"OCR confidence is {avg_conf:.0%}. "
            "Math OCR is tricky - please check and fix the extracted text."
        )

        return {
            "text": cleaned_text,
            "raw_text": raw_text,  # keep original for reference
            "confidence": round(avg_conf, 3),
            "details": details,
            "needs_hitl": needs_hitl,
            "error": None,
            "message": message,
        }

    except Exception as e:
        return {
            "text": "",
            "confidence": 0.0,
            "details": [],
            "needs_hitl": True,
            "error": str(e),
            "message": f"Image processing failed: {e}",
        }
