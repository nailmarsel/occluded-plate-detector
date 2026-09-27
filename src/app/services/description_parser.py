"""
Split a free-text car description into a plate fragment and a visual part.

Examples::

    "белая Toyota Camry, номер А1**ВС77"  -> plate "A1??BC77", text "белая Toyota Camry"
    "red hatchback A123BC 777"             -> plate "A123BC777", text "red hatchback"
    "чёрный BMW X5 2015 года"              -> plate "",          text unchanged

The plate fragment is later used as an Elasticsearch wildcard filter and the
remaining text is embedded with the CLIP text encoder.
"""

import re
from dataclasses import dataclass

from app.services.pipeline import CYRILLIC_TO_LATIN_PLATE_CHARS

PLATE_LETTERS = set("ABEKMHOPCTYX")
DIGITS = set("0123456789")
WILDCARDS = set("*?")
MAX_PLATE_LEN = 9

# Russian civil plate: L DDD LL DD(D). Used to validate fragments that were
# found without an explicit "номер"/"plate" keyword.
PLATE_TEMPLATE = "LDDDLLDDD"

PLATE_KEYWORD_RE = re.compile(
    r"(?<![\w/])("
    r"гос\.?\s*номер\w*|госномер\w*|номерн\w*\s+знак\w*|номер\w*|"
    r"г/н|грз|гнз|"
    r"license\s+plate|plate(?:\s+number)?|number\s+plate|reg(?:istration)?\.?"
    r")(?!\w)\s*[:№#\-]?",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[^\s,;:()\[\]{}\"'«»]+")
# "... с номером X" / "with plate X": drop the preposition together with the plate.
LEADING_PREPOSITION_RE = re.compile(
    r"(?:(?<=\s)|^)(?:с|со|with|и|and)\s*$", re.IGNORECASE
)
# "X 77 регион" / "X rus": region words belong to the plate, not the look.
TRAILING_REGION_RE = re.compile(
    r"^\s*(?:регион\w*|рег\.?|рус|rus|region)(?!\w)", re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedDescription:
    original: str
    plate_fragment: str
    text: str


def _normalize_token(token: str) -> str:
    token = token.upper().translate(CYRILLIC_TO_LATIN_PLATE_CHARS)
    token = token.replace("_", "?").replace("*", "?")
    return token.strip(".!")


def _is_plate_chars(token: str) -> bool:
    return bool(token) and all(
        ch in PLATE_LETTERS or ch in DIGITS or ch in WILDCARDS for ch in token
    )


def _fits_template(fragment: str) -> bool:
    """True if fragment can be a contiguous part of an L DDD LL DDD plate."""
    if not fragment or len(fragment) > len(PLATE_TEMPLATE):
        return False
    for offset in range(len(PLATE_TEMPLATE) - len(fragment) + 1):
        ok = True
        for ch, slot in zip(fragment, PLATE_TEMPLATE[offset:], strict=False):
            if ch in WILDCARDS:
                continue
            if slot == "L" and ch not in PLATE_LETTERS:
                ok = False
                break
            if slot == "D" and ch not in DIGITS:
                ok = False
                break
        if ok:
            return True
    return False


def _has_digit_or_wildcard(fragment: str) -> bool:
    return any(ch in DIGITS or ch in WILDCARDS for ch in fragment)


def _looks_like_plate_without_keyword(fragment: str) -> bool:
    if not _fits_template(fragment):
        return False
    has_wildcard = any(ch in WILDCARDS for ch in fragment)
    has_letter = any(ch in PLATE_LETTERS for ch in fragment)
    has_digit = any(ch in DIGITS for ch in fragment)
    if has_wildcard:
        return len(fragment) >= 3 and (has_letter or has_digit)
    # Short tokens like "E200", "X5M" are usually model names, not plates.
    return len(fragment) >= 5 and has_letter and has_digit


def _tokens(text: str, start: int = 0):
    for match in TOKEN_RE.finditer(text, start):
        yield match.start(), match.end(), _normalize_token(match.group())


def _plate_after_keyword(text: str) -> tuple[str, int, int] | None:
    for keyword in PLATE_KEYWORD_RE.finditer(text):
        fragment = ""
        span_end = keyword.end()
        for _tok_start, tok_end, token in _tokens(text, keyword.end()):
            gap = text[span_end:_tok_start]
            if fragment and ("," in gap or ";" in gap or "\n" in gap):
                break
            if not _is_plate_chars(token):
                break
            if len(fragment) + len(token) > MAX_PLATE_LEN:
                break
            fragment += token
            span_end = tok_end
        if fragment and _has_digit_or_wildcard(fragment):
            return fragment, keyword.start(), span_end
    return None


def _plate_without_keyword(text: str) -> tuple[str, int, int] | None:
    tokens = list(_tokens(text))
    for i, (start, end, token) in enumerate(tokens):
        if not _is_plate_chars(token) or not _fits_template(token):
            continue
        fragment = token
        span_end = end
        best: tuple[str, int] | None = None
        if _looks_like_plate_without_keyword(fragment):
            best = (fragment, span_end)
        # Plates are often typed with spaces: "А 123 ВС 77".
        for _next_start, next_end, next_token in tokens[i + 1:i + 4]:
            candidate = fragment + next_token
            if not _is_plate_chars(next_token) or not _fits_template(candidate):
                break
            fragment, span_end = candidate, next_end
            if _looks_like_plate_without_keyword(fragment):
                best = (fragment, span_end)
        if best:
            return best[0], start, best[1]
    return None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    return text.strip(" ,;:-—")


def parse_description(description: str) -> ParsedDescription:
    original = (description or "").strip()
    if not original:
        return ParsedDescription(original="", plate_fragment="", text="")

    found = _plate_after_keyword(original) or _plate_without_keyword(original)
    if not found:
        return ParsedDescription(
            original=original, plate_fragment="", text=_clean_text(original)
        )

    fragment, start, end = found
    before = LEADING_PREPOSITION_RE.sub("", original[:start])
    after = TRAILING_REGION_RE.sub("", original[end:])
    remaining = f"{before} {after}"
    return ParsedDescription(
        original=original,
        plate_fragment=fragment,
        text=_clean_text(remaining),
    )


def has_visual_content(text: str) -> bool:
    """True if text has at least one word worth embedding with CLIP."""
    return bool(re.search(r"[^\W\d_]{2,}", text))
