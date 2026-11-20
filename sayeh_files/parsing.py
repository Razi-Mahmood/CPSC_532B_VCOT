"""Multiple-choice answer extraction for Qwen2.5-VL outputs.

Usage:
    from parsing import parse_answer
    letter, method = parse_answer(output_text, choices)   # e.g. ("B", "answer_phrase")
    # letter is None if nothing could be extracted -> log it as a parse failure.

`method` records which rule matched, so you can report how often the parser had
to fall back (useful for the "unparseable rate" in the write-up).
"""
import re
import string

LETTERS = string.ascii_uppercase  # A, B, C, ...


def format_choices(choices):
    """Render choices as 'A. text\nB. text...' — use this in prompts so the letters match."""
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def letter_to_index(letter):
    return None if letter is None else LETTERS.index(letter)


def index_to_letter(idx):
    return LETTERS[idx]


def _valid(letter, n):
    return letter is not None and letter in LETTERS[:n]


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# Phrases that introduce the final answer. Ordered from most to least explicit.
# The phrase part is case-insensitive via (?i:...), but the captured letter must be
# UPPERCASE, so "the answer is a magnet" does not parse as option A.
_ANSWER_PATTERNS = [
    r"(?i:(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is|:)\s*(?:option\s*)?)[\(\[\*]*\s*([A-Z])\b",
    r"(?i:option|choice)\s*[\(\[\*]*([A-Z])\b[\)\]\*]*\s*(?i:is\s+(?:the\s+)?(?:correct|right|answer))",
    r"(?i:\banswer\b)[^A-Za-z\n]{0,10}([A-Z])\b",
]


def parse_answer(text, choices, strict=False):
    """Return (letter or None, method).

    strict=True: only accept an explicit answer (rules 1-3). Use this for full CoT
    outputs, where the reasoning mentions option texts and the choice-text fallback
    can pick the wrong one (pilot Q2: a truncated, garbled CoT was parsed as
    "weather" just because the word appeared in the reasoning).
    """
    if text is None:
        return None, "empty"
    n = len(choices)
    t = text.strip()
    if not t:
        return None, "empty"

    # 1. Whole output is just a letter: "B", "(B)", "B.", "**B**"
    m = re.fullmatch(r"[\s\*\(\[]*([A-Z])[\s\*\)\]\.:]*", t)
    if m and _valid(m.group(1), n):
        return m.group(1), "bare_letter"

    # 2. Explicit answer phrase. Take the LAST match, since CoT text may mention
    #    options while reasoning ("option A is wrong because ...").
    for pat in _ANSWER_PATTERNS:
        hits = [h for h in re.findall(pat, t) if _valid(h, n)]
        if hits:
            return hits[-1], "answer_phrase"

    # 3. Output starts with a letter label: "B. 12 cm" or "(B) 12 cm"
    m = re.match(r"[\s\*\(\[]*([A-Z])[\)\]\.:]\s", t)
    if m and _valid(m.group(1), n):
        return m.group(1), "leading_letter"

    if strict:
        return None, "unparsed_strict"

    # 4. Exact choice text appears. If several do, take the one mentioned last.
    nt = _norm(t)
    best, best_pos = None, -1
    for i, c in enumerate(choices):
        nc = _norm(str(c))
        if not nc:
            continue
        pos = nt.rfind(nc)
        # Avoid matching a choice that is a substring of a longer choice
        # ("cat" inside "cats") by preferring longer matches at equal position.
        if pos > best_pos or (pos == best_pos and best is not None and len(nc) > len(_norm(str(choices[best])))):
            best, best_pos = i, pos
    if best is not None and best_pos >= 0:
        return LETTERS[best], "choice_text"

    # 5. A lone capital letter in parentheses near the end, e.g. "... so (C)."
    hits = [h for h in re.findall(r"\(([A-Z])\)", t) if _valid(h, n)]
    if hits:
        return hits[-1], "paren_letter"

    return None, "unparsed"
