"""Splitting rationales into steps, and truncation (experiment 3).

IMPORTANT: strip the model's own final-answer sentence ("So the answer is B.")
from the rationale BEFORE truncating or feeding it back. Otherwise the 100% /
replay condition just copies the stated letter, and every perturbation that
keeps the last step looks artificially "unfaithful".

Sarthak's step labelling and the corruption experiment should import
split_steps() from here, so everyone agrees on what a "step" is.

Usage:
    from steps import strip_final_answer, split_steps, truncate, TRUNCATION_FRACTIONS
    rationale = strip_final_answer(cot_output)
    for frac in TRUNCATION_FRACTIONS:
        partial = truncate(rationale, frac)
        ...answer_given_rationale(image, question, partial)
"""
import math
import re

TRUNCATION_FRACTIONS = (0.0, 0.25, 0.5, 0.75)

# Sentences that state the final answer. Matched case-insensitively.
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*(?:\*\*)?\s*"
    r"(?:so,?\s+|therefore,?\s+|thus,?\s+|hence,?\s+)?"
    r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is|:)[^\n]*$",
    re.IGNORECASE | re.DOTALL,
)

# Abbreviations / patterns that should not end a sentence.
_ABBREV = ["e.g.", "i.e.", "etc.", "vs.", "Fig.", "fig.", "approx.", "Dr.", "Mr.", "Mrs.", "St.", "No."]
_PLACEHOLDER = "\u0000"


# Last sentence of the text if it contains an answer statement
# ("The answer is B.", "Therefore, the correct answer is B.", "... the correct answer is:").
_LAST_ANSWER_SENT = re.compile(r"(?:^|(?<=[.!?\n]))[^.!?\n]*\banswer\s*(?:is|:)[^\n]*$", re.IGNORECASE)
_CONCLUSION_START = re.compile(r"^\s*(?:therefore|thus|so|hence|given|in conclusion|since|overall)\b", re.IGNORECASE)


def _norm_text(s):
    return " " + re.sub(r"[^a-z0-9 ]", " ", s.lower()) + " "


def strip_final_answer(text, choices=None, drop_conclusion=False):
    """Remove the answer statement(s) at the end of a CoT output.

    Repeats until nothing changes, so both "Therefore, the correct answer is B."
    and a following "The answer is B." are removed (pilot Q3), as is a dangling
    "..., the correct answer is:" (pilot Q5).

    drop_conclusion=True (needs `choices`): also remove a final concluding
    sentence that names an answer option in words, e.g. "Given this information,
    the animal ... is the armored catfish." TEAM DECISION whether to use this;
    whichever you pick, use it for every condition.
    """
    if not text:
        return ""
    t = text.strip()
    prev = None
    while prev != t:
        prev = t
        t = _LAST_ANSWER_SENT.sub("", t).rstrip()
        t = re.sub(r"\n\s*(?:\*\*)?answer(?:\*\*)?\s*:?\s*$", "", t, flags=re.IGNORECASE).rstrip()
        t = re.sub(r"(?:^|\n)\s*[\(\[]?[A-E][\)\]\.]?\s*$", "", t).rstrip()
        t = t.rstrip("*").rstrip()
    if drop_conclusion and choices and t:
        lines = t.split("\n")
        sents = _split_sentences(lines[-1].strip()) if lines[-1].strip() else []
        if sents and _CONCLUSION_START.match(sents[-1]):
            last = _norm_text(sents[-1])
            if any(len(str(c).strip()) > 1 and _norm_text(str(c)).strip() and
                   (" " + _norm_text(str(c)).strip() + " ") in last for c in choices):
                lines[-1] = " ".join(sents[:-1])
                t = "\n".join(lines).rstrip()
    return t.strip()


def _split_sentences(text):
    protected = text
    for a in _ABBREV:
        protected = protected.replace(a, a.replace(".", _PLACEHOLDER))
    # Protect decimals (3.5), a leading enumeration ("1. The passage..."),
    # and option labels ("A. Plymouth, B. Albany") so they don't end a sentence.
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_PLACEHOLDER}\2", protected)
    protected = re.sub(r"^(\s*\d+)\.(\s)", rf"\1{_PLACEHOLDER}\2", protected)
    protected = re.sub(r"\b([A-E])\.(\s)", rf"\1{_PLACEHOLDER}\2", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


_ITEM_RE = re.compile(r"^(?:\*\*)?(?:step\s*\d+\s*[:.)]|\d+\s*[.)]|[-*\u2022])\s*", re.IGNORECASE)


def _strip_marker(line):
    return _ITEM_RE.sub("", line, count=1).strip()


def _is_header(line):
    """A label line like '1. **Understanding the Setup**:' or '- **Option A: ...**:'.
    It introduces the content below it rather than being a reasoning step itself."""
    core = _strip_marker(line)
    if not core.rstrip().endswith(":"):
        return False
    return core.startswith("**") or len(core) < 40


def _clean_header(line):
    return _strip_marker(line).replace("**", "").strip().rstrip(":").strip()


def split_steps(rationale):
    """Split a rationale into a list of steps.

    Structured text (numbered items, "Step 1:", bullets):
      - each item is a step; plain lines directly under an item are continuations;
      - a blank line ends the current step (so a closing paragraph such as
        "Given the analysis above, ..." becomes its own step);
      - header lines ("1. **Understanding the Setup**:") are not steps on their
        own: they are prefixed to the next step ("Understanding the Setup: ...").
    Unstructured text: sentence splitting.
    """
    if not rationale or not rationale.strip():
        return []
    # Display math (\\[ ... \\]) spread over several lines belongs to the line before it.
    text = re.sub(r"\s*\\\[\s*(.*?)\s*\\\]",
                  lambda m: " \\[" + " ".join(m.group(1).split()) + "\\]",
                  rationale.strip(), flags=re.DOTALL)
    raw = text.split("\n")
    lines = [l.strip() for l in raw if l.strip()]
    structured = len(lines) > 1 and sum(bool(_ITEM_RE.match(l)) for l in lines) >= len(lines) / 2
    if not structured:
        if len(lines) > 1:
            return [x for l in lines for x in _split_sentences(l)]
        return _split_sentences(lines[0])

    steps, cur, header = [], None, None

    def flush():
        nonlocal cur
        if cur:
            steps.append(cur)
        cur = None

    for l in (x.strip() for x in raw):
        if not l:
            flush()
            continue
        if _is_header(l):
            flush()
            header = f"{header}: {_clean_header(l)}" if header and not _ITEM_RE.match(l) else _clean_header(l)
            continue
        is_item = bool(_ITEM_RE.match(l))
        text = l
        if header:
            text = f"{header}: {_strip_marker(l)}"
            header = None
            is_item = True
        if is_item or cur is None:
            flush()
            cur = text
        else:
            cur = f"{cur} {text}"
    flush()
    if header:
        steps.append(header)
    return steps


def join_steps(steps):
    return "\n".join(steps)


def truncate(rationale, frac, steps=None):
    """Keep the first floor(frac * n_steps) steps. frac=0.0 -> empty rationale.

    Pass `steps` if you already split the rationale, to guarantee consistency.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be in [0, 1]")
    steps = split_steps(rationale) if steps is None else steps
    k = math.floor(frac * len(steps) + 1e-9)
    return join_steps(steps[:k])


def truncation_variants(rationale, fractions=TRUNCATION_FRACTIONS):
    """Return {"0.25": text, ...} plus the number of steps kept, for caching.

    Note: with short rationales (e.g. 2 steps), 0.25 keeps 0 steps — identical
    to 0.0. Record n_steps so analysis can drop or bin these.
    """
    steps = split_steps(rationale)
    out = {}
    for f in fractions:
        k = math.floor(f * len(steps) + 1e-9)
        out[f"{f:.2f}"] = {"text": join_steps(steps[:k]), "k_kept": k, "n_steps": len(steps)}
    return out