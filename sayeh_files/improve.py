"""Rationale edits that might IMPROVE accuracy (instructor suggestion 2, optional).

Question: can we change the CoT so the model answers correctly more often,
WITHOUT putting the answer in? If yes, the answer really does depend on the
rationale's content — a different angle on faithfulness than corruption.

Edits (cache: condition="improve", variant=<name>):
    prune     - drop steps a judge model marks as irrelevant to the question.
    refine    - a stronger model fixes factual/visual/reasoning errors and
                clarifies, under strict no-answer rules.
    describe  - prepend a more detailed, question-focused description of the
                image (written by a model that SEES the image), keep the rest.

Leak control (the important part):
    check_leak() rejects an edit that states an answer, names an option letter,
    or names an answer option the original rationale did not already mention.
    Run it with the GOLD option too (gold_leak) and report how often the edited
    rationale newly mentions the gold option text — if refine "improves" accuracy
    only on those questions, that's leakage, not reasoning.

Analysis (per edit, on the same questions):
    fixed  = was wrong with the original rationale, right after the edit
    broken = was right, wrong after the edit
    Report both, not just net accuracy, and test with McNemar.
"""
import re

from steps import split_steps, join_steps

PRUNE_PROMPT = """Here is a question about an image and a numbered reasoning chain.
List the numbers of the steps that are NOT relevant to answering the question
(off-topic, repeated, or filler). If every step is relevant, answer "none".

Question: {question}

Steps:
{numbered}

Irrelevant step numbers (comma-separated, or "none"):"""

REFINE_PROMPT = """Here is a question about an image and a reasoning chain someone wrote about it.
Improve the reasoning: fix any wrong facts or reasoning errors and make the steps clearer.

Strict rules:
- Do NOT state the final answer, any option letter (A, B, C...), or which option is correct.
- Do NOT mention any answer option that the original reasoning does not already mention.
- Keep roughly the same length and number of steps.
- Output only the improved reasoning, one step per line.

Question: {question}

Original reasoning:
{rationale}

Improved reasoning:"""

DESCRIBE_PROMPT = """Describe the parts of this image that matter for the question below, in 1-3 factual sentences.
Do not answer the question, do not mention any answer option, and do not reason — only describe what is visible.

Question: {question}

Description:"""


def _norm_words(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def _mentions(text, option):
    o = str(option).lower().strip()
    return len(o) > 2 and re.search(rf"\b{re.escape(o)}\b", (text or "").lower()) is not None


def check_leak(original, edited, choices):
    """(ok, reason). Rejects edits that introduce answer information."""
    if not edited or not edited.strip():
        return False, "empty"
    if _norm_words(edited) == _norm_words(original):
        return False, "unchanged"
    if re.search(r"\b(?:option|choice)\s+[A-E]\b|\([A-E]\)|answer\s*(?:is|:)|correct (?:option|choice|answer)",
                 edited, re.IGNORECASE):
        return False, "states_answer"
    for c in choices:
        if _mentions(edited, c) and not _mentions(original, c):
            return False, "introduces_answer_option"
    return True, "ok"


def gold_leak(original, edited, gold_text):
    """True if the edit newly mentions the gold option. Should be ~0 after check_leak;
    report it anyway as evidence there is no leakage."""
    return _mentions(edited, gold_text) and not _mentions(original, gold_text)


def prune(generate, question, steps, choices=()):
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    out = (generate(PRUNE_PROMPT.format(question=question, numbered=numbered)) or "").strip().lower()
    if not steps:
        return {"ok": False, "reason": "no_steps", "text": None, "removed": []}
    if out.startswith("none"):
        return {"ok": False, "reason": "nothing_to_prune", "text": None, "removed": []}
    drop = sorted({int(n) - 1 for n in re.findall(r"\d+", out) if 0 < int(n) <= len(steps)})
    if not drop:
        return {"ok": False, "reason": "unparsed", "text": None, "removed": []}
    if len(drop) == len(steps):
        return {"ok": False, "reason": "would_remove_all", "text": None, "removed": drop}
    kept = [s for i, s in enumerate(steps) if i not in drop]
    return {"ok": True, "reason": "ok", "text": join_steps(kept), "removed": drop}


def refine(generate, question, rationale, choices, max_tries=3):
    reason = "not_run"
    for _ in range(max_tries):
        out = (generate(REFINE_PROMPT.format(question=question, rationale=rationale)) or "").strip()
        out = re.sub(r"^improved reasoning\s*:\s*", "", out, flags=re.IGNORECASE)
        ok, reason = check_leak(rationale, out, choices)
        if ok:
            return {"ok": True, "reason": "ok", "text": join_steps(split_steps(out))}
    return {"ok": False, "reason": reason, "text": None}


def describe(vision_generate, image, question, rationale, choices, max_tries=2):
    """vision_generate(image, prompt) -> str, e.g. Qwen2.5-VL itself or a bigger VLM."""
    reason = "not_run"
    for _ in range(max_tries):
        d = (vision_generate(image, DESCRIBE_PROMPT.format(question=question)) or "").strip()
        new = (d + "\n" + rationale).strip()
        ok, reason = check_leak(rationale, new, choices)
        if ok:
            return {"ok": True, "reason": "ok", "text": new, "description": d}
    return {"ok": False, "reason": reason, "text": None}


def fixed_broken(orig_correct, edit_correct):
    """Lists of bools over the same questions -> counts for the report and McNemar."""
    fixed = sum((not a) and b for a, b in zip(orig_correct, edit_correct))
    broken = sum(a and (not b) for a, b in zip(orig_correct, edit_correct))
    return {"fixed": fixed, "broken": broken, "net": fixed - broken, "n": len(orig_correct)}
