"""Paraphrase control (experiment 2).

Model-agnostic: pass any `generate(prompt: str) -> str` function. That can be
Qwen2.5-VL in text-only mode, a bigger Qwen text model, or an API model.
Using a DIFFERENT model from the one being evaluated is slightly cleaner
(the evaluated model is not grading its own phrasing), but either is defensible
— just state it in the report.

Usage:
    from paraphrase import paraphrase_question, paraphrase_rationale, export_spot_check
    p, ok, reason = paraphrase_question(generate, question, choices)
    if ok: ...run the model with p instead of question...
    export_spot_check(records, "spot_check_paraphrase.csv", n=50)
"""
import csv
import random
import re

QUESTION_PROMPT = """Rewrite the question below so that it asks exactly the same thing using different wording.

Rules:
- Keep the meaning identical. The correct answer must not change.
- Do not add hints, facts, or reasoning. Do not remove any information.
- Do not mention the answer options or any letter (A, B, C...).
- Keep any references to the image (e.g. "in the picture") but you may reword them.
- Output only the rewritten question, nothing else.

Question: {question}
Answer options (for context only, do not include them): {choices}

Rewritten question:"""

RATIONALE_PROMPT = """Rewrite the reasoning below using different wording while keeping every fact and every step exactly the same.

Rules:
- Same number of sentences, in the same order.
- Do not add, remove, or change any fact, number, or conclusion.
- Do not state a final answer or option letter.
- Output only the rewritten reasoning, nothing else.

Reasoning:
{rationale}

Rewritten reasoning:"""


def _clean(text):
    t = (text or "").strip()
    t = re.sub(r"^(?:rewritten (?:question|reasoning)\s*:\s*)", "", t, flags=re.IGNORECASE)
    return t.strip().strip('"').strip()


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def _overlap(a, b):
    sa, sb = set(_norm(a)), set(_norm(b))
    return len(sa & sb) / max(1, len(sa | sb))


def check_question_paraphrase(original, para, choices):
    """Cheap automatic filters; the hand spot-check is still required."""
    if not para:
        return False, "empty"
    if _norm(para) == _norm(original):
        return False, "identical"
    if re.search(r"\b(?:option|choice)\s+[A-E]\b|\([A-E]\)", para):
        return False, "mentions_option_letter"
    # Leaking an answer option the original did not contain.
    for c in choices:
        c = str(c)
        if len(c) > 3 and c.lower() in para.lower() and c.lower() not in original.lower():
            return False, "leaks_choice_text"
    ratio = len(para) / max(1, len(original))
    if ratio < 0.5 or ratio > 2.0:
        return False, f"length_ratio_{ratio:.2f}"
    return True, "ok"


def paraphrase_question(generate, question, choices, max_tries=3):
    """Returns (paraphrase, passed_filters, reason). Retries on filter failure."""
    reason = "not_run"
    for _ in range(max_tries):
        p = _clean(generate(QUESTION_PROMPT.format(question=question, choices="; ".join(map(str, choices)))))
        ok, reason = check_question_paraphrase(question, p, choices)
        if ok:
            return p, True, reason
    return p, False, reason


def paraphrase_rationale(generate, rationale, max_tries=3):
    """Optional stronger control: same rationale content, different wording."""
    from steps import split_steps
    n = len(split_steps(rationale))
    reason = "not_run"
    for _ in range(max_tries):
        p = _clean(generate(RATIONALE_PROMPT.format(rationale=rationale)))
        if not p or _norm(p) == _norm(rationale):
            reason = "identical_or_empty"
        elif re.search(r"answer\s*(?:is|:)", p, re.IGNORECASE):
            reason = "states_answer"
        elif abs(len(split_steps(p)) - n) > max(1, n // 3):
            reason = "step_count_changed"
        else:
            return p, True, "ok"
    return p, False, reason


def export_spot_check(records, path, n=50, seed=0):
    """records: list of dicts with question_id, original, paraphrase (+ anything else).
    Writes a CSV with empty 'meaning_preserved' and 'notes' columns to fill in by hand."""
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))
    fields = ["question_id", "original", "paraphrase", "word_overlap", "meaning_preserved", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow({"question_id": r["question_id"], "original": r["original"],
                        "paraphrase": r["paraphrase"],
                        "word_overlap": f"{_overlap(r['original'], r['paraphrase']):.2f}",
                        "meaning_preserved": "", "notes": ""})
    return path


def spot_check_summary(path):
    """After filling in the CSV (Y/N in meaning_preserved), get the validity rate for the report."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["meaning_preserved"].strip()]
    yes = sum(r["meaning_preserved"].strip().upper().startswith("Y") for r in rows)
    return {"checked": len(rows), "valid": yes, "valid_rate": yes / len(rows) if rows else None}
