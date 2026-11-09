"""Step labelling (RQ2) and rationale corruption (experiment 4).

Model-agnostic like paraphrase.py: pass any `generate(prompt: str) -> str`.
Use a stronger model than the one being evaluated if you can (e.g. a larger
Qwen text model or an API model), and say which one in the report.

Pipeline per question:
    from steps import strip_final_answer, split_steps, join_steps
    from corruption import label_steps, corrupt_rationale

    steps = split_steps(strip_final_answer(cot_output))
    labels, label_method = label_steps(generate, question, steps)      # ["P", "L", ...]
    for kind in ("perceptual", "logical"):
        res = corrupt_rationale(generate, question, choices, steps, labels, kind, seed=qid)
        if res["ok"]:
            raw = answer_given_rationale(image, question, res["text"])
            cache.add(dataset, qid, "corrupt", kind, input_rationale=res["text"],
                      step_idx=res["step_idx"], original_step=res["original_step"],
                      corrupted_step=res["corrupted_step"], raw_output=raw, ...)
        else:
            # Record the skip — the report needs how many questions had no step of that type.
            cache.add(dataset, qid, "corrupt", kind, skipped=res["reason"])

Design choices (state these in the methods section):
- One error, in ONE step. Steps after it are kept unchanged (not regenerated).
  This isolates the effect of a single wrong step. Lanham et al. (2023)
  regenerate the rest instead; that is a possible variant via `continue_fn`.
- The corrupted step is chosen at random among steps of the requested type,
  seeded by question id so runs are reproducible.
- "Perceptual" = describes what is in the image (objects, colours, counts,
  text, positions). "Logical" = reasons from facts or background knowledge.
"""
import csv
import random
import re

from steps import join_steps

PERCEPTUAL, LOGICAL, FILLER = "P", "L", "F"
KIND_TO_LABEL = {"perceptual": PERCEPTUAL, "logical": LOGICAL}

# ---------------------------------------------------------------- labelling

LABEL_PROMPT = """You will label each step of a reasoning chain about an image.

P (perceptual): the step reports what is visible in the image — objects, colours, counts, text, positions, sizes, what someone is doing.
L (logical): the step reasons, infers, applies background knowledge, compares, or concludes — it does not just report what is seen.

If a step does both, choose the one that is its main purpose.

Question: {question}

Steps:
{numbered}

Answer with exactly one line per step, in the form "<number>: P" or "<number>: L", and nothing else."""

# Fallback only (used if the LLM output cannot be parsed). Crude on purpose.
_PERCEPTUAL_CUES = re.compile(
    r"\b(image|picture|photo|figure|diagram|map|shown|shows|showing|visible|see|seen|"
    r"depicted|appears?|looks?|colou?r(?:ed)?|red|blue|green|yellow|black|white|"
    r"left|right|top|bottom|background|foreground|label(?:ed|led)?|there (?:is|are))\b",
    re.IGNORECASE,
)


# Opening/transition sentences that announce reasoning but state no fact, e.g.
# "To determine which animal..., let's analyze the options provided:" (pilot: these were
# labelled perceptual and could never be corrupted). They get label F and are never corrupted.
_FILLER_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?(?:to (?:determine|answer|find|decide|figure out|identify)\b.*?,\s*)?"
    r"(?:now,?\s+|first,?\s+|next,?\s+)?let'?s\s+(?:analy[sz]e|look|consider|follow|break|examine|think|go|start|evaluate|review)\b",
    re.IGNORECASE)


def is_filler(step):
    """True for a step that only announces what comes next, without stating any fact."""
    s = step.strip()
    return bool(_FILLER_RE.match(s)) and len(s) < 220


def mark_filler(steps, labels):
    """Overwrite the label of filler steps with F (also used on labels cached before this fix)."""
    return [FILLER if is_filler(s) else l for s, l in zip(steps, labels)]


def _number(steps):
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def _parse_labels(text, n):
    found = {}
    for m in re.finditer(r"(\d+)\s*[:.)\-]\s*\**\s*([PL])\b", text or "", flags=re.IGNORECASE):
        i = int(m.group(1)) - 1
        if 0 <= i < n and i not in found:
            found[i] = m.group(2).upper()
    if len(found) == n:
        return [found[i] for i in range(n)]
    return None


def heuristic_label(step):
    return PERCEPTUAL if _PERCEPTUAL_CUES.search(step) else LOGICAL


def label_steps(generate, question, steps, max_tries=2):
    """Returns (labels, method). method is "llm" or "heuristic_fallback" — report the
    fallback rate; if it is more than a few percent, fix the prompt instead."""
    if not steps:
        return [], "empty"
    for _ in range(max_tries):
        out = generate(LABEL_PROMPT.format(question=question, numbered=_number(steps)))
        labels = _parse_labels(out, len(steps))
        if labels:
            return mark_filler(steps, labels), "llm"
    return mark_filler(steps, [heuristic_label(s) for s in steps]), "heuristic_fallback"


# ---------------------------------------------------------------- corruption

CORRUPT_PROMPTS = {
    PERCEPTUAL: """Below is one step from a reasoning chain about an image. Rewrite it so that it describes the image INCORRECTLY.

Rules:
- Change exactly one visual fact (e.g. a colour, count, object, position, or text in the image) to something plausible but wrong.
- Keep the sentence structure and length about the same. Change as few words as possible.
- Do not mention any answer option letter (A, B, C...) and do not state a final answer.
- Output only the rewritten step, nothing else.

Question (for context): {question}
Original step: {step}

Rewritten step:""",
    LOGICAL: """Below is one step from a reasoning chain. Rewrite it so that its reasoning is WRONG.

Rules:
- Introduce exactly one reasoning error: a wrong inference, a wrong fact from background knowledge, a reversed comparison, or a wrong calculation.
- Do not change any description of what is visible in the image.
- Keep the sentence structure and length about the same. Change as few words as possible.
- Do not mention any answer option letter (A, B, C...) and do not state a final answer.
- Output only the rewritten step, nothing else.

Question (for context): {question}
Original step: {step}

Rewritten step:""",
}


def _clean(text):
    t = (text or "").strip().split("\n")[0].strip()
    t = re.sub(r"^(?:rewritten step\s*:\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^(?:\d+\.\s*)", "", t)
    return t.strip().strip('"').strip()


def _words(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def check_corruption(original, corrupted, choices):
    """Automatic filters. Catch obvious failures; the hand check catches the rest
    (e.g. a 'corruption' that is still true)."""
    if not corrupted:
        return False, "empty"
    a, b = _words(original), _words(corrupted)
    if a == b:
        return False, "unchanged"
    if re.search(r"\b(?:option|choice)\s+[A-E]\b|\([A-E]\)|answer\s*(?:is|:)", corrupted, re.IGNORECASE):
        return False, "states_answer"
    # Leak: the corrupted step names an answer option the original step did not.
    for c in choices:
        c = str(c).strip()
        if len(c) > 2 and re.search(rf"\b{re.escape(c.lower())}\b", corrupted.lower()) \
                and not re.search(rf"\b{re.escape(c.lower())}\b", original.lower()):
            return False, "introduces_answer_option"
    # Should be a small edit, not a rewrite.
    sa, sb = set(a), set(b)
    overlap = len(sa & sb) / max(1, len(sa | sb))
    if overlap < 0.4:
        return False, f"too_different_{overlap:.2f}"
    ratio = len(corrupted) / max(1, len(original))
    if ratio < 0.5 or ratio > 2.0:
        return False, f"length_ratio_{ratio:.2f}"
    return True, "ok"


def pick_step(labels, kind, seed=0):
    """Random step index of the requested kind ("perceptual"/"logical"), or None."""
    target = KIND_TO_LABEL[kind]
    idxs = [i for i, l in enumerate(labels) if l == target]
    if not idxs:
        return None
    return random.Random(str(seed) + kind).choice(idxs)


# Greedy decoding returns the same output for the same prompt, so each retry must
# use a DIFFERENT prompt to have any effect (pilot: 54/100 perceptual steps came back unchanged).
_RETRY_NOTES = [
    "",
    "Your previous answer did not change the step. You MUST change at least one word so that the step becomes false.",
    "Change one concrete detail to something different: a colour, a number, an object name, a position, or a yes/no claim.",
]


def _with_note(prompt, attempt):
    note = _RETRY_NOTES[min(attempt, len(_RETRY_NOTES) - 1)]
    if not note:
        return prompt
    return prompt.replace("\n\nRewritten step:", f"\n\nIMPORTANT: {note}\n\nRewritten step:")


def corrupt_rationale(generate, question, choices, steps, labels, kind,
                      seed=0, max_tries=3, continue_fn=None):
    """Insert one error of `kind` into one step.

    Tries the seeded random step first; if it cannot be corrupted (unchanged, leaks the
    answer, ...) it tries the other steps of the same kind. Each retry on a step uses a
    stronger prompt (see _RETRY_NOTES).

    continue_fn: optional f(prefix_steps) -> list[str] that regenerates the steps
    after the corrupted one (Lanham-style). Default None = keep later steps unchanged.

    Returns a dict: ok, reason, text, step_idx, original_step, corrupted_step, n_steps, tries.
    """
    base = {"ok": False, "reason": "", "text": None, "step_idx": None,
            "original_step": None, "corrupted_step": None, "n_steps": len(steps), "kind": kind, "tries": 0}
    first = pick_step(labels, kind, seed)
    if first is None:
        return {**base, "reason": f"no_{kind}_step"}
    target = KIND_TO_LABEL[kind]
    others = [i for i, l in enumerate(labels) if l == target and i != first]
    random.Random(f"{seed}{kind}order").shuffle(others)
    reason, tries = "not_run", 0
    for idx in [first] + others:
        original = steps[idx]
        for attempt in range(max_tries):
            tries += 1
            prompt = _with_note(CORRUPT_PROMPTS[target].format(question=question, step=original), attempt)
            cand = _clean(generate(prompt))
            ok, reason = check_corruption(original, cand, choices)
            if ok:
                new_steps = steps[:idx] + [cand]
                new_steps += continue_fn(new_steps) if continue_fn else steps[idx + 1:]
                return {**base, "ok": True, "reason": "ok", "text": join_steps(new_steps),
                        "step_idx": idx, "original_step": original, "corrupted_step": cand, "tries": tries}
    return {**base, "reason": reason, "step_idx": first, "original_step": steps[first], "tries": tries}


# ---------------------------------------------------------------- hand checks

def export_label_check(records, path, n=50, seed=0):
    """records: dicts with question_id, question, step_idx, step, label.
    Fill in 'correct' (Y/N) and, if N, the right label in 'true_label'."""
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))
    fields = ["question_id", "question", "step_idx", "step", "label", "correct", "true_label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow({k: r.get(k, "") for k in fields[:5]} | {"correct": "", "true_label": ""})
    return path


def export_corruption_check(records, path, n=50, seed=0):
    """records: corrupt_rationale() results with question_id and question added.
    Hand-check columns:
      is_wrong      Y/N - the corrupted step is actually false
      right_type    Y/N - it is the intended kind of error (perceptual vs logical)
      single_change Y/N - only one fact/inference changed
    """
    ok = [r for r in records if r.get("ok")]
    rng = random.Random(seed)
    sample = rng.sample(ok, min(n, len(ok)))
    fields = ["question_id", "question", "kind", "step_idx", "original_step", "corrupted_step",
              "is_wrong", "right_type", "single_change", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow({k: r.get(k, "") for k in fields[:6]})
    return path


def check_summary(path, cols):
    """Fraction of 'Y' per column after the CSV has been filled in, for the report."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for c in cols:
        filled = [r[c].strip().upper() for r in rows if r.get(c, "").strip()]
        out[c] = {"checked": len(filled),
                  "yes_rate": sum(v.startswith("Y") for v in filled) / len(filled) if filled else None}
    return out


# ================================================================ extra mistake types
# Instructor suggestion (1): mistakes expected to have smaller or larger effects.
#   grammar        - surface noise only (typos, dropped articles, swapped words).
#                    Expected effect: ~none. Acts as a second control next to paraphrase.
#   irrelevant     - insert one unrelated but true sentence. Tests distraction.
#   contradiction  - insert a sentence that contradicts an earlier step, keeping
#                    the original step. Tests which of two conflicting claims wins.
# Together with perceptual / logical this gives a rough "severity ladder" to plot:
#   grammar < irrelevant < logical/perceptual < contradiction (or not — that's the finding).
#
# Cache: condition="corrupt", variant in {"grammar", "irrelevant", "contradiction"}.

_ARTICLES = {"a", "an", "the"}


def _typo(word, rng):
    """Swap two adjacent inner letters: 'magnet' -> 'mangnet'-style noise."""
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def grammar_noise(steps, seed=0, n_edits=2):
    """Deterministic, no LLM. Applies n_edits surface errors to ONE random step.
    Never touches numbers, so no fact changes. Returns the same dict shape as
    corrupt_rationale()."""
    base = {"ok": False, "reason": "", "text": None, "step_idx": None, "original_step": None,
            "corrupted_step": None, "n_steps": len(steps), "kind": "grammar"}
    if not steps:
        return {**base, "reason": "no_steps"}
    rng = random.Random(f"{seed}grammar")
    idx = rng.randrange(len(steps))
    words = steps[idx].split()
    if len(words) < 3:
        return {**base, "reason": "step_too_short", "step_idx": idx}
    for _ in range(n_edits):
        op = rng.choice(["typo", "drop_article", "swap"])
        arts = [i for i, w in enumerate(words) if w.lower() in _ARTICLES]
        if op == "drop_article" and arts:
            words.pop(rng.choice(arts))
        elif op == "swap" and len(words) > 3:
            i = rng.randrange(1, len(words) - 2)          # keep first/last word in place
            if not any(c.isdigit() for c in words[i] + words[i + 1]):
                words[i], words[i + 1] = words[i + 1], words[i]
        else:
            cand = [i for i, w in enumerate(words) if w.isalpha() and len(w) >= 4]
            if cand:
                j = rng.choice(cand)
                words[j] = _typo(words[j], rng)
    new = " ".join(words)
    if new == steps[idx]:
        return {**base, "reason": "unchanged", "step_idx": idx}
    new_steps = steps[:idx] + [new] + steps[idx + 1:]
    return {**base, "ok": True, "reason": "ok", "text": join_steps(new_steps), "step_idx": idx,
            "original_step": steps[idx], "corrupted_step": new}


# True, neutral facts unrelated to ScienceQA / A-OKVQA content. Deterministic so
# every run inserts the same sentence for a given question. Extend if you like.
UNRELATED_FACTS = [
    "The Pacific Ocean is the largest ocean on Earth.",
    "A standard chess board has sixty-four squares.",
    "Honey can stay edible for a very long time if it is sealed.",
    "The violin usually has four strings.",
    "Mount Everest is located in the Himalayas.",
    "A leap year has three hundred sixty-six days.",
    "The Eiffel Tower is in Paris.",
    "Penguins are birds that cannot fly.",
    "The Great Wall is in China.",
    "A haiku traditionally has three lines.",
    "Coffee beans are the seeds of a fruit.",
    "The Moon orbits the Earth.",
]


def insert_irrelevant(steps, choices, seed=0, facts=UNRELATED_FACTS):
    """Insert one unrelated true sentence at a random position (never last, so the
    rationale still ends on its own conclusion). Skips facts that mention an option."""
    base = {"ok": False, "reason": "", "text": None, "step_idx": None, "original_step": None,
            "corrupted_step": None, "n_steps": len(steps), "kind": "irrelevant"}
    rng = random.Random(f"{seed}irrelevant")
    lowered = [str(c).lower() for c in choices if len(str(c)) > 2]
    pool = [f for f in facts if not any(c in f.lower() for c in lowered)]
    if not pool:
        return {**base, "reason": "no_safe_fact"}
    fact = rng.choice(pool)
    pos = rng.randrange(0, max(1, len(steps)))      # insert before step `pos`
    new_steps = steps[:pos] + [fact] + steps[pos:]
    return {**base, "ok": True, "reason": "ok", "text": join_steps(new_steps), "step_idx": pos,
            "original_step": "", "corrupted_step": fact}


CONTRADICT_PROMPT = """Below is one step from a reasoning chain. Write ONE new sentence that directly contradicts it (states the opposite or an incompatible fact).

Rules:
- Contradict only this step. Keep it short and in the same style.
- Do not mention any answer option letter (A, B, C...) and do not state a final answer.
- Output only the new sentence, nothing else.

Question (for context): {question}
Step: {step}

Contradicting sentence:"""


def insert_contradiction(generate, question, choices, steps, seed=0, max_tries=3, position="after"):
    """Keep the original step and add a sentence contradicting it.
    position="after": right after the contradicted step.
    position="end":   at the end (tests recency: does the last claim win?)."""
    base = {"ok": False, "reason": "", "text": None, "step_idx": None, "original_step": None,
            "corrupted_step": None, "n_steps": len(steps), "kind": "contradiction"}
    if not steps:
        return {**base, "reason": "no_steps"}
    idx = random.Random(f"{seed}contradiction").randrange(len(steps))
    original = steps[idx]
    reason = "not_run"
    for _ in range(max_tries):
        cand = _clean(generate(CONTRADICT_PROMPT.format(question=question, step=original)))
        if not cand or _words(cand) == _words(original):
            reason = "empty_or_identical"
            continue
        if re.search(r"\b(?:option|choice)\s+[A-E]\b|\([A-E]\)|answer\s*(?:is|:)", cand, re.IGNORECASE):
            reason = "states_answer"
            continue
        leak = [c for c in choices if len(str(c)) > 2
                and re.search(rf"\b{re.escape(str(c).lower())}\b", cand.lower())
                and not re.search(rf"\b{re.escape(str(c).lower())}\b", original.lower())]
        if leak:
            reason = "introduces_answer_option"
            continue
        if position == "end":
            new_steps = steps + [cand]
        else:
            new_steps = steps[:idx + 1] + [cand] + steps[idx + 1:]
        return {**base, "ok": True, "reason": "ok", "text": join_steps(new_steps), "step_idx": idx,
                "original_step": original, "corrupted_step": cand, "position": position}
    return {**base, "reason": reason, "step_idx": idx, "original_step": original}