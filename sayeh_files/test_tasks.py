import os
import tempfile

from parsing import parse_answer
from steps import strip_final_answer, split_steps, truncate, truncation_variants
from cache import ResultCache
from paraphrase import check_question_paraphrase, paraphrase_question

CH = ["Paris", "London", "Berlin", "Madrid"]


def test_parse():
    cases = {
        "B": "B", "(C)": "C", "**D**": "D", "A.": "A",
        "The answer is B.": "B",
        "Answer: (C)": "C",
        "Option A is wrong because ... so the final answer is D": "D",
        "The correct answer is option (B).": "B",
        "B. London": "B",
        "It is clearly Berlin.": "C",
        "I think Paris could be right, but actually it's Madrid.": "D",
        "ANSWER: c": None,  # lowercase letter is not trusted
        "The answer is a magnet, so B.": None,  # handled below

    }
    for text, want in cases.items():
        if want is None:
            continue
        got, m = parse_answer(text, CH)
        assert got == want, (text, got, m)
    # "a" as an article must not become option A
    assert parse_answer("The answer is a magnet.", ["magnet", "rock"])[0] == "A"  # via choice text, not the article
    assert parse_answer("The answer is a city in Spain: Madrid.", CH)[0] == "D"
    assert parse_answer("E", CH)[0] is None          # out of range
    assert parse_answer("I am not sure.", CH)[0] is None
    assert parse_answer("", CH) == (None, "empty")


def test_strip_and_split():
    cot = ("The image shows a magnet near a paper clip. Magnets attract iron, e.g. steel clips. "
           "The clip moves 3.5 cm toward the magnet. So the answer is B.")
    r = strip_final_answer(cot)
    assert "answer" not in r.lower()
    s = split_steps(r)
    assert len(s) == 3, s
    assert "e.g. steel" in s[1] and "3.5 cm" in s[2]

    numbered = "Step 1: Look at the map.\nStep 2: The red state is Ohio.\nwhich borders Lake Erie.\nStep 3: Ohio is in the Midwest.\nAnswer: A"
    s = split_steps(strip_final_answer(numbered))
    assert len(s) == 3 and "Lake Erie" in s[1], s


def test_truncate():
    r = "One. Two. Three. Four."
    assert truncate(r, 0.0) == ""
    assert truncate(r, 0.5) == "One.\nTwo."
    assert truncate(r, 0.75) == "One.\nTwo.\nThree."
    v = truncation_variants("One. Two.")
    assert v["0.25"]["k_kept"] == 0 and v["0.50"]["k_kept"] == 1 and v["0.00"]["n_steps"] == 2


def test_cache_resume():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "r.jsonl")
    c = ResultCache(p, meta={"model": "qwen7b"})
    c.add("scienceqa", 1, "truncate", "0.50", parsed_answer="B")
    with open(p, "a") as f:
        f.write('{"broken": ')            # simulate job killed mid-write
    c2 = ResultCache(p)
    assert c2.has("scienceqa", "1", "truncate", "0.50")
    assert not c2.has("scienceqa", 1, "truncate", "0.25")
    assert c2.get("scienceqa", 1, "truncate", "0.50")["model"] == "qwen7b"


def test_paraphrase_filters():
    q = "Which city is the capital of France?"
    assert check_question_paraphrase(q, "What is France's capital city?", CH)[0]
    assert not check_question_paraphrase(q, q, CH)[0]
    assert not check_question_paraphrase(q, "Is the capital of France Paris?", CH)[0]
    assert not check_question_paraphrase(q, "Which is it, option B?", CH)[0]
    fake = iter(["Which city is the capital of France?", "What is France's capital city?"])
    p, ok, _ = paraphrase_question(lambda _: next(fake), q, CH)
    assert ok and p == "What is France's capital city?"


# ---------------- corruption.py
from corruption import (label_steps, corrupt_rationale, check_corruption, pick_step,
                        export_corruption_check, check_summary)

STEPS = ["The image shows a red apple on a table.",
         "Red apples are usually ripe.",
         "So the fruit is ready to eat."]
FRUIT = ["ripe", "unripe"]


def test_label_steps():
    labels, m = label_steps(lambda _: "1: P\n2: L\n3: **L**", "Is it ripe?", STEPS)
    assert labels == ["P", "L", "L"] and m == "llm"
    labels, m = label_steps(lambda _: "I am not sure", "Is it ripe?", STEPS)
    assert m == "heuristic_fallback" and labels[0] == "P"
    assert label_steps(lambda _: "", "q", []) == ([], "empty")


def test_check_corruption():
    o = STEPS[0]
    assert check_corruption(o, "The image shows a green apple on a table.", FRUIT)[0]
    assert not check_corruption(o, o, FRUIT)[0]
    assert not check_corruption(o, "The image shows an unripe apple on a table.", FRUIT)[0]  # leak
    assert not check_corruption(o, "The answer is B.", FRUIT)[0]
    assert not check_corruption(o, "Bananas grow in tropical climates worldwide.", FRUIT)[0]


def test_corrupt_rationale():
    labels = ["P", "L", "L"]
    assert pick_step(labels, "perceptual", seed=1) == 0
    assert pick_step(labels, "logical", seed=7) == pick_step(labels, "logical", seed=7)  # reproducible
    gen = lambda _: "Rewritten step: The image shows a green apple on a table."
    r = corrupt_rationale(gen, "Is it ripe?", FRUIT, STEPS, labels, "perceptual", seed=1)
    assert r["ok"] and r["step_idx"] == 0
    assert r["text"].split("\n") == ["The image shows a green apple on a table.", STEPS[1], STEPS[2]]
    r = corrupt_rationale(gen, "q", FRUIT, STEPS, ["L", "L", "L"], "perceptual")
    assert not r["ok"] and r["reason"] == "no_perceptual_step"


def test_corruption_export(tmp_path):
    rec = [{"ok": True, "question_id": "1", "question": "q", "kind": "perceptual", "step_idx": 0,
            "original_step": "a", "corrupted_step": "b"}]
    p = export_corruption_check(rec, str(tmp_path / "c.csv"))
    import csv
    rows = list(csv.DictReader(open(p)))
    rows[0]["is_wrong"] = "Y"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert check_summary(p, ["is_wrong"])["is_wrong"]["yes_rate"] == 1.0


# ---------------- extra mistake types + improve.py
from corruption import grammar_noise, insert_irrelevant, insert_contradiction
from improve import prune, refine, check_leak, gold_leak, fixed_broken


def test_grammar_noise():
    r = grammar_noise(STEPS, seed=3)
    assert r["ok"] and r["corrupted_step"] != r["original_step"]
    assert grammar_noise(STEPS, seed=3)["text"] == r["text"]          # reproducible
    assert len(r["text"].split("\n")) == 3                            # same number of steps
    assert not grammar_noise([], seed=0)["ok"]


def test_irrelevant_and_contradiction():
    r = insert_irrelevant(STEPS, FRUIT, seed=1)
    assert r["ok"] and len(r["text"].split("\n")) == 4 and r["text"].split("\n")[-1] == STEPS[-1]
    gen = lambda _: "The apple on the table is green."
    YN = ["yes", "no"]
    c = insert_contradiction(gen, "q", YN, STEPS, seed=2)
    assert c["ok"] and len(c["text"].split("\n")) == 4 and c["original_step"] in c["text"]
    c_end = insert_contradiction(gen, "q", YN, STEPS, seed=2, position="end")
    assert c_end["text"].split("\n")[-1] == "The apple on the table is green."
    leak = insert_contradiction(lambda _: "The fruit is unripe.", "q", FRUIT, ["The fruit is red."], seed=0)
    assert not leak["ok"] and leak["reason"] == "introduces_answer_option"


def test_improve():
    p = prune(lambda _: "2", "Is it ripe?", STEPS)
    assert p["ok"] and p["removed"] == [1] and "usually ripe" not in p["text"]
    assert not prune(lambda _: "none", "q", STEPS)["ok"]
    assert not prune(lambda _: "1, 2, 3", "q", STEPS)["ok"]
    orig = join_steps(STEPS)
    assert not check_leak(orig, "The apple is unripe.", FRUIT)[0]
    assert not check_leak(orig, "So the answer is A.", FRUIT)[0]
    r = refine(lambda _: "Improved reasoning: The image shows a red apple.\nColour indicates the apple has matured.",
               "Is it ripe?", orig, ["sweet", "sour"])
    assert r["ok"] and len(r["text"].split("\n")) == 2
    assert gold_leak("x", "it is sour", "sour") and not gold_leak("sour x", "sour y", "sour")
    assert fixed_broken([0, 1, 1, 0], [1, 1, 0, 0]) == {"fixed": 1, "broken": 1, "net": 0, "n": 4}


from steps import join_steps


# ---------------- regression tests from pilot run (job 41007)
Q1_COT = """To determine which of the options Wendy's test could show, let's break down the information provided and analyze each option step by step:

1. **Understanding the Experiment Setup**:
   - Wendy aimed to increase the production of insulin by 20%.
   - She introduced additional nutrients to some bacteria.

2. **Analyze Each Option**:

   - **Option A: Whether producing more insulin would help the bacteria grow faster**:
     - The experiment directly measures insulin production but does not measure bacterial growth rate explicitly.

Given the analysis above, Wendy's test specifically aims at confirming the 20% increase.

**Final Answer: The answer is C."""


def test_pilot_split_headers():
    steps = split_steps(strip_final_answer(Q1_COT))
    assert "Final Answer" not in " ".join(steps)
    assert not any(s.rstrip().endswith("**:") or s.rstrip().endswith("Setup") for s in steps), steps
    assert steps[1].startswith("Understanding the Experiment Setup: Wendy aimed"), steps
    assert any(s.startswith("Option A: Whether producing more insulin would help the bacteria grow faster: The experiment") for s in steps), steps
    assert steps[-1].startswith("Given the analysis above"), steps
    assert len(steps) == 5, steps


def test_strict_parse():
    garbled = "Weather refers to the short-term state... Climate is the long-term pattern of weather addCriterion"
    assert parse_answer(garbled, ["weather", "climate"], strict=True)[0] is None
    assert parse_answer("... so the answer is B", ["weather", "climate"], strict=True)[0] == "B"


# ---------------- regression tests from the clean pilot run (job 41476)
Q3_TAIL = """New Zealand falcons are indeed birds of prey, with sharp talons.

Therefore, the correct answer is B.

The answer is B."""

Q5_COT = """To determine the capital of Massachusetts, let's follow these steps:

1. Identify the state in question: The state highlighted in dark green on the map is Massachusetts.
2. Recall the capital of Massachusetts: The capital city of Massachusetts is Boston.
3. Match the correct option with the information: The options provided are A. Plymouth, B. Albany, C. Wichita, and D. Boston.

Given that the capital of Massachusetts is Boston, the correct answer is:

The answer is D"""

Q4_COT = r"""To determine if the statement is true or false, we need to compare the volumes of Jupiter and Mars.

1. Jupiter's volume is given as \(1.43 \times 10^{15} \, \text{km}^3\).
2. Mars's volume is given as \(1.63 \times 10^{11} \, \text{km}^3\).

Next, we calculate how many times larger Jupiter's volume is compared to Mars's volume:

\[
\frac{\text{Jupiter's volume}}{\text{Mars's volume}} = \frac{1.43 \times 10^{15}}{1.63 \times 10^{11}}
\]

Simplify the expression:

\[
\frac{1.43}{1.63} \times \frac{10^{15}}{10^{11}} = 0.879 \times 10^4 = 8790
\]

Jupiter's volume is approximately 8790 times larger than Mars's volume. Since 8790 is indeed more than 10,000, the statement is false.

The answer is B."""

Q0_COT = """To determine which animal's mouth is adapted for bottom feeding, let's analyze the options provided:

1. **Discus**: Discus fish are known to be surface feeders.

2. **Armored Catfish**: Armored catfish are bottom-dwelling fish.

Given this information, the animal whose mouth is adapted for bottom feeding is the armored catfish.

The answer is B."""


def test_strip_repeated_answer_statements():
    r = strip_final_answer(Q3_TAIL)
    assert "answer" not in r.lower() and r.endswith("sharp talons."), r
    r = strip_final_answer(Q5_COT)
    assert "answer" not in r.lower(), r
    assert r.rstrip().endswith("D. Boston."), r


def test_split_enumeration_and_latex():
    steps = split_steps(strip_final_answer(Q4_COT))
    assert not any(s.strip() in {"1.", "2.", "\\[", "\\]"} for s in steps), steps
    assert steps[1].startswith("1. Jupiter's volume"), steps
    assert any("Mars's volume: \\[" in s and "\\frac" in s for s in steps), steps
    assert any(s.startswith("Simplify the expression: \\[") and "8790\\]" in s for s in steps), steps
    assert steps[-1].startswith("Since 8790"), steps


def test_option_labels_not_split():
    steps = split_steps(strip_final_answer(Q5_COT))
    assert any("A. Plymouth, B. Albany, C. Wichita, and D. Boston." in s for s in steps), steps


def test_drop_conclusion_optional():
    ch = ["discus", "armored catfish"]
    keep = strip_final_answer(Q0_COT, ch)
    assert keep.endswith("is the armored catfish.")
    drop = strip_final_answer(Q0_COT, ch, drop_conclusion=True)
    assert drop.endswith("bottom-dwelling fish."), drop
    # Not dropped if it doesn't start like a conclusion
    assert strip_final_answer("Fish swim. The catfish is an armored catfish.", ch, drop_conclusion=True).endswith("armored catfish.")


# ---------------- run_experiments.py + analyze.py end to end (fake model)
import sys, types
if "torch" not in sys.modules:
    try:
        import torch  # noqa
    except ImportError:
        fake = types.ModuleType("torch"); fake.float32 = "float32"
        fake.no_grad = lambda: __import__("contextlib").nullcontext()
        sys.modules["torch"] = fake


class FakeQwen:
    """Answers depend on the rationale only when it mentions 'key fact'; truncation to
    'first half' makes it pick option A (simulates the option-order effect)."""
    def __init__(self, ex_gold):
        self.ex_gold = ex_gold
    def chat(self, image, user_text, assistant_prefix="", max_new_tokens=1024):
        if "letter only" in user_text:
            return "A"
        return ("To decide, look at the options.\n\n1. **First option**: not right.\n\n"
                "2. **Second option**: key fact supports it.\n\nThe answer is B.")
    def text(self, prompt, max_new_tokens=512):
        if "label each step" in prompt:
            return "1: P\n2: L\n3: L"
        if "Original step:" in prompt:                      # perceptual / logical corruption
            step = prompt.split("Original step: ")[1].split("\n")[0]
            return step.replace("look at", "ignore").replace("supports", "weakens").replace("not right", "right")
        if "directly contradicts" in prompt:
            return "Actually the options cannot be compared at all."
        if "Rewrite the question" in prompt:
            q = prompt.split("Question: ")[1].split("\n")[0]
            return "Please tell me: " + q.lower()
        r = prompt.split("Reasoning:\n")[1].split("\n\nRewritten")[0]
        return r.replace("not right", "incorrect")
    def answer_given_rationale(self, ex, rationale, question=None):
        if "key fact" in rationale:
            return "(B)"
        return "(A)" if rationale.strip() else "(A)"


def _fake_args(**kw):
    a = types.SimpleNamespace(drop_conclusion=False, skip_paraphrase=False, corrupt=False, corrupt_extra=False)
    a.__dict__.update(kw)
    return a


def test_run_and_analyze(tmp_path):
    from run_experiments import run_question, reverse_example
    from analyze import summarize, option_order
    from cache import ResultCache
    exs = [dict(qid=str(i), image=None, question="Which option is best for this image?",
                choices=["cat", "dog"], gold_idx=1, hint="") for i in range(5)]
    for name, rev in (("n", False), ("r", True)):
        c = ResultCache(str(tmp_path / f"{name}.jsonl"))
        m = FakeQwen(None)
        for ex in exs:
            run_question(m, c, "toy", reverse_example(ex) if rev else ex, _fake_args())
    rows = [json.loads(l) for l in open(tmp_path / "n.jsonl")]
    conds = {(r["condition"], r["variant"]) for r in rows}
    for k in [("direct", "orig"), ("cot", "orig"), ("replay", "orig"), ("truncate", "0.50"),
              ("paraphrase_q", "text"), ("paraphrase_q", "cot"), ("paraphrase_r", "orig")]:
        assert k in conds, k
    res = summarize(str(tmp_path / "n.jsonl"), out=lambda *a: None)
    assert res[("truncate", "0.00")]["rate"] == 1.0          # no rationale -> A, replay said B
    assert res[("truncate", "0.75")]["rate"] == 1.0          # 3 steps -> keeps 2, drops the key-fact step
    assert res[("paraphrase_r", "orig")]["rate"] == 0.0
    lines = []
    option_order(str(tmp_path / "n.jsonl"), str(tmp_path / "r.jsonl"), out=lines.append)
    assert any("0.00" in l for l in lines)
    # resume: running again adds no rows
    before = sum(1 for _ in open(tmp_path / "n.jsonl"))
    c = ResultCache(str(tmp_path / "n.jsonl"))
    for ex in exs:
        run_question(FakeQwen(None), c, "toy", ex, _fake_args())
    assert sum(1 for _ in open(tmp_path / "n.jsonl")) == before


def test_mcnemar():
    from analyze import mcnemar_p
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(10, 0) < 0.01
    assert abs(mcnemar_p(3, 3) - 1.0) < 1e-9


import json


def test_corruption_in_runner(tmp_path):
    from run_experiments import run_question
    from analyze import summarize, corruption_check_csvs
    from cache import ResultCache
    exs = [dict(qid=str(i), image=None, question="Which option is best for this image?",
                choices=["cat", "dog"], gold_idx=1, hint="") for i in range(4)]
    p = str(tmp_path / "c.jsonl")
    c = ResultCache(p)
    for ex in exs:
        run_question(FakeQwen(None), c, "toy", ex, _fake_args(corrupt=True, corrupt_extra=True, skip_paraphrase=True))
    rows = [json.loads(l) for l in open(p)]
    got = {(r["condition"], r["variant"]) for r in rows}
    for k in [("labels", "steps"), ("corrupt", "perceptual"), ("corrupt", "logical"),
              ("corrupt", "grammar"), ("corrupt", "irrelevant"), ("corrupt", "contradiction")]:
        assert k in got, (k, got)
    assert not any(r["condition"].startswith("paraphrase") for r in rows)     # skip_paraphrase respected
    lab = next(r for r in rows if r["condition"] == "labels")
    assert lab["labels"] == ["P", "L", "L"] and len(lab["steps"]) == 3
    per = next(r for r in rows if r["condition"] == "corrupt" and r["variant"] == "perceptual")
    assert per["corrupt_ok"] and per["step_idx"] == 0 and "ignore" in per["corrupted_step"]
    res = summarize(p, out=lambda *a: None)
    assert ("corrupt", "perceptual") in res and ("corrupt", "logical") in res
    # perceptual error in the intro step leaves the key-fact step -> answer unchanged
    assert res[("corrupt", "perceptual")]["rate"] == 0.0
    corruption_check_csvs(p, str(tmp_path / "chk"))
    assert os.path.exists(str(tmp_path / "chk_labels.csv")) and os.path.exists(str(tmp_path / "chk_corruptions.csv"))
    # resume adds nothing
    n = len(rows)
    c2 = ResultCache(p)
    for ex in exs:
        run_question(FakeQwen(None), c2, "toy", ex, _fake_args(corrupt=True, corrupt_extra=True, skip_paraphrase=True))
    assert sum(1 for _ in open(p)) == n


def test_analyze_handles_rows_without_choices(tmp_path):
    """Pilot rows had no 'choices' field (job crash, Nov 8)."""
    from analyze import summarize, option_order
    p = tmp_path / "old.jsonl"
    rows = [dict(question_id="0", condition="direct", variant="orig", parsed_answer="A", gold="B"),
            dict(question_id="0", condition="cot", variant="orig", parsed_answer="B", gold="B"),
            dict(question_id="0", condition="replay", variant="orig", parsed_answer="B", gold="B")]
    rows += [dict(question_id="0", condition="truncate", variant=f, parsed_answer="A", gold="B")
             for f in ("0.00", "0.25", "0.50", "0.75")]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    summarize(str(p), out=lambda *a: None)                 # must not raise
    option_order(str(p), str(p), out=lambda *a: None)      # must not raise
    rows.append(dict(question_id="0", condition="paraphrase_q", variant="text", choices=["x", "y"],
                     ok=False, reason="identical", paraphrase="", original="q", gold="B"))
    p.write_text("\n".join(json.dumps(r) for r in rows))
    summarize(str(p), out=lambda *a: None)


def test_corruption_retries_and_fallback_step():
    """Greedy decoding: same prompt -> same output, so retries need a different prompt."""
    steps = ["The image shows a red apple.", "The image shows a wooden table.", "So it is ripe."]
    labels = ["P", "P", "L"]
    def gen(prompt):
        step = prompt.split("Original step: ")[1].split("\n")[0]
        if "IMPORTANT" in prompt:                       # only the stronger prompt works
            return step.replace("red", "green").replace("wooden", "metal")
        return step                                      # first attempt: unchanged
    r = corrupt_rationale(gen, "q", ["yes", "no"], steps, labels, "perceptual", seed=0)
    assert r["ok"] and r["tries"] == 2, r
    # a step that can never be corrupted falls back to the other perceptual step
    def gen2(prompt):
        step = prompt.split("Original step: ")[1].split("\n")[0]
        return step.replace("wooden", "metal")          # only step 1 can change
    r = corrupt_rationale(gen2, "q", ["yes", "no"], steps, labels, "perceptual", seed=0)
    assert r["ok"] and r["step_idx"] == 1, r


def test_skipped_corruptions_not_counted(tmp_path):
    from analyze import summarize
    p = tmp_path / "s.jsonl"
    rows = [dict(question_id="0", condition="replay", variant="orig", parsed_answer="B", gold="B"),
            dict(question_id="0", condition="cot", variant="orig", parsed_answer="B", gold="B"),
            dict(question_id="0", condition="corrupt", variant="perceptual", corrupt_ok=False,
                 corrupt_reason="unchanged", gold="B"),
            dict(question_id="1", condition="replay", variant="orig", parsed_answer="A", gold="A"),
            dict(question_id="1", condition="cot", variant="orig", parsed_answer="A", gold="A"),
            dict(question_id="1", condition="corrupt", variant="perceptual", corrupt_ok=True,
                 parsed_answer="A", gold="A")]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    res = summarize(str(p), out=lambda *a: None)
    r = res[("corrupt", "perceptual")]
    assert r["n"] == 1 and r["unparsed"] == 0, r


def test_filler_steps_never_corrupted():
    """Real failing steps from job 42068: intro sentences were labelled perceptual."""
    from corruption import is_filler, mark_filler
    fillers = ["To determine which animal's mouth is adapted for bottom feeding, let's analyze the options provided:",
               "Let's analyze each object to determine their properties:",
               "Now, let's look at the options provided:",
               "To determine the name of the colony shown in the image, let's follow these steps:",
               "To answer this question, let's analyze the information provided and the image:"]
    facts = ["The image shows a fish with its mouth on the underside of its head.",
             "1. **Discus**: Discus fish are known to be surface feeders.",
             "Let me note that the map highlights Ohio in dark green.",
             "The state highlighted in dark green on the map is Massachusetts."]
    assert all(is_filler(s) for s in fillers), [s for s in fillers if not is_filler(s)]
    assert not any(is_filler(s) for s in facts), [s for s in facts if is_filler(s)]
    steps = [fillers[0], facts[1], "So it is a catfish."]
    assert mark_filler(steps, ["P", "L", "L"]) == ["F", "L", "L"]
    gen = lambda p: p.split("Original step: ")[1].split("\n")[0].replace("options", "answers")
    labels, _ = label_steps(lambda _: "1: P\n2: L\n3: L", "q", steps)
    assert labels == ["F", "L", "L"]
    r = corrupt_rationale(gen, "q", ["discus", "armored catfish"], steps, labels, "perceptual")
    assert not r["ok"] and r["reason"] == "no_perceptual_step"