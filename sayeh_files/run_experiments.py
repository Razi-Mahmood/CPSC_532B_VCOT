"""Main experiment runner: baseline, truncation, paraphrase controls, option order.

Examples (run through run_experiments.sbatch):
    python run_experiments.py --dataset scienceqa --n 100                      # same set as the pilot
    python run_experiments.py --dataset scienceqa --n 100 --reverse_choices    # option-order check
    python run_experiments.py --dataset scienceqa --n 100 --drop_conclusion    # conclusion decision
    python run_experiments.py --dataset aokvqa --n 500
    python run_experiments.py --dataset scienceqa --n 500 --offset 500         # split big runs
    python run_experiments.py --dataset scienceqa --n 100 --corrupt --corrupt_extra

Each configuration writes its own cache file (name printed at start), e.g.
results/scienceqa.jsonl, results/scienceqa_rev.jsonl, results/scienceqa_dropconc.jsonl.
Re-running the same command resumes: finished (question, condition, variant) rows are skipped.
At the end it prints the analysis (same as `python analyze.py <file>`).

Conditions (cache "condition" / "variant"):
    direct        orig          answer only
    cot           orig          rationale + answer
    replay        orig          original rationale fed back (reference for rationale perturbations)
    truncate      0.00..0.75    first k% of steps fed back
    paraphrase_q  text          the paraphrased question (+ whether it passed the filters)
    paraphrase_q  cot / direct  CoT / direct answer to the paraphrased question
                                -> noise floor vs cot / direct on the original question
    paraphrase_r  text          the paraphrased rationale
    paraphrase_r  orig          paraphrased rationale fed back
                                -> noise floor for rationale perturbations (vs replay)

With --corrupt (experiment 4, RQ2):
    labels        steps         each step labelled P (perceptual) or L (logical)
    corrupt       perceptual    one error inserted into a perceptual step, fed back
    corrupt       logical       one error inserted into a logical step, fed back
With --corrupt_extra as well (instructor's optional suggestion):
    corrupt       grammar / irrelevant / contradiction
All corrupt variants are compared against replay.
The labels and corruptions are written by Qwen-7B itself in text-only mode.
"""
import argparse
import os

import torch

from cache import ResultCache
from parsing import format_choices, parse_answer, index_to_letter
from steps import strip_final_answer, truncation_variants
from paraphrase import paraphrase_question, paraphrase_rationale
from steps import split_steps, join_steps
from corruption import (label_steps, corrupt_rationale, grammar_noise,
                        insert_irrelevant, insert_contradiction, mark_filler, is_filler)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ATTN = "eager"            # bf16 is broken on the B200 (job 41012); keep float32 + eager
DTYPE = torch.float32
PROMPT_VERSION = "v1"

DIRECT_TMPL = ("{context}Question: {question}\nOptions:\n{options}\n\n"
               "Answer with the option's letter only.")
COT_TMPL = ("{context}Question: {question}\nOptions:\n{options}\n\n"
            "Think step by step about the image, then give your final answer "
            "as 'The answer is X' where X is the option letter.")


# ------------------------------------------------------------------ data

def load_examples(dataset, n, offset=0, split=None, reverse_choices=False):
    """Returns a list of dicts: qid, image, question, choices, gold_idx, hint."""
    from datasets import load_dataset
    if dataset == "scienceqa":
        ds = load_dataset("derek-thomas/ScienceQA", split=split or "validation")
        ds = ds.filter(lambda ex: ex["image"] is not None)
        idx = range(offset, min(offset + n, len(ds)))
        exs = [dict(qid=str(i), image=ds[i]["image"], question=ds[i]["question"],
                    choices=list(ds[i]["choices"]), gold_idx=int(ds[i]["answer"]),
                    hint=(ds[i].get("hint") or "").strip()) for i in idx]
    elif dataset == "aokvqa":
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split or "validation")
        idx = range(offset, min(offset + n, len(ds)))
        exs = [dict(qid=str(ds[i]["question_id"]), image=ds[i]["image"], question=ds[i]["question"],
                    choices=list(ds[i]["choices"]), gold_idx=int(ds[i]["correct_choice_idx"]),
                    hint="") for i in idx]
    else:
        raise ValueError(dataset)
    if reverse_choices:
        exs = [reverse_example(e) for e in exs]
    return exs


def reverse_example(ex):
    k = len(ex["choices"])
    return {**ex, "choices": ex["choices"][::-1], "gold_idx": k - 1 - ex["gold_idx"]}


def build_prompt(tmpl, ex, question=None):
    ctx = f"Context: {ex['hint']}\n" if ex.get("hint") else ""
    return tmpl.format(context=ctx, question=question or ex["question"],
                       options=format_choices(ex["choices"]))


def out_path(args):
    name = args.dataset + ("_rev" if args.reverse_choices else "") + \
        ("_dropconc" if args.drop_conclusion else "")
    return os.path.join(args.out_dir, name + ".jsonl")


# ------------------------------------------------------------------ model

class Qwen:
    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=DTYPE, device_map="auto", attn_implementation=ATTN)
        self.processor = AutoProcessor.from_pretrained(MODEL)

    def _generate(self, text, images, max_new_tokens):
        kw = dict(text=[text], return_tensors="pt")
        if images:
            kw["images"] = images
        inputs = self.processor(**kw).to(self.model.device)
        if hasattr(self.model, "model") and hasattr(self.model.model, "rope_deltas"):
            self.model.model.rope_deltas = None
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def chat(self, image, user_text, assistant_prefix="", max_new_tokens=1024):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": user_text}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self._generate(text + assistant_prefix, [image], max_new_tokens)

    def text(self, prompt, max_new_tokens=512):
        """Text-only generation (used to write paraphrases)."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self._generate(text, None, max_new_tokens)

    def answer_given_rationale(self, ex, rationale, question=None):
        prefix = (rationale.strip() + "\n" if rationale.strip() else "") + "The answer is ("
        return "(" + self.chat(ex["image"], build_prompt(COT_TMPL, ex, question),
                               assistant_prefix=prefix, max_new_tokens=5)


# ------------------------------------------------------------------ one question

def run_question(m, cache, ds_name, ex, args):
    qid, ch = ex["qid"], ex["choices"]
    gold = index_to_letter(ex["gold_idx"])
    base = dict(dataset=ds_name, question_id=qid)

    def done(cond, var):
        return cache.has(ds_name, qid, cond, var)

    def save(cond, var, raw, strict=False, **extra):
        letter, method = parse_answer(raw, ch, strict=strict)
        cache.add(ds_name, qid, cond, var, raw_output=raw, parsed_answer=letter,
                  parse_method=method, gold=gold, choices=ch, **extra)

    # baseline
    if not done("direct", "orig"):
        save("direct", "orig", m.chat(ex["image"], build_prompt(DIRECT_TMPL, ex), max_new_tokens=10))
    if not done("cot", "orig"):
        save("cot", "orig", m.chat(ex["image"], build_prompt(COT_TMPL, ex)), strict=True)
    cot_raw = cache.get(ds_name, qid, "cot", "orig")["raw_output"]
    rationale = strip_final_answer(cot_raw, ch, drop_conclusion=args.drop_conclusion)

    if not done("replay", "orig"):
        save("replay", "orig", m.answer_given_rationale(ex, rationale), input_rationale=rationale)

    # truncation
    for frac, v in truncation_variants(rationale).items():
        if not done("truncate", frac):
            save("truncate", frac, m.answer_given_rationale(ex, v["text"]),
                 input_rationale=v["text"], k_kept=v["k_kept"], n_steps=v["n_steps"])

    if getattr(args, "corrupt", False) and rationale.strip():
        run_corruption(m, cache, ds_name, ex, rationale, save, done,
                       extra=getattr(args, "corrupt_extra", False))

    if args.skip_paraphrase:
        return

    # question paraphrase (noise floor for the question wording)
    if not done("paraphrase_q", "text"):
        p, ok, reason = paraphrase_question(m.text, ex["question"], ch)
        cache.add(ds_name, qid, "paraphrase_q", "text", paraphrase=p, ok=ok, reason=reason,
                  original=ex["question"], choices=ch, gold=gold)
    prow = cache.get(ds_name, qid, "paraphrase_q", "text")
    if prow["ok"]:
        if not done("paraphrase_q", "direct"):
            save("paraphrase_q", "direct",
                 m.chat(ex["image"], build_prompt(DIRECT_TMPL, ex, prow["paraphrase"]), max_new_tokens=10))
        if not done("paraphrase_q", "cot"):
            save("paraphrase_q", "cot",
                 m.chat(ex["image"], build_prompt(COT_TMPL, ex, prow["paraphrase"])), strict=True)

    # rationale paraphrase (noise floor for rationale perturbations)
    if rationale.strip():
        if not done("paraphrase_r", "text"):
            p, ok, reason = paraphrase_rationale(m.text, rationale)
            cache.add(ds_name, qid, "paraphrase_r", "text", paraphrase=p, ok=ok, reason=reason,
                      original=rationale, choices=ch, gold=gold)
        rrow = cache.get(ds_name, qid, "paraphrase_r", "text")
        if rrow["ok"] and not done("paraphrase_r", "orig"):
            save("paraphrase_r", "orig", m.answer_given_rationale(ex, rrow["paraphrase"]),
                 input_rationale=rrow["paraphrase"])


def run_corruption(m, cache, ds_name, ex, rationale, save, done, extra=False):
    """Experiment 4: label steps, insert one error, feed the corrupted rationale back."""
    qid, ch = ex["qid"], ex["choices"]
    steps = split_steps(rationale)

    # 1. label every step P / L (saved once, reused on resume)
    if not done("labels", "steps"):
        labels, how = label_steps(m.text, ex["question"], steps)
        cache.add(ds_name, qid, "labels", "steps", steps=steps, labels=labels, method=how,
                  question=ex["question"], choices=ch)
    labels = mark_filler(steps, cache.get(ds_name, qid, "labels", "steps")["labels"])

    # 2. build the corrupted rationales
    kinds = ["perceptual", "logical"] + (["grammar", "irrelevant", "contradiction"] if extra else [])
    for kind in kinds:
        prev = cache.get(ds_name, qid, "corrupt", kind)
        if prev and prev.get("corrupt_ok") and prev.get("original_step") and is_filler(prev["original_step"]):
            prev = None                   # an earlier run corrupted a filler sentence: redo it
        if prev and (prev.get("corrupt_ok")
                     or kind in ("grammar", "irrelevant")                  # rule-based: same result again
                     or str(prev.get("corrupt_reason", "")).startswith("no_")):   # no step of this kind
            continue                      # failed model-written corruptions are retried on a rerun
        if kind in ("perceptual", "logical"):
            res = corrupt_rationale(m.text, ex["question"], ch, steps, labels, kind, seed=qid)
        elif kind == "grammar":
            res = grammar_noise(steps, seed=qid)
        elif kind == "irrelevant":
            res = insert_irrelevant(steps, ch, seed=qid)
        else:
            res = insert_contradiction(m.text, ex["question"], ch, steps, seed=qid)

        info = dict(step_idx=res["step_idx"], original_step=res["original_step"],
                    corrupted_step=res["corrupted_step"], corrupt_ok=res["ok"],
                    corrupt_reason=res["reason"], n_steps=len(steps), question=ex["question"],
                    corrupt_tries=res.get("tries"))
        if res["ok"]:
            # 3. feed it back, exactly like truncation
            save("corrupt", kind, m.answer_given_rationale(ex, res["text"]),
                 input_rationale=res["text"], **info)
        else:
            # record the skip (e.g. no perceptual step) - the report needs these counts
            cache.add(ds_name, qid, "corrupt", kind, choices=ch, gold=index_to_letter(ex["gold_idx"]), **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["scienceqa", "aokvqa"], default="scienceqa")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", default=None, help="default: validation")
    ap.add_argument("--reverse_choices", action="store_true")
    ap.add_argument("--drop_conclusion", action="store_true")
    ap.add_argument("--skip_paraphrase", action="store_true")
    ap.add_argument("--corrupt", action="store_true", help="experiment 4: perceptual/logical corruption")
    ap.add_argument("--corrupt_extra", action="store_true", help="also grammar/irrelevant/contradiction")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    path = out_path(args)
    print(f"Writing to {path}", flush=True)
    exs = load_examples(args.dataset, args.n, args.offset, args.split, args.reverse_choices)
    cache = ResultCache(path, meta={"model": MODEL, "dtype": str(DTYPE), "attn": ATTN,
                                    "temperature": 0, "prompt_version": PROMPT_VERSION,
                                    "reverse_choices": args.reverse_choices,
                                    "drop_conclusion": args.drop_conclusion})
    m = Qwen()
    for i, ex in enumerate(exs):
        try:
            run_question(m, cache, args.dataset, ex, args)
        except Exception as e:                      # one bad example shouldn't kill a long run
            print(f"[error] qid={ex['qid']}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(exs)} done", flush=True)

    from analyze import summarize
    summarize(path)


if __name__ == "__main__":
    main()