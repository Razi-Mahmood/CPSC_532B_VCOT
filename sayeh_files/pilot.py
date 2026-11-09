"""10-question pilot: see real Qwen2.5-VL outputs and test parsing/steps on them.

Run on a GPU node (7B in float32 needs ~35 GB of GPU memory):
    pip install "transformers>=4.49" accelerate datasets qwen-vl-utils pillow
    python pilot.py --n 10

Writes results/pilot.jsonl (via ResultCache) and prints a readable summary.
This is a throwaway pilot, NOT the team harness: Carlos/Razi's
answer_given_rationale replaces answer_given_rationale_pilot() below.
"""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache import ResultCache
from parsing import format_choices, parse_answer, index_to_letter
from steps import strip_final_answer, split_steps, truncation_variants

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
# bf16 gave garbled text on the B200 (job 41012: "feedinging", "addCriterion").
# float32 + eager attention was the only clean setting. Every team script must use the same.
ATTN = "eager"
DTYPE = torch.float32

DIRECT_TMPL = ("{context}Question: {question}\nOptions:\n{options}\n\n"
               "Answer with the option's letter only.")
COT_TMPL = ("{context}Question: {question}\nOptions:\n{options}\n\n"
            "Think step by step about the image, then give your final answer "
            "as 'The answer is X' where X is the option letter.")


def build_prompt(tmpl, ex):
    hint = (ex.get("hint") or "").strip()
    return tmpl.format(context=f"Context: {hint}\n" if hint else "",
                       question=ex["question"], options=format_choices(ex["choices"]))


def chat(model, processor, image, user_text, assistant_prefix="", max_new_tokens=1024):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += assistant_prefix          # forced start of the assistant's reply
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None   # don't carry position state over from the previous call
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def answer_given_rationale_pilot(model, processor, ex, rationale):
    """Feed a fixed rationale as the start of the model's reply; generate only the letter."""
    prefix = (rationale.strip() + "\n" if rationale.strip() else "") + "The answer is ("
    out = chat(model, processor, ex["image"], build_prompt(COT_TMPL, ex),
               assistant_prefix=prefix, max_new_tokens=5)
    return "(" + out          # so the parser sees "(B)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    ds = load_dataset("derek-thomas/ScienceQA", split=args.split)
    ds = ds.filter(lambda ex: ex["image"] is not None).select(range(args.n))

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=DTYPE, device_map="auto",
        attn_implementation=ATTN)
    processor = AutoProcessor.from_pretrained(MODEL)
    cache = ResultCache("results/pilot.jsonl",
                        meta={"model": MODEL, "temperature": 0, "prompt_version": "pilot2",
                              "dtype": str(DTYPE), "attn": ATTN})

    for qid, ex in enumerate(ds):
        gold = index_to_letter(ex["answer"])
        ch = ex["choices"]

        def run(condition, variant, raw, **extra):
            letter, method = parse_answer(raw, ch, strict=(condition == "cot"))
            cache.add("scienceqa", qid, condition, variant, raw_output=raw,
                      parsed_answer=letter, parse_method=method, gold=gold, **extra)
            return letter

        d = run("direct", "orig", chat(model, processor, ex["image"], build_prompt(DIRECT_TMPL, ex), max_new_tokens=10))
        cot_raw = chat(model, processor, ex["image"], build_prompt(COT_TMPL, ex))
        c = run("cot", "orig", cot_raw)
        rationale = strip_final_answer(cot_raw)
        r = run("replay", "orig", answer_given_rationale_pilot(model, processor, ex, rationale),
                input_rationale=rationale)
        truncs = {}
        for frac, v in truncation_variants(rationale).items():
            truncs[frac] = run("truncate", frac, answer_given_rationale_pilot(model, processor, ex, v["text"]),
                               input_rationale=v["text"], k_kept=v["k_kept"], n_steps=v["n_steps"])

        print("=" * 80)
        print(f"Q{qid}: {ex['question']}  | choices={ch} | gold={gold}")
        print(f"direct={d}  cot={c}  replay={r}  " + "  ".join(f"t{k}={a}" for k, a in truncs.items()))
        print("--- raw CoT output ---\n" + cot_raw)
        print("--- steps after strip_final_answer + split_steps ---")
        for i, s in enumerate(split_steps(rationale)):
            print(f"  [{i}] {s}")

    df = cache.to_dataframe()
    print("\nParse methods:\n", df.groupby("condition")["parse_method"].value_counts())
    rep = df[df.condition == "replay"].set_index("question_id").parsed_answer
    cot = df[df.condition == "cot"].set_index("question_id").parsed_answer
    print(f"\nReplay agreement with CoT: {(rep == cot).mean():.0%}  (should be close to 100%)")


if __name__ == "__main__":
    main()