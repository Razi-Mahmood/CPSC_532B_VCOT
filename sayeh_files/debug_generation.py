"""Find out why Qwen2.5-VL outputs are degraded ("not not not", "feedinging", "addCriterion").

Runs the same 3 CoT prompts under different settings and prints a degeneracy score
for each. Whichever setting gives clean text is the fix to put in pilot.py / the harness.

    sbatch run_pilot.sbatch   # after changing its python line to: python debug_generation.py

Settings tested:
  A  default: sdpa attention, bf16, fast image processor
  B  A + reset model.model.rope_deltas before every generate (stale position state)
  C  eager attention, bf16
  D  eager attention, bf16, slow image processor (use_fast=False)
  E  eager attention, float32  (rules out bf16 numerics on the B200)
"""
import re

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from parsing import format_choices

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = ("{context}Question: {question}\nOptions:\n{options}\n\n"
          "Think step by step about the image, then give your final answer "
          "as 'The answer is X' where X is the option letter.")

CONFIGS = {
    "A_default":        dict(attn="sdpa",  dtype=torch.bfloat16, use_fast=True,  reset_rope=False),
    "B_reset_rope":     dict(attn="sdpa",  dtype=torch.bfloat16, use_fast=True,  reset_rope=True),
    "C_eager":          dict(attn="eager", dtype=torch.bfloat16, use_fast=True,  reset_rope=True),
    "D_eager_slowproc": dict(attn="eager", dtype=torch.bfloat16, use_fast=False, reset_rope=True),
    "E_eager_fp32":     dict(attn="eager", dtype=torch.float32,  use_fast=True,  reset_rope=True),
}


def degeneracy(text):
    """Higher = worse. Counts repeated words, doubled suffixes, junk tokens."""
    rep_words = len(re.findall(r"\b(\w+)(?:\s+\1\b)+", text, flags=re.IGNORECASE))
    doubled_suffix = len(re.findall(r"\b\w*(ly|ing|s)\1\b", text))
    junk = text.count("addCriterion") + text.count("matchCondition")
    no_answer = 0 if re.search(r"answer is", text, re.IGNORECASE) else 1
    return rep_words + doubled_suffix + 5 * junk + 3 * no_answer


def main():
    ds = load_dataset("derek-thomas/ScienceQA", split="validation")
    ds = ds.filter(lambda ex: ex["image"] is not None).select(range(3))
    scores = {}
    for name, cfg in CONFIGS.items():
        print("#" * 80, f"\n{name}: {cfg}", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=cfg["dtype"], attn_implementation=cfg["attn"], device_map="auto")
        processor = AutoProcessor.from_pretrained(MODEL, use_fast=cfg["use_fast"])
        total = 0
        for i, ex in enumerate(ds):
            hint = (ex.get("hint") or "").strip()
            text_prompt = PROMPT.format(context=f"Context: {hint}\n" if hint else "",
                                        question=ex["question"], options=format_choices(ex["choices"]))
            # Two calls per question, like the pilot, so stale state between calls would show up.
            for call in range(2):
                messages = [{"role": "user", "content": [{"type": "image", "image": ex["image"]},
                                                         {"type": "text", "text": text_prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[ex["image"]], return_tensors="pt").to(model.device)
                if cfg["reset_rope"] and hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
                    model.model.rope_deltas = None
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
                gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                s = degeneracy(gen)
                total += s
                print(f"--- Q{i} call{call} degeneracy={s} ---\n{gen[:700]}\n", flush=True)
        scores[name] = total
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80 + "\nDEGENERACY BY SETTING (lower is better, 0 = clean):")
    for name, s in scores.items():
        print(f"  {name:18s} {s}")


if __name__ == "__main__":
    main()
