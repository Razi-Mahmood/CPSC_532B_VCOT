# Made w/ help from Claude (only to setup the common format)
""" Load samples from both datasets, format them. 

Each example is stored as a dict:
    id
    dataset ("scienceqa" or "aokvqa")
    question, context, choices, answer_idx (tuple)
    image_path (./processed/images/)
    meta (extra info)
"""
import ast
import json
import argparse
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent / "processed"

# TODO we can modify this as we go along, i.e. if we need need one more column from the data, specific # of samples, or split from the data, etc.
def scienceqa_samples(n, seed=0):
    """ Choose `n` random ScienceQA validation questions that have an image."""
    ds = load_dataset("derek-thomas/ScienceQA", split="validation")
    ds = ds.add_column("row", range(len(ds)))
    ds = ds.filter(lambda ex: ex["image"] is not None).shuffle(seed=seed).select(range(n))
    for ex in ds:
        yield {
            "id": f"sqa_{ex['row']}",
            "dataset": "scienceqa",
            "question": ex["question"],
            "choices": ex["choices"],
            "answer_idx": ex["answer"],
            "context": ex["hint"],
            "meta": {k: ex[k] for k in ["subject", "category", "grade", "solution", "skill"]},
        }, ex["image"]


def aokvqa_samples(n, seed=0):
    """ Choose `n1 random A-OKVQA train questions, """
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train").shuffle(seed=seed).select(range(n))
    for ex in ds:
        yield {
            "id": f"aok_{ex['question_id']}",
            "dataset": "aokvqa",
            "question": ex["question"],
            "choices": ex["choices"],
            "answer_idx": ex["correct_choice_idx"],
            "context": "",
            "meta": {
                "rationales": ex["rationales"],
                "direct_answers": ast.literal_eval(ex["direct_answers"]),  # stored as a string
            },
        }, ex["image"]

def save(name, samples):
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for example, image in samples:
            example["image_path"] = f"images/{example['id']}.png"
            image.convert("RGB").save(OUT_DIR / example["image_path"])
            f.write(json.dumps(example) + "\n")
            print(f"{example['id']}: {example['question'][:70]}")


def load_processed(name):
    """ Read saved examples ("scienceqa" or "aokvqa") as a list of dicts, with absolute image paths."""
    examples = [json.loads(line) for line in open(OUT_DIR / f"{name}.jsonl")]
    for ex in examples:
        ex["image_path"] = str(OUT_DIR / ex["image_path"])
    return examples


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="examples per dataset")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("ScienceQA:")
    save("scienceqa", scienceqa_samples(args.n, args.seed))
    print("A-OKVQA:")
    save("aokvqa", aokvqa_samples(args.n, args.seed))
    print(f"Saved to {OUT_DIR}")
