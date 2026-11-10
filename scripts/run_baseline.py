# Runs baselines: baseline CoT and direct QA
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from vlm import LETTERS, build_prompt, answer, load_model, parse_answer, split_steps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.data import load_processed

# import model / experiment configs
p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/baseline.yaml")
args = p.parse_args()

config_path = Path(args.config)
cfg = yaml.safe_load(open(config_path))

out_dir = ROOT / "results" / cfg["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, out_dir / "config.yaml") # copy cfg for reproducibility

# load model, processor, and data
model, processor = load_model(cfg["model"])
data = [ex for name in cfg["datasets"] for ex in load_processed(name)]
print(f"Loaded {len(data)} questions from datasets {cfg['datasets']}")

# run the two baselines
# for each baseline, save pred/outputs/rationales per example in a json file
for mode in ["direct", "cot"]:
    print(f"\n=== {mode} ===")
    correct = defaultdict(list)
    with open(out_dir / f"{mode}.jsonl", "w") as f:
        for ex in data:
            # build the prompt, e.g.: Q + options + CoT instruction
            # then get model's answer, parse the answer, store the results
            prompt = build_prompt(ex, cfg["prompts"][mode])
            raw_output = answer(model, processor, ex["image_path"], prompt, cfg["max_new_tokens"][mode], cfg["generation"])
            pred = parse_answer(raw_output)
            gold = LETTERS[ex["answer_idx"]]
            record = {
                "id": ex["id"],
                "dataset": ex["dataset"],
                "prompt": prompt,
                "raw_output": raw_output,
                "pred": pred,
                "gold": gold,
                "correct": pred == gold,
            }
            if mode == "cot":
                record["rationale"] = split_steps(raw_output)
            f.write(json.dumps(record) + "\n")
            correct[ex["dataset"]].append(pred == gold)
            print(ex["id"], "pred:", pred, "gold:", gold)

    for name, hits in correct.items():
        print(f"{mode} accuracy on {name}: {sum(hits)}/{len(hits)}")

print(f"\nSaved to {out_dir}")
