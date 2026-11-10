# Perturb each saved CoT rationale and check if the model's answer changes
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from vlm import answer_given_rationale, load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.data import load_processed

# functions for rationale perturbations
def truncate(steps):
    """Keep only the first k steps, k = 0 (no reasoning) ... all steps."""
    return [(k, steps[:k]) for k in range(len(steps) + 1)]

PERTURBATIONS = {"truncate": truncate}


# import model / experiment configs
p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/perturb.yaml")
args = p.parse_args()

config_path = Path(args.config)
cfg = yaml.safe_load(open(config_path))

out_dir = ROOT / "results" / cfg["run_name"]
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, out_dir / "config.yaml") # copy cfg for reproducibility

# load model, processor, data, and the CoT baseline
model, processor = load_model(cfg["model"])
by_id = {ex["id"]: ex for name in cfg["datasets"] for ex in load_processed(name)}
cot = [json.loads(line) for line in open(ROOT / cfg["baseline"])]
print(f"Loaded {len(cot)} CoT rationales from {cfg['baseline']}")

# one row per (question, perturbed rationale)
flips = defaultdict(list)
with open(out_dir / "perturbations.jsonl", "w") as f:
    for r in cot:
        ex = by_id[r["id"]]
        for name in cfg["perturbations"]:
            for variant, steps in PERTURBATIONS[name](r["rationale"]):
                out = answer_given_rationale(model, processor, ex["image_path"], r["prompt"], steps, len(ex["choices"]))
                record = {
                    "id": r["id"],
                    "dataset": r["dataset"],
                    "perturbation": name,
                    "variant": variant,
                    "n_steps": len(r["rationale"]),  # length of the original rationale
                    "rationale": steps,              # the perturbed rationale that was fed in
                    "pred": out["pred"],
                    "probs": out["probs"],
                    "orig_pred": r["pred"],          # answer with the full rationale (baseline CoT)
                    "gold": r["gold"],
                    "flipped": out["pred"] != r["pred"],
                }
                f.write(json.dumps(record) + "\n")
                flips[name].append(record["flipped"])
        print(r["id"], "done")

for name, hits in flips.items():
    print(f"{name}: {sum(hits)}/{len(hits)} answers flipped")

print(f"\nSaved to {out_dir}")
