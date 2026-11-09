"""Append-only JSONL cache for every model call.

One row per (dataset, question_id, condition, variant). Re-running a script
skips rows already on disk, so a crashed job on the cluster resumes where it stopped.

Usage:
    from cache import ResultCache
    cache = ResultCache("results/qwen7b.jsonl")
    key = dict(dataset="scienceqa", question_id=qid, condition="truncate", variant="0.50")
    if not cache.has(**key):
        raw = answer_given_rationale(image, question, rationale)
        letter, method = parse_answer(raw, choices)
        cache.add(**key, input_rationale=rationale, raw_output=raw,
                  parsed_answer=letter, parse_method=method, gold=gold_letter)
    df = cache.to_dataframe()   # for analysis

Conditions we use (keep these strings consistent across the team):
    direct          - no rationale, answer directly
    cot             - model writes rationale + answer (variant="orig")
    replay          - original rationale fed back via answer_given_rationale (sanity check)
    paraphrase_q    - paraphrased question (variant = paraphrase id, e.g. "p0")
    truncate        - variant = fraction kept, e.g. "0.00", "0.25", "0.50", "0.75"
    corrupt         - variant = "perceptual", "logical", "grammar", "irrelevant", "contradiction"
    improve         - (optional) variant = "prune", "refine", "describe"
"""
import json
import os
import time
from typing import Iterator

KEY_FIELDS = ("dataset", "question_id", "condition", "variant")


def _key(dataset, question_id, condition, variant="orig"):
    return (str(dataset), str(question_id), str(condition), str(variant))


class ResultCache:
    def __init__(self, path, meta=None):
        """meta: dict written into every row, e.g. {"model": "Qwen2.5-VL-7B-Instruct",
        "temperature": 0, "prompt_version": "v1"}. Bump prompt_version if you change a
        template, so old rows are not silently mixed with new ones."""
        self.path = path
        self.meta = meta or {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._rows = {}
        if os.path.exists(path):
            with open(path) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A job killed mid-write can leave a partial last line.
                        print(f"[cache] skipping corrupt line {line_no} in {path}")
                        continue
                    self._rows[_key(*(row.get(k, "orig") for k in KEY_FIELDS))] = row

    def has(self, dataset, question_id, condition, variant="orig"):
        return _key(dataset, question_id, condition, variant) in self._rows

    def get(self, dataset, question_id, condition, variant="orig"):
        return self._rows.get(_key(dataset, question_id, condition, variant))

    def add(self, dataset, question_id, condition, variant="orig", **fields):
        row = {"dataset": str(dataset), "question_id": str(question_id),
               "condition": str(condition), "variant": str(variant),
               **self.meta, **fields, "timestamp": time.time()}
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._rows[_key(dataset, question_id, condition, variant)] = row
        return row

    def rows(self, **filters) -> Iterator[dict]:
        for r in self._rows.values():
            if all(str(r.get(k)) == str(v) for k, v in filters.items()):
                yield r

    def __len__(self):
        return len(self._rows)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(list(self._rows.values()))
