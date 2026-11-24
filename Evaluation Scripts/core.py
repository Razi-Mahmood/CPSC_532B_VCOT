"""CPU-only, read-only analysis of the team's append-only experiment caches.

Run from the repository root:
  python evaluation/analyze.py
No model imports, API requests, or source-cache writes occur in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import csv
from collections import Counter, defaultdict
from pathlib import Path

FRACTIONS = ("0.00", "0.25", "0.50", "0.75")
ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n):
    """Two-sided 95% Wilson score interval; undefined for n=0."""
    if not n:
        return None, None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., min(p, center - half)), min(1., max(p, center + half))


def mcnemar(b, c):
    """Exact two-sided binomial McNemar test on discordant pairs."""
    n = b + c
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (1 << n)) if n else 1.


def holm(values):
    """Holm adjusted p values in original order."""
    adjusted = [None] * len(values)
    previous = 0.
    for j, i in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        previous = max(previous, min(1., (len(values) - j) * values[i]))
        adjusted[i] = previous
    return adjusted


def load_cache(path):
    """Last physical line wins, matching ResultCache; malformed input is fatal."""
    rows, duplicate_log, n_lines = {}, [], 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        n_lines += 1
        try:
            r = json.loads(line)
            key = tuple(str(r[k]) for k in ("dataset", "question_id", "condition", "variant"))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path.name}:{line_no}: invalid cache row: {e}") from e
        if key in rows:
            duplicate_log.append(dict(dataset=key[0], question_id=key[1], condition=key[2],
                                      variant=key[3], replacement_line=line_no,
                                      old_answer=rows[key].get("parsed_answer"),
                                      new_answer=r.get("parsed_answer"),
                                      old_corrupt_ok=rows[key].get("corrupt_ok"),
                                      new_corrupt_ok=r.get("corrupt_ok")))
        rows[key] = r
    info = dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_rows=n_lines, unique_rows=len(rows), superseded_rows=len(duplicate_log),
                deduplication="last physical line per (dataset, question_id, condition, variant)")
    return rows, info, duplicate_log


class Dataset:
    def __init__(self, rows, name):
        self.name = name
        self.rows = {(q, c, v): r for (d, q, c, v), r in rows.items() if d == name}
        self.gold, self.choices = {}, {}
        for (q, c, v), r in self.rows.items():
            if r.get("gold") is not None:
                if q in self.gold and self.gold[q] != r["gold"]:
                    raise ValueError(f"Conflicting gold answers for {name}/{q}")
                self.gold[q] = r["gold"]
            if r.get("choices"):
                if q in self.choices and self.choices[q] != r["choices"]:
                    raise ValueError(f"Conflicting choice order for {name}/{q}; do not mix runs")
                self.choices[q] = r["choices"]
        self.ids = sorted({q for q, c, v in self.rows if c in ("direct", "cot")})

    def row(self, q, condition, variant="orig"):
        return self.rows.get((q, condition, variant))

    def answer(self, r):
        if not r or r.get("corrupt_ok") is False:
            return None
        a = r.get("parsed_answer")
        q = str(r["question_id"])
        n = len(self.choices[q]) if q in self.choices else 26
        return a if isinstance(a, str) and len(a) == 1 and a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n] else None

    def comparison(self, condition, variant, ref_condition, ref_variant="orig"):
        attempts = {q: r for (q, c, v), r in self.rows.items() if (c, v) == (condition, variant)}
        eligible = {q: r for q, r in attempts.items() if r.get("corrupt_ok") is not False}
        if not attempts:
            return None, []
        for q in eligible:
            if q not in self.gold:
                raise ValueError(f"Missing gold for {self.name}/{q}")
        parsed = {q: self.answer(r) for q, r in eligible.items() if self.answer(r) is not None}
        pairs = []
        for q, a in parsed.items():
            ref = self.answer(self.row(q, ref_condition, ref_variant))
            if ref is None:
                continue
            gold = self.gold[q]
            pairs.append(dict(dataset=self.name, question_id=q, condition=condition, variant=variant,
                              reference_condition=ref_condition, reference_variant=ref_variant,
                              answer=a, reference_answer=ref, gold=gold,
                              flipped=int(a != ref), correct=int(a == gold), reference_correct=int(ref == gold),
                              harmed=int(ref == gold and a != gold), helped=int(ref != gold and a == gold),
                              wrong_to_wrong=int(a != ref and a != gold and ref != gold)))
        n = len(pairs)
        flips = sum(p["flipped"] for p in pairs)
        helped, harmed = (sum(p[k] for p in pairs) for k in ("helped", "harmed"))
        lo, hi = wilson(flips, n)
        correct = sum(a == self.gold[q] for q, a in parsed.items())
        alo, ahi = wilson(correct, len(eligible))
        result = dict(dataset=self.name, condition=condition, variant=variant,
                      reference_condition=ref_condition, reference_variant=ref_variant,
                      cohort_n=len(self.ids), attempted_n=len(attempts), skipped_n=len(attempts)-len(eligible),
                      eligible_n=len(eligible), parsed_n=len(parsed), unparsed_n=len(eligible)-len(parsed),
                      paired_n=n, parsed_without_reference_n=len(parsed)-n,
                      correct_n=correct, accuracy=correct/len(eligible) if eligible else None,
                      accuracy_ci_low=alo, accuracy_ci_high=ahi,
                      parsed_accuracy=correct/len(parsed) if parsed else None,
                      flips=flips, flip_rate=flips/n if n else None, flip_ci_low=lo, flip_ci_high=hi,
                      helped=helped, harmed=harmed, wrong_to_wrong=sum(p["wrong_to_wrong"] for p in pairs),
                      accuracy_mcnemar_p=mcnemar(harmed, helped) if n else None)
        return result, pairs


def paired_flip_test(ds, first, second, allowed=None):
    pairs = []
    for q in ds.ids:
        if allowed is not None and q not in allowed:
            continue
        ref = ds.answer(ds.row(q, "replay"))
        a = ds.answer(ds.row(q, *first))
        b = ds.answer(ds.row(q, *second))
        if None not in (ref, a, b):
            pairs.append((q, a != ref, b != ref))
    n = len(pairs)
    fa, fb = (sum(p[i] for p in pairs) for i in (1, 2))
    only_a = sum(a and not b for _, a, b in pairs)
    only_b = sum(b and not a for _, a, b in pairs)
    alo, ahi = wilson(fa, n)
    blo, bhi = wilson(fb, n)
    return dict(dataset=ds.name, first_condition=first[0], first_variant=first[1],
                second_condition=second[0], second_variant=second[1], n=n,
                first_flips=fa, second_flips=fb, first_rate=fa/n if n else None,
                second_rate=fb/n if n else None, first_ci_low=alo, first_ci_high=ahi,
                second_ci_low=blo, second_ci_high=bhi, only_first=only_a, only_second=only_b,
                difference_first_minus_second=(fa-fb)/n if n else None,
                exact_mcnemar_p=mcnemar(only_a, only_b) if n else None), [p[0] for p in pairs]


def rq3(ds):
    rows, membership = [], []
    groups = defaultdict(list)
    for q in ds.ids:
        ref = ds.answer(ds.row(q, "replay"))
        trunc = [ds.answer(ds.row(q, "truncate", f)) for f in FRACTIONS]
        cot = ds.answer(ds.row(q, "cot"))
        if None in (ref, cot, *trunc):
            group = "incomplete"
        else:
            group = "sensitive" if any(a != ref for a in trunc) else "stable"
        membership.append(dict(dataset=ds.name, question_id=q, group=group,
                               definition="all four truncations, replay and CoT must parse"))
        groups[group].append(q)
    for group in ("sensitive", "stable", "incomplete"):
        qs = groups[group]
        for mode in ("cot", "direct"):
            present = [q for q in qs if ds.row(q, mode) is not None]
            correct = sum(ds.answer(ds.row(q, mode)) == ds.gold[q] for q in present)
            lo, hi = wilson(correct, len(present))
            rows.append(dict(dataset=ds.name, group=group, mode=mode, group_n=len(qs),
                             n=len(present), correct=correct, accuracy=correct/len(present) if present else None,
                             ci_low=lo, ci_high=hi,
                             unparsed=sum(ds.answer(ds.row(q, mode)) is None for q in present)))
    return rows, membership


def read_review_csv(path):
    """Read explicit overall decisions; blank entries remain pending."""
    required = {"question_id", "kind", "step_idx", "original_step", "corrupted_step", "valid_corruption"}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV is missing required columns: " + ", ".join(sorted(required - set(reader.fieldnames or []))))
        records = list(reader)
    for line, r in enumerate(records, 2):
        if None in r or any(v is None for v in r.values()):
            raise ValueError(f"Review CSV row {line} has the wrong number of fields")
        decision = r["valid_corruption"].strip().upper()
        if decision not in ("Y", "N", ""):
            raise ValueError(f"Review CSV row {line}: valid_corruption must be Y, N, or blank")
        r["valid_corruption"] = decision
    return records


def review_summary(ds, review_csv):
    """Apply only exact matching ScienceQA sample decisions; never use stale edits."""
    counts, exclusions, details = {}, set(), []
    if ds.name != "scienceqa" or review_csv is None or not review_csv.exists():
        return counts, exclusions, details
    records = read_review_csv(review_csv)
    for r in records:
        q, kind = r["question_id"], r["kind"]
        current = ds.row(q, "corrupt", kind)
        same = bool(current) and all(str(current.get(k)) == r[k]
                                    for k in ("step_idx", "original_step", "corrupted_step"))
        details.append(dict(dataset=ds.name, question_id=q, kind=kind,
                            matches_current_cache=same, reviewed_valid=r["valid_corruption"],
                            applied_exclusion=same and r["valid_corruption"] == "N"))
        if same and r["valid_corruption"] == "N":
            exclusions.add((q, kind))
    counts = dict(review_rows=len(records), matches_current=sum(d["matches_current_cache"] for d in details),
                  valid_counts=dict(Counter(r["valid_corruption"] for r in records)),
                  status="Provisional sample decisions; independent validation pending")
    return counts, exclusions, details


def analyze(ds, review_dir):
    metrics, pairs = [], []
    tests = [("direct", "orig", "direct", "orig"), ("cot", "orig", "direct", "orig"),
             ("replay", "orig", "cot", "orig")]
    tests += [("truncate", f, "replay", "orig") for f in FRACTIONS]
    kinds = sorted({v for q, c, v in ds.rows if c == "corrupt"})
    tests += [("corrupt", k, "replay", "orig") for k in kinds]
    tests += [("paraphrase_r", "orig", "replay", "orig"),
              ("paraphrase_q", "cot", "cot", "orig"), ("paraphrase_q", "direct", "direct", "orig")]
    for c, v, rc, rv in tests:
        result, records = ds.comparison(c, v, rc, rv)
        if result:
            metrics.append(result)
            pairs.extend(records)
    family = [m for m in metrics if m["condition"] in ("truncate", "corrupt", "paraphrase_r", "paraphrase_q")
              and m["accuracy_mcnemar_p"] is not None]
    for m, p in zip(family, holm([m["accuracy_mcnemar_p"] for m in family])):
        m["accuracy_holm_p"] = p
    rq2, shared = paired_flip_test(ds, ("corrupt", "perceptual"), ("corrupt", "logical"))
    control_tests = []
    for c, v in [("truncate", f) for f in FRACTIONS] + [("corrupt", k) for k in kinds]:
        t, _ = paired_flip_test(ds, (c, v), ("paraphrase_r", "orig"))
        if t["n"]:
            control_tests.append(t)
    for t, p in zip(control_tests, holm([t["exact_mcnemar_p"] for t in control_tests])):
        t["holm_p"] = p
    group_results, groups = rq3(ds)
    checks, exclusions, review_matches = review_summary(ds, review_dir)
    allowed = {q for q in ds.ids if (q, "perceptual") not in exclusions and (q, "logical") not in exclusions}
    sensitivity, sensitivity_ids = paired_flip_test(ds, ("corrupt", "perceptual"), ("corrupt", "logical"), allowed)
    sensitivity["analysis"] = "exclude matching rejected sample interventions; remaining rows are not fully validated"
    skip_reasons = Counter((r["variant"], r.get("corrupt_reason", "unknown"))
                           for (q, c, v), r in ds.rows.items() if c == "corrupt" and r.get("corrupt_ok") is False)
    coverage = [dict(dataset=ds.name, kind=k, reason=r, count=n) for (k, r), n in sorted(skip_reasons.items())]
    versions = Counter(r.get("prompt_version", "missing") for r in ds.rows.values())
    metadata = dict(dataset=ds.name, n_questions=len(ds.ids), prompt_versions=dict(versions),
                    model_values=sorted({str(r.get("model")) for r in ds.rows.values()}),
                    dtype_values=sorted({str(r.get("dtype")) for r in ds.rows.values()}),
                    attention_values=sorted({str(r.get("attn")) for r in ds.rows.values()}),
                    rq2_shared_ids=shared, sensitivity_shared_ids=sensitivity_ids, review=checks,
                    mixed_prompt_versions=len(versions) > 1)
    return dict(metrics=metrics, pairs=pairs, rq2=[rq2], controls=control_tests, rq3=group_results,
                groups=groups, sensitivity=[sensitivity], coverage=coverage, review_matches=review_matches,
                metadata=metadata)

