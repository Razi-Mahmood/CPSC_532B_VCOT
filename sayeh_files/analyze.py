"""Analysis of a results cache (and optionally a reversed-choices cache).

    python analyze.py results/scienceqa.jsonl
    python analyze.py results/scienceqa.jsonl --reversed results/scienceqa_rev.jsonl
    python analyze.py results/scienceqa.jsonl --compare results/scienceqa_dropconc.jsonl
    python analyze.py results/scienceqa.jsonl --spot_check spot_check_paraphrase.csv
    python analyze.py results/scienceqa.jsonl --corruption_check check   # check_labels.csv, check_corruptions.csv

Flip rate = fraction of questions (with both answers parsed) where the answer differs
from the reference. References:
    truncate, paraphrase_r   vs replay
    paraphrase_q/cot         vs cot
    paraphrase_q/direct      vs direct
McNemar (exact) tests whether accuracy changed vs the reference.
"""
import argparse
import json
import math
from collections import defaultdict

LETTERS = "ABCDEFGHIJ"


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["question_id"], r["condition"], r["variant"])] = r   # later lines win
    return rows


def choices_for(rows, q):
    """Answer options for question q, from any row that stored them (pilot rows didn't)."""
    for (qq, c, v), r in rows.items():
        if qq == q and r.get("choices"):
            return r["choices"]
    return None


def answers(rows, cond, var):
    """Parsed answers for one condition. Corruption rows that were skipped (no valid
    corruption, corrupt_ok=False) are not answers and are left out."""
    return {q: r.get("parsed_answer") for (q, c, v), r in rows.items()
            if c == cond and v == var and r.get("corrupt_ok", True) is not False}


def golds(rows):
    return {q: r["gold"] for (q, c, v), r in rows.items() if r.get("gold") and c in ("direct", "cot")}


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(ans, ref, gold):
    qs = [q for q in ans if q in ref and ans[q] is not None and ref[q] is not None]
    flips = [q for q in qs if ans[q] != ref[q]]
    b = sum(ref[q] == gold.get(q) and ans[q] != gold.get(q) for q in qs)   # ref right, new wrong
    c = sum(ref[q] != gold.get(q) and ans[q] == gold.get(q) for q in qs)   # ref wrong, new right
    return dict(n=len(qs), flips=len(flips), rate=len(flips) / len(qs) if qs else float("nan"),
                broke=b, fixed=c, p=mcnemar_p(b, c), flip_qs=flips,
                unparsed=sum(a is None for a in ans.values()))


def summarize(path, out=print):
    rows = load(path)
    gold = golds(rows)
    out("=" * 80 + f"\nANALYSIS: {path}   ({len(gold)} questions)")

    # accuracy
    out("\nAccuracy (parsed answers only; unparsed counted as wrong in brackets):")
    keys = sorted({(c, v) for (_, c, v), r in rows.items() if "parsed_answer" in r})
    for c, v in keys:
        a = answers(rows, c, v)
        if not a:
            continue
        right = sum(a[q] == gold.get(q) for q in a)
        parsed = sum(x is not None for x in a.values())
        out(f"  {c:13s} {v:7s} {right / max(1, parsed):6.1%}  [{right / len(a):6.1%}]  n={len(a)}")

    direct, cot, replay = answers(rows, "direct", "orig"), answers(rows, "cot", "orig"), answers(rows, "replay", "orig")

    out("\nFlip rates vs reference (McNemar p on accuracy change):")
    tests = [("truncate", f, replay, "replay") for f in ("0.00", "0.25", "0.50", "0.75")]
    tests += [("corrupt", k, replay, "replay")
              for k in ("perceptual", "logical", "grammar", "irrelevant", "contradiction")]
    tests += [("paraphrase_r", "orig", replay, "replay"),
              ("paraphrase_q", "cot", cot, "cot"),
              ("paraphrase_q", "direct", direct, "direct")]
    results = {}
    for c, v, ref, refname in tests:
        a = answers(rows, c, v)
        if not a:
            continue
        r = compare(a, ref, gold)
        results[(c, v)] = r
        out(f"  {c:13s} {v:6s} vs {refname:6s}: {r['flips']:4d}/{r['n']:<4d} = {r['rate']:6.1%}   "
            f"fixed {r['fixed']:3d} broke {r['broke']:3d}  p={r['p']:.3f}  unparsed={r['unparsed']}")

    floor = [results[k]["rate"] for k in (("paraphrase_r", "orig"), ("paraphrase_q", "cot")) if k in results]
    if floor:
        out(f"\n  Noise floor (paraphrase): rationale {results.get(('paraphrase_r','orig'),{}).get('rate',float('nan')):.1%}, "
            f"question {results.get(('paraphrase_q','cot'),{}).get('rate',float('nan')):.1%}")

    # paraphrase filter pass rates
    for kind in ("paraphrase_q", "paraphrase_r"):
        t = [r for (_, c, v), r in rows.items() if c == kind and v == "text"]
        if t:
            reasons = defaultdict(int)
            for r in t:
                reasons[r["reason"]] += 1
            out(f"  {kind} filter: {sum(r['ok'] for r in t)}/{len(t)} passed  {dict(reasons)}")

    # corruption: coverage, labels, RQ2 on a fair subset
    corr = [r for (_, c, v), r in rows.items() if c == "corrupt"]
    if corr:
        out("\nCorruption coverage (how many questions got a valid corruption of each kind):")
        by = defaultdict(lambda: [0, 0, defaultdict(int)])
        for r in corr:
            b = by[r["variant"]]
            b[1] += 1
            if r.get("corrupt_ok"):
                b[0] += 1
            else:
                b[2][r.get("corrupt_reason")] += 1
        for k, (ok, tot, reasons) in by.items():
            out(f"  {k:13s} {ok}/{tot} ok   skipped: {dict(reasons)}")
        labs = [r for (_, c, v), r in rows.items() if c == "labels"]
        if labs:
            from corruption import mark_filler
            fixed = [mark_filler(r["steps"], r["labels"]) for r in labs]
            allp = sum(l == "P" for ls in fixed for l in ls)
            alll = sum(l == "L" for ls in fixed for l in ls)
            allf = sum(l == "F" for ls in fixed for l in ls)
            withp = sum("P" in ls for ls in fixed)
            fb = sum(r["method"] == "heuristic_fallback" for r in labs)
            out(f"  step labels: {allp} perceptual, {alll} logical, {allf} filler; "
                f"{withp}/{len(labs)} rationales contain at least one perceptual step; "
                f"heuristic fallback used for {fb}/{len(labs)} questions")
        # RQ2: compare perceptual vs logical on the SAME questions (both corrupted ok)
        pa, la = answers(rows, "corrupt", "perceptual"), answers(rows, "corrupt", "logical")
        both = [q for q in pa if q in la and pa[q] and la[q] and replay.get(q)]
        if both:
            pf = sum(pa[q] != replay[q] for q in both)
            lf = sum(la[q] != replay[q] for q in both)
            only_p = sum(pa[q] != replay[q] and la[q] == replay[q] for q in both)
            only_l = sum(la[q] != replay[q] and pa[q] == replay[q] for q in both)
            out(f"\nRQ2 (questions with both kinds, n={len(both)}): perceptual flips {pf} ({pf/len(both):.1%}), "
                f"logical flips {lf} ({lf/len(both):.1%}); only-perceptual {only_p}, only-logical {only_l}, "
                f"McNemar p={mcnemar_p(only_p, only_l):.3f}")

    # direct vs cot
    qs = [q for q in direct if q in cot and direct[q] and cot[q]]
    if qs:
        dis = [q for q in qs if direct[q] != cot[q]]
        t0 = answers(rows, "truncate", "0.00")
        back = sum(t0.get(q) == direct[q] for q in dis)
        out(f"\nDirect vs CoT disagree on {len(dis)}/{len(qs)}; truncation 0% returns to the direct answer in {back}/{len(dis)}.")
        out(f"CoT fixed {sum(direct[q] != gold.get(q) and cot[q] == gold.get(q) for q in qs)}, "
            f"broke {sum(direct[q] == gold.get(q) and cot[q] != gold.get(q) for q in qs)} direct answers.")

    # where do truncation flips go?
    out("\nWhere truncation flips land (position):")
    for f in ("0.00", "0.25", "0.50", "0.75"):
        r = results.get(("truncate", f))
        if not r or not r["flips"]:
            continue
        a = answers(rows, "truncate", f)
        first = sum(a[q] == "A" for q in r["flip_qs"])
        last = sum(a[q] == LETTERS[len(ch) - 1] for q in r["flip_qs"]
                   if (ch := choices_for(rows, q)))
        out(f"  {f}: {r['flips']} flips -> first option (A) {first}, last option {last}")

    # never flip
    trunc = [answers(rows, "truncate", f) for f in ("0.00", "0.25", "0.50", "0.75")]
    if trunc[0]:
        never = sum(all(t.get(q) == replay.get(q) for t in trunc) for q in replay)
        out(f"\nNever flip under any truncation: {never}/{len(replay)}")
    return results


def option_order(normal_path, reversed_path, out=print):
    """Do truncation flips follow option POSITION (same letter in both runs) or CONTENT
    (same option text in both runs)? Position-following flips = option-order bias."""
    N, R = load(normal_path), load(reversed_path)
    out("=" * 80 + f"\nOPTION ORDER: {normal_path} vs {reversed_path}")
    for f in ("0.00", "0.25", "0.50", "0.75"):
        n_ans, r_ans = answers(N, "truncate", f), answers(R, "truncate", f)
        n_rep, r_rep = answers(N, "replay", "orig"), answers(R, "replay", "orig")
        same_pos = same_content = both = 0
        first_n = first_r = fl_n = fl_r = 0
        for q in n_ans:
            if q not in r_ans or None in (n_ans[q], r_ans.get(q), n_rep.get(q), r_rep.get(q)):
                continue
            ch_n, ch_r = choices_for(N, q), choices_for(R, q)
            if not ch_n or not ch_r:
                continue
            fn, fr = n_ans[q] != n_rep[q], r_ans[q] != r_rep[q]
            fl_n += fn
            fl_r += fr
            first_n += fn and n_ans[q] == "A"
            first_r += fr and r_ans[q] == "A"
            if fn or fr:
                both += 1
                same_pos += n_ans[q] == r_ans[q]
                same_content += ch_n[LETTERS.index(n_ans[q])] == ch_r[LETTERS.index(r_ans[q])]
        out(f"  {f}: flips normal {fl_n} (to A: {first_n}), reversed {fl_r} (to A: {first_r}); "
            f"among {both} flipped questions: same letter {same_pos}, same option text {same_content}")
    out("  If flips go to 'A' in BOTH runs (same letter, different text), the model follows position, not content.")


def compare_runs(path_a, path_b, out=print):
    """E.g. default vs --drop_conclusion: flip rates side by side."""
    out("=" * 80 + f"\nCOMPARE: {path_a}  vs  {path_b}")
    for p in (path_a, path_b):
        rows, gold = load(p), None
        gold = golds(rows)
        rep = answers(rows, "replay", "orig")
        parts = []
        for f in ("0.00", "0.25", "0.50", "0.75"):
            r = compare(answers(rows, "truncate", f), rep, gold)
            parts.append(f"t{f}={r['rate']:.1%}")
        acc = sum(rep[q] == gold.get(q) for q in rep) / max(1, len(rep))
        out(f"  {p}: replay acc {acc:.1%}  " + "  ".join(parts))


def corruption_check_csvs(path, prefix, n=50):
    """Writes <prefix>_labels.csv and <prefix>_corruptions.csv for hand checking."""
    from corruption import export_label_check, export_corruption_check, is_filler as _is_filler
    rows = load(path)
    lab = [dict(question_id=q, question=r.get("question", ""), step_idx=i, step=s, label=l)
           for (q, c, v), r in rows.items() if c == "labels"
           for i, (s, l) in enumerate(zip(r["steps"], r["labels"])) if not _is_filler(s)]
    export_label_check(lab, prefix + "_labels.csv", n=n)
    cor = [dict(ok=True, question_id=q, question=r.get("question", ""), kind=v, step_idx=r.get("step_idx"),
                original_step=r.get("original_step"), corrupted_step=r.get("corrupted_step"))
           for (q, c, v), r in rows.items()
           if c == "corrupt" and v in ("perceptual", "logical") and r.get("corrupt_ok")]
    export_corruption_check(cor, prefix + "_corruptions.csv", n=n)
    print(f"Wrote {prefix}_labels.csv and {prefix}_corruptions.csv - fill in the Y/N columns.")


def spot_check_csv(path, csv_path, n=50):
    from paraphrase import export_spot_check
    rows = load(path)
    recs = [dict(question_id=q, original=r["original"], paraphrase=r["paraphrase"])
            for (q, c, v), r in rows.items() if c == "paraphrase_q" and v == "text" and r.get("ok")]
    export_spot_check(recs, csv_path, n=n)
    print(f"Wrote {min(n, len(recs))} paraphrases to {csv_path} - fill in meaning_preserved with Y/N.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--reversed")
    ap.add_argument("--compare")
    ap.add_argument("--spot_check")
    ap.add_argument("--corruption_check", help="prefix for the two hand-check CSVs")
    a = ap.parse_args()
    summarize(a.path)
    if a.reversed:
        option_order(a.path, a.reversed)
    if a.compare:
        compare_runs(a.path, a.compare)
    if a.spot_check:
        spot_check_csv(a.path, a.spot_check)
    if a.corruption_check:
        corruption_check_csvs(a.path, a.corruption_check)