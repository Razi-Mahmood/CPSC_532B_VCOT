# Sarthak's contribution: evaluation and review

This adds two folders to the existing `sayeh-work` branch. It uses the team's
saved 100-question ScienceQA run. No new model inference was run.

## The project in plain language

The model looks at a picture and answers a multiple-choice question. Sometimes
it also writes an explanation. We then shorten or change that explanation and
ask it to answer again. If its answer changes, the written explanation affected
that answer under this test. If it stays the same, we have not shown that the
model used that explanation. An unchanged answer alone does not prove that
the model ignored all reasoning.

The three research questions are: does changing the explanation affect the
answer; do mistakes about the picture and mistakes in reasoning have different
effects; and are answers that change more or less likely to have been correct?

## What the team had already built

- Loading ScienceQA and A-OKVQA and running Qwen2.5-VL-7B-Instruct.
- Direct answers, explanations followed by answers, and replay of a saved explanation.
- Truncation, paraphrase controls, step labels and error-insertion conditions.
- Saved model responses, answer parsing, initial metrics and sample review CSVs.

The supplied ScienceQA cache contains the 100-question results used here.
The runner uses float32 and eager attention; the teammate reported garbled
bf16 output on the B200 setup. This contribution does not independently test
that hardware issue.

## What this contribution adds

| Folder / file | Purpose |
|---|---|
| `evaluation/analyze.py` | Recompute paired results, uncertainty, tests and plots from saved outputs, using CPU only. |
| `evaluation/test_analysis.py` | Checks the denominators, pairing, missing answers, duplicate handling and review safeguards. |
| `evaluation/requirements.txt` | The single plotting dependency. Numerical analysis otherwise uses Python's standard library. |
| `evaluation/truncation_flip_rate.png` | Answer changes at each retained explanation length, with uncertainty and a paraphrase reference. |
| `evaluation/error_type_flip_rate.png` | Answer changes by error type, with counts and uncertainty. |
| `evaluation/README.md` | This explanation, results, methods, limitations and instructions. |
| `review/check_labels.csv` | The original 50 sampled steps, with the existing review fields filled and a short notes column added. |
| `review/check_corruptions.csv` | The original 50 sampled edits, with the existing review fields filled and an overall validity decision added. |

Review method: automated review of the step text, question context and images.
Independent human validation remains pending. The annotations are provisional
judgments; they are not presented as completed human checks.

## Results from the existing run

Direct answering was correct on **86/100** questions; answering with an
explanation was correct on **90/100**. Replaying the full explanation matched
the original explanation-based answer on **100/100** questions. The 4-point
accuracy difference is not statistically conclusive (paired exact p = 0.549,
99 parsed pairs).

| Explanation retained | Changed answers / parsed pairs | Flip rate | Accuracy / attempted questions |
|---|---:|---:|---:|
| 0% | 9/99 | 9.1% | 89/100 |
| 25% | 10/99 | 10.1% | 86/100 |
| 50% | 17/97 | 17.5% | 77/100 |
| 75% | 10/99 | 10.1% | 86/100 |
| Paraphrased explanation | 0/90 | 0.0% | 81/90 |

These are requested percentages of segmented steps, not exact percentages of
words. Shorter explanations did not produce a smooth increase or decrease in
answer changes. On matched questions, all four truncation conditions had more
flips than the rationale-paraphrase control after Holm adjustment (adjusted
p = 0.00014–0.0273). The control's zero observed flips still has a 95% interval
of **0–4.1%**; zero observations do not establish a zero true rate.

The question-paraphrase controls are separate: 3/94 flips for explanation-based
answering and 3/92 for direct answering. They do not measure the same change as
paraphrasing the explanation.

| Error type | Changed answers / parsed eligible pairs | Flip rate |
|---|---:|---:|
| Perceptual: a claim about the picture | 0/43 | 0.0% |
| Logical: an inference or reasoning claim | 4/96 | 4.2% |
| Grammar | 0/97 | 0.0% |
| Unrelated content | 0/100 | 0.0% |
| Contradiction | 2/92 | 2.2% |

These raw counts use the runner's validity flags. The eligible question sets
differ, so comparing 0/43 with 4/96 directly is not the paired RQ2 test.
Grammar, unrelated-content and contradiction outputs were already in the
cache; this contribution analyzes them rather than claiming new experiments.

**RQ2, comparing the same questions:** on 41 shared eligible questions,
perceptual errors changed **0/41** answers and logical errors changed **3/41**.
Exact paired McNemar p = **0.250**. The current sample is inconclusive about a
reliable difference; it also does not show that the two error types are equivalent.

**Review sample:** under the broader definitions below, 44/50 original step
labels are defensible and 6 need correction. Of the sampled corruptions,
34/50 are accepted and 16 are rejected for clear problems. This is a sample
audit, not a validity estimate for every generated edit. Excluding the matching
rejected sample interventions leaves 31 paired questions: 0/31 perceptual
flips versus 3/31 logical flips, exact paired p = 0.250. This sensitivity check
is still not a fully validated comparison and does not establish a reliable
difference between the error types.

The review criteria were broadened on 14 November 2026 after inspecting the
examples. The earlier stricter interpretation accepted 36/50 labels and
20/50 corruptions. The revised counts reflect changed category boundaries and
the definition of one conceptual edit, not new model runs or improved model
performance. Treat the review-filtered comparison as exploratory.

**RQ3, accuracy and sensitivity:** 77 questions kept the same answer across
all four truncations; 20 changed at least once; 3 had an unparsed truncation
answer and cannot be assigned to either group. Original explanation-based
accuracy was **73/77 (94.8%)** in the stable group and **14/20 (70.0%)** in
the changing group. This is an association; it does not show that making
an explanation more influential would cause accuracy to improve.

## How to use the two CSVs

Both files already existed in `sayeh_files/` with 50 rows each and blank review
fields. The copies in `review/` retain the original row order, identifiers,
questions and model text. They fill the existing review fields and add just
one column to each file. The original files in `sayeh_files/` are unchanged.
These are plain text tables: open them in Excel or any CSV editor. Notes are filled only for labels marked incorrect (`correct=N`) and corruptions marked invalid (`valid_corruption=N`); other notes are blank. The previous
workbook and HTML page are omitted from this simplified package.

In `check_labels.csv`, `label` is the original classification. `correct` says
whether that classification agrees with the proposed decision; `true_label`
contains the proposed classification. These two columns already existed.
The only added column is `notes`, explaining a clear label problem. P covers
direct image descriptions, map entities and spatial relations, and apparent
properties of pictured objects. L covers reasoning, comparisons, background
knowledge, nonvisual premises and planning. For a step that substantially
combines observation and reasoning, either P or L can be defensible. Planning
headers, transitions and option evaluation can remain L; they are not rejected
just because a separate filler category could have been used. The revised
decisions use P and L only.

A pure image readout is still perceptual, and a fact supplied only in the
written passage is not a visual observation. A factually wrong statement can
still have a defensible type. `correct=Y` means the original category is
acceptable under these definitions, not that the sentence is factually true.
Check the underlying question and image when confirming a decision. Step
indices start at zero, matching the source files.

In `check_corruptions.csv`, `is_wrong` asks whether the edit introduces a
substantive factual, relational or task-reference error. `right_type` asks
whether the stated type is defensible under the broader definitions above.
`single_change` asks whether the edit changes one concept or relation, not
whether it changes exactly one word. A coordinated reversal of one causal
relation, or replacement of one paired location description, can count as one
conceptual change. These columns and `notes` already existed. The only added
column is `valid_corruption`: the overall Y/N decision.

Accept mixed visual/logical interpretations when the stated category is
reasonable. An already-wrong original does not automatically invalidate an
edit if the edit adds a distinct false claim or internal contradiction. An
inference-bearing conclusion can count as a whole-step logical error; for
example, Q69 changes a premise-and-conclusion step into a contradiction.
These broader interventions do not isolate the effect of an intermediate
premise, so do not interpret them as doing so.

Still reject edits that make no meaningful error, merely repair the original,
have a clearly wrong category, rewrite the answer options or their letters,
or substitute a bare final answer without changing supporting reasoning.
Notes appear only for these rejected cases. No rejection is based solely on
a debatable P/L boundary or the number of changed words.

Decisions are proposed entries to confirm or revise. CSVs contain values,
not formulas: if you change a label, update `correct` and `true_label` together;
if you change a corruption judgment, update `valid_corruption` and its notes
as well. Leave `valid_corruption` blank for an undecided row. The analysis
uses that explicit overall decision and never treats blanks as rejections.

Several concrete issues explain the rejected edits:

- Q7, Q58 and Q88 only remove numbering: the claim is unchanged.
- Q74 changes Georgia from west of Mississippi to east, repairing the original
  mistake. It does not introduce a new error.
- Q28 changes a plotted speed value: that is a perceptual change even if the
  original label called it logical.
- Q5 changes answer letter D to nonexistent option E; Q78's logical edit
  replaces the bare final answer. These do not test a supporting reasoning step.

Examples accepted under the broader definitions include Q71's towel-opacity
reversal as a perceptual error, Q24's reversed melting direction within a
physical-change explanation as logical, and Q27's mistaken state selection
within a step combining a map observation and option evaluation. Q8's
perceptual edit substitutes North Carolina for the labelled option South
Carolina; the task-reference mismatch counts as an error even though both
states are southeastern. These examples explain the revised boundaries.

Q31 and Q74 are useful qualitative examples of repairing a mistaken step.
Neither repaired replay fixes the final answer. Some shorter explanations do
correct previously wrong answers: in the 0% condition, four answers improve,
four worsen, and one changes from one wrong choice to another. These are
examples of improvement without inserting an answer, not a general accuracy gain.

Rerun the script after review edits to refresh the sample-exclusion sensitivity
result. It does not overwrite either CSV. The experimental results reported
above remain the dated snapshot; no new inference is part of this revision.

## Add these folders to the existing branch

Extract `evaluation/` and `review/` into the repository root alongside
`sayeh_files/`. Use these instead of the earlier `sarthak_work/` package.
No new branch is needed. On the teammate's branch, run:

```bash
python -m pip install -r evaluation/requirements.txt
python -m unittest discover -s evaluation -p 'test_*.py'
python evaluation/analyze.py --out /tmp/vcot-analysis
git add evaluation review
git commit -m "Add cached evaluation, review CSVs and result plots"
```

The analysis writes one `analysis.json` and the two PNG plots to the specified
output folder. The JSON contains per-condition results, paired records, RQ2,
RQ3, control tests, review matches and source provenance. It is generated on
demand rather than duplicated in this download. Omit `--out` to regenerate
inside `evaluation/`; use `--no-plots` to run without matplotlib. Use `--input`
to analyze a later compatible cache and `--reviews` to choose a corruption
review CSV. The default is `review/check_corruptions.csv`.

Existing source scripts and caches are unchanged. The download does not contain
the original model, datasets, caches, duplicate CSV/SVG exports or temporary files.
It is an addition to the repository, not a standalone copy of the entire project.

## Method and source details

Source branch: `sayeh-work`, commit
`d2076b5244b3015f6471d8dfe00ae27ccd256ad8`. Source cache:
`sayeh_files/results/scienceqa.jsonl`, SHA-256
`afb584546b2c601761f1ebb9f1d23bc32d8b1bd0e514ce4648e32ff5c9b90788`.
The two review samples come from `sayeh_files/check_labels.csv` and
`sayeh_files/check_corruptions.csv`. Snapshot date: 12 November 2026.

The cache has 1,960 records but only 1,782 distinct experiment keys. The last
record per dataset/question/condition/variant wins, matching the runner.
Duplicate retries are not extra observations. The 100 questions are the first
image-bearing validation examples, not a random sample. The cache mixes 70
`pilot2` records with 1,712 `v1` records, so it is not a single clean rerun.

Flip rates use pairs where both answers parse to valid choices. Accuracy counts
unparsed eligible responses as wrong. Failed/skipped interventions are excluded
and reported separately. All pairing is by question ID. Confidence intervals
are 95% Wilson intervals. McNemar tests use the exact two-sided binomial test.
Holm correction is applied separately to 12 perturbation/control accuracy
comparisons and nine flip-rate comparisons against rationale paraphrasing.
The 50%-retained accuracy comparison has unadjusted p = 0.0213 but adjusted
p = 0.255; it is not significant after that family correction.

The review sensitivity check applies only to ScienceQA rows whose question ID,
error type, step index and both step texts exactly match the reviewed example.
Unmatched, unfinished and unreviewed decisions do not remove rows.

Evidence comes from `derek-thomas/ScienceQA`, validation split. Question text,
choice order and gold answer matched all 100 cached IDs. The original run did
not preserve image bytes, so current image evidence cannot prove byte-for-byte
identity with the historical inputs.

## What is still pending

1. A person should confirm or revise the 50 label decisions and 50 corruption
   decisions. Resolve ambiguous steps before describing the data as validated.
2. Repair invalid edits and run a consistently configured larger evaluation,
   approximately 500 questions each on ScienceQA and A-OKVQA. Then refresh RQ2,
   the control comparisons and plots. No such larger results are included here.
3. Finish the joint report using the larger results, with each member's methods
   and these limitations clearly stated.
4. Optional: image blurring/masking and other model sizes. Preparing modified
   images can run on CPU, but measuring their effects requires new model
   inference. This contribution leaves that experiment out.
