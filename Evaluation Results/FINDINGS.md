# What happens when we change a model’s explanation?

**Main finding:** the way we change the explanation matters. A single inserted error rarely changes the answer. Keeping that error and asking the model to rewrite the following steps produces substantially more answer changes.

## The experiment in plain words

We give Qwen2.5-VL-7B a picture and a multiple-choice question. It writes an explanation and an answer. We then feed an edited explanation back with the same picture and question, and check whether the answer changes. We evaluate 500 questions from ScienceQA and 500 from A-OKVQA.

“Visual” means a statement about what is in the picture. “Logical” means a reasoning step based on those observations. An answer change is measured against replaying the original explanation. It can be a correction, a new mistake, or a different wrong answer.

## 1. Rewording is mostly harmless; removing steps has a larger effect

| Dataset | Reworded explanation | Shortened explanation, across four lengths |
|---|---:|---:|
| ScienceQA | 3/476 (0.6%) | 9.8%–15.4% |
| A-OKVQA | 1/493 (0.2%) | 5.5%–10.3% |

The answer is not equally sensitive to every edit. Removing explanation steps changes answers far more often than rewording the explanation. The effect is not a simple “shorter is always worse” pattern: ScienceQA has its largest change rate at 50% retained. See truncation.png.

## 2. Rewriting after an error is the strongest result

| Dataset | Error type | Single error | Error + rewritten following steps | Holm-adjusted p |
|---|---|---:|---:|---:|
| ScienceQA | perceptual | 0/208 (0.0%) | 12/208 (5.8%) | 0.000488 |
| ScienceQA | logical | 2/453 (0.4%) | 38/453 (8.4%) | 8.51e-10 |
| A-OKVQA | perceptual | 2/187 (1.1%) | 15/187 (8.0%) | 0.000488 |
| A-OKVQA | logical | 8/419 (1.9%) | 54/419 (12.9%) | 9.07e-12 |

Each row compares exactly the same questions. All four increases remain significant after Holm correction. The error-type chart uses each condition’s full eligible subset; the matched table above and regeneration.png use the common subset required for a fair comparison.

**Concrete observation:** evaluating only an isolated edited step gives a different picture from allowing the rest of the explanation to change. The first experiment measures resistance to a local edit; the second measures sensitivity when that edit is followed by a newly generated continuation.

## 3. Visual errors are not clearly more influential than logical errors

The matched visual-versus-logical comparisons do not reach significance after Holm correction, for either dataset or either corruption method. The stronger supported distinction is single-error versus error-plus-rewrite. The results do not establish that visual and logical errors have identical effects.

## 4. Stable answers are much more often correct

| Dataset | Answer changed under shortening | Answer stayed stable |
|---|---:|---:|
| ScienceQA | 86/118 (72.9%) | 347/361 (96.1%) |
| A-OKVQA | 29/73 (39.7%) | 397/420 (94.5%) |

These groups use questions with valid answers in the original explanation run, replay, and all four shortened conditions. Questions with incomplete outputs remain separately reported in rq3.csv. Stability is a useful descriptive signal in these runs; it is not a guarantee that an individual answer is correct.

## 5. Writing an explanation does not consistently improve accuracy

| Dataset | Direct answer | Explanation then answer |
|---|---:|---:|
| ScienceQA | 433/500 (86.6%) | 446/500 (89.2%) |
| A-OKVQA | 443/500 (88.6%) | 427/500 (85.4%) |

ScienceQA improves numerically with explanations; A-OKVQA declines. The paired baseline accuracy tests do not establish a significant gain on either dataset. Therefore, a written explanation should not automatically be taken as evidence of a better answer.

## Conclusion for the class

> Our model is robust to small edits in its written explanation, but its answers become more sensitive when we remove steps or let it rewrite what follows an error. This pattern appears in both datasets. The clearest finding is that the editing method matters more than the visual-versus-logical distinction established by these experiments. Stable answers also tend to be more accurate. Together, these results show why testing explanation reliability requires more than inserting one mistake and checking whether the final answer changes.

## Image blur response

| Dataset | Correct | Close | Incorrect annotation | Strict accuracy |
|---|---:|---:|---:|---:|
| ScienceQA | 348 | 122 | 30 | 69.6% |
| A-OKVQA | 340 | 59 | 101 | 68.0% |

Close is retained as a supplied annotation, but every Close answer is incorrect under exact multiple-choice scoring. Combined strict accuracy is 688/1,000 (68.8%); 312 answers are wrong, comprising 181 Close and 131 Incorrect annotations.

## Evaluation details

- Figures use 95% Wilson intervals. Exact counts, eligibility, parse failures and intervals are in conditions.csv. Error-type bars may have different denominators; statistical comparisons use matched questions.
- Exact two-sided McNemar tests compare paired correctness or paired flip indicators. Holm families are explicit in comparisons.csv: four single-versus-rewritten comparisons across datasets; four visual-versus-logical comparisons; four prompt-version sensitivity comparisons; and a separate paraphrase-control family per dataset. Perturbation accuracy tests form a separate family per dataset in conditions.csv. Baseline direct-versus-explanation tests are exploratory and unadjusted.
- Accuracy counts an eligible unparsed answer as incorrect. Flip comparisons require both answers to parse. The current corruption validity flags are used; historical review decisions are not applied to changed edits.
- Duplicate cache keys use the last record, matching the inference cache. Regenerated edits must match the current single-error edit. Source hashes, per-question results and prompt metadata are retained in analysis.json.
- ScienceQA retains older prompt-version records on ten questions. Removing those questions preserves the central findings at the 0.05 level; see the prompt-version family in comparisons.csv.
- Rewriting changes the continuation as well as retaining the inserted error. A clean-prefix rewrite control would separate those effects. RQ3 groups can also differ in question difficulty. The observations concern responses to edited explanations, not direct measurement of internal reasoning.

## Completed contribution

Reproducible CPU evaluation of the frozen caches; confidence intervals; matched tests and Holm correction; RQ3; four clear figures; prompt-version sensitivity checks; a unified findings report; light-blur preparation and a runner for the next experiment.
