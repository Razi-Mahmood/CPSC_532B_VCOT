# Run the complete evaluation

Place these four folders at the root of the sayeh-work checkout. Keep the existing
sayeh_files/results/scienceqa.jsonl and aokvqa.jsonl unchanged. The package contains
analysis outputs, not duplicate copies of those large source caches.

```bash
python -m pip install numpy matplotlib Pillow python-docx
python "Evaluation Scripts/evaluate_all.py"
python "Evaluation Scripts/build_report.py"
python "Evaluation Scripts/test_statistics.py"
```

If the frozen caches are in results, add `--results results` to evaluate_all.py.
Evaluation Results is flat: FINDINGS.md, CSV tables, analysis JSON and six plots.
The two supplied CSVs are preserved byte-for-byte in Blurred Image Results.

blur_row_index.csv maps question/options/scene signatures to question IDs, because
the uploaded blur tables omit IDs and some ScienceQA questions repeat. The evaluator
rejects any changed signature or mismatched option text. If a future run changes
row order/content, remap explicitly instead of silently matching by question text.
Its earlier-forecast column records the observed source overlap for audit purposes.

## Code included

- evaluate_all.py: complete CPU evaluation and blur-table integration.
- text_evaluation.py and core.py: explanation metrics, paired tests, RQ3 and plots.
- build_report.py: Word report with embedded plots.
- run_blur.py: original/light-blur validation experiment, using cached explanations.
- prepare_images.py: pilot image pairs and numbered index.
- run_experiments.py, cache.py, parsing.py, steps.py, paraphrase.py, corruption.py,
  team_summary.py: copied frozen inference implementation and dependencies. The only
  inference-runner edit redirects its final summary import to team_summary.
- test_statistics.py and test_delivery.py: statistical and data-integrity checks.

## Future GPU runs

Use the team's configured model environment with torch, transformers and datasets.
The CPU packages above are not a substitute for those model dependencies.

```bash
python "Evaluation Scripts/run_blur.py" --dataset scienceqa --n 500
python "Evaluation Scripts/run_blur.py" --dataset aokvqa --n 500
```

These commands write raw JSONL answers and manifests; they do not overwrite supplied
CSV tables. The imported CSV comparison uses historical direct-answer baselines.
A new run should instead compare with its freshly measured original-image controls.
Use summarize_raw_blur.py for that paired analysis. Keep raw logs and image hashes.

The 20 numbered original/blurred pairs are pilot illustrations, not the two validation
cohorts. Their blur radius is 1% of the shorter side. No radius can be verified for
the imported CSV responses from the supplied fields.
