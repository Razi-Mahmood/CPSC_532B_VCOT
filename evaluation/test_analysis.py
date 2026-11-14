"""Small synthetic counterexamples for denominator and pairing mistakes."""
import importlib.util
import json
import tempfile
import unittest
import csv
from unittest.mock import patch
from pathlib import Path

FILE = Path(__file__).resolve().parent / "analyze.py"
spec = importlib.util.spec_from_file_location("analysis", FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(q, c, a="A", v="orig", **extra):
    return dict(dataset="scienceqa", question_id=str(q), condition=c, variant=v,
                gold="A", choices=["first", "second", "third"], parsed_answer=a, **extra)


def ds(records):
    return m.Dataset({tuple(r[k] for k in ("dataset", "question_id", "condition", "variant")):r
                      for r in records}, "scienceqa")


class AnalysisTests(unittest.TestCase):
    def test_zero_events_still_have_uncertainty(self):
        lo, hi = m.wilson(0, 43)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, .08)
        self.assertLess(hi, .09)
        self.assertEqual(m.wilson(0, 0), (None, None))

    def test_exact_discordant_pairs(self):
        self.assertEqual(m.mcnemar(0, 3), .25)
        self.assertEqual(m.mcnemar(0, 0), 1)
        self.assertEqual(m.mcnemar(4, 4), 1)
        self.assertAlmostEqual(m.mcnemar(0, 10), 2/1024)

    def test_holm_preserves_order(self):
        self.assertEqual(m.holm([.04, .01, .03]), [.06, .03, .06])

    def test_parse_failures_are_not_flips(self):
        d = ds([row(1,"cot"),row(2,"cot"),row(3,"cot"),
                row(1,"replay"),row(2,"replay"),row(3,"replay"),
                row(1,"truncate","B","0.00"),row(2,"truncate",None,"0.00"),
                row(3,"truncate","Z","0.00")])
        result, _ = d.comparison("truncate","0.00","replay")
        self.assertEqual((result["paired_n"],result["flips"],result["unparsed_n"]),(1,1,2))
        self.assertEqual(result["accuracy"],0)

    def test_skipped_corruptions_are_not_answers(self):
        d = ds([row(1,"cot"),row(1,"replay"),row(1,"corrupt","B","logical",corrupt_ok=False)])
        r,_=d.comparison("corrupt","logical","replay")
        self.assertEqual((r["skipped_n"],r["paired_n"]),(1,0))
        self.assertIsNone(r["flip_rate"])

    def test_rq2_uses_same_questions(self):
        records=[row(q,"cot") for q in [1,2,3]]+[row(q,"replay") for q in [1,2,3]]
        records += [row(1,"corrupt","A","perceptual",corrupt_ok=True),
                    row(1,"corrupt","B","logical",corrupt_ok=True),
                    row(2,"corrupt","B","logical",corrupt_ok=True),
                    row(3,"corrupt","B","perceptual",corrupt_ok=True)]
        result, ids = m.paired_flip_test(ds(records),("corrupt","perceptual"),("corrupt","logical"))
        self.assertEqual(ids,["1"])
        self.assertEqual((result["n"],result["first_flips"],result["second_flips"]),(1,0,1))

    def test_wrong_to_wrong_is_not_accuracy_change(self):
        d=ds([row(1,"cot","B"),row(1,"replay","B"),row(1,"truncate","C","0.25")])
        r,_=d.comparison("truncate","0.25","replay")
        self.assertEqual((r["flips"],r["wrong_to_wrong"],r["helped"],r["harmed"]),(1,1,0,0))
        self.assertEqual(r["accuracy_mcnemar_p"],1)

    def test_missing_truncation_is_not_stable(self):
        records=[row(1,"cot"),row(1,"replay"),row(1,"direct")]
        records += [row(1,"truncate","A",f) for f in m.FRACTIONS[:-1]]
        _, groups=m.rq3(ds(records))
        self.assertEqual(groups[0]["group"],"incomplete")

    def test_last_line_wins_and_datasets_do_not_collide(self):
        a=row(1,"cot","B"); b=row(1,"cot","A"); c={**row(1,"cot","C"),"dataset":"aokvqa"}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"cache.jsonl";p.write_text("\n".join(map(json.dumps,[a,b,c])))
            rows,info,dups=m.load_cache(p)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[("scienceqa","1","cot","orig")]["parsed_answer"],"A")
        self.assertEqual(info["superseded_rows"],1)

    def test_malformed_input_is_visible(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"bad.jsonl";p.write_text('{"broken":')
            with self.assertRaisesRegex(ValueError,"invalid cache row"):
                m.load_cache(p)

    def test_conflicting_gold_fails(self):
        with self.assertRaisesRegex(ValueError,"Conflicting gold"):
            ds([row(1,"cot"),{**row(1,"replay"),"gold":"B"}])

    def test_stale_review_cannot_remove_current_intervention(self):
        d=ds([row(1,"cot"),row(1,"replay"),row(1,"corrupt","B","logical",corrupt_ok=True,
              step_idx=1,original_step="old",corrupted_step="new")])
        records=[dict(question_id="1",kind="logical",step_idx="1",original_step="old",
                      corrupted_step="different",valid_corruption="N")]
        with patch.object(m,"read_review_csv",return_value=records):
            counts, exclusions, details=m.review_summary(d,Path(__file__))
        self.assertFalse(exclusions)
        self.assertFalse(details[0]["matches_current_cache"])

    def test_review_csv_snapshot(self):
        path=Path(__file__).resolve().parents[1]/"review/check_corruptions.csv"
        records=m.read_review_csv(path)
        self.assertEqual(len(records),50)
        self.assertTrue(all(r["valid_corruption"] in ("Y","N","") for r in records))

    def test_review_csv_preserves_pending_and_rejects_invalid_decisions(self):
        fields=["question_id","kind","step_idx","original_step","corrupted_step","valid_corruption"]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"reviews.csv"
            with path.open("w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f);w.writerow(fields)
                w.writerow(["1","logical","0","old, with comma","new"," y "])
                w.writerow(["2","logical","0","old","new",""])
            records=m.read_review_csv(path)
            self.assertEqual([r["valid_corruption"] for r in records],["Y",""])
            self.assertEqual(records[0]["original_step"],"old, with comma")
            with path.open("a",newline="") as f:
                csv.writer(f).writerow(["3","logical","0","old","new","maybe"])
            with self.assertRaisesRegex(ValueError,"must be Y, N, or blank"):
                m.read_review_csv(path)

    def test_reviews_do_not_cross_datasets(self):
        d=ds([row(1,"cot")]);d.name="aokvqa"
        counts,exclusions,details=m.review_summary(d,Path(__file__))
        self.assertEqual((counts,exclusions,details),({},set(),[]))


if __name__=="__main__":
    unittest.main()
