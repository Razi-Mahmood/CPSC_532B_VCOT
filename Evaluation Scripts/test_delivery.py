"""Check packaged results and imported row consistency after evaluate_all.py."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'Evaluation Results'
    rows=list(csv.DictReader((out/'blur_question_results.csv').open()))
    assert len(rows)==1000
    assert len({(r['dataset'],r['question_id']) for r in rows})==1000
    summary=list(csv.DictReader((out/'blur_summary.csv').open()))
    for s in summary:
        rs=[r for r in rows if r['dataset']==s['dataset']]
        correct=sum(r['supplied_blur_answer']==r['gold'] for r in rs)
        assert correct==int(s['correct'])
        assert int(s['correct'])+int(s['close'])+int(s['annotated_incorrect'])==500
        assert int(s['baseline_correct'])-int(s['correct'])==int(s['correct_to_wrong'])-int(s['wrong_to_correct'])
        assert int(s['paired_n'])+int(s['baseline_unparsed'])==500
    data=json.loads((out/'analysis.json').read_text())
    assert all(r['metadata']['n_questions']==500 for r in data['datasets'].values())
    images=list(csv.DictReader((ROOT/'Blurred Image Examples'/'image_index.csv').open()))
    assert len(images)==20
    assert all((ROOT/'Blurred Image Examples'/r[k]).is_file() for r in images for k in ['original_file','blurred_file'])
    print('1000 mapped blur rows, accuracy identities, cohort sizes and 20 image pairs passed.')
if __name__=='__main__':main()
