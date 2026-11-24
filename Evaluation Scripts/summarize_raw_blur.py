"""Evaluate measured responses from run.py; never generates model responses."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import wilson, mcnemar, holm

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('responses',type=Path,nargs='+')
    p.add_argument('--out',type=Path,default=Path('Evaluation Results/raw_blur_summary.csv'))
    a=p.parse_args()
    rows={}
    for line in '\n'.join(p.read_text() for p in a.responses).splitlines():
        r=json.loads(line)
        key=(r['dataset'],r['question_id'],r['mode'],r['sigma_fraction'])
        if key in rows:raise ValueError('Duplicate response '+str(key))
        rows[key]=r
    def answer(r):
        v=r.get('parsed_answer')
        return v if v in [chr(65+i) for i in range(len(r['choices']))] else None
    results=[]
    for dataset in sorted({k[0] for k in rows}):
        family=[]
        for mode in ['direct','fixed_rationale']:
            for severity in [0,.01]:
                current=[r for k,r in rows.items() if k[0]==dataset and k[2:]==(mode,severity)]
                if not current:raise ValueError(f'Missing condition: {dataset}/{mode}/{severity}')
                pairs=[]
                for r in current:
                    base=rows.get((dataset,r['question_id'],mode,0))
                    if not base:raise ValueError('Missing radius-zero control')
                    if (r['gold'],r['choices'])!=(base['gold'],base['choices']):raise ValueError('Mismatched question metadata')
                    if answer(r) is not None and answer(base) is not None:pairs.append((r,base))
                n=len(current);pn=len(pairs)
                correct=sum(answer(r)==r['gold'] for r in current)
                flips=sum(answer(r)!=answer(b) for r,b in pairs)
                harmed=sum(answer(b)==b['gold'] and answer(r)!=r['gold'] for r,b in pairs)
                helped=sum(answer(b)!=b['gold'] and answer(r)==r['gold'] for r,b in pairs)
                lo,hi=wilson(correct,n);flo,fhi=wilson(flips,pn)
                result=dict(dataset=dataset,mode=mode,sigma_fraction=severity,n=n,
                    unparsed=sum(answer(r) is None for r in current),correct=correct,accuracy=correct/n,
                    accuracy_ci_low=lo,accuracy_ci_high=hi,paired_n=pn,flips=flips,
                    flip_rate=flips/pn if pn else None,flip_ci_low=flo,flip_ci_high=fhi,
                    harmed=harmed,helped=helped,accuracy_mcnemar_p=mcnemar(harmed,helped) if pn and severity else None,
                    accuracy_holm_p=None)
                results.append(result)
                if result['accuracy_mcnemar_p'] is not None:family.append(result)
        for r,pv in zip(family,holm([r['accuracy_mcnemar_p'] for r in family])):r['accuracy_holm_p']=pv
    output=a.out
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(results[0]));writer.writeheader();writer.writerows(results)
    print(output)

if __name__=='__main__':main()
