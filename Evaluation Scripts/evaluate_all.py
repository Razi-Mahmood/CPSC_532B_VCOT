"""Evaluate frozen text caches and supplied blur response tables."""
import argparse,csv,json,re,hashlib,subprocess,sys
from pathlib import Path
from collections import Counter
import numpy as np
import core
from text_evaluation import table,NAMES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('sayeh_files/results'))
    p.add_argument('--blur',type=Path,default=Path('Blurred Image Results'))
    p.add_argument('--out',type=Path,default=Path('Evaluation Results'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('text_evaluation.py')),'--results',str(a.results),'--out',str(a.out)],check=True)
    index=list(csv.DictReader(Path(__file__).with_name('blur_row_index.csv').open()))
    summaries=[];pairs=[];audit=[];annotations=[]
    for name in NAMES:
        source=a.blur/(name+'_blur_results.csv')
        uploaded=list(csv.DictReader(source.open()))
        known=[r for r in index if r['dataset']==name]
        if len(uploaded)!=500 or len(known)!=500:raise ValueError('Expected 500 rows per dataset')
        rows,info,_=core.load_cache(a.results/(name+'.jsonl'));ds=core.Dataset(rows,name)
        correct=0;harm=0;helped=0;flips=0;pn=0;baseline_correct=0;overlap=0
        categories=Counter();outcomes=Counter()
        for number,(r,k) in enumerate(zip(uploaded,known),1):
            signature=hashlib.sha256('\x1f'.join([r['question'],r['options'],r['image_content']]).encode()).hexdigest()
            if signature!=k['input_signature']:raise ValueError(f'{name} row {number}: identity signature mismatch; remap rows before analysis')
            q=k['question_id'];gold=ds.gold[q]
            match=re.fullmatch(r'([A-Z]):\s*(.*)',r['qwen_answer'],re.S)
            if not match:raise ValueError('Invalid response format')
            answer,text=match.groups();choices=ds.choices[q]
            if ord(answer)-65 not in range(len(choices)) or text!=choices[ord(answer)-65]:raise ValueError('Letter/text mismatch')
            truth=answer==gold;correct+=truth
            category=r['correctness'];categories[category]+=1
            if category not in ['Correct','Close','Incorrect']:raise ValueError('Unexpected annotation')
            if (category=='Correct')!=truth:raise ValueError('Correctness annotation conflicts with answer key')
            original=ds.answer(ds.row(q,'direct'));base_ok=original==gold
            baseline_correct+=base_ok
            harm+=base_ok and not truth;helped+=not base_ok and truth
            outcomes[(int(base_ok),int(truth))]+=1
            if original is not None:pn+=1;flips+=original!=answer
            same=answer==k['earlier_forecast_answer'];overlap+=same
            pairs.append(dict(dataset=name,source_row=number,question_id=q,question=r['question'],original_direct=original,
                supplied_blur_answer=answer,gold=gold,strict_correct=truth,annotation=category,
                baseline_correct=base_ok,flip=(original!=answer) if original is not None else None,
                identical_to_earlier_forecast=same))
        lo,hi=core.wilson(correct,500);flo,fhi=core.wilson(flips,pn)
        # Paired percentile bootstrap of accuracy difference, including unparsed baseline as incorrect.
        cells=[(0,0),(0,1),(1,0),(1,1)]
        rng=np.random.default_rng(532)
        draws=rng.multinomial(500,[outcomes[c]/500 for c in cells],size=20000)
        changes=(draws[:,1]-draws[:,2])/500
        dlo,dhi=np.quantile(changes,[.025,.975])
        summary=dict(dataset=name,n=500,correct=correct,wrong=500-correct,close=categories['Close'],
            annotated_incorrect=categories['Incorrect'],accuracy=correct/500,ci_low=lo,ci_high=hi,
            baseline_correct=baseline_correct,baseline_accuracy=baseline_correct/500,
            delta_accuracy=(correct-baseline_correct)/500,delta_ci_low=float(dlo),delta_ci_high=float(dhi),
            paired_n=pn,flips=flips,flip_rate=flips/pn,flip_ci_low=flo,flip_ci_high=fhi,
            baseline_unparsed=500-pn,correct_to_wrong=harm,wrong_to_correct=helped,
            exact_mcnemar_p=core.mcnemar(harm,helped),identical_forecast_answers=overlap,
            evidence_status='User-reported GPU results; answers and scenes exactly match earlier text-only forecasts; raw logs unavailable')
        summaries.append(summary)
        audit.append(dict(dataset=name,rows=500,matched_unique_ids=len({r['question_id'] for r in known}),
            signature_checks_passed=500,answer_text_checks_passed=500,correct_label_checks_passed=500,
            forecast_answer_matches=overlap,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            matching_method='Ordered question/options/scene signatures mapped to earlier forecast question IDs; repeated questions not matched by question alone'))
        annotations.extend(dict(dataset=name,category=k,count=v) for k,v in categories.items())
    for s,pv in zip(summaries,core.holm([s['exact_mcnemar_p'] for s in summaries])):s['holm_p']=pv
    table(a.out/'blur_summary.csv',summaries);table(a.out/'blur_question_results.csv',pairs)
    table(a.out/'blur_annotations.csv',annotations);table(a.out/'input_audit.csv',audit)
    (a.out/'blur_analysis.json').write_text(json.dumps(dict(summary=summaries,audit=audit),indent=2))
    blur_plots(summaries,a.out)
    old=(a.out/'FINDINGS.md').read_text()
    start=old.index('## Image-blur experiment');end=old.index('## Evaluation details')
    section=['## Supplied image blur response tables','',
      '| Dataset | Correct | Close | Incorrect annotation | Strict accuracy |','|---|---:|---:|---:|---:|']
    for s in summaries:section.append(f"| {NAMES[s['dataset']]} | {s['correct']} | {s['close']} | {s['annotated_incorrect']} | {s['accuracy']:.1%} |")
    section+=['','Close is retained as a supplied annotation, but every Close answer is incorrect under exact multiple-choice scoring. Combined strict accuracy is 688/1,000 (68.8%); 312 answers are wrong, comprising 181 Close and 131 Incorrect annotations.',
      '', 'The tables were identified by the submitter as GPU-run results. All 1,000 answer choices and scene descriptions match the earlier text-only forecasts. The files contain no generation logs, image identifiers, blur settings or decoding settings. Their descriptive statistics are reproducible, but they are not independently verified blur measurements. Paired tests are conditional comparisons of these supplied responses with historical direct-answer caches; they do not validate the origin of the responses or isolate blur as the cause.',
      '', 'The 20 paired images are pilot illustrations. Their Gaussian radius is 1% of the shorter image side; that setting cannot be inferred for the supplied 500-question CSVs. Fine-detail vulnerability and semantic resilience remain hypotheses rather than demonstrated mechanisms without validated categories and genuine image/run evidence.','']
    (a.out/'FINDINGS.md').write_text(old[:start]+'\n'.join(section)+old[end:])
    print(json.dumps(summaries,indent=2))

def blur_plots(summaries,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,s in zip(axes,summaries):
        vals=[s['baseline_accuracy'],s['accuracy']]
        intervals=[core.wilson(s['baseline_correct'],500),(s['ci_low'],s['ci_high'])]
        ax.bar([0,1],vals,color=['#245d91','#be6528'],width=.6)
        ax.errorbar([0,1],vals,yerr=[[v-ci[0] for v,ci in zip(vals,intervals)],[ci[1]-v for v,ci in zip(vals,intervals)]],fmt='none',capsize=4,color='#333333')
        for i,v in enumerate(vals):ax.text(i,v+.06,f'{v:.1%}',ha='center')
        ax.set_xticks([0,1],['Original direct','Supplied blur']);ax.set_ylim(0,1.05)
        ax.set_title(NAMES[s['dataset']]);ax.set_ylabel('Strict answer accuracy');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'blur_accuracy.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    left=[0,0]
    for key,label,color in [('correct','Correct','#245d91'),('close','Close','#d8a347'),('annotated_incorrect','Incorrect','#9a5555')]:
        vals=[s[key] for s in summaries]
        ax.barh([0,1],vals,left=left,label=label,color=color,height=.5)
        for i,v in enumerate(vals):ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if key!='close' else 'black')
        left=[l+v for l,v in zip(left,vals)]
    ax.set_yticks([0,1],[NAMES[s['dataset']] for s in summaries]);ax.set_xlabel('Questions');ax.set_xlim(0,500)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'blur_categories.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
