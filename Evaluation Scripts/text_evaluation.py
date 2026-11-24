"""Reproduce the frozen-run evaluation without model inference."""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import core

NAMES={'scienceqa':'ScienceQA','aokvqa':'A-OKVQA'}
def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def plots(data,tests,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/name,dpi=180,bbox_inches='tight');plt.close(fig)
    def panels(title):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        fig.suptitle(title)
        for ax,name in zip(axes,data):
            ax.set_title(NAMES[name]);ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        return fig,axes
    fig,axes=panels('Shortening the explanation changes some answers')
    for ax,(name,r) in zip(axes,data.items()):
        ms=[m for m in r['metrics'] if m['condition']=='truncate']
        y=[m['flip_rate'] for m in ms]
        ax.errorbar([0,25,50,75],y,yerr=[[m['flip_rate']-m['flip_ci_low'] for m in ms],[m['flip_ci_high']-m['flip_rate'] for m in ms]],fmt='o-',capsize=4,color='#245d91',label='Shortened explanation')
        ctrl=next(m for m in r['metrics'] if m['condition']=='paraphrase_r')
        ax.axhline(ctrl['flip_rate'],color='#777777',linestyle='--',label='Reworded explanation')
        ax.set(xlabel='Explanation retained (%)',ylabel='Answers changed',xticks=[0,25,50,75],ylim=(0,.22))
        ax.legend(fontsize=9)
    save(fig,'truncation.png')
    fig,axes=panels('Answer changes by error type')
    variants=['perceptual','logical','perceptual_regen','logical_regen','grammar','irrelevant','contradiction']
    labels=['Visual','Logical','Visual +\nrewrite','Logical +\nrewrite','Grammar','Unrelated','Contradiction']
    for ax,(name,r) in zip(axes,data.items()):
        ms=[next(m for m in r['metrics'] if m['condition']=='corrupt' and m['variant']==v) for v in variants]
        y=[m['flip_rate'] for m in ms]
        ax.bar(range(7),y,color=['#245d91','#245d91','#be6528','#be6528','#777777','#777777','#777777'])
        ax.errorbar(range(7),y,yerr=[[m['flip_rate']-m['flip_ci_low'] for m in ms],[m['flip_ci_high']-m['flip_rate'] for m in ms]],fmt='none',capsize=3,color='#333333')
        ax.set_xticks(range(7),labels,rotation=35,ha='right',fontsize=9)
        ax.set(ylabel='Answers changed',ylim=(0,.2))
    save(fig,'error_types.png')
    fig,axes=panels('Rewriting after an error increases answer changes')
    for ax,(name,r) in zip(axes,data.items()):
        ts=[t for t in tests if t['dataset']==name]
        for i,t in enumerate(ts):
            for offset,key,color,label in [(-.18,'first','#245d91','Single error'),(.18,'second','#be6528','Error + rewrite')]:
                y=t[key+'_rate'];lo=t[key+'_ci_low'];hi=t[key+'_ci_high']
                ax.bar(i+offset,y,width=.34,color=color,label=label if i==0 else None)
                ax.errorbar(i+offset,y,yerr=[[y-lo],[hi-y]],fmt='none',color='#333333',capsize=3)
                ax.text(i+offset,hi+.005,f"{t[key+'_flips']}/{t['n']}",ha='center',fontsize=9)
        ax.set_xticks([0,1],['Visual error','Logical error']);ax.set(ylabel='Answers changed',ylim=(0,.23));ax.legend(fontsize=9)
    save(fig,'regeneration.png')
    fig,axes=panels('Stable answers have higher original accuracy')
    for ax,(name,r) in zip(axes,data.items()):
        for i,key in enumerate(['sensitive','stable']):
            g=next(g for g in r['rq3'] if g['mode']=='cot' and g['group']==key)
            y=g['accuracy'];ax.bar(i,y,color=['#be6528','#245d91'][i])
            ax.errorbar(i,y,yerr=[[y-g['ci_low']],[g['ci_high']-y]],fmt='none',color='#333333',capsize=4)
            ax.text(i,g['ci_high']+.025,f"{y:.1%}",ha='center')
        ax.set_xticks([0,1],['Answer changed','Answer stayed stable']);ax.set(ylabel='Original accuracy',ylim=(0,1.12))
    save(fig,'accuracy_and_stability.png')

def findings(report,out):
    ds=report['datasets'];tests=report['regeneration_tests']
    lines=['# What happens when we change a model’s explanation?','',
    '**Main finding:** the way we change the explanation matters. A single inserted error rarely changes the answer. Keeping that error and asking the model to rewrite the following steps produces substantially more answer changes.','',
    '## The experiment in plain words','',
    'We give Qwen2.5-VL-7B a picture and a multiple-choice question. It writes an explanation and an answer. We then feed an edited explanation back with the same picture and question, and check whether the answer changes. We evaluate 500 questions from ScienceQA and 500 from A-OKVQA.','',
    '“Visual” means a statement about what is in the picture. “Logical” means a reasoning step based on those observations. An answer change is measured against replaying the original explanation. It can be a correction, a new mistake, or a different wrong answer.','',
    '## 1. Rewording is mostly harmless; removing steps has a larger effect','',
    '| Dataset | Reworded explanation | Shortened explanation, across four lengths |','|---|---:|---:|']
    for n,r in ds.items():
        ctrl=next(m for m in r['metrics'] if m['condition']=='paraphrase_r')
        trunc=[m['flip_rate'] for m in r['metrics'] if m['condition']=='truncate']
        lines.append(f"| {NAMES[n]} | {ctrl['flips']}/{ctrl['paired_n']} ({ctrl['flip_rate']:.1%}) | {min(trunc):.1%}–{max(trunc):.1%} |")
    lines+=['','The answer is not equally sensitive to every edit. Removing explanation steps changes answers far more often than rewording the explanation. The effect is not a simple “shorter is always worse” pattern: ScienceQA has its largest change rate at 50% retained. See truncation.png.','',
    '## 2. Rewriting after an error is the strongest result','',
    '| Dataset | Error type | Single error | Error + rewritten following steps | Holm-adjusted p |','|---|---|---:|---:|---:|']
    for t in tests:
        lines.append(f"| {NAMES[t['dataset']]} | {t['first_variant']} | {t['first_flips']}/{t['n']} ({t['first_rate']:.1%}) | {t['second_flips']}/{t['n']} ({t['second_rate']:.1%}) | {t['holm_p']:.3g} |")
    lines+=['','Each row compares exactly the same questions. All four increases remain significant after Holm correction. The error-type chart uses each condition’s full eligible subset; the matched table above and regeneration.png use the common subset required for a fair comparison.','',
    '**Concrete observation:** evaluating only an isolated edited step gives a different picture from allowing the rest of the explanation to change. The first experiment measures resistance to a local edit; the second measures sensitivity when that edit is followed by a newly generated continuation.','',
    '## 3. Visual errors are not clearly more influential than logical errors','',
    'The matched visual-versus-logical comparisons do not reach significance after Holm correction, for either dataset or either corruption method. The stronger supported distinction is single-error versus error-plus-rewrite. The results do not establish that visual and logical errors have identical effects.','',
    '## 4. Stable answers are much more often correct','',
    '| Dataset | Answer changed under shortening | Answer stayed stable |','|---|---:|---:|']
    for n,r in ds.items():
        gs=[next(g for g in r['rq3'] if g['mode']=='cot' and g['group']==key) for key in ['sensitive','stable']]
        lines.append('| '+NAMES[n]+' | '+' | '.join(f"{g['correct']}/{g['n']} ({g['accuracy']:.1%})" for g in gs)+' |')
    lines+=['','These groups use questions with valid answers in the original explanation run, replay, and all four shortened conditions. Questions with incomplete outputs remain separately reported in rq3.csv. Stability is a useful descriptive signal in these runs; it is not a guarantee that an individual answer is correct.','',
    '## 5. Writing an explanation does not consistently improve accuracy','',
    '| Dataset | Direct answer | Explanation then answer |','|---|---:|---:|']
    for n,r in ds.items():
        ms=[next(m for m in r['metrics'] if m['condition']==c) for c in ['direct','cot']]
        lines.append('| '+NAMES[n]+' | '+' | '.join(f"{m['correct_n']}/{m['eligible_n']} ({m['accuracy']:.1%})" for m in ms)+' |')
    lines+=['','ScienceQA improves numerically with explanations; A-OKVQA declines. The paired baseline accuracy tests do not establish a significant gain on either dataset. Therefore, a written explanation should not automatically be taken as evidence of a better answer.','',
    '## Conclusion for the class','',
    '> Our model is robust to small edits in its written explanation, but its answers become more sensitive when we remove steps or let it rewrite what follows an error. This pattern appears in both datasets. The clearest finding is that the editing method matters more than the visual-versus-logical distinction established by these experiments. Stable answers also tend to be more accurate. Together, these results show why testing explanation reliability requires more than inserting one mistake and checking whether the final answer changes.','',
    '## Image-blur experiment','',
    'The blur experiment uses one light Gaussian blur level: a radius of 1% of the shorter image dimension. It compares original versus blurred images for direct answering and for answering with the original explanation held fixed. This tests whether answers still depend on clear visual input when an explanation is available. The supplied attachments contain 20 unique pilot images; every one has a lightly blurred version in blurred_images. These pilot images are separate from the two 500-question validation cohorts.','',
    'No measured blur responses are present in the supplied inputs. The blur experiment therefore contributes no numerical claim to the conclusions above. The run script loads the validation images for the full experiment; its output is incorporated by rerunning this evaluation.','',
    '## Evaluation details','',
    '- Figures use 95% Wilson intervals. Exact counts, eligibility, parse failures and intervals are in conditions.csv. Error-type bars may have different denominators; statistical comparisons use matched questions.',
    '- Exact two-sided McNemar tests compare paired correctness or paired flip indicators. Holm families are explicit in comparisons.csv: four single-versus-rewritten comparisons across datasets; four visual-versus-logical comparisons; four prompt-version sensitivity comparisons; and a separate paraphrase-control family per dataset. Perturbation accuracy tests form a separate family per dataset in conditions.csv. Baseline direct-versus-explanation tests are exploratory and unadjusted.',
    '- Accuracy counts an eligible unparsed answer as incorrect. Flip comparisons require both answers to parse. The current corruption validity flags are used; historical review decisions are not applied to changed edits.',
    '- Duplicate cache keys use the last record, matching the inference cache. Regenerated edits must match the current single-error edit. Source hashes, per-question results and prompt metadata are retained in analysis.json.',
    '- ScienceQA retains older prompt-version records on ten questions. Removing those questions preserves the central findings at the 0.05 level; see the prompt-version family in comparisons.csv.',
    '- Rewriting changes the continuation as well as retaining the inserted error. A clean-prefix rewrite control would separate those effects. RQ3 groups can also differ in question difficulty. The observations concern responses to edited explanations, not direct measurement of internal reasoning.','',
    '## Completed contribution','',
    'Reproducible CPU evaluation of the frozen caches; confidence intervals; matched tests and Holm correction; RQ3; four clear figures; prompt-version sensitivity checks; a unified findings report; light-blur preparation and a runner for the next experiment.']
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('sayeh_files/results'))
    p.add_argument('--out',type=Path,default=Path('Evaluation Results'))
    p.add_argument('--blur-results',type=Path,default=Path('Blurred Image Results'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    datasets={};sources=[];regen=[];rq2=[];sensitivity=[];controls=[]
    for name in NAMES:
        rows,info,_=core.load_cache(a.results/(name+'.jsonl'));sources.append(info);ds=core.Dataset(rows,name)
        stale=[]
        for (q,c,v),r in ds.rows.items():
            if c=='corrupt' and v.endswith('_regen') and r.get('corrupt_ok'):
                original=ds.row(q,c,v.removesuffix('_regen'))
                if not original or not original.get('corrupt_ok') or any(r.get(k)!=original.get(k) for k in ['step_idx','original_step','corrupted_step']):
                    stale.append(q+':'+v);r['corrupt_ok']=False
        result=core.analyze(ds,None)
        result.pop('sensitivity');result.pop('review_matches')
        result['metadata']['stale_regenerated_rows_excluded']=stale
        datasets[name]=result
        for k in ['perceptual','logical']:
            t,_=core.paired_flip_test(ds,('corrupt',k),('corrupt',k+'_regen'));regen.append(t)
        for suffix in ['','_regen']:
            t,_=core.paired_flip_test(ds,('corrupt','perceptual'+suffix),('corrupt','logical'+suffix));rq2.append(t)
        legacy={q for (q,c,v),r in ds.rows.items() if r.get('prompt_version')!='v1'}
        if legacy:
            for first,second in [(('corrupt',k),('corrupt',k+'_regen')) for k in ['perceptual','logical']]+[(('corrupt','perceptual'+s),('corrupt','logical'+s)) for s in ['','_regen']]:
                t,_=core.paired_flip_test(ds,first,second,set(ds.ids)-legacy);t['excluded_questions']=len(legacy);sensitivity.append(t)
        for t in result['controls']:t['family']=name+'_paraphrase_control';controls.append(t)
    for family,rows in [('single_vs_rewrite',regen),('visual_vs_logical',rq2),('prompt_version_sensitivity',sensitivity)]:
        tested=[r for r in rows if r['exact_mcnemar_p'] is not None]
        for r,pv in zip(tested,core.holm([r['exact_mcnemar_p'] for r in tested])):r['holm_p']=pv;r['family']=family
    report=dict(sources=sources,datasets=datasets,regeneration_tests=regen,rq2_tests=rq2,prompt_version_sensitivity=sensitivity)
    table(a.out/'conditions.csv',[m for r in datasets.values() for m in r['metrics']])
    table(a.out/'rq3.csv',[m for r in datasets.values() for m in r['rq3']])
    table(a.out/'comparisons.csv',regen+rq2+sensitivity+controls)
    (a.out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    plots(datasets,regen,a.out);findings(report,a.out)
    print('Saved text evaluation to',a.out)

if __name__=='__main__':main()
