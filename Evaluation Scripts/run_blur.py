"""Measure original versus lightly blurred validation images with frozen prompts."""
import argparse, hashlib, json, sys
from pathlib import Path
from PIL import ImageFilter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['scienceqa','aokvqa'])
    p.add_argument('--n',type=int,default=500)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    from run_experiments import load_examples,Qwen,DIRECT_TMPL,build_prompt
    from parsing import parse_answer
    cache_path=ROOT/'sayeh_files/results'/f'{a.dataset}.jsonl'
    rows={}
    for line in cache_path.read_text().splitlines():
        r=json.loads(line);rows[(str(r['question_id']),r['condition'],r['variant'])]=r
    examples=load_examples(a.dataset,a.n)
    for ex in examples:
        q=ex['qid'];known=rows[(q,'replay','orig')]
        if known['choices']!=ex['choices'] or known['gold']!=chr(65+ex['gold_idx']):raise ValueError('Dataset mismatch '+q)
        if 'input_rationale' not in known:raise ValueError('Missing original rationale '+q)
    out=ROOT/'Blurred Image Results';out.mkdir(exist_ok=True)
    images=ROOT/'Blurred Image Examples';images.mkdir(exist_ok=True)
    target=out/f'{a.dataset}.jsonl'
    if not a.prepare_only and target.exists() and target.stat().st_size:raise SystemExit('Existing results protected. Archive the prior run before starting a new one.')
    manifest=out/f'{a.dataset}_manifest.jsonl'
    if manifest.exists():raise SystemExit('Existing manifest protected. Archive it before starting a new run.')
    model=None if a.prepare_only else Qwen()
    with manifest.open('w') as f:
        f.write(json.dumps(dict(record_type='configuration',dataset=a.dataset,n=len(examples),
          model='Qwen/Qwen2.5-VL-7B-Instruct',dtype='float32',temperature=0,attention='eager',
          sigma_fraction=.01,prepare_only=a.prepare_only,cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest()))+'\n')
    for ex in examples:
        q=ex['qid'];rationale=rows[(q,'replay','orig')]['input_rationale']
        original=ex['image'].convert('RGB');radius=.01*min(original.size)
        blurred=original.filter(ImageFilter.GaussianBlur(radius))
        path=images/f"{a.dataset}_{q}_light.png";blurred.save(path)
        for sigma,image in [(0,original),(.01,blurred)]:
            entry=dict(dataset=a.dataset,question_id=q,sigma_fraction=sigma,
                radius_pixels=0 if sigma==0 else radius,width=image.width,height=image.height,
                image_file=path.name if sigma else None,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            with manifest.open('a') as f:f.write(json.dumps(entry)+'\n')
            if model:
                modified={**ex,'image':image}
                for mode in ['direct','fixed_rationale']:
                    raw=(model.chat(image,build_prompt(DIRECT_TMPL,modified),max_new_tokens=10)
                         if mode=='direct' else model.answer_given_rationale(modified,rationale))
                    answer,method=parse_answer(raw,ex['choices'])
                    record={**entry,'mode':mode,'raw_output':raw,'parsed_answer':answer,
                        'parse_method':method,'gold':chr(65+ex['gold_idx']),'choices':ex['choices']}
                    with target.open('a') as f:f.write(json.dumps(record)+'\n')
    print('Finished',a.dataset,'image preparation' if a.prepare_only else 'inference')

if __name__=='__main__':main()
