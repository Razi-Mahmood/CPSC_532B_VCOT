"""Build the concise Word report from the computed evaluation outputs."""
import json,csv
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'Evaluation Results'
data=json.loads((OUT/'analysis.json').read_text());blur=json.loads((OUT/'blur_analysis.json').read_text())['summary']
N={'scienceqa':'ScienceQA','aokvqa':'A-OKVQA'}
doc=Document();sec=doc.sections[0]
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.7)
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(10.5)
doc.styles['Normal'].paragraph_format.space_after=Pt(6)
doc.styles['Normal'].paragraph_format.line_spacing=1.08
doc.styles['Title'].font.size=Pt(23)
doc.styles['Heading 1'].font.size=Pt(15)
doc.styles['Heading 1'].paragraph_format.space_before=Pt(8)
doc.styles['Heading 2'].font.size=Pt(12)
doc.styles['Caption'].font.size=Pt(9)
def para(text,style=None):return doc.add_paragraph(text,style)
def heading(text):doc.add_heading(text,1)
def table(headers,rows,widths=None):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 if widths:
  for col,w in zip(t.columns,widths):col.width=Inches(w)
 for cell,text in zip(t.rows[0].cells,headers):cell.text=str(text)
 for row in rows:
  for cell,text in zip(t.add_row().cells,row):cell.text=str(text)
 for ri,row in enumerate(t.rows):
  for ci,cell in enumerate(row.cells):
   if widths:cell.width=Inches(widths[ci])
   cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   tcPr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
   tcPr.append(borders)
   fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'DFE8EF' if ri==0 else ('F4F6F8' if ri%2==0 else 'FFFFFF'));tcPr.append(fill)
   margins=OxmlElement('w:tcMar')
   for edge in ['top','left','bottom','right']:
    m=OxmlElement('w:'+edge);m.set(qn('w:w'),'80');m.set(qn('w:type'),'dxa');margins.append(m)
   tcPr.append(margins)
   for p in cell.paragraphs:
    p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
    if ci:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:run.font.size=Pt(9);run.bold=ri==0
  trPr=row._tr.get_or_add_trPr();avoid=OxmlElement('w:cantSplit');trPr.append(avoid)
 repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
 para('')
 return t
def figure(file,caption,width=6.65):
 p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(2)
 p.add_run().add_picture(str(OUT/file),width=Inches(width));p.paragraph_format.keep_with_next=True
 para(caption,'Caption')
def page(title):doc.add_page_break();heading(title)
def metric(ds,c,v='orig'):return next(m for m in data['datasets'][ds]['metrics'] if m['condition']==c and m['variant']==v)

doc.add_heading('Explanation reliability in visual question answering',0)
para('Results and interpretation across ScienceQA and A-OKVQA','Subtitle')
para('The strongest result is that the intervention matters: a single inserted error rarely changes the answer, while allowing the model to rewrite the following explanation produces substantially more changes. This holds in both 500-question datasets after multiple-testing correction. Stable answers also tend to be more accurate.')
para('This report covers Qwen2.5-VL-7B-Instruct, the frozen explanation experiments, and the two supplied image-blur response tables. The blur tables score 68.8% overall under exact-answer scoring. Their numerical analysis is presented separately because their run origin is not independently established; see the final section.')
heading('What was tested')
table(['Condition','What changes','Reference'],[
 ['Direct and explanation','Whether the model writes reasoning','Direct answer'],
 ['Replay','Original explanation is supplied again','Original explanation answer'],
 ['Paraphrase','Question or explanation is reworded','Matching original condition'],
 ['Truncation','Retain 0%, 25%, 50% or 75% of steps','Full explanation replay'],
 ['Error insertion','Visual, logical, grammar, unrelated or contradictory edit','Full explanation replay'],
 ['Error plus rewrite','Keep the inserted error and regenerate later steps','Single error on matched questions'],
 ['Image blur tables','Supplied answers attributed to blurred images','Historical direct answers']], [1.45,3.45,1.95])
heading('Baseline accuracy and replay check')
rows=[]
for n in N:
 d=metric(n,'direct');c=metric(n,'cot');r=metric(n,'replay')
 rows.append([N[n],f"{d['correct_n']}/500 ({d['accuracy']:.1%})",f"{c['correct_n']}/500 ({c['accuracy']:.1%})",f"{r['flips']}/{r['paired_n']} ({r['flip_rate']:.1%})"])
table(['Dataset','Direct accuracy','Explanation accuracy','Replay answer changes'],rows,[1.25,1.8,1.9,1.9])
para('Writing an explanation improves ScienceQA by 2.6 percentage points but lowers A-OKVQA by 3.2 points. Neither paired baseline accuracy test establishes a significant change at 0.05. Replay is mostly stable, supporting its use as the reference for explanation edits; its nonzero changes still need to be acknowledged.')

page('Removing steps and inserting errors')
figure('truncation.png','Figure 1. Answer changes relative to full replay. Error bars are 95% Wilson intervals; the dashed line is rationale paraphrasing.')
para('Rationale paraphrasing changes 3/476 ScienceQA answers (0.6%) and 1/493 A-OKVQA answers (0.2%). Truncation changes 9.8–15.4% and 5.5–10.3%, respectively. ScienceQA peaks at 50% retained, so the effect is not a simple monotonic relationship between explanation length and answer stability.')
figure('error_types.png','Figure 2. Each bar uses that condition’s eligible, valid-answer subset. Visual means perceptual. Rewrite means regenerating the steps after the error.')
para('Single visual errors change 0/210 ScienceQA and 2/190 A-OKVQA answers; single logical errors change 6/477 and 13/456. Grammar and unrelated additions change at most 0.6%; contradictions change 1.4% and 1.9%. These observations show resistance to local edits, not proof that the model ignores all reasoning.')
para('Question paraphrasing is a different control: it changes direct answers by 4.8% in both datasets, and explanation-mode answers by 7.0% and 12.2%. Rewording the question cannot be substituted for rewording the explanation as the reference noise level.')

page('Rewriting after an error and answer stability')
figure('regeneration.png','Figure 3. Same questions in each pair of bars. All four exact paired comparisons survive Holm correction with adjusted p < 0.001.')
para('Matched single-error versus rewritten-continuation rates are 0.0% versus 5.8% and 0.4% versus 8.4% in ScienceQA; 1.1% versus 8.0% and 1.9% versus 12.9% in A-OKVQA. This is the clearest cross-dataset finding. The visual-versus-logical contrasts are not significant after correction, for either intervention style.')
figure('accuracy_and_stability.png','Figure 4. Original explanation-mode accuracy, grouped by whether any of four truncations changes the replay answer. Complete cases only.')
para('ScienceQA stable questions score 347/361 (96.1%), versus 86/118 (72.9%) for sensitive questions. A-OKVQA scores 397/420 (94.5%) versus 29/73 (39.7%). Another 21 and 7 questions have incomplete outputs. Stability is associated with correctness, but question difficulty can explain part of that association.')
para('Regeneration changes the downstream explanation as well as retaining the inserted error. A clean-prefix regeneration control is needed before attributing the full increase specifically to propagation of the error.')

page('Supplied image blur responses')
table(['Dataset','Correct','Close','Incorrect label','Strict accuracy'],[
 [N[s['dataset']],s['correct'],s['close'],s['annotated_incorrect'],f"{s['accuracy']:.1%} [{s['ci_low']:.1%}, {s['ci_high']:.1%}]"] for s in blur], [1.3,.8,.75,1.1,2.3])
para('Across 1,000 rows, 688 are correct and 312 are wrong. The wrong answers comprise 181 labelled Close and 131 labelled Incorrect. Close remains a separate descriptive annotation, but does not receive partial credit in multiple-choice accuracy. No documented rubric establishes how those two wrong-answer categories were assigned.')
figure('blur_categories.png','Figure 5. Supplied annotations. ScienceQA has more Close labels; this does not establish better visual recognition.',6.4)
figure('blur_accuracy.png','Figure 6. Historical direct accuracy versus the supplied blur responses, with 95% Wilson intervals. These are separate runs, not verified same-run original/blurred controls.',6.4)
para('The numerical gaps are −17.0 percentage points for ScienceQA and −20.6 points for A-OKVQA. The pooled strict score is 68.8%, compared with 87.6% in the original direct-answer caches. The supplied settings do not establish that blur alone explains the difference.')

page('Interpretation and conclusions')
heading('What the paired blur comparison contains')
table(['Dataset','Correct to wrong','Wrong to correct','Changed answers','Holm p'],[
 [N[s['dataset']],s['correct_to_wrong'],s['wrong_to_correct'],f"{s['flips']}/{s['paired_n']}",f"{s['holm_p']:.2g}"] for s in blur],[1.25,1.25,1.25,1.65,.85])
para('Correctness transitions include all 500 questions per dataset, treating five unparsed ScienceQA baseline answers as incorrect. Answer flips require valid answers in both conditions, leaving 495 ScienceQA pairs. Paired bootstrap intervals for the accuracy difference are −21.6 to −12.4 points and −24.8 to −16.4 points. These tests describe the supplied tables conditional on their authenticity and comparability; they do not validate either.')
heading('What can be concluded for the class')
para('First, explanation sensitivity depends strongly on how the explanation is changed. Near-zero effects from local mistakes coexist with much larger effects from truncation and rewritten continuations. Second, a visual-versus-logical difference is not established at this sample size. Third, stable answers are markedly more accurate in these subsets, making stability an informative descriptive signal rather than a correctness guarantee.')
para('The supplied blur tables contain a large accuracy gap, but cannot establish semantic resilience, vulnerability to fine details, or knowledge compensation as measured mechanisms. Those are useful hypotheses for a validated blur study. To test them, label question requirements before inspecting outcomes, measure original and blurred images with identical settings, and compare those groups using matched question IDs.')
heading('Evidence and evaluation details')
para('The submitter identifies the two blur CSVs as outputs from a GPU Qwen run. All 1,000 selected answers and scene descriptions exactly match the earlier text-only heuristic forecast rows. Raw generation logs, image IDs, blur parameters and decoding settings were not supplied. Consequently, this report retains the numerical analysis but does not treat these tables as independently verified experimental evidence.')
para('The explanation evaluation uses the frozen 500-question caches from sayeh-work. Smaller main/extra-testing runs are not pooled with them. Last-record-wins cache handling, corruption validity flags and valid paired outputs determine denominators. Historical review decisions are not applied to regenerated edits. Excluding ten ScienceQA questions with older prompt records preserves the central conclusions at the 0.05 level.')
para('Intervals are 95% Wilson for proportions. Exact McNemar tests compare paired outcomes. Holm correction is applied separately to four single-versus-rewrite tests, four visual-versus-logical tests, four prompt-version sensitivity tests, control comparisons within each dataset, perturbation accuracy tests within each dataset, and two supplied-blur accuracy comparisons. Blur difference intervals use 20,000 paired bootstrap resamples with seed 532. Full counts, individual rows, source hashes and matching checks accompany the report.')
para('The image example folder contains all 20 pilot originals and their light-blur versions, numbered in pairs. These illustrate a Gaussian radius of 1% of the shorter side; they are not the images behind all 1,000 validation questions and do not establish the setting used in the submitted response tables.')

doc.core_properties.title='Explanation reliability in visual question answering'
doc.core_properties.subject='Evaluation of explanation interventions and supplied image blur response tables'
doc.core_properties.author=''
for root in [doc._element,doc.styles.element]:
 for border in list(root.xpath('.//w:pBdr')):border.getparent().remove(border)
doc.save(ROOT/'Results Report.docx')
print(ROOT/'Results Report.docx')
