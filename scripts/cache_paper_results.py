"""Extract versioned UniADet numeric references from a local paper HTML file.

Normal reporting reads the resulting cache offline; this tool never downloads.
"""
from html.parser import HTMLParser
from pathlib import Path
import re,json
class Node:
 def __init__(self,tag='',attrs=()):self.tag=tag;self.attrs=dict(attrs);self.children=[]
 def all(self,tag):
  for c in self.children:
   if isinstance(c,Node):
    if c.tag==tag:yield c
    yield from c.all(tag)
 def text(self):
  if self.tag in ['annotation','script','style']:return ''
  return ''.join(c.text() if isinstance(c,Node) else c for c in self.children)
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.root=Node();self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  n=Node(tag,attrs);self.stack[-1].children.append(n)
  if tag not in ['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']:self.stack.append(n)
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:del self.stack[i:];break
 def handle_data(self,d):self.stack[-1].children.append(d)
def clean(s):return re.sub(r'\s+',' ',s).strip()
def tables(path):
 p=Parser();p.feed(Path(path).read_text());out=[]
 for fig in p.root.all('figure'):
  if 'ltx_table' not in fig.attrs.get('class',''):continue
  cap=clean(next(fig.all('figcaption')).text());number=int(re.search(r'Table (\d+):',cap)[1]);rows=[]
  for row in next(fig.all('table')).all('tr'):
   cells=[c for c in row.children if isinstance(c,Node) and c.tag in ('th','td')]
   rows.append([dict(text=clean(c.text()),rowspan=int(c.attrs.get('rowspan',1)),colspan=int(c.attrs.get('colspan',1))) for c in cells])
  out.append(dict(number=number,id=fig.attrs['id'],caption=cap,rows=rows))
 return out

METRICS = ['I-AUROC','I-AUPR','I-F1max','P-AUROC','P-AUPR','P-F1max','P-AUPRO']
BACKBONES = ['clip','dinov2','dinov3']
SOURCE = 'https://arxiv.org/html/2601.05552v1'

def measurement(cell):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?', cell)
    if not match:
        raise ValueError(f'Unrecognized numeric cell: {cell!r}')
    return dict(value_percent=float(match[1]), std_percent=float(match[2]) if match[2] else None,
                reported=cell)

def extract(path):
    import hashlib
    extracted = tables(path)
    categories, datasets, main = [], [], []
    def common(backbone,dataset,shots,table):
        return dict(backbone=backbone,dataset=dataset,training_dataset='MVTecAD' if dataset=='VisA' else 'VisA',
                    regime='zero_shot' if shots==0 else 'few_shot',shots=shots,paper_version='2601.05552v1',
                    source_table=table,source_url=SOURCE+'#'+next(t['id'] for t in extracted if t['number']==table))
    for table in extracted:
        n=table['number']
        if n in list(range(12,27))+list(range(29,35)):
            if n<29:
                dataset=['MVTecAD','VisA','BTAD','DTD','KSDD'][(n-12)//3]
                backbone=BACKBONES[(n-12)%3]
            else:
                dataset='Real-IAD';backbone=BACKBONES[(n-29)//2]
            shots=None
            for row in table['rows'][3:]:
                cells=[c['text'] for c in row]
                assert len(cells)==10,(n,cells)
                if cells[0]:shots=int(re.fullmatch(r'(\d+)-shot',cells[0])[1])
                assert shots in (0,1,2,4)
                values={m:measurement(cells[i]) for m,i in zip(METRICS,[2,3,4,6,7,8,9])}
                item=dict(**common(backbone,dataset,shots,n),category=cells[1],metrics=values)
                if cells[1].lower()=='mean':
                    item.update(category='macro_mean',origin='detailed_table_mean');datasets.append(item)
                else:
                    categories.append(item)
                    if dataset=='KSDD':datasets.append(dict(item,category='macro_mean',origin='single_category_dataset'))
        elif n in (27,28):
            names=METRICS[:3] if n==27 else METRICS[3:]
            for row in table['rows'][1:]:
                cells=[c['text'] for c in row]
                if cells[0]=='Mean':continue
                for backbone,cell in zip(BACKBONES,cells[1:]):
                    parts=[x.strip() for x in cell.split(',')];assert len(parts)==len(names)
                    datasets.append(dict(**common(backbone,cells[0],0,n),category='macro_mean',origin='medical_dataset_row',
                                         metrics={m:measurement(v) for m,v in zip(names,parts)}))
        elif n in (2,3,5,10,11):
            shots={2:0,3:0,5:1,10:2,11:4}[n];task=None
            for row in table['rows'][2:]:
                cells=[c['text'] for c in row]
                if cells[0]:task='I' if cells[0].startswith('Image') else 'P'
                if cells[1]=='Mean':continue
                dataset={'MVTec':'MVTecAD','RealIAD':'Real-IAD'}.get(cells[1],cells[1])
                for backbone,cell in zip(BACKBONES,cells[-3:]):
                    values={task+'-'+m:measurement(v.strip()) for m,v in zip(['AUROC','AUPR'],cell.split(','))}
                    main.append(dict(**common(backbone,dataset,shots,n),metrics=values))
    assert len(categories)==876,len(categories)
    assert len(datasets)==96,len(datasets)
    assert len({(r['backbone'],r['dataset'],r['shots'],r['category']) for r in categories})==876
    assert len({(r['backbone'],r['dataset'],r['shots']) for r in datasets})==96
    discrepancies=[]
    for r in main:
        detail=next(d for d in datasets if all(d[k]==r[k] for k in ['backbone','dataset','shots']))
        for metric,v in r['metrics'].items():
            other=detail['metrics'][metric]
            if v['value_percent']!=other['value_percent'] or v['std_percent']!=other['std_percent']:
                discrepancies.append(dict(backbone=r['backbone'],dataset=r['dataset'],shots=r['shots'],metric=metric,
                    main_table=r['source_table'],main_reported=v['reported'],detail_table=detail['source_table'],detail_reported=other['reported']))
    import statistics
    macro_discrepancies=[]
    for record in datasets:
        selected=[c for c in categories if all(c[k]==record[k] for k in ['backbone','dataset','shots'])]
        if not selected:continue
        for metric,value in record['metrics'].items():
            mean=statistics.mean(c['metrics'][metric]['value_percent'] for c in selected)
            # One-decimal category and macro rounding can differ by up to 0.1 pp.
            if abs(mean-value['value_percent'])>0.11:
                macro_discrepancies.append(dict(backbone=record['backbone'],dataset=record['dataset'],shots=record['shots'],
                    metric=metric,source_table=record['source_table'],reported_macro=value['value_percent'],
                    mean_of_rounded_categories=mean,difference_pp=mean-value['value_percent']))
    return dict(paper_version='2601.05552v1',source_url=SOURCE,html_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                units='percent',datasets=datasets,categories=categories,main_table_results=main,discrepancies=discrepancies,
                macro_discrepancies=macro_discrepancies,
                header_corrections=['Tables 27/28 repeat dagger for second column; identified as DINOv2 from column order and main Table 3.',
                                    'Table 2 caption calls DINOv2-R ViT-L/16; implementation details specify ViT-L/14. Backbone key dinov2 retains intended identity.'])

def write_cache(reference,output):
    import csv
    from datetime import datetime,timezone
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'reference.json').write_text(json.dumps(reference,indent=2,ensure_ascii=False)+'\n')
    def write_csv(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    def flatten(records):
        rows=[]
        for record in records:
            row={k:v for k,v in record.items() if k!='metrics'}
            for m in METRICS:
                value=record['metrics'].get(m,{})
                row[m]=value.get('value_percent');row[m+'_std']=value.get('std_percent');row[m+'_reported']=value.get('reported')
            rows.append(row)
        return rows
    write_csv('dataset_results.csv',flatten(reference['datasets']))
    write_csv('category_results.csv',flatten(reference['categories']))
    write_csv('main_table_results.csv',flatten(reference['main_table_results']))
    if reference['discrepancies']:write_csv('discrepancies.csv',reference['discrepancies'])
    if reference['macro_discrepancies']:write_csv('macro_discrepancies.csv',reference['macro_discrepancies'])
    provenance=dict(paper_version=reference['paper_version'],source_url=SOURCE,html_sha256=reference['html_sha256'],
                    extracted_at=datetime.now(timezone.utc).isoformat(),dataset_rows=len(reference['datasets']),category_rows=len(reference['categories']),
                    tables=list(range(12,35)),main_tables=[2,3,5,10,11],std_definition='Reported ± values retained; do not assume local ddof=1 or known paper seeds.',
                    policy='Use detailed-table dataset rows for full-metric comparisons; retain main-table differences explicitly; never silently reconcile.',
                    extraction='scripts/cache_paper_results.py; local input only; no need to parse paper for ordinary reporting.')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html',type=Path,required=True,help='Already-downloaded arXiv v1 HTML')
    parser.add_argument('--output',type=Path,default=Path('reports/paper/uniadet_2601.05552v1'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv);reference=extract(args.html)
    if not args.dry_run:write_cache(reference,args.output)
    print(json.dumps(dict(dataset_rows=len(reference['datasets']),category_rows=len(reference['categories']),
                         discrepancies=len(reference['discrepancies']),output=str(args.output)),indent=2))

if __name__=='__main__':main()
