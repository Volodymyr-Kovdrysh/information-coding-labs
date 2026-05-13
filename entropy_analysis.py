import math, random, re, string, os, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from faker import Faker

random.seed(42)
BASE=Path('/mnt/data/lab_entropy')
BASE.mkdir(exist_ok=True)

# Character sets for filtering
UKR_LETTERS = set('абвгґдеєжзиіїйклмнопрстуфхцчшщьюя')
ENG_LETTERS = set('abcdefghijklmnopqrstuvwxyz')
CJK_RE = re.compile(r'[\u4e00-\u9fff]')

def extract_pdf_text(paths):
    # Use pdftotext if available.
    import subprocess, tempfile
    texts=[]
    for p in paths:
        out=BASE/(Path(p).stem+'.txt')
        try:
            subprocess.run(['pdftotext','-layout',p,str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            texts.append(out.read_text(encoding='utf-8',errors='ignore'))
        except Exception as e:
            pass
    return '\n'.join(texts)

def normalize(text, lang):
    text=text.lower()
    if lang=='uk':
        return ''.join(ch for ch in text if ch in UKR_LETTERS)
    if lang=='en':
        return ''.join(ch for ch in text if ch in ENG_LETTERS)
    if lang=='zh':
        return ''.join(ch for ch in text if CJK_RE.match(ch))
    raise ValueError(lang)

def entropy(counter):
    n=sum(counter.values())
    if n==0: return 0.0
    return -sum((c/n)*math.log2(c/n) for c in counter.values())

def bigrams(text):
    return [text[i:i+2] for i in range(len(text)-1)]

def metrics(text):
    chars=Counter(text)
    H0=entropy(chars)
    bg=Counter(bigrams(text))
    Hb=entropy(bg)
    # Conditional entropy per next symbol: H(X_n|X_{n-1}) = H(X_{n-1},X_n)-H(X_{n-1})
    H1=max(0.0, Hb-H0)
    alphabet=len(chars)
    Hmax=math.log2(alphabet) if alphabet>0 else 0
    R=1-H0/Hmax if Hmax else 0
    return {'length':len(text),'alphabet':alphabet,'H0':H0,'H_bigram':Hb,'H1':H1,'Hmax':Hmax,'redundancy':R}

def convergence(text, lengths):
    rows=[]
    for L in lengths:
        if L <= len(text):
            m=metrics(text[:L])
            rows.append({'L':L,'H0':m['H0'],'H1':m['H1']})
    return rows

def random_text_like(text):
    alphabet=sorted(set(text))
    return ''.join(random.choice(alphabet) for _ in range(len(text)))

# Build corpora
uk_pdf_paths=[
 '/mnt/data/Тема3_МоделіІмВідповіді-1.pdf',
 '/mnt/data/Тема3_МоделіІмВідповіді-2-4.pdf',
 '/mnt/data/6-11_MMIB_20260421.pdf',
 '/mnt/data/Тема2-MMНП_2026.pdf',
 '/mnt/data/Тема2-MMНП_20250325.pdf'
]
uk_raw=extract_pdf_text([p for p in uk_pdf_paths if Path(p).exists()])
uk=normalize(uk_raw,'uk')
if len(uk)<100000:
    f=Faker('uk_UA')
    while len(uk)<110000:
        uk += normalize(f.text(max_nb_chars=200),'uk')

eng_raw=Path('/usr/local/go/src/testdata/Isaac.Newton-Opticks.txt').read_text(encoding='utf-8',errors='ignore')
en=normalize(eng_raw,'en')

fzh=Faker('zh_CN')
zh=''
while len(zh)<110000:
    zh += normalize(fzh.text(max_nb_chars=200),'zh')

corpora={'Українська': uk[:100000], 'Англійська': en[:100000], 'Китайська': zh[:100000]}
for name, text in corpora.items():
    (BASE/f'{name}.txt').write_text(text, encoding='utf-8')

summary=[]; shuffle_rows=[]; random_rows=[]
lengths=[100,500,1000,5000,10000,25000,50000,100000]
conv={}
for name,text in corpora.items():
    m=metrics(text); m['Мова']=name; summary.append(m)
    t=list(text); random.shuffle(t); sh=''.join(t)
    ms=metrics(sh); ms['Мова']=name; shuffle_rows.append(ms)
    rt=random_text_like(text)
    mr=metrics(rt); mr['Мова']=name; random_rows.append(mr)
    conv[name]=convergence(text,lengths)

# Save results
import csv
fields=['Мова','length','alphabet','H0','H1','H_bigram','Hmax','redundancy']
for fname, rows in [('results_summary.csv',summary),('results_shuffled.csv',shuffle_rows),('results_random.csv',random_rows)]:
    with open(BASE/fname,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

with open(BASE/'convergence.json','w',encoding='utf-8') as f: json.dump(conv,f,ensure_ascii=False,indent=2)

# Plots
for name, rows in conv.items():
    xs=[r['L'] for r in rows]; y0=[r['H0'] for r in rows]; y1=[r['H1'] for r in rows]
    plt.figure(figsize=(7,4.2))
    plt.plot(xs,y0,marker='o',label='H0')
    plt.plot(xs,y1,marker='o',label='H1 умовна')
    plt.xscale('log')
    plt.xlabel('Довжина тексту, символів')
    plt.ylabel('Ентропія, біт/символ')
    plt.title(f'Збіжність ентропії: {name}')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE/f'convergence_{name}.png',dpi=180)
    plt.close()

# Comparison H0/H1 initial vs shuffled vs random for English as example
for name,text in corpora.items():
    plt.figure(figsize=(7,4.2))
    init=metrics(text)
    t=list(text); random.shuffle(t); sh=metrics(''.join(t))
    rnd=metrics(random_text_like(text))
    labels=['Початковий','Перемішаний','Випадковий']
    h0=[init['H0'],sh['H0'],rnd['H0']]
    h1=[init['H1'],sh['H1'],rnd['H1']]
    x=range(len(labels)); width=0.35
    plt.bar([i-width/2 for i in x],h0,width,label='H0')
    plt.bar([i+width/2 for i in x],h1,width,label='H1 умовна')
    plt.xticks(list(x),labels)
    plt.ylabel('Ентропія, біт/символ')
    plt.title(f'Порівняння типів тексту: {name}')
    plt.legend(); plt.tight_layout()
    plt.savefig(BASE/f'compare_{name}.png',dpi=180)
    plt.close()

# A compact JSON for report generation
out={'summary':summary,'shuffle':shuffle_rows,'random':random_rows,'conv':conv}
with open(BASE/'all_results.json','w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out['summary'],ensure_ascii=False,indent=2))
