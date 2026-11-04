import glob,os,re
import numpy as np,pandas as pd
from scipy import stats
def vals(p):
    d=pd.read_csv(p,dtype=str,usecols=['DV']); return pd.to_numeric(d['DV'].replace('.',None),errors='coerce').dropna().values
out={}
for tr in sorted(glob.glob('results/paper_repro/w*/results/*/virtual_cohort_train.csv')):
    te=tr.replace('train','test')
    if not os.path.exists(te): continue
    m=re.search(r'results/(\d)\d+/',tr)
    if not m: continue
    sc=m.group(1); run=re.search(r'results/(\d+)/',tr).group(1)
    if run in out: continue
    a=vals(tr); b=vals(te); al=np.concatenate([a,b]); mt,ma=a.max(),al.max()
    out[run]=(int(sc),ma/mt,stats.boxcox(a[a>0]/mt)[1],stats.boxcox(al[al>0]/ma)[1])
O=np.array(list(out.values()))
print('unique datasets',len(O))
for k in (1,2,3,None):
    x=O if k is None else O[O[:,0]==k]
    q=x[:,1]; d=np.abs(x[:,2]-x[:,3])
    print(('all' if k is None else f'Scenario {k}'),f'n={len(x)} max unchanged in {100*np.mean(q==1):.0f}%, median ratio {np.median(q):.3f}, p90 {np.percentile(q,90):.3f}, largest {q.max():.3f}; lambda |diff| median {np.median(d):.3f}, largest {d.max():.3f}')
