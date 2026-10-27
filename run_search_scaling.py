import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,csv,time,itertools,argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from model import *

def job(row):
    T,k,pa,pb,la,lb,reference=row
    starts=starting_schedules(T);ta=marginal(pa,la);tb=marginal(pb,lb)
    t=time.perf_counter();seq,val,n,m=exchange_search(starts,k,pa,pb,ta,tb);elapsed=time.perf_counter()-t
    return dict(T=T,k=k,pa=pa,pb=pb,la=la,lb=lb,schedule=schedule_string(seq),upper=val,optimum=reference,relative_gap=(val/reference-1) if reference>0 else 0,evaluations=n,moves=m,seconds=elapsed)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);ap.add_argument('--output',type=Path,default=ROOT/'results');args=ap.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    with (args.output/'benchmark.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['policy']=='robust-opt']
    jobs=[(8,int(r['k']),float(r['pa']),float(r['pb']),float(r['la']),float(r['lb']),float(r['upper'])) for r in rows]
    jobs.extend((T,T//2,pa,pb,la,lb,-1) for T in [16,32,64] for pa,pb,la,lb in [(.1,.2,4,2),(.1,.1,3,3),(.02,.2,8,1.25)])
    exchange_search(starting_schedules(4),2,.1,.2,marginal(.1,4),marginal(.2,2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:res=list(pool.map(job,jobs))
    with (args.output/'search_scaling.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(res[0]));w.writeheader();w.writerows(res)
    gaps=[r['relative_gap'] for r in res if r['T']==8]
    print('Global optimum reached',sum(g<1e-9 for g in gaps),'of',len(gaps),'max gap',max(gaps),flush=True)
    for r in res[-9:]:print(r,flush=True)
if __name__=='__main__':main()
