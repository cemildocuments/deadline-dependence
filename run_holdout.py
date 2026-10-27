import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,csv,time,argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from model import *
from run_experiments import choose

def job(j):
    i,pa,pb,la,lb,k=j;ta=marginal(pa,la);tb=marginal(pb,lb)
    ss=schedules(8,8);v=evaluate(ss,k,pa,pb,ta,tb);opt=v[choose(v,2),2]
    st=time.perf_counter();seq,val,n,m=exchange_search(starting_schedules(8),k,pa,pb,ta,tb);sec=time.perf_counter()-st
    return dict(instance=i,pa=pa,pb=pb,la=la,lb=lb,k=k,upper=val,optimum=opt,relative_gap=val/opt-1,evaluations=n,moves=m,seconds=sec,schedule=schedule_string(seq))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);ap.add_argument('--output',type=Path,default=ROOT/'results');args=ap.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(240027)
    jobs=[(i,*rng.uniform(.005,.35,2),*np.exp(rng.uniform(np.log(1.05),np.log(12),2)),int(rng.integers(2,8))) for i in range(200)]
    exchange_search(starting_schedules(4),2,.1,.2,marginal(.1,4),marginal(.2,2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(job,jobs))
    with (args.output/'holdout.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gaps=np.array([r['relative_gap'] for r in rows]);print('Optimal',np.sum(gaps<1e-10),'of 200; max gap',gaps.max(),'median',np.median(gaps),flush=True)
if __name__=='__main__':main()
