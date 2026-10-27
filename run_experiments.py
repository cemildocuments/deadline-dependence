#!/usr/bin/env python3
"""Regenerate the exhaustive benchmark, sensitivity, timing, and simulations."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,csv,json,itertools,time,sys,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from model import marginal,schedules,evaluate,risk,witness,simulate,schedule_string


def savecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def choose(values,column):
    best=np.min(values[:,column]);tol=1e-12*max(best,1e-12)
    tied=np.flatnonzero(values[:,column]<=best+tol)
    secondary=0 if column==2 else 2
    return int(tied[np.argmin(values[tied,secondary])])


def scenario(job):
    sid,pa,pb,la,lb,k=job;T=8;B=8
    ss=schedules(T,B);ta=marginal(pa,la);tb=marginal(pb,lb)
    start=time.perf_counter();v=evaluate(ss,k,pa,pb,ta,tb);duration=time.perf_counter()-start
    candidates={'independent-opt':ss[choose(v,0)],'positive-opt':ss[choose(v,1)],'robust-opt':ss[choose(v,2)],
      'round-robin':np.array([1,2]*4,dtype=np.int8),'spread-pair':np.array([3,0,3,0,0,3,0,3],dtype=np.int8)}
    single=[np.ones(T,dtype=np.int8),np.full(T,2,dtype=np.int8)]
    candidates['best-single']=min(single,key=lambda a:risk(a,k,pa,pb,ta,tb,0))
    rows=[]
    for name,seq in candidates.items():
        vals=[risk(seq,k,pa,pb,ta,tb,m) for m in [0,2,3,4]]
        rows.append(dict(scenario=sid,pa=pa,pb=pb,la=la,lb=lb,k=k,T=T,budget=B,policy=name,
             schedule=schedule_string(seq),independent=vals[0],positive=vals[1],upper=vals[2],lower=vals[3]))
    summary=dict(scenario=sid,pa=pa,pb=pb,la=la,lb=lb,k=k,schedules=len(ss),seconds=duration,
       max_positive_underestimate=float(np.max(v[:,2]/np.maximum(v[:,1],1e-300))))
    return rows,summary


def mcjob(job):
    name,seq,k,pa,pb,la,lb,mode,n,seed=job
    a=np.array(['.ABX'.index(c) for c in seq],dtype=np.int8);ta=marginal(pa,la);tb=marginal(pb,lb)
    failures=simulate(a,k,pa,pb,ta,tb,n,seed,mode)
    m={'independent':0,'positive':2,'worst':3}[mode]
    exact=risk(a,k,pa,pb,ta,tb,m)
    return dict(policy=name,schedule=seq,k=k,pa=pa,pb=pb,la=la,lb=lb,mode=mode,seed=seed,blocks=n,failures=failures,exact=exact)


def budgets(job):
    pa,pb,la,lb,k=job;ta=marginal(pa,la);tb=marginal(pb,lb);rows=[]
    for b in range(k,17):
        ss=schedules(8,b);v=evaluate(ss,k,pa,pb,ta,tb)
        for name,col in [('independent-opt',0),('positive-opt',1),('robust-opt',2)]:
            ix=choose(v,col);rows.append(dict(pa=pa,pb=pb,la=la,lb=lb,k=k,budget=b,policy=name,schedule=schedule_string(ss[ix]),independent=v[ix,0],positive=v[ix,1],upper=v[ix,2]))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);ap.add_argument('--output',type=Path,default=ROOT/'results');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/experiment.json').read_text())
    p=cfg['erasure_probabilities'];L=cfg['mean_burst_lengths'];ks=cfg['source_packets']
    start=time.perf_counter()
    # Compile once before spawning workers; each worker loads the cached machine code.
    risk(np.array([3],dtype=np.int8),1,.1,.2,marginal(.1,3),marginal(.2,3))
    jobs=[(i,*x) for i,x in enumerate(itertools.product(p,p,L,L,ks))]
    rows=[];summary=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r,s in pool.map(scenario,jobs):rows.extend(r);summary.append(s)
    savecsv(out/'benchmark.csv',rows);savecsv(out/'search_timings.csv',summary)
    print('Exhaustive benchmark completed:',len(jobs),'scenarios;',len(jobs)*12870,'schedules',flush=True)
    q=cfg['illustration'];illustration=[]
    for k in ks:
        r,_=scenario((-1,q['pa'],q['pb'],q['la'],q['lb'],k));illustration.extend(r)
    savecsv(out/'illustration.csv',illustration)
    mc=[]
    for r in illustration:
        if r['policy'] not in ['independent-opt','positive-opt','robust-opt']:continue
        for mode in ['independent','positive','worst']:
            for j in range(cfg['mc_seeds']):
                seed=cfg['seed']+10000*r['k']+1000*['independent-opt','positive-opt','robust-opt'].index(r['policy'])+100*['independent','positive','worst'].index(mode)+j
                mc.append((r['policy'],r['schedule'],r['k'],r['pa'],r['pb'],r['la'],r['lb'],mode,cfg['mc_blocks_per_seed'],seed))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:mcrows=list(pool.map(mcjob,mc))
    savecsv(out/'monte_carlo.csv',mcrows);print('Monte Carlo blocks:',sum(r['blocks'] for r in mcrows),flush=True)
    # All budgets for all 81 path-pair scenarios, source size 4.
    bj=list(itertools.product(p,p,L,L,[4]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:br=[r for rows_ in pool.map(budgets,bj) for r in rows_]
    br.extend(budgets((q['pa'],q['pb'],q['la'],q['lb'],4)))
    savecsv(out/'budget_frontiers.csv',br)
    # Continuous-radius sensitivity: re-optimize at every specified radius.
    ss=schedules(8,8);sens=[]
    for k in ks:
        for lam in np.linspace(0,1,41):
            v=evaluate(ss,k,q['pa'],q['pb'],marginal(q['pa'],q['la']),marginal(q['pb'],q['lb']),lam)
            ix=choose(v,2);ix0=choose(v,0)
            sens.append(dict(k=k,radius=lam,schedule=schedule_string(ss[ix]),independent=v[ix,0],upper=v[ix,2],nominal_opt_upper=v[ix0,2]))
    savecsv(out/'radius_sensitivity.csv',sens)
    # Linear evaluator scaling; warmed compilation and repeated independent timings.
    timing=[]
    for T in [8,16,32,64,128,256,512,1024]:
        k=T//2;seq=np.array([1,2]*(T//2),dtype=np.int8);ta=marginal(.1,4);tb=marginal(.2,2)
        risk(seq,k,.1,.2,ta,tb,3)
        for repeat in range(30):
            tt=time.perf_counter()
            for _ in range(20):risk(seq,k,.1,.2,ta,tb,3)
            timing.append(dict(T=T,k=k,repeat=repeat,microseconds=(time.perf_counter()-tt)*1e6/20))
    savecsv(out/'evaluator_timings.csv',timing)
    meta=dict(config=cfg,python=platform.python_version(),numpy=np.__version__,platform=platform.system(),machine=platform.machine(),cpu_count=os.cpu_count(),workers=args.workers,elapsed_seconds=time.perf_counter()-start,scenarios=len(jobs),schedules_per_scenario=12870,schedule_instances=len(jobs)*12870,mc_blocks=sum(r['blocks'] for r in mcrows))
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
