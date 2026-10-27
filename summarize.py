"""Compute benchmark statistics and Monte Carlo uncertainty from saved results."""
import csv,json,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'results');a=ap.parse_args();p=a.results
    by={}
    for r in read(p/'benchmark.csv'):by.setdefault(int(r['scenario']),{})[r['policy']]=r
    stats=[];allg=[]
    for k in [2,4,6]:
        sub=[x for x in by.values() if int(x['robust-opt']['k'])==k]
        gain=np.array([1-float(x['robust-opt']['upper'])/float(x['independent-opt']['upper']) for x in sub]);allg.extend(gain)
        gap=[float(x['independent-opt']['upper'])/float(x['independent-opt']['independent']) for x in sub]
        posgap=[float(x['positive-opt']['upper'])/float(x['positive-opt']['positive']) for x in sub]
        price=[float(x['robust-opt']['independent'])/float(x['independent-opt']['independent']) for x in sub]
        single=[1-float(x['robust-opt']['upper'])/float(x['best-single']['upper']) for x in sub]
        stats.append(dict(k=k,n=len(sub),gain_median=float(np.median(gain)),gain_min=float(np.min(gain)),gain_max=float(np.max(gain)),nominal_optimism_median=float(np.median(gap)),nominal_optimism_max=float(np.max(gap)),positive_optimism_median=float(np.median(posgap)),positive_optimism_max=float(np.max(posgap)),nominal_cost_median=float(np.median(price)),nominal_cost_max=float(np.max(price)),single_gain_median=float(np.median(single)),single_gain_max=float(np.max(single))))
    groups={}
    for r in read(p/'monte_carlo.csv'):
        key=(r['policy'],r['k'],r['mode']);groups.setdefault(key,[]).append(r)
    mc=[]
    for key,rr in groups.items():
        n=sum(int(r['blocks']) for r in rr);f=sum(int(r['failures']) for r in rr);q=f/n;exact=float(rr[0]['exact']);z=1.959963984540054
        center=(q+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/(1+z*z/n)
        mc.append(dict(policy=key[0],k=int(key[1]),mode=key[2],n=n,failures=f,estimate=q,exact=exact,lo=center-half,hi=center+half,covered=bool(center-half<=exact<=center+half),absolute_error=abs(q-exact),standardized_error=(q-exact)/np.sqrt(exact*(1-exact)/n)))
    with (p/'monte_carlo_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(mc[0]));w.writeheader();w.writerows(mc)
    front={}
    for r in read(p/'budget_frontiers.csv'):
        key=tuple(float(r[x]) for x in ['pa','pb','la','lb'])
        front.setdefault(key,{}).setdefault(r['policy'],[]).append(r)
    cert=[]
    for target in [.01,.05,.1]:
        nominal_feasible=0;robust_feasible=0;false=0;positive_false=0;extra=[]
        for key,pol in front.items():
            if key==(.1,.2,4.,2.):continue
            nominal=[r for r in pol['independent-opt'] if float(r['independent'])<=target]
            positive=[r for r in pol['positive-opt'] if float(r['positive'])<=target]
            robust=[r for r in pol['robust-opt'] if float(r['upper'])<=target]
            ni=min(nominal,key=lambda r:int(r['budget'])) if nominal else None
            po=min(positive,key=lambda r:int(r['budget'])) if positive else None
            ro=min(robust,key=lambda r:int(r['budget'])) if robust else None
            nominal_feasible+=ni is not None;robust_feasible+=ro is not None
            false+=ni is not None and float(ni['upper'])>target+1e-12
            positive_false+=po is not None and float(po['upper'])>target+1e-12
            if ni and ro:extra.append(int(ro['budget'])-int(ni['budget']))
        cert.append(dict(target=target,scenarios=81,nominal_feasible=nominal_feasible,nominal_unsupported=false,positive_unsupported=positive_false,robust_feasible=robust_feasible,extra_packets_median=float(np.median(extra)) if extra else None,extra_packets_max=max(extra) if extra else None))
    sr=read(p/'search_scaling.csv');hold=read(p/'holdout.csv')
    timing={}
    for r in read(p/'evaluator_timings.csv'):timing.setdefault(int(r['T']),[]).append(float(r['microseconds']))
    out=dict(grid=stats,aggregate_gain_median=float(np.median(allg)),aggregate_improved=int(sum(v>1e-10 for v in allg)),mc_groups=len(mc),mc_covered=sum(r['covered'] for r in mc),mc_max_absolute_error=max(r['absolute_error'] for r in mc),mc_max_standardized_error=max(abs(r['standardized_error']) for r in mc),certification=cert,heuristic_grid_optimal=sum(float(r['relative_gap'])<1e-10 for r in sr if int(r['T'])==8),heuristic_holdout_optimal=sum(float(r['relative_gap'])<1e-10 for r in hold),heuristic_holdout_max_gap=max(float(r['relative_gap']) for r in hold),evaluator_median_us={T:float(np.median(x)) for T,x in timing.items()})
    (p/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
