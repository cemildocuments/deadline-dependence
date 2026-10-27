#!/usr/bin/env python3
"""Regenerate experiments and compare numerical results with reference tables."""
from pathlib import Path
import argparse,csv,json,math,re,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
IGNORED={'seconds','microseconds','elapsed_seconds'}
FILES=['benchmark.csv','illustration.csv','budget_frontiers.csv','radius_sensitivity.csv','monte_carlo.csv','monte_carlo_summary.csv','search_scaling.csv','holdout.csv']

def compare(a,b):
    with a.open() as f:aa=list(csv.DictReader(f))
    with b.open() as f:bb=list(csv.DictReader(f))
    assert len(aa)==len(bb),(a.name,'row count')
    for i,(x,y) in enumerate(zip(aa,bb)):
        assert x.keys()==y.keys(),(a.name,'columns')
        for k in x:
            if k in IGNORED:continue
            if x[k]==y[k]:continue
            if re.fullmatch(r'-?\d+', x[k]) and re.fullmatch(r'-?\d+', y[k]):
                raise AssertionError((a.name,i,k,'integer mismatch',x[k],y[k]))
            try:u=float(x[k]);v=float(y[k])
            except ValueError:raise AssertionError((a.name,i,k,x[k],y[k]))
            assert math.isclose(u,v,rel_tol=1e-10,abs_tol=1e-13),(a.name,i,k,u,v)
    return len(aa)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);args=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/'tests/validate.py')],check=True,cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='deadline-reproduction-') as d:
        out=Path(d)
        for script in ['run_experiments.py','run_search_scaling.py','run_holdout.py']:
            subprocess.run([sys.executable,str(ROOT/script),'--workers',str(args.workers),'--output',str(out)],check=True,cwd=ROOT)
        subprocess.run([sys.executable,str(ROOT/'summarize.py'),'--results',str(out)],check=True,cwd=ROOT)
        checked={name:compare(ROOT/'results'/name,out/name) for name in FILES}
    report={'status':'passed','checked_csv_rows':checked,'probability_relative_tolerance':1e-10,'probability_absolute_tolerance':1e-13,'integer_fields':'exact comparison','timing_columns':'machine-dependent'}
    (ROOT/'results/reproduction_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
