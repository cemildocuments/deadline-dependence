#!/usr/bin/env python3
"""Independent occupancy LP, path enumeration, and analytical boundary checks."""
from pathlib import Path
import sys,json,itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from model import risk,marginal,witness


def lp_bound(seq,k,pa,pb,ta,tb,lam=1.,maximize=True):
    T=len(seq);C=k+1
    nx=T*C*4;nf=max(T-1,0)*C*16;n=nx+nf
    def x(t,c,s):return (t*C+c)*4+s
    def f(t,c,s,z):return nx+((t*C+c)*4+s)*4+z
    eq=[];beq=[];ineq=[];bineq=[]
    def add(row,b=0):eq.append(row);beq.append(b)
    add({x(0,0,s):1 for s in range(4)},1)
    add({x(0,0,2):1,x(0,0,3):1},pa)
    add({x(0,0,1):1,x(0,0,3):1},pb)
    q=pa*pb;lo=q+lam*(max(0,pa+pb-1)-q);hi=q+lam*(min(pa,pb)-q)
    ineq.extend([{x(0,0,3):1},{x(0,0,3):-1}]);bineq.extend([hi,-lo])
    bounds=[(0,None)]*n
    for c in range(1,C):
        for s in range(4):bounds[x(0,c,s)]=(0,0)
    for t in range(T-1):
        incoming={(c,z):{x(t+1,c,z):1} for c in range(C) for z in range(4)}
        for c in range(C):
            for s in range(4):
                a=ta[s//2];b=tb[s%2];v=x(t,c,s)
                row={v:-1};row.update({f(t,c,s,z):1 for z in range(4)});add(row)
                add({v:-a,f(t,c,s,2):1,f(t,c,s,3):1})
                add({v:-b,f(t,c,s,1):1,f(t,c,s,3):1})
                q=a*b;lo=q+lam*(max(0,a+b-1)-q);hi=q+lam*(min(a,b)-q)
                ineq.extend([{f(t,c,s,3):1,v:-hi},{f(t,c,s,3):-1,v:lo}]);bineq.extend([0,0])
                gain=(int(seq[t])&1)*(1-s//2)+((int(seq[t])>>1)&1)*(1-s%2)
                nxt=min(k,c+gain)
                for z in range(4):incoming[nxt,z][f(t,c,s,z)]=-1
        for row in incoming.values():add(row)
    objective=np.zeros(n)
    for c in range(C):
        for s in range(4):
            gain=(int(seq[-1])&1)*(1-s//2)+((int(seq[-1])>>1)&1)*(1-s%2)
            objective[x(T-1,c,s)]=float(c+gain<k)
    def mat(rows):
        a=lil_matrix((len(rows),n))
        for i,r in enumerate(rows):
            for j,v in r.items():a[i,j]=v
        return a.tocsr()
    ans=linprog(-objective if maximize else objective,A_ub=mat(ineq),b_ub=bineq,A_eq=mat(eq),b_eq=beq,bounds=bounds,method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not ans.success:raise RuntimeError(ans.message)
    return float(objective@ans.x)


def enumerate_paths(seq,k,pa,pb,ta,tb):
    p0=np.array([(1-pa)*(1-pb),(1-pa)*pb,pa*(1-pb),pa*pb]);P=np.zeros((4,4))
    for s in range(4):
        a=ta[s//2];b=tb[s%2];P[s]=[(1-a)*(1-b),(1-a)*b,a*(1-b),a*b]
    fail=0.
    for path in itertools.product(range(4),repeat=len(seq)):
        prob=p0[path[0]];received=0
        for t,s in enumerate(path):
            if t:prob*=P[path[t-1],s]
            received+=(int(seq[t])&1)*(1-s//2)+((int(seq[t])>>1)&1)*(1-s%2)
        if received<k:fail+=prob
    return fail


def forward_witness(seq,k,pa,pb,ta,tb,maximize=True):
    _,q,policy,_=witness(seq,k,pa,pb,ta,tb,maximize=maximize)
    mass=np.zeros((k+1,4));mass[0]=[1-pa-pb+q,pb-q,pa-q,q]
    for t,ac in enumerate(seq):
        nxtmass=np.zeros_like(mass)
        for c in range(k+1):
            for s in range(4):
                gain=(int(ac)&1)*(1-s//2)+((int(ac)>>1)&1)*(1-s%2);d=min(k,c+gain)
                if t==len(seq)-1:nxtmass[d,s]+=mass[c,s]
                else:
                    a=ta[s//2];b=tb[s%2];q=policy[t,c,s]
                    nxtmass[d]+=mass[c,s]*np.array([1-a-b+q,b-q,a-q,q])
        mass=nxtmass
        assert abs(mass.sum()-1)<2e-12
    return mass[:k].sum()


def main():
    rng=np.random.default_rng(71731);maxlp=0.;maxenumeration=0.;maxwitness=0.
    for i in range(72):
        T=1+i%6;k=1+i%max(1,2*T);seq=rng.integers(0,4,T,dtype=np.int8)
        pa,pb=rng.uniform(.01,.35,2);la,lb=rng.uniform(1.1,10,2);ta=marginal(pa,la);tb=marginal(pb,lb)
        lam=[0.,.35,1.][i%3]
        for maximize,mode in [(True,3),(False,4)]:
            exact=risk(seq,k,pa,pb,ta,tb,mode,lam);lp=lp_bound(seq,k,pa,pb,ta,tb,lam,maximize)
            maxlp=max(maxlp,abs(lp-exact));assert abs(lp-exact)<1e-8
        lo=risk(seq,k,pa,pb,ta,tb,4);hi=risk(seq,k,pa,pb,ta,tb,3);ind=risk(seq,k,pa,pb,ta,tb,0)
        assert -1e-12<=lo<=ind+1e-12 and ind<=hi+1e-12<=1+2e-12
        rads=[risk(seq,k,pa,pb,ta,tb,3,l) for l in np.linspace(0,1,11)]
        assert np.min(np.diff(rads))>=-1e-12
        for maximize,mode in [(True,3),(False,4)]:
            diff=abs(forward_witness(seq,k,pa,pb,ta,tb,maximize)-risk(seq,k,pa,pb,ta,tb,mode));maxwitness=max(maxwitness,diff);assert diff<1e-12
    # All 256 schedules for a four-slot instance, independently enumerate all 256 channel paths.
    pa,pb=.13,.27;ta=marginal(pa,3.2);tb=marginal(pb,1.7)
    for tup in itertools.product(range(4),repeat=4):
        seq=np.array(tup,dtype=np.int8)
        for k in [1,3,5]:
            diff=abs(enumerate_paths(seq,k,pa,pb,ta,tb)-risk(seq,k,pa,pb,ta,tb,0));maxenumeration=max(maxenumeration,diff);assert diff<2e-12
    # Closed-form counterexample and single-path invariance.
    seq=np.array([3],dtype=np.int8)
    assert abs(risk(seq,2,.1,.1,ta,tb,3)-.2)<1e-14
    assert abs(risk(seq,2,.1,.1,ta,tb,2)-.1)<1e-14
    for path in [1,2]:
        seq=np.full(9,path,dtype=np.int8)
        vals=[risk(seq,4,pa,pb,ta,tb,m) for m in range(5)]
        assert max(vals)-min(vals)<1e-12
    # Exhaustive local MDS decoding check over a prime field using Vandermonde columns.
    prime=257;K=4;N=8
    G=np.array([[pow(j+1,i,prime) for j in range(N)] for i in range(K)],dtype=np.int64)
    def rankmod(A):
        A=A.copy();r=0
        for c in range(A.shape[1]):
            piv=next((j for j in range(r,len(A)) if A[j,c]%prime),None)
            if piv is None:continue
            A[[r,piv]]=A[[piv,r]];A[r]=(A[r]*pow(int(A[r,c]),-1,prime))%prime
            for j in range(len(A)):
                if j!=r:A[j]=(A[j]-A[j,c]*A[r])%prime
            r+=1
            if r==len(A):break
        return r
    for cols in itertools.combinations(range(N),K):assert rankmod(G[:,cols])==K
    result=dict(lp_instances=144,max_lp_error=maxlp,path_enumeration_comparisons=768,max_path_error=maxenumeration,witness_comparisons=144,max_witness_error=maxwitness,mds_subsets=70,status='passed')
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
