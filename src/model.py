"""Exact deadline failure probabilities for two causally coupled erasure chains.

State 2*x+y records path erasures (0=success, 1=erasure).
An action is a bit mask: 0=idle, 1=A, 2=B, 3=both, with distinct MDS symbols.
mode: 0 independent, 1 lower endpoint, 2 upper endpoint, 3 worst, 4 best.
"""
from itertools import combinations
import numpy as np
from numba import njit


def marginal(p, burst):
    """Stationary erasure probability p and mean erasure-run duration burst."""
    beta=1.0/burst
    alpha=p*beta/(1.0-p)
    if not (0<=alpha<=1 and 0<=beta<=1):
        raise ValueError('Invalid two-state chain')
    return np.array([alpha,1-beta],dtype=np.float64)

@njit(cache=True)
def expectation(v,a,b,mode,lam):
    q0=a*b
    lo=q0+lam*(max(0.,a+b-1.)-q0)
    hi=q0+lam*(min(a,b)-q0)
    delta=v[0]-v[1]-v[2]+v[3]
    q=q0
    if mode==1: q=lo
    elif mode==2: q=hi
    elif mode==3: q=hi if delta>=0 else lo
    elif mode==4: q=lo if delta>=0 else hi
    return (1-a-b+q)*v[0]+(b-q)*v[1]+(a-q)*v[2]+q*v[3]

@njit(cache=True)
def risk(schedule,k,pa,pb,ta,tb,mode=3,lam=1.0):
    T=len(schedule)
    V=np.ones((k+1,4),dtype=np.float64)
    V[k,:]=0.
    for t in range(T-1,-1,-1):
        W=np.zeros((k+1,4),dtype=np.float64)
        ac=schedule[t]
        for c in range(k):
            for s in range(4):
                x=s//2; y=s%2
                gain=(ac&1)*(1-x)+((ac>>1)&1)*(1-y)
                nxt=min(k,c+gain)
                if t==T-1:
                    W[c,s]=1. if nxt<k else 0.
                else:
                    W[c,s]=expectation(V[nxt],ta[x],tb[y],mode,lam)
        V=W
    return expectation(V[0],pa,pb,mode,lam)

@njit(cache=True)
def evaluate(schedules,k,pa,pb,ta,tb,lam=1.):
    out=np.empty((len(schedules),4))
    for i in range(len(schedules)):
        for j,mode in enumerate((0,2,3,4)):
            out[i,j]=risk(schedules[i],k,pa,pb,ta,tb,mode,lam)
    return out


def schedules(T,budget):
    rows=[]
    for selected in combinations(range(2*T),budget):
        a=np.zeros(T,dtype=np.int8)
        for i in selected: a[i//2] |= 1<<(i%2)
        rows.append(a)
    return np.asarray(rows)


def schedule_string(s):
    return ''.join('.ABX'[int(x)] for x in s)


def witness(schedule,k,pa,pb,ta,tb,lam=1.,maximize=True):
    """Backward values and attaining causal transition-coupling policy."""
    T=len(schedule); V=np.ones((T+1,k+1,4));V[:,k,:]=0
    choices=np.zeros((T,k+1,4))
    def pick(v,a,b):
        q0=a*b;lo=q0+lam*(max(0,a+b-1)-q0);hi=q0+lam*(min(a,b)-q0)
        d=v[0]-v[1]-v[2]+v[3]
        q=hi if ((d>=0)==maximize) else lo
        return np.dot([1-a-b+q,b-q,a-q,q],v),q
    for t in range(T-1,-1,-1):
        for c in range(k+1):
            for s in range(4):
                x,y=divmod(s,2)
                gain=(int(schedule[t])&1)*(1-x)+((int(schedule[t])>>1)&1)*(1-y)
                nxt=min(k,c+gain)
                if t==T-1:V[t,c,s]=float(nxt<k)
                else: V[t,c,s],choices[t,c,s]=pick(V[t+1,nxt],ta[x],tb[y])
    val,qinit=pick(V[0,0],pa,pb)
    return val,qinit,choices,V


def simulate(schedule,k,pa,pb,ta,tb,n,seed,mode='worst',lam=1.):
    """Independent vectorized forward simulation; mode worst attains DP bound."""
    rng=np.random.default_rng(seed)
    if mode=='worst': _,qinit,choices,_=witness(schedule,k,pa,pb,ta,tb,lam)
    elif mode=='independent': qinit=pa*pb
    elif mode=='positive': qinit=pa*pb+lam*(min(pa,pb)-pa*pb)
    else:raise ValueError(mode)
    u=rng.random(n)
    s=(u>=1-pa-pb+qinit).astype(np.int8)+(u>=1-pa).astype(np.int8)+(u>=1-qinit).astype(np.int8)
    count=np.zeros(n,dtype=np.int16)
    for t,act in enumerate(schedule):
        prev=count.copy()
        count=np.minimum(k,count+(int(act)&1)*(1-s//2)+((int(act)>>1)&1)*(1-s%2))
        if t==len(schedule)-1: break
        a=ta[s//2];b=tb[s%2]
        if mode=='worst':q=choices[t,prev,s]
        elif mode=='independent':q=a*b
        else:q=a*b+lam*(np.minimum(a,b)-a*b)
        u=rng.random(n)
        s=(u>=1-a-b+q).astype(np.int8)+(u>=1-a).astype(np.int8)+(u>=1-q).astype(np.int8)
    return int(np.sum(count<k))

@njit(cache=True)
def exchange_search(starts,k,pa,pb,ta,tb,lam=1.):
    """Best-improvement one-packet exchanges from fixed deterministic starts.

    Returns a local optimum, not a global-optimality certificate.
    """
    best=starts[0].copy(); bestvalue=2.; evaluations=0; moves=0
    T=starts.shape[1]
    for ii in range(len(starts)):
        seq=starts[ii].copy();value=risk(seq,k,pa,pb,ta,tb,3,lam);evaluations+=1
        while True:
            nextvalue=value;remove=-1;add=-1
            for i in range(2*T):
                ti=i//2;bi=1<<(i%2)
                if seq[ti]&bi==0:continue
                for j in range(2*T):
                    tj=j//2;bj=1<<(j%2)
                    if seq[tj]&bj:continue
                    seq[ti]^=bi;seq[tj]^=bj
                    z=risk(seq,k,pa,pb,ta,tb,3,lam);evaluations+=1
                    seq[tj]^=bj;seq[ti]^=bi
                    if z<nextvalue-1e-13:nextvalue=z;remove=i;add=j
            if remove<0:break
            seq[remove//2]^=1<<(remove%2);seq[add//2]^=1<<(add%2)
            value=nextvalue;moves+=1
        if value<bestvalue-1e-13:bestvalue=value;best=seq.copy()
    return best,bestvalue,evaluations,moves


def starting_schedules(T):
    """Five deterministic starts, each with T transmissions (T must be even)."""
    if T%2:raise ValueError('Even horizon required')
    pair=np.zeros(T,dtype=np.int8);pair[np.round(np.linspace(0,T-1,T//2)).astype(int)]=3
    return np.array([np.ones(T,dtype=np.int8),np.full(T,2,dtype=np.int8),np.array([1,2]*(T//2),dtype=np.int8),np.r_[np.full(T//2,3,dtype=np.int8),np.zeros(T//2,dtype=np.int8)],pair])
