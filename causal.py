"""Explicitly synthetic causal inference benchmark for ATE identification limits.
Requires numpy and scikit-learn; no real individuals or interventions.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

@dataclass(frozen=True)
class CausalData:
    x: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    sample_true_ate: float
    p_true: np.ndarray
    mu0_true: np.ndarray
    mu1_true: np.ndarray
    mode: str


def simulate(n=1800, seed=0, mode='observed'):
    if n < 100: raise ValueError('n must be >=100')
    if mode not in ('observed','randomized','hidden'):
        raise ValueError('mode not recognized')
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,2)); hidden=rng.normal(size=n)
    tau=1.8+0.6*x[:,0]
    logits=-0.3+1.1*x[:,0]-0.8*x[:,1]
    if mode=='hidden': logits+=1.4*hidden
    propensity=expit(logits) if mode!='randomized' else np.full(n,0.5)
    treatment=rng.binomial(1,propensity)
    # 'mu' here is expected outcome conditional on BOTH x and hidden
    mu0=0.5+1.4*x[:,0]-0.9*x[:,1]+0.4*x[:,0]**2
    if mode=='hidden':mu0+=2.0*hidden
    mu1=mu0+tau
    outcome=mu0+treatment*tau+rng.normal(size=n)
    return CausalData(x,treatment,outcome,float(tau.mean()),propensity,mu0,mu1,mode)


def estimate_from_nuisance(y,t,m0,m1,p,clip=0.05):
    """AIPW influence score. Clipping is explicitly a sensitivity choice."""
    y,t,m0,m1,p=(np.asarray(a) for a in (y,t,m0,m1,p))
    if not(y.shape==t.shape==m0.shape==m1.shape==p.shape):raise ValueError('shape mismatch')
    if np.any((t!=0)&(t!=1)):raise ValueError('binary treatment required')
    if not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('invalid propensities')
    if not 0<clip<.5:raise ValueError('clip must be (0,0.5)')
    p=np.clip(p,clip,1-clip)
    return m1-m0 +t*(y-m1)/p -(1-t)*(y-m0)/(1-p)


def crossfit_nuisance(data: CausalData, folds=3,seed=777,clip=.05):
    x,t,y=data.x,data.treatment,data.outcome
    if folds<2 or folds>len(y)//20:raise ValueError('invalid number of folds')
    polynomial=PolynomialFeatures(degree=2,include_bias=False)
    features=polynomial.fit_transform(x)
    m0=np.empty(len(y));m1=np.empty(len(y));prop=np.empty(len(y));foldid=np.empty(len(y),dtype=int)
    for fold,(tr,te) in enumerate(KFold(n_splits=folds,shuffle=True,random_state=seed).split(x)):
        if not (set(np.unique(t[tr]))=={0,1}):raise ValueError('both treatment groups needed in training')
        clf=LogisticRegression(C=100,max_iter=1000,solver='lbfgs')
        clf.fit(x[tr],t[tr]);prop[te]=clf.predict_proba(x[te])[:,1]
        for target,out in ((0,m0),(1,m1)):
            model=Ridge(alpha=0.01)
            model.fit(features[tr][t[tr]==target],y[tr][t[tr]==target])
            out[te]=model.predict(features[te])
        foldid[te]=fold
    return m0,m1,prop,foldid


def evaluate(data: CausalData, seed=777, folds=3,clip=.05):
    t,y=data.treatment,data.outcome
    if t.sum()==0 or t.sum()==len(t): raise ValueError('both groups required')
    m0,m1,p,folds_used=crossfit_nuisance(data,folds=folds,seed=seed,clip=clip)
    pp=np.clip(p,clip,1-clip)
    aipw=estimate_from_nuisance(y,t,m0,m1,p,clip)
    ipw=t*y/pp-(1-t)*y/(1-pp)
    naive=float(y[t==1].mean()-y[t==0].mean())
    standard_error=float(np.std(aipw,ddof=1)/np.sqrt(len(y)))
    v=float(aipw.mean())
    return {'mode':data.mode,'n':len(y),'sample_true_ate':data.sample_true_ate,
            'naive_ate':naive,'ipw_ate':float(ipw.mean()),
            'outcome_regression_ate':float((m1-m0).mean()),
            'aipw_ate':v,'aipw_error':v-data.sample_true_ate,
            'aipw_se':standard_error,'aipw_ci_low':v-1.96*standard_error,
            'aipw_ci_high':v+1.96*standard_error,
            'oracle_in_ci':bool(v-1.96*standard_error <= data.sample_true_ate <= v+1.96*standard_error),
            'observed_treatment_rate':float(t.mean()),
            'min_predicted_propensity':float(p.min()),
            'max_predicted_propensity':float(p.max()),
            'fraction_clipped':float(((p<clip)|(p>1-clip)).mean()),
            'fold_counts':[int((folds_used==i).sum()) for i in range(folds)]}
