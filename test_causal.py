import unittest
import numpy as np
from causal import simulate,estimate_from_nuisance,crossfit_nuisance,evaluate
class TestCausal(unittest.TestCase):
    def test_invalid_mode(self):
        with self.assertRaises(ValueError):simulate(mode='wrong')
    def test_small_n(self):
        with self.assertRaises(ValueError):simulate(n=50)
    def test_repeatability(self):
        a,b=simulate(n=500,seed=5),simulate(n=500,seed=5)
        np.testing.assert_allclose(a.outcome,b.outcome)
        np.testing.assert_array_equal(a.treatment,b.treatment)
    def test_true_ate_known(self):
        d=simulate(n=15000,seed=1);self.assertAlmostEqual(d.sample_true_ate,1.8,delta=.02)
    def test_true_propensity_range(self):
        d=simulate(n=1000);self.assertTrue(np.all((d.p_true>0)&(d.p_true<1)))
    def test_randomized_propensity(self):
        d=simulate(n=500,mode='randomized');np.testing.assert_array_equal(d.p_true,np.full(500,.5))
    def test_expected_outcome_diff(self):
        d=simulate(n=500,seed=7);np.testing.assert_allclose(d.mu1_true-d.mu0_true,1.8+.6*d.x[:,0])
    def test_nuisance_shape(self):
        d=simulate(n=500);arrays=crossfit_nuisance(d)
        for a in arrays:self.assertEqual(len(a),500)
    def test_nuisance_folds_all_used(self):
        d=simulate(n=510);_,_,_,ids=crossfit_nuisance(d);self.assertEqual(set(ids),{0,1,2})
    def test_invalid_folds(self):
        with self.assertRaises(ValueError):crossfit_nuisance(simulate(n=100),folds=1)
    def test_perfect_outcomes_stable_for_wrong_propensity(self):
        d=simulate(n=1000,seed=2)
        # use Y set equal to conditional mean: noiseless potential outcomes
        y=d.mu0_true+d.treatment*(d.mu1_true-d.mu0_true)
        score=estimate_from_nuisance(y,d.treatment,d.mu0_true,d.mu1_true,np.full(1000,.8))
        self.assertAlmostEqual(score.mean(),d.sample_true_ate,places=10)
    def test_misspecified_propensity_bounds(self):
        with self.assertRaises(ValueError):estimate_from_nuisance(np.ones(3),np.ones(3),np.ones(3),np.ones(3),np.array([-.1,.2,.3]))
    def test_invalid_treatment(self):
        with self.assertRaises(ValueError):estimate_from_nuisance(np.ones(2),np.array([0,2]),np.ones(2),np.ones(2),np.ones(2)*.5)
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):estimate_from_nuisance(np.ones(3),np.ones(3),np.ones(3),np.ones(2),np.ones(3))
    def test_invalid_clip(self):
        with self.assertRaises(ValueError):estimate_from_nuisance(np.ones(2),np.ones(2),np.ones(2),np.ones(2),np.ones(2)*.5,clip=0)
    def test_result_fields(self):
        e=evaluate(simulate(n=500,seed=0));self.assertTrue(e['aipw_ci_low']<e['aipw_ci_high'])
    def test_propensity_clipping(self):
        d=simulate(n=500);p=np.array([0.,1.]);t=np.array([1,0]);y=np.array([2.,3.]);a=estimate_from_nuisance(y,t,np.zeros(2),np.ones(2),p)
        self.assertTrue(np.all(np.isfinite(a)))
    def test_naive_bias_observed(self):
        errs=[]
        for seed in range(4):
            r=evaluate(simulate(n=1000,seed=seed,mode='observed'))
            errs.append(r['naive_ate']-r['sample_true_ate'])
        self.assertGreater(np.mean(errs),.4)
    def test_aipw_recovers_observed(self):
        errs=[]
        for seed in range(4):
            r=evaluate(simulate(n=1000,seed=seed,mode='observed'))
            errs.append(abs(r['aipw_ate']-r['sample_true_ate']))
        self.assertLess(np.mean(errs),.2)
    def test_hidden_confounder_not_magically_removed(self):
        errs=[]
        for seed in range(4):
            r=evaluate(simulate(n=1000,seed=seed,mode='hidden'))
            errs.append(r['aipw_ate']-r['sample_true_ate'])
        self.assertGreater(np.mean(errs),.3)
if __name__=='__main__':unittest.main()
