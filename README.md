# Causal Inference Lab (P11)

**What it proves:** reproducible comparison of naive group differences versus cross-fitted AIPW (augmented inverse probability weighting) on **synthetic data where true average treatment effect is known**. Demonstrates a key failure mode: controlling for *observed* confounders does not eliminate bias from an omitted confounder.

**What it does NOT prove:** causal effects in real-world markets, business products, medicine, or any observational dataset without an appropriate identification design.

## Run

Python 3.10+, NumPy, SciPy, scikit-learn.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python demo.py
```

The demo runs 18 seeds per scenario, N=1,400 per seed, 3 cross-fitting folds. Use seeded experiments for reproducibility; library versions can affect small numerical differences.

## Measured results

| Scenario | Known effect (mean oracle) | Naive estimate | Cross-fitted AIPW | AIPW bias | AIPW 95% interval empirical coverage |
|---|---:|---:|---:|---:|---:|
| Randomized | 1.79839 | 1.81065 | 1.78898 | −0.00942 | 17/18 |
| Observed confounding | 1.79839 | 3.75019 | 1.82237 | +0.02398 | 17/18 |
| Hidden confounding | 1.79839 | 5.13446 | 3.92773 | +2.12933 | 0/18 |

**20/20 unit tests PASS.** Coverage is the fraction of 18 synthetic replications whose approximate AIPW confidence interval contains the replication's sample oracle effect. Do not extrapolate that number to real-world coverage.

## Math

For observed outcome Y, treatment T∈{0,1}, estimated propensity `e(X)`, outcome models `m_t(X)`:

`psi = m1(X) - m0(X) + T*(Y-m1(X))/e(X) - (1-T)*(Y-m0(X))/(1-e(X))`.

ATE = sample mean of the cross-fitted score. CI here uses `±1.96 × std(psi)/sqrt(n)`, an approximate influence-score standard error under assumptions; clipping to `[0.05,0.95]` changes the estimator and can introduce bias.

`crossfit_nuisance` trains 3 folds with disjoint train/test data. Logistic regression models treatment propensity; separate degree-two Ridge outcome models estimate potential outcomes for treated and control groups. The hidden scenario has an omitted variable that affects both treatment and outcome; this invalidates conditional exchangeability based on X alone.

## Files

- `causal.py`: three data-generating processes; cross fitting, IPW, outcome regression, AIPW, diagnostics.
- `demo.py`: 54 runs and aggregate bias/RMSE/interval coverage.
- `test_causal.py`: 20 unit tests.
- `results/all_runs.csv`, `results/summary.json`, `results/example_run.json`, `results/ate_comparison.png`.

## Scope and limitations

Simulated scalar outcomes; no real intervention or verified causal identification. Model specification is deliberately simple, results depend on sampling, positivity, nuisance learning, clipping and the unknown-confounder assumption. Confidence intervals are approximate and were not evaluated beyond the chosen simulation.

## References

- https://docs.doubleml.org/stable/guide/guide.html
- https://github.com/py-why/EconML/blob/main/doc/spec/estimation/dr.rst

Independent mechanism-level implementation, not a benchmark of DoubleML/EconML packages.
