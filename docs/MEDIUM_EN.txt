# A Better Causal Estimator Still Fails When the Confounder Is Missing

*Fifty-four synthetic experiments in randomized assignment, observed confounding, and hidden confounding.*

A dataset can show a strong correlation between an action and an outcome while providing misleading evidence about the action's effect. This distinction matters in growth experiments, financial research, recommendation systems, and scientific decision-making.

I built **Causal Inference Lab** to test three situations with a known ground-truth answer.

## The design

Each simulation generates 1,400 observations. Individuals have two measured variables, and the true treatment effect varies with one of them. I compare three data-generating processes:

1. **Randomized assignment:** treatment is allocated independently of measured variables.
2. **Observed confounding:** treatment assignment depends on the two recorded covariates, which also affect the outcome.
3. **Hidden confounding:** treatment and outcomes additionally depend on an **unobserved** variable, absent from the estimator's inputs.

The true causal effect is known because I wrote the data-generating process. The mean oracle effect over 18 seeds was **1.79839**.

## Four estimators

The program computes a naive treated-minus-control mean, inverse probability weighting, outcome regression, and augmented inverse probability weighting (AIPW).

AIPW combines a treatment-propensity model with separate potential-outcome regression models. To avoid evaluating each nuisance model on the exact observations used to fit it, the project uses **three-fold cross-fitting**.

For each observation, the AIPW score is:

`m1(x) - m0(x) + t*(y-m1(x))/e(x) - (1-t)*(y-m0(x))/(1-e(x))`.

That score is valuable under identification and model-quality assumptions. The mathematical label "doubly robust" does not mean it can repair missing causal information.

## What happened

Across **18 seeds per scenario**, the mean estimates were:

| Data-generating process | Naive difference | Cross-fitted AIPW | Known oracle |
|---|---:|---:|---:|
| Randomized | 1.81065 | 1.78898 | 1.79839 |
| Observed confounding | 3.75019 | 1.82237 | 1.79839 |
| Hidden confounding | 5.13446 | 3.92773 | 1.79839 |

The lab passed **20/20 tests**.

*[Insert `results/ate_comparison.png` here.]*

The observed-confounding simulation demonstrates a case where conditioning on recorded variables sharply improves the estimate. But in the hidden-confounding case, AIPW still overshot the true effect by about **2.13 units**, on average.

The approximate 95% AIPW intervals covered the per-run simulated oracle in **17/18**, **17/18**, and **0/18** runs respectively. These counts are descriptive for only 18 synthetic replications and should not be generalized to empirical interval performance.

## Why the boundary matters

Statistical sophistication is not a substitute for identification. When a common cause of treatment and outcome is omitted, conditional exchangeability can fail. Cross-fitting reduces a particular form of estimation bias; it cannot create a missing variable or establish causation from arbitrary observational data.

The prototype also clips propensity scores to [0.05, 0.95], uses parametric nuisance models, and estimates standard errors from a simple influence-score approximation. Those choices deserve sensitivity analysis.

## Next research direction

I would introduce negative controls, instrumental-variable experiments, alternative propensity models, overlapping treatment diagnostics, sensitivity bounds, and independent benchmarks. Most importantly, I would define the intervention, outcome, treatment assignment mechanism, and identification strategy *before* estimating any effect in a real dataset.

**Code, tests, seeds, and complete synthetic results:** [INSERT VERIFIED PUBLIC GITHUB URL]

**Scope:** A controlled educational simulation, not causal identification in a real dataset.

References: [DoubleML Guide](https://docs.doubleml.org/stable/guide/guide.html) · [EconML Doubly Robust Learning](https://github.com/py-why/EconML/blob/main/doc/spec/estimation/dr.rst)

**Medium Topics:** Causal Inference · Data Science · Statistics · Machine Learning · Python
