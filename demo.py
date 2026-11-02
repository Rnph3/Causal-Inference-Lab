import json,csv,math,statistics
from pathlib import Path
from causal import simulate,evaluate
out=Path('results');out.mkdir(exist_ok=True)
all_results=[]
for mode in ['randomized','observed','hidden']:
    for seed in range(18):
        x=evaluate(simulate(n=1400,seed=seed,mode=mode),seed=900+seed,folds=3)
        all_results.append({'seed':seed,**x})
with (out/'all_runs.csv').open('w',newline='') as f:
    # serialize vector fold counts as JSON, portable for CSV
    columns=[k for k in all_results[0] if k!='fold_counts']
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r[k] for k in columns} for r in all_results)
summary={'scope':'synthetic data with known oracle causal effect, no real-world causal identification',
         'n_per_run':1400,'seeds_per_scenario':18,'folds':3,'scenarios':{}}
for mode in ['randomized','observed','hidden']:
    group=[r for r in all_results if r['mode']==mode]
    summary['scenarios'][mode]={
        'mean_sample_true_ate':round(statistics.mean(r['sample_true_ate'] for r in group),5),
        'mean_naive_estimate':round(statistics.mean(r['naive_ate'] for r in group),5),
        'mean_aipw_estimate':round(statistics.mean(r['aipw_ate'] for r in group),5),
        'naive_bias':round(statistics.mean(r['naive_ate']-r['sample_true_ate'] for r in group),5),
        'aipw_bias':round(statistics.mean(r['aipw_error'] for r in group),5),
        'naive_rmse':round(math.sqrt(statistics.mean((r['naive_ate']-r['sample_true_ate'])**2 for r in group)),5),
        'aipw_rmse':round(math.sqrt(statistics.mean(r['aipw_error']**2 for r in group)),5),
        'aipw_interval_coverage_fraction':round(statistics.mean(r['oracle_in_ci'] for r in group),5),
        'mean_fraction_propensity_clipped':round(statistics.mean(r['fraction_clipped'] for r in group),5)
    }
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(out/'example_run.json').write_text(json.dumps(all_results[19],indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
