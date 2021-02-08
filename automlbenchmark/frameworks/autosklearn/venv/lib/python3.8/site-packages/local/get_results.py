from collections import defaultdict
import glob
import json
import os

import hpolib.benchmarks.synthetic_functions.branin
import hpolib.benchmarks.synthetic_functions.bohachevsky
import hpolib.benchmarks.synthetic_functions.camelback
import hpolib.benchmarks.synthetic_functions.forrester
import hpolib.benchmarks.synthetic_functions.goldstein_price
import hpolib.benchmarks.synthetic_functions.hartmann3
import hpolib.benchmarks.synthetic_functions.hartmann6
import hpolib.benchmarks.synthetic_functions.levy
import hpolib.benchmarks.synthetic_functions.rosenbrock
import hpolib.benchmarks.synthetic_functions.sin_one
import hpolib.benchmarks.synthetic_functions.sin_two
import joblib
import numpy as np
import pandas as pd
import scipy.stats

pd.set_option('display.width', 999)
pd.set_option('display.max_columns', 999)

benchmarks = (
    hpolib.benchmarks.synthetic_functions.bohachevsky.Bohachevsky(),
    hpolib.benchmarks.synthetic_functions.branin.Branin(),
    hpolib.benchmarks.synthetic_functions.camelback.Camelback(),
    hpolib.benchmarks.synthetic_functions.forrester.Forrester(),
    hpolib.benchmarks.synthetic_functions.goldstein_price.GoldsteinPrice(),
    hpolib.benchmarks.synthetic_functions.hartmann3.Hartmann3(),
    hpolib.benchmarks.synthetic_functions.hartmann6.Hartmann6(),
    hpolib.benchmarks.synthetic_functions.levy.Levy1D(),
    hpolib.benchmarks.synthetic_functions.levy.Levy2D(),
    hpolib.benchmarks.synthetic_functions.rosenbrock.Rosenbrock2D(),
    hpolib.benchmarks.synthetic_functions.rosenbrock.Rosenbrock5D(),
    #hpolib.benchmarks.synthetic_functions.rosenbrock.Rosenbrock10D(),
    hpolib.benchmarks.synthetic_functions.sin_one.SinOne(),
    hpolib.benchmarks.synthetic_functions.sin_two.SinTwo(),
)


results = defaultdict(lambda: defaultdict(list))

paths = glob.glob('/home/feurerm/projects/smac3test/*/*/*')
for path in paths:
    splitpath = path.split('/')
    benchmark_name = splitpath[-3]
    competitor = splitpath[-2]
    seed = int(splitpath[-1])

    if competitor not in (
        #'daq_marginalized',
        #'default',
        'default_fixed_acq',
        'daq_lhd-maximin1000',
        'daq_lhd-center1000',
        'daq_lhd-centermaximin1000',
        #'default_fixed_acq_2',
        #'default_rf', 'default_rf_fixed_acq', 'default_rf_fixed_acq_2',
        #'a'
        #'salinas',
        #'salinas_fixed_acq',
        #'salinas_fixed_acq_2',
        #'default_emcee',
        #'default_nuts',
    ):
        continue

    hit = False
    for benchmark in benchmarks:
        if benchmark_name == benchmark.get_meta_information()['name']:
            hit = True

            runhistory_path = os.path.join(path, 'run_%d' % seed, 'runhistory.json')
            if not os.path.exists(runhistory_path):
                print(benchmark_name, competitor, seed)
            else:
                with open(runhistory_path) as fh:
                    jason = json.load(fh)
                    if len(jason['configs']) != 50:
                        print(benchmark_name, competitor, seed, len(jason['configs']))

            with open(os.path.join(path, 'run_%d' % seed, 'traj.json')) as fh:
                lines = fh.read().strip().split('\n')
            jason = json.loads(lines[-1].strip())
            result = jason['cost']
            optimum = benchmark.get_meta_information()['f_opt']
            regret = result - optimum + 1e-14
            results[benchmark_name][competitor].append(regret)

    if not hit:
        raise ValueError((benchmark_name, path))

for key1 in results:
    for key2 in results[key1]:
        results[key1][key2] = np.log10(np.mean(results[key1][key2]) + 1e-7)

results = pd.DataFrame(results)
results = results[sorted(results.columns)]
ranks = defaultdict(list)
for i, benchmark_name in enumerate(results.columns):
    rank = scipy.stats.rankdata(results.values[:, i])
    for j, method_name in enumerate(results.index):
        ranks[method_name].append(rank[j])
results['Rank'] = pd.Series()
for key in ranks:
    ranks[key] = np.mean(ranks[key])
    results['Rank'][key] = ranks[key]
results['Average'] = pd.Series()
for key in results.index:
    results['Average'][key] = np.nanmean(results.loc[key].values)
print(results)
