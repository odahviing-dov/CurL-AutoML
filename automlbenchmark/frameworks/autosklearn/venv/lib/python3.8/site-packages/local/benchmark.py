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

from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.runhistory.runhistory2epm import (
    RunHistory2EPM4Cost,
    #RunHistory2EPM4QuantileTransformedCost,
    #RunHistory2EPM4GaussianCopula,
    #RunHistory2EPM4GaussianCopulaCorrect,
    #RunHistory2EPM4GaussianCopulaTurbo,
    #RunHistory2EMPCopulaOriginal,
    RunHistory2EPM4LogScaledCost,
)

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


def evaluate(seed, benchmark, rh2epm, rh2epm_name):
    benchmark_name = benchmark.get_meta_information()['name']

    output_dir = "/home/feurerm/projects/smac3test/%s/%s/%d" % (benchmark_name, rh2epm_name, seed)

    if os.path.exists(output_dir):
        print('Output directory %s exists - skip' % output_dir)
        return

    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 50,
                         # max. number of function evaluations; for this example set to a low number
                         "cs": benchmark.get_configuration_space(),  # configuration space
                         "deterministic": "true",
                         'limit_resources': False,
                         "output_dir": output_dir,
                         })

    import local.treegp

    smac = SMAC4BO(scenario=scenario,
                   rng=seed,
                   tae_runner=benchmark,
                   runhistory2epm=rh2epm,
                   model_type='gp',
                   initial_design=LHDesign,
                   )

    smac.optimize()


jobs = []

for seed in range(10):
    for benchmark in benchmarks:
        for rh2epm, rh2epm_name in (
            (RunHistory2EPM4Cost, 'daq_lhd-centermaximin1000'),
            #(RunHistory2EPM4QuantileTransformedCost, 'sklearn_nonorm'),
            #(RunHistory2EPM4GaussianCopula, 'marius_nonorm'),
            #(RunHistory2EPM4GaussianCopulaCorrect, 'salinas_fixed_acq'),
            #(RunHistory2EPM4GaussianCopulaTurbo, 'turbo'),
            #(RunHistory2EMPCopulaOriginal, 'salinas_original'),
        ):
            jobs.append((seed, benchmark, rh2epm, rh2epm_name))


joblib.Parallel(n_jobs=4, backend='multiprocessing', batch_size=1)(joblib.delayed(evaluate)(*job) for job in jobs)
