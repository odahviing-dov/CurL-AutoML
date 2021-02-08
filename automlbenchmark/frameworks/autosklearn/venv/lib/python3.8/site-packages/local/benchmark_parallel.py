import sys
import time

import numpy as np

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.intensification.simple_intensifier import SimpleIntensifier

import branin

cs = ConfigurationSpace()
x0 = UniformFloatHyperparameter("x0", -5, 10)
x1 = UniformFloatHyperparameter("x1", 0, 15)
cs.add_hyperparameters([x0, x1])


if __name__ == '__main__':
    # Fully parallel SMAC

    histories = []

    for n_workers in (1, 2):
        for seed in range(10):

            print('#' * 80)
            print('#' * 80)
            print('#' * 80)
            print('#' * 80)
            print('#' * 80)

            # SMAC scenario object
            scenario = Scenario({"run_obj": "quality",
                                 "runcount-limit": 25,
                                 "cs": cs,
                                 "deterministic": "true",
                                 "output-dir": "/home/feurerm/projects/smac3parallel/new/%s" % str(n_workers),
                                 })

            # To optimize, we pass the function to the SMAC-object
            smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(seed),
                            tae_runner=branin.branin if n_workers == 1 else branin.branin_sleep,
                            n_jobs=n_workers, intensifier=SimpleIntensifier)
            incumbent = smac.optimize()
            del smac

            time.sleep(1)
            sys.stdout.flush()
            sys.stderr.flush()
