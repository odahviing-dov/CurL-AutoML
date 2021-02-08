# type: ignore
# mypy: ignore-errors
# flake8: noqa

import logging
import typing

import numpy as np
from pyrfr import regression
import sklearn.ensemble
import sklearn.tree
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.gaussian_process

from smac.configspace import ConfigurationSpace
from smac.epm.base_epm import AbstractEPM
from smac.utils.constants import N_TREES
from smac.epm.gaussian_process import GaussianProcess

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


N_EST = 2


class RandomForestWithInstances(AbstractEPM):
    """Random forest that takes instance features into account.

    Attributes
    ----------
    rf_opts : regression.rf_opts
        Random forest hyperparameter
    n_points_per_tree : int
    rf : regression.binary_rss_forest
        Only available after training
    hypers: list
        List of random forest hyperparameters
    unlog_y: bool
    seed : int
    types : np.ndarray
    bounds : list
    rng : np.random.RandomState
    logger : logging.logger
    """

    def __init__(self,
                 configspace: ConfigurationSpace,
                 types: np.ndarray,
                 bounds: typing.List[typing.Tuple[float, float]],
                 log_y: bool = False,
                 num_trees: int = N_TREES,
                 do_bootstrapping: bool = True,
                 n_points_per_tree: int = -1,
                 ratio_features: float = 5. / 6.,
                 min_samples_split: int = 3,
                 min_samples_leaf: int = 3,
                 max_depth: int = 3,
                 eps_purity: float = 1e-8,
                 max_num_nodes: int = 2 ** 20,
                 seed: int = 42,
                 **kwargs):
        """
        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : list
            Specifies the bounds for continuous features.
        log_y: bool
            y values (passed to this RF) are expected to be log(y) transformed;
            this will be considered during predicting
        num_trees : int
            The number of trees in the random forest.
        do_bootstrapping : bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree : int
            Number of points per tree. If <= 0 X.shape[0] will be used
            in _train(X, y) instead
        ratio_features : float
            The ratio of features that are considered for splitting.
        min_samples_split : int
            The minimum number of data points to perform a split.
        min_samples_leaf : int
            The minimum number of data points in a leaf.
        max_depth : int
            The maximum depth of a single tree.
        eps_purity : float
            The minimum difference between two target values to be considered
            different
        max_num_nodes : int
            The maxmimum total number of nodes in a tree
        seed : int
            The seed that is passed to the random_forest_run library.
        """
        super().__init__(configspace, types, bounds, seed, **kwargs)

        self.log_y = log_y

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features > 1.0 else max(1, int(len(types) * ratio_features))
        self.rf_opts.tree_opts.max_features = max_features
        self.rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self.rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self.rf_opts.tree_opts.max_depth = max_depth
        self.rf_opts.tree_opts.epsilon_purity = eps_purity
        self.rf_opts.tree_opts.max_num_nodes = max_num_nodes
        self.rf_opts.compute_law_of_total_variance = False

        self.n_points_per_tree = n_points_per_tree
        self.rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, seed]
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.logger = logging.getLogger(self.__module__ + "." +
                                        self.__class__.__name__)

    def _train(self, X: np.ndarray, y: np.ndarray):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """

        self.X = X
        self.y = y.flatten()

        from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
        from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior

        self.rf = sklearn.ensemble.RandomForestRegressor(
            max_features=1, bootstrap=True, max_depth=2, min_samples_leaf=5, n_estimators=N_EST,
        )
        self.rf.fit(X, np.log(y - np.min(y) + 1e-7).ravel())
        indicators = np.array(self.rf.apply(X))
        all_datasets = []
        all_targets = []
        all_mappings = []
        for est in range(N_EST):
            unique = np.unique(indicators[:, est])
            mapping = {j: i for i, j in enumerate(unique)}
            datasets = [[] for _ in unique]
            targets = [[] for _ in indicators]
            for indicator, x, y_ in zip(indicators[:, est], X, y):
                index = mapping[indicator]
                datasets[index].append(x)
                targets[index].append(y_)
            all_mappings.append(mapping)
            all_datasets.append(datasets)
            all_targets.append(targets)

        # print('Before')
        # for est in range(N_EST):
        #     for dataset in all_datasets[est]:
        #         print(len(dataset))

        for est in range(N_EST):
            n_nodes = self.rf.estimators_[est].tree_.node_count
            children_left = self.rf.estimators_[est].tree_.children_left
            children_right = self.rf.estimators_[est].tree_.children_right
            feature = self.rf.estimators_[est].tree_.feature
            threshold = self.rf.estimators_[est].tree_.threshold

            # The tree structure can be traversed to compute various properties such
            # as the depth of each node and whether or not it is a leaf.
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1

                # If we have a test node
                if (children_left[node_id] != children_right[node_id]):
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))
                else:
                    is_leaves[node_id] = True

            rules = {}
            import copy

            def extend(rule, idx):
                if is_leaves[idx]:
                    rules[idx] = rule
                else:
                    rule_left = copy.deepcopy(rule)
                    rule_left.append((threshold[idx], '<=', feature[idx]))
                    extend(rule_left, children_left[idx])
                    rule_right = copy.deepcopy(rule)
                    rule_right.append((threshold[idx], '>', feature[idx]))
                    extend(rule_right, children_right[idx])

            extend([], 0)
            #print(rules)

            for key, rule in rules.items():
                lower = -np.ones((X.shape[1], )) * np.inf
                upper = np.ones((X.shape[1],)) * np.inf
                for element in rule:
                    if element[1] == '<=':
                        if element[0] < upper[element[2]]:
                            upper[element[2]] = element[0]
                    else:
                        if element[0] > lower[element[2]]:
                            lower[element[2]] = element[0]

                for feature_idx in range(X.shape[1]):
                    closest_lower = -np.inf
                    closes_lower_idx = None
                    closest_upper = np.inf
                    closest_upper_idx = None
                    for x in X:
                        if x[feature_idx] > lower[feature_idx] and x[feature_idx] < upper[feature_idx]:
                            continue
                        if x[feature_idx] <= lower[feature_idx]:
                            if x[feature_idx] > closest_lower:
                                closest_lower = x[feature_idx]
                                closes_lower_idx = feature_idx
                        if x[feature_idx] >= upper[feature_idx]:
                            if x[feature_idx] < closest_upper:
                                closest_upper = x[feature_idx]
                                closest_upper_idx = feature_idx

                    if closest_upper_idx is not None:
                        all_datasets[est][all_mappings[est][key]].append(X[closest_upper_idx])
                        all_targets[est][all_mappings[est][key]].append(y[closest_upper_idx])
                    if closes_lower_idx is not None:
                        all_datasets[est][all_mappings[est][key]].append(X[closes_lower_idx])
                        all_targets[est][all_mappings[est][key]].append(y[closes_lower_idx])

        # print('After')
        # for est in range(N_EST):
        #     for dataset in all_datasets[est]:
        #         print(len(dataset))

        self.all_mappings = all_mappings
        self.models = []
        for est in range(N_EST):
            models = []
            for dataset, targets_ in zip(all_datasets[est], all_targets[est]):

                cov_amp = ConstantKernel(
                    2.0,
                    constant_value_bounds=(np.exp(-10), np.exp(2)),
                    prior=LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng),
                )

                cont_dims = np.nonzero(self.types == 0)[0]
                cat_dims = np.nonzero(self.types != 0)[0]

                if len(cont_dims) > 0:
                    exp_kernel = Matern(
                        np.ones([len(cont_dims)]),
                        [(np.exp(-10), np.exp(2)) for _ in range(len(cont_dims))],
                        nu=2.5,
                        operate_on=cont_dims,
                    )

                if len(cat_dims) > 0:
                    ham_kernel = HammingKernel(
                        np.ones([len(cat_dims)]),
                        [(np.exp(-10), np.exp(2)) for _ in range(len(cat_dims))],
                        operate_on=cat_dims,
                    )

                noise_kernel = WhiteKernel(
                    noise_level=1e-8,
                    noise_level_bounds=(np.exp(-25), np.exp(2)),
                    prior=HorseshoePrior(scale=0.1, rng=self.rng),
                )

                if len(cont_dims) > 0 and len(cat_dims) > 0:
                    # both
                    kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
                elif len(cont_dims) > 0 and len(cat_dims) == 0:
                    # only cont
                    kernel = cov_amp * exp_kernel + noise_kernel
                elif len(cont_dims) == 0 and len(cat_dims) > 0:
                    # only cont
                    kernel = cov_amp * ham_kernel + noise_kernel
                else:
                    raise ValueError()

                gp = GaussianProcess(
                    configspace=self.configspace,
                    types=self.types,
                    bounds=self.bounds,
                    kernel=kernel,
                    normalize_y=True,
                    seed=self.rng.randint(low=0, high=10000),
                )
                gp.train(np.array(dataset), np.array(targets_))
                models.append(gp)
            self.models.append(models)
        return self

    def predict_marginalized_over_instances(self, X: np.ndarray):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Note
        ----
        This method overwrites the same method of ~smac.epm.base_epm.AbstractEPM;
        the following method is random forest specific
        and follows the SMAC2 implementation;
        it requires no distribution assumption
        to marginalize the uncertainty estimates

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if self.instance_features is None or \
                len(self.instance_features) == 0:

            means = []
            vars = []
            indicators = np.array(self.rf.apply(X))
            for indicator, x in zip(indicators, X):
                means_tmp = []
                vars_tmp = []
                for est in range(N_EST):
                    mapping = self.all_mappings[est]
                    index = mapping[indicator[est]]
                    m, v = self.models[est][index].predict(x.reshape((1, -1)))
                    means_tmp.append(m[0, 0])
                    vars_tmp.append(v[0, 0])
                means_tmp = np.array(means_tmp)
                vars_tmp = np.array(vars_tmp)
                #v = 1 / (np.sum(1 / vars_tmp))
                #m = v * np.sum([1 / vars_tmp[i] * means_tmp[i] for i in range(N_EST)])
                m = np.mean(means_tmp)
                v = np.average(vars_tmp, weights=[(1 / len(vars_tmp)) ** 2] * len(vars_tmp))
                means.append(m)
                vars.append(v)

            mean = np.array(means).reshape((-1, 1))
            var = np.array(vars).reshape((-1, 1))

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        else:
            raise NotImplementedError()
