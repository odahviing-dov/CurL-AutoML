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

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


def pdist(X, metric=None):
    rval = []
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if j >= i:
                continue
            rval.append(np.sqrt(np.sum([
                np.abs(X[i][k] - X[j][k]) if X[i][k] != 0 and X[j][k] != 0 else (0 if X[i][k] == 0 and X[j][k] == 0 else 1)
                    for k in range(X.shape[1])
            ])))
    rval = np.array(rval)
    #import scipy.spatial
    #rval = scipy.spatial.distance.pdist(X, metric='cosine')
    return rval


def cdist(X, Y, metric=None):
    rval = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            rval[i][j] = np.sqrt(np.sum([
                np.abs(X[i][k] - Y[j][k]) if X[i][k] != 0 and Y[j][k] != 0 else (0 if X[i][k] == 0 and Y[j][k] == 0 else 1)
                for k in range(X.shape[1])
            ]))
    #import scipy.spatial
    #rval = scipy.spatial.distance.cdist(X, Y, metric='cosine')
    return rval


sklearn.gaussian_process.kernels.pdist = pdist
sklearn.gaussian_process.kernels.cdist = cdist


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
                 max_depth: int = 2 ** 20,
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
        self.rng = regression.default_random_engine(seed)

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features > 1.0 else \
            max(1, int(types.shape[0] * ratio_features))
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

        self.rf = sklearn.ensemble.RandomForestRegressor(
            max_features=1.0, bootstrap=False, n_estimators=1, max_depth=None,
        )
        #self.rf = sklearn.tree.DecisionTreeRegressor(max_depth=10)
        self.rf.fit(X, y)

        new_X = []
        for tree in self.rf.estimators_:
            tree_X = []
            # There's no reason to also take the leaves into account!
            path = tree.decision_path(X)
            for i in range(path.shape[0]):
                row = path.getrow(i).toarray().flatten().copy()
                new_row = []
                for j in range(len(row)):
                    if row[j] == 0:
                        new_row.append(np.NaN)
                    else:
                        threshold = tree.tree_.threshold[j]
                        feature_idx = tree.tree_.feature[j]
                        diff = (threshold - X[i][feature_idx])
                        new_row.append(diff)
                tree_X.append(new_row)
            new_X.append(np.array(tree_X))
        new_X = np.hstack(new_X)
        assert X.shape[0] == new_X.shape[0]
        X_min = np.nanmin(new_X, axis=0)
        X_max = np.nanmax(new_X, axis=0)
        diff = X_max - X_min
        diff[diff == 0] = 1
        self.X_min_ = X_min
        self.diff_ = diff
        new_X = (new_X - self.X_min_) / self.diff_
        new_X[np.isnan(new_X)] = 0
        self.max_length_ = np.max(np.sum(new_X, axis=1))
        new_X = new_X / self.max_length_

        # TODO compute the kernel manually by computing the tree similarities and then only compute an additive kernel within each tree...
        # only compare 'same' paths of a tree
        self.gp = sklearn.pipeline.Pipeline([
            # Cannot use the scaler here as it would destroy all knowledge about where the zeros are
            #['preproc', sklearn.preprocessing.MinMaxScaler()],
            [
                'regressor', sklearn.gaussian_process.GaussianProcessRegressor(
                    kernel=sklearn.gaussian_process.kernels.ConstantKernel() *
                    sklearn.gaussian_process.kernels.Matern(
                    ),# + sklearn.gaussian_process.kernels.WhiteKernel(
                    #    noise_level=1e-7, noise_level_bounds=(1e-14, 1e-6)
                    #),
                    n_restarts_optimizer=10,
                    normalize_y=True,
                )
            ]
        ])
        print(new_X.shape)
        self.gp.fit(new_X, y)
        print(self.gp.steps[-1][-1].kernel_)
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

            new_X = []
            for tree in self.rf.estimators_:
                tree_X = []
                path = tree.decision_path(X)
                for i in range(path.shape[0]):
                    row = path.getrow(i).toarray().flatten().copy()
                    new_row = []
                    for j in range(len(row)):
                        if row[j] == np.NaN:
                            new_row.append(0)
                        else:
                            threshold = tree.tree_.threshold[j]
                            feature_idx = tree.tree_.feature[j]
                            diff = (threshold - X[i][feature_idx])
                            new_row.append(diff)
                    tree_X.append(new_row)
                new_X.append(np.array(tree_X))
            new_X = np.hstack(new_X)
            new_X = (new_X - self.X_min_) / self.diff_
            new_X[np.isnan(new_X)] = 0
            new_X = new_X / self.max_length_

            #new_X = self.scaler.transform(new_X)
            mean, std = self.gp.predict(new_X, return_std=True)

            mean = mean.reshape((-1, 1))
            var = (std).reshape((-1, 1)) ** 2
            #print(new_X, mean, var)

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        else:
            raise NotImplementedError()
