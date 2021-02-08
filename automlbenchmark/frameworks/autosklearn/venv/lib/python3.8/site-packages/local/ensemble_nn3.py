import torch
import typing
import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.configspace import ConfigurationSpace
from smac.epm.bnn3 import SimpleNetworkEmbedding
from smac.utils.logging import PickableLoggerAdapter


class Dummy:

    def __init__(self, **kwargs):
        self.rng = np.random.RandomState(kwargs.get("seed", None))
        self.q = None

    def reset(self):
        pass

    def train(self, X, y, **kwargs):
        self.rng = np.random.RandomState(1)
        self.q = self.rng.uniform(25, 75)
        self.value = np.percentile(y, self.q)
        if not np.isfinite(self.value):
            # Unclear why this could happen
            self.value = np.mean(y)

    def predict(self, X):
        m = np.ones([X.shape[0], 1]) * self.value
        v = np.zeros([X.shape[0], 1])
        return np.concatenate([m, v], axis=1)


class EnsembleNN(AbstractEPM):

    def __init__(
            self,
            configspace: ConfigurationSpace,
            types: typing.List[int],
            bounds: typing.List[typing.Tuple[float, float]],
            seed: int,
            hidden_dims: typing.List[int] = [50, 50, 50],
            lr: float = 1e-3,
            momentum: float = 0.999,
            weight_decay: float = 1e-4,
            iterations: int = 5000,
            batch_size: int = 16,
            number_of_networks: int = 5,
            var: bool = True,
            train_with_lognormal_llh=False,
            compute_mean_in_logspace=False,
            max_cat: int = np.inf,
            ignore_cens: bool = False,
            learned_weight_init: bool = False,
            optimization_algorithm: str = 'sgd',
            **kwargs
    ):
        super().__init__(configspace, types, bounds, seed, **kwargs)
        #self.types[self.types == 0] = -1
        self.types = [int(f) for f in self.types]
        assert not (train_with_lognormal_llh and compute_mean_in_logspace)

        if type(self.seed) != int:
            self.seed = self.seed[0]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_loss = 1000
        self.log_error = 5000

        self.var = var
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.momentum = momentum
        self.iterations = iterations
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.number_of_networks = number_of_networks
        self.train_with_lognormal = train_with_lognormal_llh
        self.compute_mean_in_logspace = compute_mean_in_logspace
        self.max_cat = max_cat
        self.ignore_cens = ignore_cens
        self.learned_weight_init = learned_weight_init
        self.optimization_algorithm = optimization_algorithm

        self._my = None
        self._sy = None

        # Quick check, should not take too long
        a = np.random.normal(42, 23, 1000)
        m1, v1 = (np.mean(a), np.var(a))
        a = self._preprocess_y(a)
        m2, v2 = self._postprocess_mv(np.mean(a), np.var(a))
        assert np.abs(m1 - m2) < 1e-3, (m1, m2)
        assert np.abs(v1 - v2) < 1e-3, (v1, v2)
        self._my = None
        self._sy = None

        self.nns = None
        self.logger = PickableLoggerAdapter(
            self.__module__ + "." + self.__class__.__name__
        )

    def _preprocess_y(self, y: np.ndarray, redo=False):
        if self._my is None or redo:
            self._my = np.mean(y)
            self._sy = np.std(y)
            if self._sy == 0:
                # all y's are the same
                self._sy = 1

        if not self.train_with_lognormal:
            y -= self._my
            y /= self._sy

        return y

    def _postprocess_mv(self, m: np.ndarray, v: np.ndarray):
        # zero mean scaling
        m = m * self._sy + self._my
        v = v * self._sy ** 2
        return m, v

    def _preprocess_x(self, x: np.ndarray, redo: bool=False):
        # Replace nans with 0, should be fine for both cats and conts
        # TODO: Maybe refine this and replace cont with mean
        x = np.nan_to_num(x)
        return x

    def _train(self, X: np.ndarray, Y: np.ndarray, C: np.ndarray = None):
        self.logger.critical("Not using C as this is not a Tobit model")
        Y = self._preprocess_y(Y, redo=True)
        X = self._preprocess_x(X, redo=True)
        self.train_data = (X, Y)
        self.nns = []
        self.logger.debug("Start Training %d networks" % self.number_of_networks)
        for i in range(self.number_of_networks):
            nn = SimpleNetworkEmbedding(hidden_dims=self.hidden_dims,
                                        feat_types=self.types,
                                        lr=self.lr,
                                        seed=self.seed + i,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay,
                                        iterations=self.iterations,
                                        batch_size=self.batch_size,
                                        var=self.var,
                                        lognormal_nllh=self.train_with_lognormal,
                                        var_bias_init=np.std(Y),
                                        max_cat=self.max_cat,
                                        learned_weight_init=self.learned_weight_init,
                                        optimization_algorithm=self.optimization_algorithm,
                                        )
            nn.reset()
            nn.train(X, Y)
            self.nns.append(nn)

    def _predict_individual(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        X = self._preprocess_x(X, redo=True)
        ms = np.zeros([X.shape[0], self.number_of_networks])
        vs = np.zeros([X.shape[0], self.number_of_networks])
        for i_nn, nn in enumerate(self.nns):
            pred = nn.predict(X)
            m = pred[:, 0]
            v = pred[:, 1]

            if not self.train_with_lognormal:
                m, v = self._postprocess_mv(m, v)

            ms[:, i_nn] = m
            vs[:, i_nn] = v

        return ms, vs

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        ms, _ = self._predict_individual(X)
        m = ms.mean(axis=1)
        v = ms.var(axis=1)
        return m.reshape((-1, 1)), v.reshape((-1, 1))

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
            mean_, var = self.predict(X)
            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean_, var

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != len(self.bounds):
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (len(self.bounds),
                              X.shape[1]))

        mean_ = np.zeros((X.shape[0], 1))
        var = np.zeros(X.shape[0])

        for i, x in enumerate(X):

            # marginalize over instance
            # 1. Get predictions for all networks

            # Not very efficient
            # preds_nns1 = np.zeros([len(self.instance_features), self.number_of_networks])
            #for i_f, feat in enumerate(self.instance_features):
            #    x_ = np.concatenate([x, feat]).reshape([1, -1])
            #    print(i_f, x_)
            #    m, _ = self._predict_individual(x_)
            #    preds_nns1[i_f, :] = m

            input = np.concatenate((np.tile(x, (len(self.instance_features), 1)), self.instance_features), axis=1)
            preds_nns, _ = self._predict_individual(input)

            # 2. Average in each NN for all instances
            pred_per_nn = []
            for nn_id in range(self.number_of_networks):
                if self.compute_mean_in_logspace:
                    pred_per_nn.append(np.log(np.mean(np.exp(preds_nns[:, nn_id]))))
                else:
                    pred_per_nn.append(np.mean(preds_nns[:, nn_id]))

            # 3. compute statistics across trees
            mean_x = np.mean(pred_per_nn)
            var_x = np.var(pred_per_nn)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean_[i] = mean_x

        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean_, var
