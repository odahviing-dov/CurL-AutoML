# type: ignore
# mypy: ignore-errors
# flake8: noqa

import torch
import typing
import numpy as np

from smac.epm.base_rf import BaseModel
from smac.configspace import ConfigurationSpace
from smac.epm.bnn2 import SimpleNetworkEmbedding
from smac.utils.logging import PickableLoggerAdapter


class EnsembleNN(BaseModel):

    def __init__(
            self,
            configspace: ConfigurationSpace,
            types: np.ndarray,
            bounds: typing.List[typing.Tuple[float, float]],
            seed: int,
            hidden_dims: typing.List[int] = [50, 50, 50],
            lr: float = 1e-3,
            momentum: float = 0.999,
            weight_decay: float = 1e-4,
            iterations: int = 10000,
            batch_size: int = 8,
            number_of_networks: int = 10,
            var: bool = True,
            train_with_lognormal_llh = False,
            compute_mean_in_logspace = True,
            **kwargs
    ):
        super().__init__(configspace, types, bounds, seed, **kwargs)

        assert not (train_with_lognormal_llh and compute_mean_in_logspace)

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

        self.nns = None
        self.logger = PickableLoggerAdapter(
            self.__module__ + "." + self.__class__.__name__
        )

    def _train(self, X: np.ndarray, y: np.ndarray):

        self._my = np.mean(y)
        self._sy = np.std(y)

        if not self.train_with_lognormal:
            y -= self._my
            y /= self._sy

        self.train_data = (X, y)
        self.nns = []
        self.logger.debug("Start Training %d networks" % self.number_of_networks)
        for i in range(self.number_of_networks):
            nn = SimpleNetworkEmbedding(hidden_dims=self.hidden_dims,
                                        lr=self.lr,
                                        seed=self.seed + i,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay,
                                        iterations=self.iterations,
                                        batch_size=self.batch_size,
                                        var=self.var,
                                        lognormal_nllh=self.train_with_lognormal
                                        )
            nn.train(X, y)
            self.nns.append(nn)

    def _predict_individual(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        ms = np.zeros([X.shape[0], self.number_of_networks])
        vs = np.zeros([X.shape[0], self.number_of_networks])
        for i_nn, nn in enumerate(self.nns):
            pred = nn.predict(X)
            m = pred[:, 0]
            v = pred[:, 1]

            if not self.train_with_lognormal:
                m = m * self._sy + self._my
                v = v * self._sy ** 2

            ms[:, i_nn] = m
            vs[:, i_nn] = v

        return ms, vs

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        ms, _ = self._predict_individual(X)
        m = ms.mean(axis=1)
        v = ms.var(axis=1)
        return m, v

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

        mean_ = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])

        for i, x in enumerate(X):

            # marginalize over instance
            # 1. Get predictions for all networks

            # Not very efficient
            preds_nns1 = np.zeros([len(self.instance_features), self.number_of_networks])
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
