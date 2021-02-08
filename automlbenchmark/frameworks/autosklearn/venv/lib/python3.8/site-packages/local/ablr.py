from typing import Dict, List, Tuple, Union

from ConfigSpace import Configuration
import numpy as np
import scipy.optimize
import torch.nn as nn
import torch

from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.optimizer.acquisition import EI
from smac.initial_design.latin_hypercube_design import LHDesign

D = 16  # Hidden layer size


precision = 32
if precision == 32:
    t_dtype = torch.float32
    np_dtype = np.float32
else:
    t_dtype = torch.float64
    np_dtype = np.float64


########################################################################################################################
########################################################################################################################
# The actual PyTorch model
########################################################################################################################
########################################################################################################################

class Net(torch.nn.Module):
    def __init__(self, num_tasks, n_attributes, alpha_min, alpha_max, beta_min, beta_max, meta_data=None):
        """
        num_tasks : int
            Number of Tasks
        n_attributes : int
            Number of attributes/features of the training data
        meta_data : dict
            Meta-data, mapping from task Id to tuple (x, y, meta-features)
        """
        self.num_tasks = num_tasks
        self.n_attributes = n_attributes
        self.meta_data = meta_data

        self.mean_ = None
        self.std_ = None

        super().__init__()
        self.total_n_params = 0

        hidden1 = nn.Linear(self.n_attributes, D)
        hidden2 = nn.Linear(D, D)
        #hidden3 = nn.Linear(D, D)
        self.layers = [
            hidden1, hidden2, #hidden3
        ]
        if precision == 32:
            self.layers = [layer.float() for layer in self.layers]
        else:
            self.layers = [layer.double() for layer in self.layers]

        # initialization of alpha and beta
        # Instead of alpha, we model 1/alpha and use a different range for the values
        # (i.e. 1e-6 to 1 instead of 1 to 1e6)
        print(10**(alpha_min + (alpha_max - alpha_min) / 2), 10**(beta_min + (beta_max - beta_min) / 2))
        self.alpha_t = torch.tensor([10**(alpha_min + (alpha_max - alpha_min) / 2)] * self.num_tasks, requires_grad=True, dtype=t_dtype)
        self.total_n_params += len(self.alpha_t)
        # params for each log likelihood
        self.beta_t = torch.tensor([10**(beta_min + (beta_max - beta_min) / 2)] * self.num_tasks, requires_grad=True, dtype=t_dtype)
        self.total_n_params += len(self.beta_t)

        # initialization of the weights
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            if len(layer.weight.shape) == 1:
                size = layer.weight.shape[0]
            else:
                size = layer.weight.shape[0] * layer.weight.shape[1]
            self.total_n_params += size

        # initialize arrays for the optimization of sum log-likelihood
        self.K_t = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]
        self.L_t = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]
        self.L_t_inv = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]
        self.e_t = [torch.tensor(0.0, dtype=t_dtype) for i in range(self.num_tasks)]

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x):
        """
        Simple forward pass through the neural network
        """

        for layer in self.layers:
            x = layer(x)
            x = torch.tanh(x)

        return x

    def loss(self, hp, training_datasets):
        """
        Negative log marginal likelihood of multi-task ABLR
        hp : np.ndarray
            Contains the weights of the network, alpha and beta
        training_datasets : list
            tuples (X, y) for the meta-datasets and the current dataset
        """
        # I have to separate the variables because LBFGS needs a flattened array

        if precision == 32:
            hp = hp.astype(np.float32)

        idx = 0
        for layer in self.layers:
            weights = layer.weight.data.numpy().astype(np_dtype)
            if len(weights.shape) == 1:
                size = weights.shape[0]
            else:
                size = weights.shape[0] * weights.shape[1]
            layer.weight.data = torch.from_numpy(hp[idx: idx + size].reshape(weights.shape))
            layer.weight.requires_grad_()
            idx += size

        self.alpha_t.data = torch.from_numpy(hp[idx: idx + self.num_tasks])
        idx += self.num_tasks
        self.alpha_t.requires_grad_()
        self.beta_t.data = torch.from_numpy(hp[idx: idx + self.num_tasks])
        idx += self.num_tasks
        self.beta_t.requires_grad_()
        assert idx == self.total_n_params

        self.likelihood = None

        for i, (x, y) in enumerate(training_datasets):

            out = self.forward(x)

            # Loss function calculations, see 6th Equation on the first page of the Appendix
            # https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning-supplemental.zip
            assert (torch.t(out).shape == (D, x.shape[0]))
            # Remember that we model 1/alpha instead of alpha
            r = self.beta_t[i] * self.alpha_t[i]
            K_t = torch.add(
                torch.eye(D, dtype=t_dtype),
                r * torch.matmul(torch.t(out), out)
            )
            self.K_t[i] = K_t.clone()
            assert (K_t.shape == (D, D))

            L_t = torch.cholesky(K_t, upper=False)
            self.L_t[i] = L_t.clone()
            #self.L_t_inv[i] = torch.inverse(L_t)
            #e_t = torch.matmul(self.L_t_inv[i], torch.matmul(torch.t(out), y))
            e_t = torch.triangular_solve(torch.matmul(torch.t(out), y), L_t, upper=False).solution
            self.e_t[i] = e_t.view((D, 1)).clone()
            assert (self.e_t[i].shape == (D, 1))

            norm_y_t = torch.norm(y, 2, 0)
            norm_c_t = torch.norm(e_t[i], 2, 0)

            L1 = -(x.shape[0] / 2 * torch.log(self.beta_t[i]))
            L2 = self.beta_t[i] / 2 * (torch.pow(norm_y_t, 2) -r * torch.pow(norm_c_t, 2))
            L3 = torch.sum(torch.log(torch.diag(L_t)))
            L = L1 + L2 + L3

            if self.likelihood is None:
                self.likelihood = L
            else:
                self.likelihood = torch.add(self.likelihood, L)

        g = np.zeros((self.total_n_params))
        self.likelihood.backward()

        idx = 0
        for layer in self.layers:
            gradients = layer.weight.grad.data.numpy().astype(np_dtype)
            if len(gradients.shape) == 1:
                size = gradients.shape[0]
            else:
                size = gradients.shape[0] * gradients.shape[1]
            g[idx: idx + size] = gradients.flatten()
            idx += size
            layer.weight.grad.zero_()

        g[idx: idx + self.num_tasks] = self.alpha_t.grad.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        g[idx: idx + self.num_tasks] = self.beta_t.grad.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        self.alpha_t.grad.data.zero_()
        self.beta_t.grad.data.zero_()
        self._gradient = g

        return self.likelihood

    def gradient(self, hp, training_datasets):
        """
        Gradient of the parameters of the network that are optimized through LBFGS
        hp :np.array((tot_par,))
            array that contains the weights of the network, alpha and beta
        """

        # Initialization

        return self._gradient

    def optimize(self, training_datasets):
        """
        Optimize weights, alpha and beta with LBFGSB
        """

        # Initial flattened array
        init = np.ones((self.total_n_params), dtype=np_dtype)

        idx = 0
        for layer in self.layers:
            weights = layer.weight.data.numpy().astype(np_dtype)
            if len(weights.shape) == 1:
                size = weights.shape[0]
            else:
                size = weights.shape[0] * weights.shape[1]
            init[idx: idx + size] = weights.flatten()
            idx += size
        mybounds = [[None, None] for i in range(idx)]

        init[idx: idx + self.num_tasks] = self.alpha_t.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        print(self.alpha_min, alpha_max, beta_min, beta_max)
        mybounds.extend([[10**self.alpha_min, 10**self.alpha_max]] * self.num_tasks)
        init[idx: idx + self.num_tasks] = self.beta_t.data.numpy().astype(np_dtype)
        idx += self.num_tasks
        mybounds.extend([[10**self.beta_min, 10**self.beta_max]] * self.num_tasks)

        assert self.total_n_params == len(mybounds), (self.total_n_params, len(mybounds))
        assert self.total_n_params == idx

        res = scipy.optimize.fmin_l_bfgs_b(
            lambda *args: float(self.loss(*args)),
            x0=init,
            bounds=mybounds,
            fprime=self.gradient,
            args=(training_datasets, ),
        )
        self.loss(res[0], training_datasets)  # This updates the internal states
        #print('L-BFGS-S result', res)

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        First optimized the hyperparameters if do_optimize is True and then computes
        the posterior distribution of the weights.
        X: np.ndarray (N, D)
            Input data points.
        y: np.ndarray (N,)
            The corresponding target values.
        """
        y = y.reshape((y.shape[0], 1))

        training_datasets = []
        for meta_task in self.meta_data:
            meta_task_data = self.meta_data[meta_task]
            X_t = meta_task_data[0]
            y_t = meta_task_data[1]
            if X_t.shape[1] != self.n_attributes:
                raise ValueError((X_t.shape[1], self.n_attributes))

            mean = y_t.mean()
            std = y_t.std()
            if std == 0:
                std = 1
            y_t = (y_t.copy() - mean) / std
            y_t = y_t.reshape(y_t.shape[0], 1)

            training_datasets.append((
                torch.tensor(X_t, dtype=t_dtype),
                torch.tensor(y_t, dtype=t_dtype),
            ))

        if X.shape[1] != self.n_attributes:
            raise ValueError((X.shape[1], self.n_attributes))

        self.mean_ = y.mean()
        self.std_ = y.std()
        if self.std_ == 0:
            self.std_ = 1
        y_ = (y.copy() - self.mean_) / self.std_

        training_datasets.append((
            torch.tensor(X, dtype=t_dtype),
            torch.tensor(y_, dtype=t_dtype),
        ))
        if len(training_datasets) != self.num_tasks:
            raise ValueError((len(training_datasets), self.num_tasks))

        self.optimize(training_datasets)
        print(self.alpha_t[-1], self.beta_t[-1])
        #print('Alpha^{-1} (target task)', self.alpha_t)
        #print('Beta (target task)', self.beta_t)
        #print('Beta / Alpha', self.beta_t[-1] * self.alpha_t[-1])

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.
        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points
        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance
        """
        X_test = torch.tensor(X_test, dtype=t_dtype)
        out = self.forward(X_test)

        #m = torch.matmul(torch.matmul(torch.t(self.e_t[-1]), self.L_t_inv[-1]), torch.t(out))
        m = torch.matmul(
            torch.t(self.e_t[-1]),
            torch.triangular_solve(torch.t(out), self.L_t[-1], upper=False).solution,
        )
        # Remember that we model 1/alpha instead of alpha
        m = (self.beta_t[-1] * self.alpha_t[-1]) * m.reshape((m.shape[1], 1))
        assert (m.shape == (X_test.shape[0], 1))
        if not torch.isfinite(m).all():
            raise ValueError('Infinite predictions %s for input %s' % (m, X_test))
        m = m * self.std_ + self.mean_

        #v = torch.matmul(self.L_inv_t[-1], torch.t(out))
        v = torch.triangular_solve(torch.t(out), self.L_t[-1], upper=False).solution
        # Remember that we model 1/alpha instead of alpha
        v = self.alpha_t[-1] * torch.pow(torch.norm(v, dim=0), 2)
        v = v.reshape((-1, 1))
        assert (v.shape == (X_test.shape[0], 1)), v.shape
        if not torch.isfinite(v).all():
            raise ValueError('Infinite predictions %s for input %s' % (v, X_test))
        v = v * (self.std_ ** 2)

        return m.detach().numpy(), v.detach().numpy()


########################################################################################################################
########################################################################################################################
# A SMAC wrapper around the PyTorch code
########################################################################################################################
########################################################################################################################


class ABLR(AbstractEPM):
    """
    Class to use with SMAC
    """

    def __init__(
            self,
            alpha_min,
            alpha_max,
            beta_min,
            beta_max,
            training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]] = None,
            **kwargs
    ):
        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        if training_data is None:
            self.training_data = dict()
        else:
            self.training_data = training_data
        self.nn = None
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

        self.categorical_mask = np.array(self.types) > 0
        self.n_categories = np.sum(self.types)

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM:
        meta_data = dict()
        for id_ in self.training_data:
            configs = self.training_data[id_]['configurations']
            X_ = convert_configurations_to_array(configs)
            X_ = self._preprocess(X_)
            meta_data[id_] = (
                X_,
                self.training_data[id_]['y'].flatten(),
                None,
            )

        X = self._preprocess(X)
        for i in range(10):
            try:
                if self.nn is None:
                    self.nn = Net(
                        num_tasks=len(self.training_data) + 1,
                        n_attributes=X.shape[1],
                        meta_data=meta_data,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        beta_min=beta_min,
                        beta_max=beta_max,
                    )
                self.nn.train(X, Y)
                break
            except Exception as e:
                print('Training failed %d/%d!' % (i + 1, 10))
                print(e)
                self.nn = None

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._preprocess(X)
        if self.nn:
            return self.nn.predict(X)
        else:
            return self.rng.randn(X.shape[0], 1), self.rng.randn(X.shape[0], 1)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        categories_array = np.zeros((X.shape[0], self.n_categories))
        categories_idx = 0
        for idx in range(len(self.types)):
            if self.types[idx] == 0:
                continue
            else:
                for j in range(self.types[idx]):
                    mask = X[:, idx] == j
                    categories_array[mask, categories_idx] = 1
                    categories_idx += 1
        numerical_array = X[:, ~self.categorical_mask]
        X = np.concatenate((numerical_array, categories_array), axis=1)
        X[np.isnan(X)] = -1.0
        return X


########################################################################################################################
########################################################################################################################
# Using the model for the Rosenbrock function in SMAC
########################################################################################################################
########################################################################################################################


from smac.facade.func_facade import fmin_smac


def rosenbrock_2d(x):
    x1 = x[0]
    x2 = x[1]
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return ret


results = {}
for alpha_min in range(-6, 7, 3):
    for alpha_max in range(alpha_min + 3, 7, 3):
        for beta_min in range(-6, 7, 3):
            for beta_max in range(beta_min + 3, 7, 3):
                costs = []
                for i in range(10):
                    # fmin_smac assumes that the function is deterministic
                    # and uses under the hood the SMAC4HPO
                    x, cost, _ = fmin_smac(func=rosenbrock_2d,
                                           model=ABLR,
                                           model_kwargs={
                                               'alpha_min': alpha_min,
                                               'alpha_max': alpha_max,
                                               'beta_min': beta_min,
                                               'beta_max': beta_max,
                                           },
                                           runhistory2epm=RunHistory2EPM4Cost,
                                           acquisition_function=EI,
                                           x0=[2.5, 7.5],
                                           bounds=[(-5, 10), (0, 15)],
                                           maxfun=50,
                                           initial_design=LHDesign,
                                           rng=i)  # Passing a seed makes fmin_smac determistic

                    print("Best x: %s; with cost: %f" % (str(x), cost))
                    costs.append(cost)
                costs = np.mean(costs)
                results[(alpha_min, alpha_max, beta_min, beta_max)] = costs
                print(alpha_min, alpha_max, beta_min, beta_max, costs)

for key in results:
    print(key, results[key])
