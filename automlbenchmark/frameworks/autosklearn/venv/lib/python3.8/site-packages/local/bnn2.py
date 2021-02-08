# type: ignore
# mypy: ignore-errors
# flake8: noqa

import torch.nn as nn
from torch.utils.data import Dataset
import torch
import typing
import numpy as np
import copy
from scipy import optimize
from sklearn.metrics import mean_squared_error

from smac.epm.base_rf import BaseModel
from smac.configspace import ConfigurationSpace
from smac.utils.logging import PickableLoggerAdapter


def rmse(pred_y, true_y, cutoff=np.log10(300), min_val=np.log10(0.005)):

    if not np.isfinite(pred_y).all():
        print("Cannot compute metric, inputs are not finite")
        return np.NAN

    pred_y = np.array(pred_y).reshape([-1, 1])
    true_y = np.array(true_y).reshape([-1, 1])

    pred_y[pred_y > cutoff] = cutoff
    true_y[true_y > cutoff] = cutoff

    pred_y[pred_y < min_val] = min_val
    true_y[true_y < min_val] = min_val

    mse = mean_squared_error(y_pred=pred_y, y_true=true_y)
    return np.sqrt(mse)


class NLLHLoss(nn.Module):

    def forward(self, input, target):
        # Assuming network outputs var
        #std = torch.sqrt(torch.exp(input[:, 1].view(-1, 1)))
        print(input, target)
        std = torch.sqrt(input[:, 1].view(-1, 1))
        mu = input[:, 0].view(-1, 1)
        # std = torch.exp(input[:, 1])
        # std = torch.clamp_min(std, 0.1)
        # std = torch.zeros_like(mu) + 1

        # PDF
        # t1 = - torch.log(var) - math.log(math.sqrt(2 * math.pi))
        # t2 = - ((target - mu) ** 2.0) / (2.0 * var**2)
        # print(-(t1 + t2))

        n = torch.distributions.normal.Normal(mu, std)
        loss = n.log_prob(target)
        return -torch.mean(loss)


class NLLHLogNormalLoss(nn.Module):

    def forward(self, input, target):
        # Assuming network outputs var
        #std = torch.sqrt(torch.exp(input[:, 1].view(-1, 1)))
        std = torch.sqrt(input[:, 1].view(-1, 1))
        mu = input[:, 0].view(-1, 1)
        # std = torch.exp(input[:, 1])
        # std = torch.clamp_min(std, 0.1)
        # std = torch.zeros_like(mu) + 1

        # PDF
        # t1 = - torch.log(var) - math.log(math.sqrt(2 * math.pi))
        # t2 = - ((target - mu) ** 2.0) / (2.0 * var**2)
        # print(-(t1 + t2))

        n = torch.distributions.log_normal.LogNormal(mu, std)
        loss = n.log_prob(target)
        return -torch.mean(loss)


class XYDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = torch.Tensor(X).double()
        if y is not None:
            self.y = torch.Tensor(y).reshape([self.X.shape[0], -1]).double()
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx, :], self.y[idx, :]
        else:
            return self.X[idx, :]


class DNGO(BaseModel):

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
            var: bool = True,
            **kwargs
    ):
        super().__init__(configspace, types, bounds, seed, **kwargs)
        print("USE DNGO")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_loss = 100
        self.log_error = 1000

        self.var = var
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.momentum = momentum
        self.iterations = iterations
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.nn = None
        self.blr = None

        self.logger = PickableLoggerAdapter(self.__module__ + "." + self.__class__.__name__)

    def _train(self, X: np.ndarray, y: np.ndarray):
        self.nn = SimpleNetworkEmbedding(hidden_dims=self.hidden_dims,
                                         lr=self.lr,
                                         seed=self.seed,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay,
                                         iterations=self.iterations,
                                         batch_size=self.batch_size,
                                         var=self.var,
                                         )
        self.blr = BayesianLinearRegressionLayer()

        self._my = np.mean(y)
        self._sy = np.std(y)

        y -= self._my
        y /= self._sy

        #print(X, y)
        #import matplotlib.pyplot as plt

        self.nn.train(X, y)
        #plt.scatter(X, y)

        #x_dense = np.linspace(-0.1, 1.1, 100)
        #pred = self._predict_nn(x_dense.reshape([-1, 1]))
        #m = pred[:, 0].flatten()
        #v = pred[:, 1].flatten()
        #plt.plot(x_dense, m, label="nn")
        #plt.fill_between(x_dense, m - v, m + v, alpha=0.5)
        self.blr.optimize_alpha_beta(self.nn.model, X, y)

        #m, v = self.blr.predict(self.model, x_dense.reshape([-1, 1]))
        #m = m.data.numpy().flatten()
        #v = v.data.numpy().flatten()
        #plt.scatter(X, y)
        #plt.plot(x_dense, m, label="blr")
        #plt.fill_between(x_dense, m-v, m+v, alpha=0.5)
        #plt.legend()
        #plt.ylim([-10, 10])
        #plt.show()

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        means, vars = self.blr.predict(self.nn.model, X)
        means = means.data.numpy().flatten()
        vars = vars.data.numpy().flatten()

        means = np.array(means * self._sy + self._my).reshape([-1, 1])
        vars = np.array(vars * self._sy ** 2).reshape([-1, 1])

        if not np.isfinite(means).any():
            self.logger.critical("All DNGO predictions are NaN. Fall back to random predictions")
            return np.random.randn(means.shape[0], means.shape[1]), np.zeros_like(vars)
        else:
            return means, vars


class SimpleNetworkEmbedding:

    def __init__(
            self,
            seed: int,
            hidden_dims: typing.List[int] = [50, 50, 50],
            lr: float = 1e-3,
            momentum: float = 0.999,
            weight_decay: float = 1e-4,
            iterations: int = 10000,
            batch_size: int = 8,
            var: bool = True,
            lognormal_nllh: bool = False,
            **kwargs
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_loss = 1000
        self.log_error = iterations

        self.var = var
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.momentum = momentum
        self.iterations = iterations
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lognormal_nllh = lognormal_nllh

        if self.lognormal_nllh:
            assert self.var, "Can't train with lognormal nllh if no var is selected"

        self.model = None

    def _initialize_optimizer(self, len_train_loader):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay,
                                         momentum=self.momentum,
                                         nesterov=True)

        self.numbers_of_epoch = self.iterations / len_train_loader
        self.restart_after_num_epochs = np.ceil(self.numbers_of_epoch / 1)

        self.scheduler = torch.optim.lr_scheduler. \
            CosineAnnealingLR(optimizer=self.optimizer,
                              T_max=self.restart_after_num_epochs,
                              eta_min=1e-7, last_epoch=-1)
        if self.var:
            if self.lognormal_nllh:
                self.criterion = NLLHLogNormalLoss()
            else:
                self.criterion = NLLHLoss()
        else:
            self.criterion = nn.MSELoss()

    def train(self, X: np.ndarray, y: np.ndarray):
        train_dataset = XYDataset(X, y)
        print(X.shape, y.shape)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                   shuffle=True)
        self.model = NeuralNet(hidden_dims=self.hidden_dims, input_size=X.shape[1], feat_type=None, var=self.var)
        self.model.train()
        self.model.initialize_weights(X)
        self.model = self.model.double()
        if self.device.type == "cuda":
            # Move network to GPU
            self.model = self.model.cuda(device=self.device)

        try:
            from torchsummary import summary
            summary(self.model, input_size=(1, 1))
        except:
            print("torchsummary not installed")
        ###################

        self._initialize_optimizer(len_train_loader=len(train_loader))

        iteration = 1
        current_loss = []
        self.loss_curve = []
        self.error_curve = []
        self.lr_curve = []
        epoch_ct = 0

        x0 = []
        for param in self.model.parameters():
            print(param.data.dtype)
            x0.extend(param.data.detach().numpy().flatten())

        def loss_and_grad(x):
            self.model.zero_grad()
            self.criterion.zero_grad()
            x = np.array(x, float)
            x_idx = 0
            for param in self.model.parameters():
                if len(param.data.shape) == 2:
                    num_weights = int(param.data.shape[0] * param.data.shape[1])
                else:
                    num_weights = int(param.data.shape[0])
                shape = param.data.shape
                param.data = torch.tensor(x[x_idx: x_idx + num_weights], dtype=param.data.dtype).reshape(shape)
                x_idx += num_weights
            outputs = self.model(torch.tensor(X, dtype=param.data.dtype))
            loss = self.criterion(outputs, torch.tensor(y, dtype=param.data.dtype))
            rval = float(loss.item())
            loss.backward()
            x_idx = 0
            grad = np.zeros(x.shape)
            for param in self.model.parameters():
                if len(param.data.shape) == 2:
                    num_weights = int(param.data.shape[0] * param.data.shape[1])
                else:
                    num_weights = int(param.data.shape[0])
                grad[x_idx: x_idx + num_weights] = param.grad.detach().type(torch.DoubleTensor).numpy().flatten()
                x_idx += num_weights
            return rval, grad

        import scipy.optimize
        print('Gradient check', scipy.optimize.check_grad(lambda x: loss_and_grad(x)[0], lambda x: loss_and_grad(x)[0], x0))



        while iteration <= self.iterations:

            for iter, (x_train, y_train) in enumerate(train_loader):
                if iteration > self.iterations:
                    break
                self.optimizer.zero_grad()

                # Move tensors to the configured device
                x_train.requires_grad = True
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # Forward pass
                outputs = self.model(x_train)
                loss = self.criterion(outputs, y_train)

                # Backward and optimize
                loss.backward()
                from torch.autograd import gradcheck
                print('Grad check', gradcheck(self.criterion, (self.model(x_train), y_train)))

                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.01)
                # self.plot_grad_flow(self.model.named_parameters())

                self.optimizer.step()

                current_loss.append(loss.item())
                # Log stats
                if iteration % self.log_loss == 0:
                    print('iteration [{}/{}], Step [{}/{}], Avg.Loss: {:.4f}'
                          .format(iteration, self.iterations, iter + 1,
                                  len(train_loader), np.mean(current_loss[:-500])))

                if iteration % self.log_error == 0:
                    c = [self._eval(X=X, y=y), ]
                    print("{} iter: Train RMSE = {}, Validation RMSE = {}".format(iteration, c[0],
                                                                                  None if len(c) == 1 else c[
                                                                                      1]))
                    self.model.train()

                iteration += 1

                if epoch_ct % self.restart_after_num_epochs == 0 and epoch_ct > 0:
                    print("Reset scheduler")
                    self.scheduler.step(epoch=-1)
                elif (self.iterations - iteration) < len(train_loader):
                    # We can't make another full epoch
                    self.scheduler.step(epoch=self.restart_after_num_epochs)
                else:
                    self.scheduler.step()

            epoch_ct += 1

        x0 = []
        for param in self.model.parameters():
            x0.extend(param.data.detach().numpy().flatten())
        print(scipy.optimize.check_grad(lambda x: loss_and_grad(x)[0], lambda x: loss_and_grad(x)[0], x0))

        return self

    def predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        test_dataset = XYDataset(X, y=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset),
                                                  shuffle=False)
        y = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(self.device)
                out = self.model(x)
                if self.device.type == "cuda":
                    # Move network to GPU
                    out = out.cpu()
                y.extend(out.data.numpy())

        y = np.array(y).reshape([X.shape[0], -1])

        if self.lognormal_nllh:
            loc = copy.copy(y[:, 0])
            sigma_2 = copy.copy(y[:, 1])
            y[:, 0] = np.exp(loc + sigma_2 / 2)
            y[:, 1] = (np.exp(sigma_2) - 1) * np.exp(2 * loc + sigma_2)

        #y[:, 1] = np.exp(y[:, 1])
        return y

    def _eval(self, X, y):
        # Only if y is already transformed, not to be used from outside
        pred = self.predict(X)
        pred = np.array(pred[:, 0]).reshape([-1, 1])
        sc = rmse(pred_y=pred,
                  true_y=y, min_val=-np.inf,
                  cutoff=np.inf)
        return sc


class NeuralNet(nn.Module):

    def __init__(self, hidden_dims, input_size, feat_type=None, var=True):
        super(NeuralNet, self).__init__()
        self.feat_type = feat_type
        self.input_size = input_size
        self.num_neurons = hidden_dims
        self.activation = nn.Tanh
        self.num_layer = len(hidden_dims)
        if var:
            self.n_output = 2
        else:
            self.n_output = 1

        if self.feat_type is not None:
            print("Use cat embedding")
            assert len(self.feat_type) == self.input_size
            emb = nn.ModuleList()
            sz = 0
            for f in self.feat_type:
                if f == -1:
                    emb.append(None)
                    sz += 1
                else:
                    emb.append(nn.Embedding(int(f), int(f)))
                    sz += f

            num_neurons = [sz] + self.num_neurons
            self.embedding = emb
        else:
            num_neurons = [self.input_size] + self.num_neurons

        self.weights = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(self.num_layer):
            self.weights.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            self.acts.append(self.activation())

        self.outlayer = nn.Linear(num_neurons[-1], self.n_output)

    def initialize_weights(self, X):
        # Use Xavier normal intialization, slightly modified from "Understanding the difficulty of ..."
        for i in range(len(self.weights)):
            torch.nn.init.xavier_normal_(self.weights[i].weight)
            self.weights[i].bias.data.fill_(0)
        torch.nn.init.xavier_normal_(self.outlayer.weight)
        self.outlayer.bias.data.fill_(0)

    def forward(self, x):
        out = []
        if self.feat_type is not None and self.cat_embedding:
            for idx, (emb, typ) in enumerate(zip(self.embedding, self.feat_type)):
                if typ == -1:
                    out.append(x[:, idx].view(-1, 1))
                else:
                    out.append(emb(x[:, idx].long().view(-1, 1)).view([-1, typ]))
            out = torch.cat(out, 1)
        else:
            out = x

        for i in range(self.num_layer):
            out = self.weights[i](out)
            out = self.acts[i](out)
        out = self.outlayer(out)
        if self.n_output == 2:
            out[:, 1] = torch.log(1 + torch.exp(out[:, 1])) + 10e-6
        return out



class BayesianLinearRegressionLayer:

    def __init__(self, alpha=None, beta=None):

        self.alpha = alpha
        self.beta = beta

        # Those values will be initialized in calc_values()
        # Meaning: See Paper from Snoek
        self.K = None
        self.K_inv = None
        self.m = None
        self.phi_train = None

        # Number features in hidden layer
        self.D = None

    def predict(self, model, xtest):
        xtest = torch.Tensor(xtest).float()
        phi_x = self.get_features(model, xtest)

        # make sure to run once the optimizing step before making predictions.
        assert self.K is not None \
               and self.K_inv is not None \
               and self.m is not None \
               and self.alpha is not None \
               and self.beta is not None

        mu = torch.mv(phi_x, self.m)
        s2 = torch.mul(phi_x.t(), torch.mm(self.K_inv, phi_x.t())).sum(0).add(1 / self.beta)

        return mu, np.clip(s2, a_min=1e-5, a_max=np.inf)

    def calc_values(self, model, xtrain, ytrain, alpha, beta):
        xtrain = torch.Tensor(xtrain).float().view([-1, xtrain.shape[1]])
        ytrain = torch.Tensor(ytrain).float().view([-1, 1])

        # Updates the basis matrices and stores them for later use
        #  (like predicting)
        self.phi_train = self.get_features(model, xtrain)

        self.D = self.phi_train.shape[1]

        # self.K = torch.addmm(alpha ** 2, torch.eye(self.D), beta, self.phi_train.t(), self.phi_train)
        self.K = torch.addmm(alpha, torch.eye(self.D), beta, self.phi_train.t(), self.phi_train)
        self.K_inv = torch.inverse(self.K)
        self.m = torch.mv(torch.mm(self.K_inv, self.phi_train.t()), ytrain.view((-1,))).mul(beta)

        self.alpha = alpha
        self.beta = beta

    def get_features(self, model, inputs):
        # Returns the phi for a some inputs (like train or test data)
        # https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

        ninputs = inputs.shape[0]
        nbasis = model.weights[-1].in_features
        phi = torch.Tensor(ninputs, nbasis)
        # For Phi use last hidden layer, not output layer
        #print(model.weights)
        hidden_layer = model.weights[-1]

        model.eval()

        def copy_data(m, i, o):
            phi.copy_(o.data.detach())

        h = hidden_layer.register_forward_hook(copy_data)

        model(inputs)
        h.remove()

        return phi

    def optimize_alpha_beta(self, model, xtrain, ytrain):
        xtrain = torch.tensor(xtrain).float().view([-1, xtrain.shape[1]])
        ytrain = torch.tensor(ytrain).float().view([-1, 1])
        # Use in optimization step previous alpha/beta values
        # If not available guess them random
        bounds = [(1e-2, 1e-1), (1e-3, 1e1)]
        if self.alpha is None or self.beta is None:
            alpha_beta = np.random.uniform([bounds[0][0], bounds[1][0]],
                                           [bounds[0][1], bounds[1][1]])
            # alpha_beta = np.random.rand(2)
        else:
            alpha_beta = np.array([self.alpha, self.beta])

        # TODO FIX THIS THING WITH THE ARGS
        # args = (model, xtrain, ytrain)
        # self.xtrain = xtrain
        # self.ytrain = ytrain

        # TODO: WIE GEHT DAS MIT fmin l bfgs b
        # res = optimize.fmin(self.optimizing_step, np.random.rand(2),
        #                     args=(model,), bounds=bounds)

        # self.hypers = [[np.exp(res[0]), np.exp(res[1])]]
        print(alpha_beta, bounds)
        res = optimize.minimize(
            self.optimizing_step, alpha_beta, args=(model, xtrain, ytrain),
            method='L-BFGS-B',
            tol=1e-7,
            bounds=bounds
        )
        print(res)
        # bounds=[(1e1,1e6), (1e-3,1e3)]
        self.alpha, self.beta = res['x']
        # self.calc_values(model, xtrain, ytrain, self.alpha, self.beta)

    def optimizing_step(self, alpha_beta, *args):
        """
        Wrapper to optimize alpha and beta values according to the negative mll.

        Args:
            alpha_beta (tuple):
                alpha and beta are precisions.
            arg:
                arguments for fmin call. Here a tuple with the model.

        Returns:
            negative marginal log likelihood
        """

        # 1. Receives alpha and beta values
        alpha, beta = alpha_beta
        model, xtrain, ytrain = args

        # 2. Updates the basis matrix m, K, K^-1, which are necessary
        #    for the nmll and dependant on alpha beta
        self.calc_values(model, xtrain, ytrain, alpha, beta)

        # 3. Calcuates the negative mll and returns it
        return self.marginal_loglikelihood(alpha, beta, ytrain)

    def marginal_loglikelihood(self, alpha, beta, y):
        """
        Marginalised log likelihood over weights w following [1]
        It is using precalculated values m, K, phi_train.

        [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15

        Args:
            alpha: Precision
            beta:  Precision
            y:  true labels

        Returns:
            float: negative marginal log likelihood

        """
        N = len(y)

        ll = (self.D / 2) * np.log(alpha)
        ll += (N / 2) * np.log(beta)
        ll -= (N / 2) * np.log(2 * np.pi)
        ll -= (beta / 2) * torch.norm((y.view((-1)) - torch.mv(self.phi_train, self.m)), 2)
        ll -= (alpha / 2) * np.dot(self.m, self.m)  # it calculate a/2 * m.T x m

        det = np.linalg.slogdet(self.K.numpy())
        ll -= 0.5 * np.clip(det[1], a_min=1e-10, a_max=np.inf)
        return -ll
