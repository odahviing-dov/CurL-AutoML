import torch.nn as nn
from torch.utils.data import Dataset
import torch
import typing
import numpy as np
import copy
import itertools
import logging
from scipy import optimize
from sklearn.metrics import mean_squared_error

from smac.epm.base_rf import BaseModel
from smac.configspace import ConfigurationSpace
from smac.utils.logging import PickableLoggerAdapter


import torch
from functools import reduce
from torch.optim.optimizer import Optimizer, required


class SGDHD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, hypergrad_lr=1e-6):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, hypergrad_lr=hypergrad_lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDHD, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SGDHD doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._params_numel = reduce(lambda total, p: total + p.numel(), self._params, 0)

    def _gather_flat_grad_with_weight_decay(self, weight_decay=0):
        views = []
        for p in self._params:
            if p.grad is None:
                view = torch.zeros_like(p.data)
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            if weight_decay != 0:
                view.add_(weight_decay, p.data.view(-1))
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._params_numel

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        grad = self._gather_flat_grad_with_weight_decay(weight_decay)

        # NOTE: SGDHD has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        # State initialization
        if len(state) == 0:
            state['grad_prev'] = torch.zeros_like(grad)

        grad_prev = state['grad_prev']
        # Hypergradient for SGD
        h = torch.dot(grad, grad_prev)
        # Hypergradient descent of the learning rate:
        group['lr'] += group['hypergrad_lr'] * h

        if momentum != 0:
            if 'momentum_buffer' not in state:
                buf = state['momentum_buffer'] = torch.zeros_like(grad)
                buf.mul_(momentum).add_(grad)
            else:
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad.add_(momentum, buf)
            else:
                grad = buf

        state['grad_prev'] = grad

        self._add_grad(-group['lr'], grad)

        return loss


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
        # Assuming network outputs std
        # TODO double check with Deep Learning Book (page  183)
        std = input[:, 1].view(-1, 1)
        mu = input[:, 0].view(-1, 1)
        # std = torch.exp(input[:, 1])
        # std = torch.clamp_min(std, 0.1)
        # std = torch.zeros_like(mu) + 1

        n = torch.distributions.normal.Normal(mu, std)
        loss = n.log_prob(target)

        # PDF
        #var = input[:, 1].view(-1, 1)
        #t1 = 0.5 * torch.log(var)
        #t2 = ((target - mu) ** 2.0) / (2.0 * var ** 2)
        #man_loss = -t1 - t2
        #print("M", man_loss)
        #print("Auto", loss)
        return -torch.mean(loss)


class XYCDataset(Dataset):

    def __init__(self, X, y=None, cens_idx=None):
        self.X = torch.Tensor(X)

        if y is not None:
            self.y = torch.Tensor(y).reshape([self.X.shape[0], -1])
        else:
            self.y = None

        self.cens_idx = torch.Tensor(cens_idx).reshape([self.X.shape[0], -1])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None and self.cens_idx is None:
            return self.X[idx, :], self.y[idx, :]
        if self.y is not None and self.cens_idx is not None:
            return self.X[idx, :], self.y[idx, :], self.cens_idx[idx]
        if self.y is None and self.cens_idx is not None:
            return self.X[idx, :], self.cens_idx[idx]

        else:
            return self.X[idx, :]


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
        self.X = torch.Tensor(X)
        if y is not None:
            self.y = torch.Tensor(y).reshape([self.X.shape[0], -1])
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx, :], self.y[idx, :]
        else:
            return self.X[idx, :]


class SimpleNetworkEmbedding:

    def __init__(
            self,
            seed: int,
            feat_types: typing.List[int] = None,
            hidden_dims: typing.List[int] = [50, 50, 50],
            lr: float = 1e-3,
            momentum: float = 0.999,
            weight_decay: float = 1e-4,
            iterations: int = 10000,
            batch_size: int = 8,
            var: bool = True,
            lognormal_nllh: bool = False,
            var_bias_init: float = 1,
            max_cat: int = np.inf,
            learned_weight_init: bool = False,
            optimization_algorithm: str = 'sgd',
            **kwargs
    ):
        self.logger = PickableLoggerAdapter(self.__module__ + "." + self.__class__.__name__)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_loss = 1000
        self.log_error = iterations
        self.seed = seed
        self.var = var
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.momentum = momentum
        self.iterations = iterations
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lognormal_nllh = lognormal_nllh
        self.var_bias_init = var_bias_init
        self.max_cat = max_cat
        self.learned_weight_init = learned_weight_init
        self.optimization_algorithm = optimization_algorithm

        self.feat_types = feat_types

        if self.lognormal_nllh:
            assert self.var, "Can't train with lognormal nllh if no var is selected"

        self.model = None

    def _initialize_optimizer(self, len_train_loader):
        if self.optimization_algorithm == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay,
                                             momentum=self.momentum,
                                             nesterov=True)
        elif self.optimization_algorithm == 'sgd-hd':
            self.optimizer = SGDHD(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay,
                                   momentum=self.momentum,
                                   hypergrad_lr=1e-3)
        else:
            self.optimizer = None

        self.numbers_of_epoch = self.iterations / len_train_loader
        self.restart_after_num_epochs = np.ceil(self.numbers_of_epoch / 1)

        self.scheduler = torch.optim.lr_scheduler. \
            CosineAnnealingLR(optimizer=self.optimizer,
                              T_max=self.restart_after_num_epochs,
                              eta_min=1e-7, last_epoch=-1)
        if self.var:
            if self.lognormal_nllh:
                raise NotImplementedError()
            else:
                self.criterion = NLLHLoss()
        else:
            self.criterion = nn.MSELoss()

    def reset(self):
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        train_dataset = XYDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                   shuffle=True)
        if self.model is None:
            torch.manual_seed(self.seed)
            self.model = NeuralNet(hidden_dims=self.hidden_dims, input_size=X.shape[1],
                                   feat_type=self.feat_types, var=self.var, max_cat=self.max_cat)
            self.model.train()
            if self.learned_weight_init:
                self.model.learn_initial_weights(X)
            else:
                self.model.initialize_weights(var_bias_init=self.var_bias_init)
            self._initialize_optimizer(len_train_loader=len(train_loader))
        else:
            print("Continue training")
            self._initialize_optimizer(len_train_loader=len(train_loader))

        if self.device.type == "cuda":
            # Move network to GPU
            self.model = self.model.cuda(device=self.device)

        try:
            from torchsummary import summary
            #summary(self.model, input_size=(1, 1))
        except:
            print("torchsummary not installed")
        ###################

        if self.optimization_algorithm in ['sgd', 'sgd-hd']:
            iteration = 1
            self.current_loss = []
            self.loss_curve = []
            self.error_curve = []
            self.lr_curve = []
            epoch_ct = 0
            while iteration <= self.iterations:

                for iter, (x_train, y_train) in enumerate(train_loader):
                    if iteration > self.iterations:
                        break
                    self.optimizer.zero_grad()

                    # Move tensors to the configured device
                    x_train = x_train.to(self.device)
                    y_train = y_train.to(self.device)

                    # Forward pass
                    outputs = self.model(x_train)
                    loss = self.criterion(outputs, y_train)

                    # Backward and optimize
                    loss.backward()

                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    # self.plot_grad_flow(self.model.named_parameters())

                    self.optimizer.step()

                    self.current_loss.append(loss.item())
                    # Log stats
                    if iteration % self.log_loss == 0:
                        print('iteration [{}/{}], Step [{}/{}], Avg.Loss: {:.4f}'
                              .format(iteration, self.iterations, iter + 1,
                                      len(train_loader), np.mean(self.current_loss[-500:])))

                    if iteration % self.log_error == 0:
                        c = [self._eval(X=X, y=y), ]
                        print("{} iter: Train RMSE = {}, Validation RMSE = {}".format(iteration, c[0],
                                                                                      None if len(c) == 1 else c[1]))
                        self.model.train()

                    iteration += 1

                # if epoch_ct % self.restart_after_num_epochs == 0 and epoch_ct > 0:
                #     print("Reset scheduler")
                #     self.scheduler.step(epoch=-1)
                # elif (self.iterations - iteration) < len(train_loader):
                #     # We can't make another full epoch
                #     self.scheduler.step(epoch=self.restart_after_num_epochs)
                # else:
                #     self.scheduler.step()

                epoch_ct += 1

        elif self.optimization_algorithm == 'rprop':
            iteration = 1
            self.current_loss = []
            self.loss_curve = []
            self.error_curve = []
            self.lr_curve = []
            epoch_ct = 0

            self.optimizer = torch.optim.Rprop(self.model.parameters(), lr=self.lr,)

            x_train = torch.tensor(X, dtype=torch.float32)
            y_train = torch.tensor(y, dtype=torch.float32)

            while iteration <= self.iterations:
                if iteration > self.iterations:
                    break
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x_train)
                loss = self.criterion(outputs, y_train)

                # Backward and optimize
                loss.backward()

                self.optimizer.step()

                self.current_loss.append(loss.item())
                # Log stats
                if iteration % self.log_loss == 0:
                    print('iteration [{}/{}], Loss: {:.4f}'
                          .format(iteration, self.iterations, self.current_loss[-1]))

                if iteration % self.log_error == 0:
                    c = [self._eval(X=X, y=y), ]
                    print("{} iter: Train RMSE = {}, Validation RMSE = {}".format(iteration, c[0],
                                                                                  None if len(c) == 1 else c[1]))
                    self.model.train()

                iteration += 1

            epoch_ct += 1

        elif self.optimization_algorithm == 'lbfgs':
            # import pickle
            # with open('/tmp/data.pkl', 'wb') as fh:
            #     pickle.dump((X, y), fh)
            #     exit(1)

            # Alternative optimization based on scipy's L-BFGS
            iteration = 1
            self.current_loss = []
            self.loss_curve = []
            self.error_curve = []
            self.lr_curve = []

            x0 = []
            for param in self.model.parameters():
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
                for param in self.model.parameters():
                    if len(param.shape) == 1:
                        continue
                    loss += self.weight_decay * param.data.norm(2) ** 2
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
            # L-BFGS can be tuned quite a bit. I believe this setup is kind of heavily tuned towards well-performing
            # networks
            res = scipy.optimize.fmin_l_bfgs_b(
                loss_and_grad,
                x0=x0,
                # m=50,       # Quality of approximation of the Hessian
                # maxls=50,   # Maximal number of line search steps
                # factr=1,   # Stopping criterion, according to the scipy docs, 10 stands for good approximation
            )

            self.model.zero_grad()
            self.criterion.zero_grad()
            x = np.array(res[0], float)
            x_idx = 0
            for param in self.model.parameters():
                if len(param.data.shape) == 2:
                    num_weights = int(param.data.shape[0] * param.data.shape[1])
                else:
                    num_weights = int(param.data.shape[0])
                shape = param.data.shape
                param.data = torch.tensor(x[x_idx: x_idx + num_weights], dtype=param.data.dtype).reshape(shape)
                x_idx += num_weights

            self.current_loss.append(res[1])
            print(res)
            c = [self._eval(X=X, y=y), ]
            print(
                "{} iter: Train RMSE = {}, Validation RMSE = {}".format(iteration, c[0], None if len(c) == 1 else c[1]))
            self.model.zero_grad()
            self.criterion.zero_grad()
        else:
            raise ValueError(self.optimization_algorithm)
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
                    # Move network to CPU
                    out = out.cpu()
                y.extend(out.data.numpy())

        y = np.array(y).reshape([X.shape[0], -1])
        # Torch network outputs std, but this class returns var
        y[:, 1] = y[:, 1]**2

        if self.lognormal_nllh:
            loc = copy.copy(y[:, 0])
            sigma_2 = copy.copy(y[:, 1])
            y[:, 0] = np.exp(loc + sigma_2 / 2)
            y[:, 1] = (np.exp(sigma_2) - 1) * np.exp(2 * loc + sigma_2)
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

    def __init__(self, hidden_dims, input_size, feat_type=None, var: bool = True, max_cat: int = np.inf):
        super(NeuralNet, self).__init__()
        self.logger = PickableLoggerAdapter(self.__module__ + "." + self.__class__.__name__)

        self.feat_type = feat_type
        self.input_size = input_size
        self.num_neurons = hidden_dims
        self.activation = nn.Tanh
        self.num_layer = len(hidden_dims)
        self.max_cat = max_cat
        if var:
            self.n_output = 2
        else:
            self.n_output = 1

        if np.sum(self.feat_type) == 0:
            self.feat_type = None

        if self.feat_type is not None:
            self.logger.info("Use cat embedding")
            assert len(self.feat_type) == self.input_size
            emb = nn.ModuleList()
            sz = int(0)
            for f in self.feat_type:
                if f == 0:
                    # In SMAC 0 encodes a numerical
                    emb.append(None)
                    sz += 1
                else:
                    es = min(self.max_cat, int(f))
                    emb.append(nn.Embedding(int(f), es))
                    sz += es
            assert int(sz) == sz
            sz = int(sz)
            num_neurons = [sz] + self.num_neurons
            self.embedding = emb
        else:
            num_neurons = [self.input_size] + self.num_neurons

        self.weights = nn.ModuleList()
        self.acts = nn.ModuleList()

        print(num_neurons)
        for i in range(self.num_layer):
            self.weights.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            self.acts.append(self.activation())

        self.outlayer = nn.Linear(num_neurons[-1], self.n_output)

    def initialize_weights(self, var_bias_init: float = 1):
        # Use Xavier normal intialization, slightly modified from "Understanding the difficulty of ..."
        for i in range(len(self.weights)):
            torch.nn.init.xavier_normal_(self.weights[i].weight)
            self.weights[i].bias.data.fill_(0)
        torch.nn.init.xavier_normal_(self.outlayer.weight)
        # TODO Second bias should be initialised to np.log(np.exp(x) - 1), s.t. softplus = x
        self.outlayer.bias.data[0].fill_(0)
        if var_bias_init == 0:
            self.logger.critical("Can't properly initialize bias unit, initialize wih zero")
            self.outlayer.bias.data[0].fill_(0)
        else:
            self.outlayer.bias.data[1].fill_(np.log(np.exp(var_bias_init) - 1))

    def learn_initial_weights(self, X):
        """Learn initial weights such that the mean over the data is on average zero per neuron"""
        output = torch.tensor(X, dtype=torch.float32)
        for i in range(len(self.weights)):
            torch.nn.init.xavier_normal_(self.weights[i].weight, torch.nn.init.calculate_gain('tanh'))
            self.weights[i].bias.data.fill_(0)
            output2 = self.weights[i].forward(output)
            mean = output2.mean(axis=0)
            self.weights[i].bias.data = -mean
            output = self.weights[i].forward(output)
            output = self.acts[i](output)
            # print(output.mean(axis=0), output.mean(axis=0).shape)
        torch.nn.init.xavier_normal_(self.outlayer.weight, torch.nn.init.calculate_gain('tanh'))
        self.outlayer.bias.data.fill_(0)
        # self.outlayer.bias.data[1].fill_(np.log(np.exp(1) - 1))
        # Noise can be tuned here...
        self.outlayer.bias.data[1] = -5

    def forward(self, x):
        out = []
        if self.feat_type is not None:
            for idx, (emb, typ) in enumerate(zip(self.embedding, self.feat_type)):
                if typ == 0:
                    # a numerical
                    out.append(x[:, idx].view(-1, 1))
                else:
                    # a categorical
                    out.append(emb(x[:, idx].long().view(-1, 1)).view([-1, min(self.max_cat, typ)]))
            out = torch.cat(out, 1)
        else:
            out = x

        for i in range(self.num_layer):
            out = self.weights[i](out)
            out = self.acts[i](out)
        out = self.outlayer(out)
        if self.n_output == 2:
            # Passing second output through softplus function (see Lakshminarayanan (2017))
            out[:, 1] = torch.log(1 + torch.exp(out[:, 1])) + 10e-6
        return out
