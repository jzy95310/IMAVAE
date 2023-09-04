# ivae.py: a slightly modified version of the identifiable variational autoencoder (iVAE) model
# Reference: Khemakhem, Ilyes, et al. "Variational autoencoders and nonlinear ica: A unifying framework." 
# International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

from numbers import Number

import math
import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)

ACTIVATIONS = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "elu": torch.nn.ELU(),
    "selu": torch.nn.SELU(),
    "softplus": torch.nn.Softplus(),
    "softmax": torch.nn.Softmax(),
    "erf": Erf(),
}

class AdaptiveAverageUnpool2d(nn.Module):
    def __init__(self, output_size: _size_2_t = None) -> None:
        super(AdaptiveAverageUnpool2d, self).__init__()

        self.output_size: _size_2_t = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.output_size, mode='nearest')


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class ConvNet(nn.Module):
    def __init__(self, input_width: int, input_height: int, in_channels: int, latent_feature_dim: int, num_blocks: int, 
                 num_intermediate_channels: int = 64, kernel_size: int = 3, stride: int = 1, use_batch_norm: bool = False, 
                 activation: str = 'relu', adaptive_avgpool_size: int = 7, aux_dim: int = 0, num_hidden_dense_layers: int = 2, 
                 num_dense_units: int = 512, dropout_ratio: float = 0.0, device: str = 'cpu', transpose: bool = False) -> None:
        super(ConvNet, self).__init__()
        self.input_width: int = input_width
        self.input_height: int = input_height
        self.in_channels = in_channels
        self.latent_feature_dim = latent_feature_dim
        self.num_blocks = num_blocks
        self.num_intermediate_channels = num_intermediate_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.adaptive_avgpool_size = adaptive_avgpool_size
        self.aux_dim = aux_dim
        self.num_hidden_dense_layers = num_hidden_dense_layers
        self.num_dense_units = num_dense_units
        self.dropout_ratio = dropout_ratio
        self.device = device
        self.transpose = transpose

        self.depth = self.num_blocks + self.num_hidden_dense_layers + 1
        self._validate_inputs()
        self._build_layers()
        self.to(self.device)
    
    def _validate_inputs(self) -> None:
        assert self.latent_feature_dim > 0, "The number of output features should be positive."
        assert self.num_blocks >= 0, "The number of blocks should be non-negative."
        assert self.activation in ACTIVATIONS, "The activation function should be one of the following: {}".format(ACTIVATIONS.keys())
        assert 0.0 <= self.dropout_ratio <= 1.0, "The dropout ratio should be between 0.0 and 1.0."
        assert self.input_width > 0, "The width of input features should be positive."
        assert self.input_height > 0, "The height of input features should be positive."
        assert self.in_channels > 0, "The number of input channels should be positive."
        assert self.num_intermediate_channels > 0, "The number of intermediate channels should be positive."
        assert self.kernel_size > 0, "The kernel size should be positive."
        assert self.stride > 0, "The stride should be positive."
        assert self.adaptive_avgpool_size > 0, "The adaptive average pool size should be positive."
        assert self.num_hidden_dense_layers >= 0, "The number of hidden dense layers should be non-negative."
        assert self.num_dense_units > 0, "The number of dense units should be positive."
    
    def _build_conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, 
                          use_batch_norm: bool, activation: str, transpose: bool) -> nn.Sequential:
        """
        Build a convolutional block
        """
        if not transpose:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            bn_layer = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        else:
            conv_layer = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride)
            bn_layer = nn.BatchNorm2d(in_channels) if use_batch_norm else nn.Identity()
        return nn.Sequential(
            conv_layer,
            bn_layer,
            ACTIVATIONS[activation]
        )
    
    def _build_dense_block(self, input_dim: int, output_dim: int, activation: str, dropout_ratio: float, 
                           transpose: bool) -> nn.Sequential:
        """
        Build a dense block
        """
        linear_layer = nn.Linear(input_dim, output_dim) if not transpose else nn.Linear(output_dim, input_dim)
        return nn.Sequential(
            linear_layer,
            ACTIVATIONS[activation],
            nn.Dropout(dropout_ratio)
        )
    
    def _build_layers(self) -> None:
        self.conv_blocks: nn.ModuleList = nn.ModuleList()
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
        block_expansion = self.num_intermediate_channels*self.adaptive_avgpool_size**2 if self.num_blocks > 0 else self.in_channels*self.input_width*self.input_height
        width_before_avgpool, height_before_avgpool = self.input_width, self.input_height

        if self.num_blocks > 0:
            self.conv_blocks.append(
                self._build_conv_block(self.in_channels, self.num_intermediate_channels, self.kernel_size, self.stride,
                                       self.use_batch_norm, self.activation, self.transpose)
            )
            width_before_avgpool = math.floor((width_before_avgpool - self.kernel_size) / self.stride) + 1
            height_before_avgpool = math.floor((height_before_avgpool - self.kernel_size) / self.stride) + 1
            for _ in range(self.num_blocks - 1):
                self.conv_blocks.append(
                    self._build_conv_block(self.num_intermediate_channels, self.num_intermediate_channels, self.kernel_size, 
                                           self.stride, self.use_batch_norm, self.activation, self.transpose)
                )
                width_before_avgpool = math.floor((width_before_avgpool - self.kernel_size) / self.stride) + 1
                height_before_avgpool = math.floor((height_before_avgpool - self.kernel_size) / self.stride) + 1
            if not self.transpose:
                self.conv_blocks.append(nn.AdaptiveAvgPool2d(self.adaptive_avgpool_size))
            else:
                self.conv_blocks.append(AdaptiveAverageUnpool2d((width_before_avgpool, height_before_avgpool)))
        if not self.transpose:
            self.conv_blocks.append(nn.Flatten())
        else:
            if self.num_blocks > 0:
                self.conv_blocks.append(nn.Unflatten(1, (self.num_intermediate_channels, self.adaptive_avgpool_size, self.adaptive_avgpool_size)))
            else:
                self.conv_blocks.append(nn.Unflatten(1, (self.in_channels, self.input_width, self.input_height)))
        if self.num_hidden_dense_layers == 0:    
            self.dense_blocks.append(
                nn.Linear(block_expansion + self.aux_dim, self.latent_feature_dim)
            )
        else:
            self.dense_blocks.append(
                self._build_dense_block(block_expansion + self.aux_dim, self.num_dense_units, self.activation, self.dropout_ratio, 
                                        self.transpose)
            )
            for _ in range(self.num_hidden_dense_layers - 1):
                self.dense_blocks.append(
                    self._build_dense_block(self.num_dense_units, self.num_dense_units, self.activation, self.dropout_ratio, 
                                            self.transpose)
                )
            last_linear_layer = nn.Linear(self.num_dense_units, self.latent_feature_dim) if not self.transpose else nn.Linear(self.latent_feature_dim, self.num_dense_units)
            self.dense_blocks.append(
                last_linear_layer
            )
        if self.transpose:
            self.conv_blocks = self.conv_blocks[::-1]
            self.dense_blocks = self.dense_blocks[::-1]
    
    def forward(self, x: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        if not self.transpose:
            for conv_block in self.conv_blocks:
                x = conv_block(x)
            if aux is not None:
                x = torch.cat((x, aux), dim=1)
            for dense_block in self.dense_blocks:
                x = dense_block(x)
            return x
        else:
            for dense_block in self.dense_blocks:
                x = dense_block(x)
            for conv_block in self.conv_blocks:
                x = conv_block(x)
            return x


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt().squeeze())
        scaled = scaled.reshape(-1,1) if len(scaled.shape) == 1 else scaled
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + 2*v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Normal(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)


    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        # prior_params
        # self.prior_mean = torch.zeros(1).to(device)
        self.prior_mean = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                              device=device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.logd = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        logd = self.logd(s)
        return f, logd.exp()

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean(u), logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)   # shape: (batch_size,)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)   # shape: (batch_size,)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)   # shape: (batch_size,)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)    # shape: (batch_size, batch_size, latent_dim), temporary variable
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)   # shape: (batch_size,)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)   # shape: (batch_size,)

            """
            The annealing parameters a, b, c, and d control the rate at which the weight of the KL divergence term is 
            increased during training. The parameter a controls the weight of the reconstruction term, while b, c, 
            and d control the weight of the KL divergence term at different stages of training.
            Note that the KL divergence term in ELBO is KL(q(z|x,u)||p(z|u)) = log(q(z|x,u)) - log(p(z|u))

            The KL divergence term here is decomposed into 3 terms: : b * (log_qz_xu - log_qz), c * (log_qz - log_qz_i), 
            and d * (log_qz_i - log_pz_u), where log_qz is an estimate of the log probability of the variational 
            distribution over the latent variables, and log_qz_i is the average of the log probabilities of the 
            variational distribution over the latent variables given each individual data point. We try to make the training
            process more stable by controlling the values of a, b, c, and d because the variational distribution log_qz_xu 
            can be difficult to estimate accurately, especially early in training. 
            """
            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)   # a is increasing with it
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))   # b is decreasing with it
        self._training_hyperparams[2] = min(1, it / thr)   # c is increasing with it
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))   # d is decreasing with it
        if it > thr:
            self.anneal_params = False


class ConvIVAE(iVAE):
    def __init__(self, data_width, data_height, data_channels, aux_dim, latent_feature_dim, n_layers, hidden_dim, 
                 activation='lrelu', prior=None, decoder=None, encoder=None, device='cpu', anneal=False):
        super(ConvIVAE, self).__init__(latent_feature_dim, data_width * data_height * data_channels, aux_dim, prior=prior,
                                       decoder=decoder, encoder=encoder, n_layers=n_layers, hidden_dim=hidden_dim,
                                       activation=activation, device=device, anneal=anneal)
        
        self.data_width = data_width
        self.data_height = data_height
        self.data_channels = data_channels
        self.aux_dim = aux_dim
        self.latent_feature_dim = latent_feature_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder
        
        # prior_params
        self.prior_mean = MLP(aux_dim, latent_feature_dim, hidden_dim, n_layers, activation=activation, slope=.1,
                              device=device)
        self.logl = MLP(aux_dim, latent_feature_dim, hidden_dim, n_layers, activation=activation, slope=.1, 
                        device=device)
        # decoder params
        self.f = ConvNet(data_width, data_height, data_channels, latent_feature_dim, n_layers, aux_dim=0,
                         num_hidden_dense_layers=1, activation='leaky_relu', num_dense_units=hidden_dim, device=device,
                         transpose=True)
        self.logd = ConvNet(data_width, data_height, data_channels, latent_feature_dim, n_layers, aux_dim=0,
                            num_hidden_dense_layers=1, activation='leaky_relu', num_dense_units=hidden_dim, device=device,
                            transpose=True)
        # encoder params
        self.g = ConvNet(data_width, data_height, data_channels, latent_feature_dim, n_layers, aux_dim=aux_dim,
                         num_hidden_dense_layers=1, activation='leaky_relu', num_dense_units=hidden_dim, device=device)
        self.logv = ConvNet(data_width, data_height, data_channels, latent_feature_dim, n_layers, aux_dim=aux_dim,
                            num_hidden_dense_layers=1, activation='leaky_relu', num_dense_units=hidden_dim, device=device)
        
        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]
    
    def encoder_params(self, x, u):
        g = self.g(x, u)
        logv = self.logv(x, u)
        return g, logv.exp()
    
    def decoder_params(self, s):
        f = self.f(s)
        logd = self.logd(s)
        return f.view(f.shape[0],-1), logd.exp().view(logd.shape[0],-1)
    
    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        x = x.view(x.shape[0], -1)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)   # shape: (batch_size,)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)   # shape: (batch_size,)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)   # shape: (batch_size,)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)    # shape: (batch_size, batch_size, latent_dim), temporary variable
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)   # shape: (batch_size,)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)   # shape: (batch_size,)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z
        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z