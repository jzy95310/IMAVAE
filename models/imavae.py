# imavae.py: Identifiable Mediation Analysis Variational Autoencoder (IMAVAE) model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
import warnings

from .ivae import iVAE, ConvIVAE, Normal
from .vae import VAE
from .nmf_base import NmfBase


class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)

ACTIVATIONS = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "lrelu": torch.nn.LeakyReLU(),
    "elu": torch.nn.ELU(),
    "selu": torch.nn.SELU(),
    "softplus": torch.nn.Softplus(),
    "softmax": torch.nn.Softmax(),
    "erf": Erf(),
}

class IMAVAE(NmfBase):
    """
    Identifiable Mediation Analysis Variational Autoencoder (IMAVAE) model

    Parameters
    ----------
    n_components : int, optional
        number of networks to learn or latent dimensionality.
        Defaults to ``32``.
    device : ``{'cuda','cpu','auto'}``, optional
        Torch device. Defaults to ``'auto'``
    n_intercepts : int, optional
        Number of unique intercepts for linear/logistic regression,
        defaults to 1
    n_sup_networks : int, optional
        Number of networks that will be supervised
        ``0 < n_sup_networks < n_components``. Defaults to ``1``.
    decoder_type : ``{'VAE','NMF'}``, optional
        Decoder type. Defaults to ``'VAE'``.
    recon_loss : ``{'IS', 'MSE'}``, optional
        Reconstruction loss function. Defaults to ``'MSE'``.
    recon_weight : float, optional
        Importance weight for the reconstruction. Defaults to ``1.0``.
    elbo_weight : float, optional
        Importance weight for the ELBO. Defaults to ``1.0``.
    sup_weight : float, optional
        Importance weight for the supervision. Defaults to ``1.0``.
    sup_recon_weight : float, optional
        Importance weight for the reconstruction of the supervised component.
        Defaults to ``1.0``.
    n_hidden_layers: int, optional
        Number of hidden layers in the encoder and decoder of iVAE. 
        Defaults to ``2``.
    hidden_dim: int, optional
        Number of hidden units in the encoder and decoder of iVAE. 
        Defaults to ``50``.
    n_sup_hidden_layers: int, optional
        Number of hidden layers in the supervised networks. 
        Defaults to ``0``.
    n_sup_hidden_dim: int, optional
        Number of hidden units in the supervised networks. 
        Defaults to ``30``.
    sup_conditional_gaussian: bool, optional
        Whether to use a conditional Gaussian distribution for the supervised
        network. Defaults to ``False``.
    aug_aux_dim: int, optional
        If given, the auxiliary variable will be augmented into
        the specified dimensionality. Defaults to ``None``.
    activation : ``{'lrelu','sigmoid','tanh','none'}``, optional
        Activation function for the encoder and decoder of iVAE. 
        Defaults to ``'lrelu'``.
    predictor_type : ``{'linear','logistic'}``, optional
        Type of predictor to use for supervised networks. Defaults to
        ``'linear'``.
    sup_recon_type : ``{'Residual', 'All'}``, optional
        Which supervised component reconstruction loss to use. Defaults to
        ``'Residual'``. ``'Residual'`` estimates network scores optimal for
        reconstruction and penalizes deviation of the real scores from those
        values. ``'All'`` evaluates the reconstruction loss of the supervised
        network reconstruction against all features.
        Only work for ``decoder_type='NMF'``.
    feature_groups : ``None`` or list of int, optional
        Indices of the divisions of feature types. Defaults to ``None``.
    group_weights : ``None`` or list of floats, optional
        Weights for each of the feature types. Defaults to ``None``.
    fixed_corr : ``None`` or list of str, optional
        List the same length as ``n_sup_networks`` indicating correlation
        constraints for the network. Defaults to ``None``. ``'positive'``
        constrains a supervised network to have a positive correlation between
        score and label. ``'negative'`` constrains a supervised network to have
        a negative correlation between score and label. ``'n/a'`` applies no
        constraint, meaning the supervised network can be positive or
        negatively correlated.
    sup_smoothness_weight : float, optional
        Encourages smoothness for the supervised network. Defaults to ``1.0``.
    optim_name : ``{'SGD','Adam','AdamW'}``, optional
        torch.optim algorithm to use in . Defaults to ``'AdamW'``.
    momentum : float, optional
        Momentum value if optimizer is ``'SGD'``. Defaults to ``0.9``.
    lr : float, optional
        Learning rate for the optimizer. Defaults to ``1e-3``.
    weight_decay : float, optional
        Weight decay for the optimizer. Defaults to ``0``.
    anneal : bool, optional
        Whether to apply parameter annealing when learning the iVAE.
        Defaults to ``True``.
    identifiable : bool, optional
        If True, use iVAE for the backbone network. If False, use regular VAE.
    model_name : str, optional
        Name of the model. Defaults to ``'imavae'``.
    save_folder : str, optional
        Location to save the best pytorch model parameters. 
        Defaults to './model_ckpts/'
    verbose : int, optional
        Verbosity level. 
        ``0`` - No output
        ``1`` - Output loss and metric values
        ``2`` - Output loss and metric values and visualization of latent space
        Defaults to 0.
    """

    def __init__(
        self,
        n_components=32,
        device="auto",
        n_intercepts=1,
        n_sup_networks=1,
        decoder_type="VAE",
        recon_loss="MSE",
        recon_weight=1.0,
        elbo_weight=1.0,
        sup_weight=1.0,
        sup_recon_weight=1.0,
        n_hidden_layers=2,
        hidden_dim=50,
        n_sup_hidden_layers=0,
        n_sup_hidden_dim=30,
        sup_conditional_gaussian=False,
        aug_aux_dim=None,
        activation="lrelu",
        predictor_type="linear",
        sup_recon_type="Residual",
        feature_groups=None,
        group_weights=None,
        fixed_corr=None,
        sup_smoothness_weight=1.0,
        optim_name="AdamW",
        momentum=0.9,
        lr=1e-3,
        weight_decay=0,
        anneal=True,
        identifiable=True,
        model_name="imavae",
        save_folder="./model_ckpts/",
        verbose=0,
    ):
        super(IMAVAE, self).__init__(
            n_components=n_components,
            device=device,
            n_sup_networks=n_sup_networks,
            fixed_corr=fixed_corr,
            recon_loss=recon_loss,
            recon_weight=recon_weight,
            sup_recon_type=sup_recon_type,
            sup_recon_weight=sup_recon_weight,
            sup_smoothness_weight=sup_smoothness_weight,
            feature_groups=feature_groups,
            group_weights=group_weights,
            verbose=verbose
        )
        self.n_intercepts = n_intercepts
        self.decoder_type = decoder_type
        self.sup_weight = sup_weight
        self.elbo_weight = elbo_weight
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.n_sup_hidden_layers = n_sup_hidden_layers
        self.n_sup_hidden_dim = n_sup_hidden_dim
        self.sup_conditional_gaussian = sup_conditional_gaussian
        self.aug_aux_dim = aug_aux_dim
        self.activation = activation
        self.predictor_type = predictor_type
        self.optim_name = optim_name
        self.optim_alg = self.get_optim(optim_name)
        self.pred_loss_f = nn.BCELoss if predictor_type == "logistic" else nn.MSELoss
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.anneal = anneal
        self.identifiable = identifiable
        self.model_name = model_name
        self.save_folder = save_folder
    
    def _initialize(self, dim_in, aux_dim, y_dim):
        """
        Initializes encoder and decoder parameters using the input dimensionality

        Parameters
        ----------
        dim_in : int or tuple
            Total number of features or shape of input features
        aux_dim : int
            Total number of auxiliary features
        y_dim : int
            Total number of labels to predict
        """
        assert self.n_sup_networks <= self.n_components, "n_sup_networks must be smaller or equal to n_components."
        if self.aug_aux_dim is not None:
            assert self.aug_aux_dim > aux_dim, "aug_aux_dim must be larger than aux_dim."
        self.dim_in = dim_in
        if self.aug_aux_dim is not None:
            self.aux_transform = nn.Linear(aux_dim, self.aug_aux_dim)
        self.aux_dim = aux_dim if self.aug_aux_dim is None else self.aug_aux_dim
        self.y_dim = y_dim
        # Initialize the iVAE encoder
        if self.identifiable:
            if isinstance(dim_in, int):
                self.ivae = iVAE(
                    data_dim=self.dim_in,
                    latent_dim=self.n_components,
                    aux_dim=self.aux_dim,
                    n_layers=self.n_hidden_layers,
                    hidden_dim=self.hidden_dim,
                    activation=self.activation,
                    device=self.device,
                    anneal=self.anneal
                )
            else:
                self.ivae = ConvIVAE(
                    data_width=self.dim_in[1],
                    data_height=self.dim_in[2],
                    data_channels=self.dim_in[0],
                    aux_dim=self.aux_dim,
                    latent_feature_dim=self.n_components,
                    n_layers=self.n_hidden_layers,
                    hidden_dim=self.hidden_dim,
                    activation=self.activation,
                    device=self.device,
                    anneal=self.anneal
                )
        else:
            self.ivae = VAE(
                data_dim=self.dim_in,
                latent_dim=self.n_components,
                aux_dim=self.aux_dim,
                n_layers=self.n_hidden_layers,
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                device=self.device,
                anneal=self.anneal
            )
        self.encoder = self.ivae.g
        self.prior_dist = self.ivae.prior_dist
        # Initialize the decoder based on the decoder type
        if self.decoder_type == "NMF":
            super(IMAVAE, self)._initialize(dim_in)
        elif self.decoder_type == "VAE":
            self.decoder = self.ivae.f
        else:
            raise ValueError("Decoder type must be either 'NMF' or 'VAE', not {}".format(self.decoder_type))
        assert self.predictor_type in ["linear", "logistic"], "Predictor type must be either 'linear' or 'logistic'."
        # Create the save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        # Initialize linear/logistic regression parameters OR
        # Initialize neural network parameters if n_sup_hidden_layers > 0
        if self.n_sup_networks == self.y_dim or self.y_dim == 1:
            if self.n_sup_hidden_layers == 0:
                self.phi_list = nn.ParameterList(
                    [nn.Parameter(torch.randn(self.y_dim)) for _ in range(self.n_sup_networks + self.aux_dim)]   # shape: (n_sup_networks + aux_dim, y_dim)
                )
            else:
                # shape: (n_sup_hidden_dim, n_sup_networks + aux_dim)
                setattr(self, "phi_list_0", nn.ParameterList(
                    [nn.Parameter(torch.randn(self.n_sup_networks + self.aux_dim)) for _ in range(self.n_sup_hidden_dim)]
                ))
                for idx in range(self.n_sup_hidden_layers):
                    # shape: (output_dim, n_sup_hidden_dim)
                    output_dim = self.n_sup_hidden_dim if idx < self.n_sup_hidden_layers - 1 else self.y_dim
                    setattr(self, f"phi_list_{idx+1}", nn.ParameterList(
                        [nn.Parameter(torch.randn(self.n_sup_hidden_dim)) for _ in range(output_dim)]
                    ))
                if self.sup_conditional_gaussian:
                    # shape: (n_sup_networks, n_sup_hidden_dim)
                    # Parameters for the standard deviation of the conditional Gaussian
                    setattr(self, "sigma_list_0", nn.ParameterList(
                        [nn.Parameter(torch.randn(self.n_sup_networks + self.aux_dim)) for _ in range(self.n_sup_hidden_dim)]
                    ))
                    for idx in range(self.n_sup_hidden_layers):
                        # shape: (output_dim, n_sup_hidden_dim)
                        output_dim = self.n_sup_hidden_dim if idx < self.n_sup_hidden_layers - 1 else self.y_dim
                        setattr(self, f"sigma_list_{idx+1}", nn.ParameterList(
                            [nn.Parameter(torch.randn(self.n_sup_hidden_dim)) for _ in range(output_dim)]
                        ))
        else:
            raise ValueError("y_dim must be either equal to n_sup_networks or 1.")
        if self.n_sup_hidden_layers == 0:
            self.beta_list = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(self.n_intercepts, 1))
                    for _ in range(self.n_sup_networks)   # shape: (n_sup_networks, n_intercepts, 1)
                ]
            )
        else:
            for idx in range(self.n_sup_hidden_layers+1):
                beta_dim = self.n_sup_hidden_dim if idx < self.n_sup_hidden_layers else self.y_dim
                setattr(self, f"beta_list_{idx}", nn.ParameterList(
                    [
                        nn.Parameter(torch.randn(self.n_intercepts, 1))
                        for _ in range(beta_dim)   # shape: (beta_dim, n_intercepts, 1)
                    ]
                ))
            if self.sup_conditional_gaussian:
                # Bias parameters for the standard deviation of the conditional Gaussian
                for idx in range(self.n_sup_hidden_layers+1):
                    gamma_dim = self.n_sup_hidden_dim if idx < self.n_sup_hidden_layers else self.y_dim
                    setattr(self, f"gamma_list_{idx}", nn.ParameterList(
                        [
                            nn.Parameter(torch.randn(self.n_intercepts, 1))
                            for _ in range(gamma_dim)   # shape: (gamma_dim, n_intercepts, 1)
                        ]
                    ))
        if self.sup_conditional_gaussian:
            self.sup_dist = Normal(device=self.device)
        self.to(self.device)
    
    def instantiate_optimizer(self):
        """
        Create an optimizer.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            Torch optimizer
        """
        if self.optim_name == "SGD":
            optimizer = self.optim_alg(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = self.optim_alg(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return optimizer
    
    def get_all_class_predictions(self, aux, s, intercept_mask, avg_intercept):
        """
        Get predictions for every supervised network.

        *****************************************************
        NOTE: Not tested for case where n_intercepts > 1 and 
        n_sup_hidden_layers > 0
        *****************************************************

        Parameters
        ----------
        aux : torch.Tensor
            Auxiliary features
            Shape: ``[batch_size,aux_dim]``
        s : torch.Tensor
            latent embeddings
            Shape: ``[batch_size,n_components]``
        intercept_mask : torch.Tensor
            One-hot encoded mask for linear/logistic regression intercept terms
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool
            Whether to average all intercepts together.

        Returns
        -------
        y_pred : torch.Tensor
            Predictions
            Shape: ``[batch_size, n_sup_networks]``
        """
        if intercept_mask is None and avg_intercept is False:
            warnings.warn(
                f"Intercept mask cannot be none and avg_intercept False... "
                f"Averaging Intercepts",
            )
            avg_intercept = True
        # Get predictions for each class.
        if self.n_sup_hidden_layers == 0:
            if self.n_sup_networks == self.y_dim:
                y_pred_list = []
                for sup_net in range(self.n_sup_networks):
                    if self.n_intercepts == 1:
                        # unexpected behavior if there's a batch dimension of 1
                        # somewhere.
                        logit = s[:, sup_net].view(-1, 1) * self.get_phi(sup_net) + self.beta_list[sup_net] 
                    elif self.n_intercepts > 1 and not avg_intercept:
                        logit = s[:, sup_net].view(-1, 1) * self.get_phi(sup_net) + intercept_mask @ self.beta_list[sup_net]
                    else:
                        intercept_mask = (
                            torch.ones(aux.shape[0], self.n_intercepts).to(self.device)
                            / self.n_intercepts
                        )
                        logit = s[:, sup_net].view(-1, 1) * self.get_phi(sup_net) + intercept_mask @ self.beta_list[sup_net]
                    if self.aux_dim == 1:
                        logit += aux * self.get_phi(self.n_sup_networks)
                    else:
                        logit += aux[:, sup_net].view(-1, 1) * self.get_phi(self.n_sup_networks + sup_net)
                    y_pred_sup_net = logit if self.predictor_type == "linear" else torch.sigmoid(logit)
                    y_pred_list.append(y_pred_sup_net.view(-1, 1))
                # Concatenate predictions into a single matrix [n_samples,n_tasks]
                y_pred = torch.cat(y_pred_list, dim=1)
            elif self.y_dim == 1:
                logit = torch.sum(torch.stack(
                    [s[:,sup_net].view(-1,1) * self.get_phi(sup_net) for sup_net in range(self.n_sup_networks)] + \
                    [aux[:,idx].view(-1,1) * self.get_phi(idx+self.n_sup_networks) for idx in range(self.aux_dim)], 
                    dim=1), dim=1)
                if self.n_intercepts == 1:
                    logit += self.beta_list[0]
                elif self.n_intercepts > 1 and not avg_intercept:
                    logit += intercept_mask @ self.beta_list[0]
                else:
                    intercept_mask = (
                        torch.ones(aux.shape[0], self.n_intercepts).to(self.device)
                        / self.n_intercepts
                    )
                    logit += intercept_mask @ self.beta_list[0]
                y_pred = logit if self.predictor_type == "linear" else torch.sigmoid(logit)
        else:
            if not self.sup_conditional_gaussian:
                y_pred = self.get_mlp_out(aux, s, intercept_mask, avg_intercept)
            else:
                mean = self.get_mlp_out(aux, s, intercept_mask, avg_intercept, param_name="mean")
                var = self.get_mlp_out(aux, s, intercept_mask, avg_intercept, param_name="logvar").exp()
                y_pred = self.sup_dist.sample(mean, var)

        return y_pred
    
    def get_mlp_out(self, aux, s, intercept_mask, avg_intercept, param_name="mean"):
        """
        Compute the output of an MLP with specified number of hidden layers
        and layer dimensions.

        Parameters
        ----------
        aux : torch.Tensor
            Auxiliary features
            Shape: ``[batch_size,aux_dim]``
        s : torch.Tensor
            latent embeddings
            Shape: ``[batch_size,n_components]``
        intercept_mask : torch.Tensor
            One-hot encoded mask for linear/logistic regression intercept terms
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool
            Whether to average all intercepts together.
        param_name : str, optional
            Parameter name to get from the network. Must be either ``'mean'`` or ``'logvar'``.
        """
        assert param_name in ["mean", "logvar"], "param_name must be either 'mean' or 'logvar'."
        param_list_name = "phi_list" if param_name == "mean" else "sigma_list"
        bias_list_name = "beta_list" if param_name == "mean" else "gamma_list"
        output_list = []
        for layer_idx in range(self.n_sup_hidden_layers+1):
            input_list = torch.cat([s,aux], dim=1) if layer_idx == 0 else output_list
            output_list = []
            layer_dim = len(getattr(self, f"{param_list_name}_{layer_idx}"))   # dimension of the next layer
            beta_list = getattr(self, f"{bias_list_name}_{layer_idx}")
            for idx in range(layer_dim):
                if self.n_intercepts == 1:
                    logit = input_list @ self.get_phi(idx, layer_idx, param_list_name=param_list_name) + beta_list[idx]
                elif self.n_intercepts > 1 and not avg_intercept:
                    logit = input_list @ self.get_phi(idx, layer_idx, param_list_name=param_list_name) + intercept_mask @ beta_list[idx]
                else:
                    intercept_mask = (
                        torch.ones(aux.shape[0], self.n_intercepts).to(self.device)
                        / self.n_intercepts
                    )
                    logit = input_list @ self.get_phi(idx, layer_idx, param_list_name=param_list_name) + intercept_mask @ beta_list[idx]
                if layer_idx < self.n_sup_hidden_layers:
                    output = ACTIVATIONS[self.activation](logit).squeeze()
                else:
                    output = logit.squeeze()
                output_list.append(output)
            output_list = torch.stack(output_list, dim=1)
        return output_list
    
    def get_embedding(self, X, aux):
        """
        Get the latent embedding.

        Parameters
        ----------
        X : torch.Tensor
            Input features
            Shape: ``[batch_size,dim_in]``
        aux : torch.Tensor
            Auxiliary features
            Shape: ``[batch_size,aux_dim]``

        Returns
        -------
        s : torch.Tensor
            Latent embeddings (scores)
            Shape: ``[batch_size,n_components]``
        """
        if self.aug_aux_dim is not None:
            aux = self.aux_transform(aux)
        if not self.identifiable:
            return self.encoder(X)
        if len(X.shape) == len(aux.shape):
            return self.encoder(torch.cat([X, aux], dim=1))
        else:
            return self.encoder(X, aux)
    
    def get_phi(self, sup_net, layer=0, param_list_name="phi_list"):
        """
        Return the linear/logistic regression coefficient correspoding to ``sup_net``.

        # NOTE: Raises before Parameters and Returns

        Raises
        ------
        * ``ValueError`` if ``self.fixed_corr[sup_net]`` is not in
          ``{'n/a', 'positive', 'negative'}``. This should be caught at
          initialization. # NOTE: it's better to catch this a initialization.

        Parameters
        ----------
        sup_net : int
            Index of the supervised network you would like to get a coefficient for.
        layer : int, optional
            Index of the layer of coefficients you would like to get. Defaults to ``0``.

        Returns
        -------
        phi : torch.Tensor
            The coefficient that has either been returned raw, or through a
            positive or negative softplus(phi).
            shape: ``[n_sup_networks + aux_dim, 1]``
        """
        if layer == 0:
            phi = self.phi_list[sup_net] if self.n_sup_hidden_layers == 0 else getattr(self, f"{param_list_name}_0")[sup_net]
        else:
            phi = getattr(self, f"{param_list_name}_{layer}")[sup_net]
        if layer == 0:
            fixed_corr_str = self.fixed_corr[sup_net].lower() if self.y_dim > 1 else self.fixed_corr[0].lower()
        else:
            fixed_corr_str = "n/a"
        if fixed_corr_str == "n/a":
            return phi
        elif fixed_corr_str == "positive":
            # NOTE: use nn.functional instead of nn here.
            return F.softplus(phi)
        elif fixed_corr_str == "negative":
            return -1 * F.softplus(phi)
        else:
            # NOTE: spit out an informative error
            raise ValueError(f"Unsupported fixed_corr value: {fixed_corr_str}")
            
    
    def forward(self, X, aux, y, task_mask, pred_weight, intercept_mask=None, avg_intercept=False):
        """
        IMAVAE forward pass

        Parameters
        ----------
        X : torch.Tensor
            Input Features
            Shape: ``[batch_size,dim_in]``
        aux : torch.Tensor
            Auxiliary features
            Shape: ``[batch_size,aux_dim]``
        y : torch.Tensor
            Ground truth labels
            Shape: ``[batch_size,n_sup_networks]``
        task_mask : torch.Tensor
            Per window mask for whether or not predictions should be counted
            Shape: ``[batch_size,n_sup_networks]``
        pred_weight : torch.Tensor
            Per window classification importance weighting
            Shape: ``[batch_size,1]``
        intercept_mask : ``None`` or torch.Tensor, optional
            Window specific intercept mask. Defaults to ``None``.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool, optional
            Whether or not to average intercepts. This is used in evaluation.
            Defaults to ``False``.

        Returns
        -------
        recon_loss (torch.Tensor):
            ``recon_weight*full_recon_loss + sup_recon_weight*sup_recon_loss``
        pred_loss (torch.Tensor):
            ``sup_weight * BCELoss()``
        """
        X = X.to(self.device)
        aux = self.aux_transform(aux).to(self.device) if self.aug_aux_dim is not None else aux.to(self.device)
        y = y.to(self.device)
        task_mask = task_mask.to(self.device)
        pred_weight = pred_weight.to(self.device)
        if intercept_mask is not None:
            intercept_mask = intercept_mask.to(self.device)
        
        if self.decoder_type == "NMF":
            """
            ************************************************************************
            NOTE: Needs further investigation. For decoder_type == "NMF", we need to 
            first train the full iVAE model, and then replace the decoder with NMF.
            It is not clear how we should fit the NMF parameters in this case.
            ************************************************************************
            """
            s = self.get_embedding(X, aux)
            recon_loss = self.NMF_decoder_forward(X, s)
        else:
            # Compute the evidence lower bound (ELBO)
            if self.identifiable:
                elbo, s = self.ivae.elbo(X, aux)
            else:
                elbo, s = self.ivae.elbo(X)
            recon_loss = self.recon_weight * nn.MSELoss()(X, self.decoder(s))
            elbo_loss = self.elbo_weight * elbo.mul(-1)
        # Get predictions
        y_pred = self.get_all_class_predictions(
            aux,
            s,
            intercept_mask,
            avg_intercept,
        )
        pred_loss_f = self.pred_loss_f(weight=pred_weight) if self.predictor_type == "logistic" else self.pred_loss_f()
        pred_loss = self.sup_weight * pred_loss_f(
            y_pred * task_mask,
            y * task_mask,
        )
        return recon_loss, elbo_loss, pred_loss
    
    @torch.no_grad()
    def transform(self, X, aux, intercept_mask=None, avg_intercept=True, return_npy=True):
        """
        Transform method to return reconstruction, predictions, and projections

        Parameters
        ----------
        X (torch.Tensor): Input Features
            Shape: ``[batch_size,dim_in]``
        aux (torch.Tensor): Auxiliary features
            Shape: ``[batch_size,aux_dim]``
        intercept_mask (torch.Tensor, optional): window specific intercept
            mask. Defaults to None.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept (bool, optional): Whether or not to average intercepts -
            used in evaluation. Defaults to False.
        return_npy (bool, optional): Whether or not to convert to numpy arrays.
            Defaults to True.

        Returns
        -------
        X_recon (torch.Tensor) : Full reconstruction of input features
            Shape: ``[batch_size,dim_in]``
        y_pred (torch.Tensor) : All task predictions
            Shape: ``[batch_size,n_sup_networks]``
        s (torch.Tensor) : Network activation scores
            Shape: ``[batch_size,n_components]``
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).float().to(self.device)
        else:
            X = X.to(self.device)
        if not torch.is_tensor(aux):
            aux = torch.Tensor(aux).float().to(self.device)
        else:
            aux = aux.to(self.device)
        if intercept_mask is not None:
            if not torch.is_tensor(intercept_mask):
                intercept_mask = torch.Tensor(intercept_mask).to(self.device)
            else:
                intercept_mask = intercept_mask.to(self.device)
        
        s = self.get_embedding(X, aux)
        if self.decoder_type == "NMF":
            """
            ************************************************************************
            NOTE: Needs further investigation. For decoder_type == "NMF", we need to 
            first train the full iVAE model, and then replace the decoder with NMF.
            It is not clear how we should fit the NMF parameters in this case.
            ************************************************************************
            """
            X_recon = self.get_all_comp_recon(s)
        else:
            X_recon = self.decoder(s)
        aux = self.aux_transform(aux) if self.aug_aux_dim is not None else aux
        y_pred = self.get_all_class_predictions(
            aux,
            s,
            intercept_mask,
            avg_intercept,
        )
        if return_npy:
            s = s.detach().cpu().numpy()
            X_recon = X_recon.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
        return X_recon, y_pred, s
    
    def pretrain(
        self,
        X,
        aux,
        y,
        y_pred_weights,
        task_mask,
        intercept_mask,
        sample_weights,
        n_pre_epochs=100,
        batch_size=128,
    ):
        """
        Pretrain the iVAE (or just the iVAE encoder if decoder_type == "NMF")

        Parameters
        ----------
        X (torch.Tensor): Input Features
            Shape: ``[n_samples,dim_in]``
        aux (torch.Tensor): Auxiliary features
            Shape: ``[n_samples,aux_dim]``
        y (torch.Tensor): ground truth labels
            Shape: ``[n_samples,n_sup_networks]``
        task_mask (torch.Tensor):
            per window mask for whether or not predictions should be counted
            Shape: ``[n_samples,n_sup_networks]``
        y_pred_weight (torch.Tensor):
            per window classification importance weighting
            Shape: ``[n_samples,1]``
        intercept_mask (torch.Tensor, optional):
            window specific intercept mask. Defaults to None.
            Shape: ``[n_samples,n_intercepts]``
        sample_weights (torch.Tensor):
            Gradient Descent sampling weights.
            Shape: ``[n_samples,1]
        n_pre_epochs (int,optional):
            number of epochs for pretraining the encoder. Defaults to 100
        batch_size (int,optional):
            batch size for pretraining. Defaults to 128
        """
        if self.decoder_type == "NMF":
            """
            ************************************************************************
            NOTE: Needs further investigation. For decoder_type == "NMF", we need to 
            first train the full iVAE model, and then replace the decoder with NMF.
            It is not clear how we should fit the NMF parameters in this case.
            ************************************************************************
            """
            self.W_nmf.requires_grad = False
        # Load arguments onto device
        X = torch.Tensor(X).float().to(self.device)
        aux = torch.Tensor(aux).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        y_pred_weights = torch.Tensor(y_pred_weights).float().to(self.device)
        task_mask = torch.Tensor(task_mask).long().to(self.device)
        intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        sample_weights = torch.Tensor(sample_weights).to(self.device)
        # Create a Dataset.
        dset = TensorDataset(X, aux, y, task_mask, y_pred_weights, intercept_mask)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler)
        # Instantiate Optimizer
        optimizer = self.instantiate_optimizer()
        # Define iterator
        if self.verbose > 0:
            epoch_iter = tqdm(range(n_pre_epochs))
        else:
            epoch_iter = range(n_pre_epochs)
        # Pre-training
        for epoch in epoch_iter:
            r_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                recon_loss, _, _ = self.forward(*batch)
                recon_loss.backward()
                optimizer.step()
                r_loss += recon_loss.item()
            if self.verbose > 0:
                avg_r = r_loss / len(loader)
                epoch_iter.set_description(
                    f"Encoder Pretrain Epoch: {epoch}, Recon Loss: {avg_r:.6}",
                )
        if self.decoder_type == "NMF":
            """
            ************************************************************************
            NOTE: Needs further investigation. For decoder_type == "NMF", we need to 
            first train the full iVAE model, and then replace the decoder with NMF.
            It is not clear how we should fit the NMF parameters in this case.
            ************************************************************************
            """
            self.W_nmf.requires_grad = True
    
    def fit(
        self,
        X,
        aux,
        y,
        y_pred_weights=None,
        task_mask=None,
        intercept_mask=None,
        y_sample_groups=None,
        n_epochs=100,
        n_pre_epochs=100,
        nmf_max_iter=100,
        batch_size=128,
        lr=1e-3,
        pretrain=True,
        verbose=0,
        X_val=None,
        aux_val=None,
        y_val=None,
        y_pred_weights_val=None,
        task_mask_val=None,
        best_model_name="imavae-best-model.pt"
    ):
        """
        Fit the model.

        Parameters
        ----------
        X (np.ndarray): Input Features
            Shape: ``[n_samples, dim_in]`` or ``[n_samples, width, height, channels]``
        aux (np.ndarray): Auxiliary features
            Shape: ``[n_samples, aux_dim]``
        y (np.ndarray): ground truth labels
            Shape: ``[n_samples, n_sup_networks]``
        y_pred_weights (np.ndarray, optional):
            supervision window specific importance weights. Defaults to
            ``None``.
            Shape: ``[n_samples,1]``
        task_mask (np.ndarray, optional):
            identifies which windows should be trained on which tasks. Defaults
            to ``None``.
            Shape: ``[n_samples,n_sup_networks]``
        intercept_mask (np.ndarray, optional):
            One-hot Mask for group specific intercepts in the linear/logistic
            regression model. Defaults to None.
            Shape: ``[n_samples,n_intercepts]``
        y_sample_groups (_type_, optional):
            groups for creating sample weights - each group will be sampled
            evenly. Defaults to None.
            Shape: ``[n_samples,1]``
        n_epochs (int, optional):
            number of training epochs. Defaults to 100.
        n_pre_epochs (int, optional):
            number of pretraining epochs. Defaults to 100.
        nmf_max_iter (int, optional):
            max iterations for NMF pretraining solver. Defaults to 100.
        batch_size (int, optional):
            batch size for gradient descent. Defaults to 128.
        lr (_type_, optional):
            learning rate for gradient descent. Defaults to 1e-3.
        pretrain (bool, optional):
            whether or not to pretrain the generative model. Defaults to True.
        verbose (bool, optional):
            activate or deactivate print statements. Defaults to False.
        X_val (np.ndarray, optional):
            Validation Features for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,dim_in]``
        aux_val (np.ndarray, optional):
            Validation Auxiliary Features for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,aux_dim]``
        y_val (np.ndarray, optional):
            Validation Labels for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,n_sup_networks]``
        y_pred_weights_val (np.ndarray, optional):
            window specific classification weights. Defaults to None.
            Shape: ``[n_val_samples,1]``
        task_mask_val (np.ndarray, optional):
            validation task relevant window masking. Defaults to None.
            Shape: ``[n_val_samples,n_sup_networks]``
        best_model_name (str, optional):
            save file name for the best model. Must end in ".pt". Defaults to
            ``'imavae-best-model.pt'``.

        Returns
        -------
        self : IMAVAE
        """
        # Initialize model parameters.
        if len(X.shape) == 2:
            # Shape: ``[n_samples, dim_in]``
            self._initialize(X.shape[1], aux.shape[1], y.shape[1])
        elif len(X.shape) == 4:
            # Shape: ``[n_samples, width, height, channels]``
            self._initialize(tuple([X.shape[1],X.shape[2],X.shape[3]]), aux.shape[1], y.shape[1])

        # Establish loss histories.
        self.training_hist = []  # tracks average overall loss
        self.recon_hist = []  # tracks training data mse
        self.pred_hist = []  # tracks training data aucs

        # Globaly activate/deactivate print statements.
        self.verbose = verbose

        # Fill default values
        if intercept_mask is None:
            intercept_mask = np.ones((X.shape[0], self.n_intercepts))
        if task_mask is None:
            task_mask = np.ones(y.shape)
        if y_pred_weights is None:
            y_pred_weights = np.ones((y.shape[0], 1))
        
        # Fill sampler parameters.
        if y_sample_groups is None:
            y_sample_groups = np.ones((y.shape[0]))
            samples_weights = y_sample_groups
        else:
            class_sample_counts = np.array(
                [
                    np.sum(y_sample_groups == group)
                    for group in np.unique(y_sample_groups)
                ],
            )   # number of samples in each group with length equal to number of groups
            weight = 1.0 / class_sample_counts
            samples_weights = np.array(
                [weight[t] for t in y_sample_groups.astype(int)],
            ).squeeze()
            samples_weights = torch.Tensor(samples_weights)
        
        if pretrain:
            self.lr = 1e-3
            if self.decoder_type == "NMF":
                """
                ************************************************************************
                NOTE: Needs further investigation. For decoder_type == "NMF", we need to 
                first train the full iVAE model, and then replace the decoder with NMF.
                It is not clear how we should fit the NMF parameters in this case.
                ************************************************************************
                """
                self.pretrain_NMF(X, y, nmf_max_iter)
            self.pretrain(
                X, 
                aux, 
                y, 
                y_pred_weights, 
                task_mask, 
                intercept_mask, 
                samples_weights, 
                n_pre_epochs, 
                batch_size
            )
        
        # Send training arguments to Tensors.
        X = torch.Tensor(X).float().to(self.device)
        aux = torch.Tensor(aux).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        y_pred_weights = torch.Tensor(y_pred_weights).float().to(self.device)
        task_mask = torch.Tensor(task_mask).long().to(self.device)
        intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        samples_weights = torch.Tensor(samples_weights).to(self.device)

        # If validation data is provided, set up the tensors.
        if X_val is not None and aux_val is not None and y_val is not None:
            assert best_model_name.split(".")[-1] == "pt", (
                f"Save file `{self.save_folder + best_model_name}` must be "
                f"of type .pt"
            )
            self.best_model_name = best_model_name
            self.best_performance = 1e8
            self.best_val_recon = 1e8
            self.best_val_avg_auc = 0.0
            self.val_recon_hist = []
            self.val_pred_hist = []

            if task_mask_val is None:
                task_mask_val = np.ones(y_val.shape)

            if y_pred_weights_val is None:
                y_pred_weights_val = np.ones((y_val[:, 0].shape[0], 1))
            
            X_val = torch.Tensor(X_val).float().to(self.device)
            aux_val = torch.Tensor(aux_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)
            task_mask_val = torch.Tensor(task_mask_val).long().to(self.device)
            y_pred_weights_val = torch.Tensor(y_pred_weights_val).float().to(self.device)
        
        # Instantiate the dataloader and optimizer.
        dset = TensorDataset(X, aux, y, task_mask, y_pred_weights, intercept_mask)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler)
        optimizer = self.instantiate_optimizer()
        self.lr = lr

        # Define the training iterator.
        if self.verbose > 0:
            print("Beginning Training")
            epoch_iter = tqdm(range(n_epochs))
        else:
            epoch_iter = range(n_epochs)
        
        # Training loop.
        for epoch in epoch_iter:
            epoch_loss = 0.0
            recon_e_loss = 0.0
            pred_e_loss = 0.0

            for batch in loader:
                self.train()
                optimizer.zero_grad()
                recon_loss, elbo_loss, pred_loss = self.forward(*batch)
                # Weighting happens inside of the forward call
                loss = recon_loss + elbo_loss + pred_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                recon_e_loss += recon_loss.item()
                pred_e_loss += pred_loss.item()
            self.training_hist.append(epoch_loss / len(loader))
            with torch.no_grad():
                self.eval()
                X_recon, y_pred, _ = self.transform(
                    X,
                    aux,
                    intercept_mask,
                    avg_intercept=False,
                    return_npy=True,
                )
                training_mse_loss = np.mean((X.detach().cpu().numpy() - X_recon) ** 2)
                training_metric_list = []
                for sup_net in range(self.n_sup_networks):
                    idx = sup_net if self.y_dim > 1 else 0
                    temp_mask = task_mask[:, idx].detach().cpu().numpy()
                    if self.predictor_type == "linear":
                        metric = mean_squared_error(
                            y.detach().cpu().numpy()[temp_mask == 1, idx],
                            y_pred[temp_mask == 1, idx],
                        )
                    else:
                        metric = roc_auc_score(
                            y.detach().cpu().numpy()[temp_mask == 1, idx],
                            y_pred[temp_mask == 1, idx],
                        )
                    training_metric_list.append(metric)
                self.recon_hist.append(training_mse_loss)
                self.pred_hist.append(training_metric_list)

                # If validation data is present, collect performance metrics.
                if X_val is not None and aux_val is not None and y_val is not None:
                    X_recon_val, y_pred_val, _ = self.transform(
                        X_val,
                        aux_val,
                        return_npy=True,
                    )
                    validation_mse_loss = np.mean(
                        (X_val.detach().cpu().numpy() - X_recon_val) ** 2
                    )
                    validation_metric_list = []
                    for sup_net in range(self.n_sup_networks):
                        idx = sup_net if self.y_dim > 1 else 0
                        temp_mask = task_mask_val[:, idx].detach().cpu().numpy()
                        if self.predictor_type == "linear":
                            metric = mean_squared_error(
                                y_val.detach().cpu().numpy()[temp_mask == 1, idx],
                                y_pred_val[temp_mask == 1, idx],
                            )
                        else:
                            metric = roc_auc_score(
                                y_val.detach().cpu().numpy()[temp_mask == 1, idx],
                                y_pred_val[temp_mask == 1, idx],
                            )
                        validation_metric_list.append(metric)

                    self.val_recon_hist.append(validation_mse_loss)
                    self.val_pred_hist.append(validation_metric_list)

                    mse_var_rat = validation_mse_loss / torch.std(X_val) ** 2
                    if self.predictor_type == "linear":
                        avg_err = np.mean(validation_metric_list)
                    else:
                        avg_err = 1 - np.mean(validation_metric_list)

                    if mse_var_rat + avg_err < self.best_performance:
                        self.best_epoch = epoch
                        self.best_performance = mse_var_rat + avg_err
                        self.best_val_avg_metric = np.mean(validation_metric_list)
                        self.best_val_recon = validation_mse_loss
                        self.best_val_metrics = validation_metric_list
                        torch.save(
                            self.state_dict(),
                            self.save_folder + self.best_model_name,
                        )

                    if self.verbose > 0:
                        epoch_iter.set_description(
                            "Epoch: {}, Best Epoch: {}, Best Recon MSE: {:.6}, Best Pred Metric {}, current MSE: {:.6}, current Pred Metric: {}".format(
                                epoch,
                                self.best_epoch,
                                self.best_val_recon,
                                self.best_val_metrics,
                                validation_mse_loss,
                                validation_metric_list,
                            )
                        )
                        if self.verbose > 1:
                            self.visualize_latent_space(aux_val)
                else:
                    if self.verbose > 0:
                        epoch_iter.set_description(
                            "Epoch: {}, Current Training MSE: {:.6}, Current Pred Metric: {}".format(
                                epoch, training_mse_loss, training_metric_list
                            )
                        )
                        if self.verbose > 1:
                            self.visualize_latent_space(aux)   
        
        if self.verbose > 0:
            print(
                "Saving the last epoch with training MSE: {:.6} and Pred Metric: {}".format(
                    training_mse_loss, training_metric_list
                )
            )
        
        if X_val is not None and aux_val is not None and y_val is not None:
            if self.verbose > 0:
                print(
                    "Loaded the best model from Epoch: {} with MSE: {:.6} and Pred Metric: {}".format(
                        self.best_epoch, self.best_val_recon, self.best_val_metrics
                    )
                )
            self.load_state_dict(torch.load(self.save_folder + self.best_model_name))
        return self
    
    def reconstruct(self, X, aux, component=None):
        """
        Gets full or partial reconstruction.

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,dim_in]``
        aux : numpy.ndarray
            Auxiliary Features
            Shape: ``[n_samples,aux_dim]``
        component : int, optional
            identifies which component to use for reconstruction

        Returns
        -------
        X_recon : numpy.ndarray
            Reconstructed Features
            Shape: ``[n_samples,dim_in]``
        """
        X_recon, _, s = self.transform(X, aux)
        if component is not None and self.decoder_type == "NMF":
            """
            ************************************************************************
            NOTE: Needs further investigation. For decoder_type == "NMF", we need to 
            first train the full iVAE model, and then replace the decoder with NMF.
            It is not clear how we should fit the NMF parameters in this case.
            ************************************************************************
            """
            X_recon = self.get_comp_recon(s, component)
        return X_recon
    
    def visualize_latent_space(self, aux):
        """
        Visualize the first two dimensions of latent space.

        Parameters
        ----------
        aux : numpy.ndarray
            Auxiliary Features
            Shape: ``[n_samples,aux_dim]``
        """
        N = aux.shape[0]
        t0, t1 = torch.zeros(N, 1).to(self.device), torch.ones(N, 1).to(self.device)
        z_m0 = self.prior_dist.sample(*self.ivae.prior_params(t0))
        z_m1 = self.prior_dist.sample(*self.ivae.prior_params(t1))
        z_m = torch.stack([z_m0[i,:] if aux[i,0] == 0 else z_m1[i,:] for i in range(N)]).cpu().numpy()

        _, ax = plt.subplots(1,1,figsize=(8,7))
        c_dict = {0: 'blue', 1: 'orange'}
        for g in np.unique(aux[:,0].squeeze()):
            i = np.where(aux[:,0].squeeze() == g)
            ax.scatter(z_m[i,0], z_m[i,1], c=c_dict[g], label=g, s=1)
        ax.legend()
        ax.set_title("p(Z|T)")
        plt.show()
    
    def predict_proba(self, X, aux, return_scores=False):
        """
        Returns prediction probabilities, only for self.predict_type == "logistic".

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,dim_in]``
        aux : numpy.ndarray
            Auxiliary Features
            Shape: ``[n_samples,aux_dim]``
        return_scores (bool, optional):
            Whether or not to include the projections. Defaults to False.

        Returns
        -------
        y_pred_proba (numpy.ndarray): predictions
            Shape: ``[n_samples,n_sup_networks]``
        s (numpy.ndarray): supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        assert self.predictor_type == "logistic", "predict_type must be 'logistic' for predict_proba() method."
        _, y_pred, s = self.transform(X, aux)

        if return_scores:
            return y_pred, s
        else:
            return y_pred
    
    def predict(self, X, aux, return_scores=False):
        """
        Return predictions.

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,dim_in]``
        aux : numpy.ndarray
            Auxiliary Features
            Shape: ``[n_samples,aux_dim]``
        return_scores : bool, optional
            Whether or not to include the projections. Defaults to ``False``.

        Returns
        -------
        y_pred_proba : numpy.ndarray
            Predictions in {0,1}
            Shape: ``[n_samples,n_sup_networks]``
        s : numpy.ndarray
            supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, y_pred, s = self.transform(X, aux)

        if return_scores:
            if self.predictor_type == "logistic":
                return (y_pred > 0.5).astype(int), s
            else:
                return y_pred, s
        else:
            if self.predictor_type == "logistic":
                return (y_pred > 0.5).astype(int)
            else:
                return y_pred
    
    def project(self, X, aux):
        """
        Get projections

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,dim_in]
        aux : numpy.ndarray
            Auxiliary Features
            Shape: ``[n_samples,aux_dim]``
        return_scores : bool, optional
            Whether or not to include the projections. Defaults to False.

        Returns
        -------
        s : numpy.ndarray
            supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, _, s = self.transform(X, aux)
        return s
    
    def auc_score(self, X, aux, y, groups=None, return_dict=False):
        """
        Gets a list of task AUCs either by group or for all samples. Can return
        a dictionary with AUCs for each group with each group label as a key.
        Only works for self.predict_type == "logistic".

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,dim_in]
        aux : numpy.ndarray
            Auxiliary Features
            Shape: ``[n_samples,aux_dim]``
        y : numpy.ndarray
            Ground Truth Labels
            Shape: ``[n_samples,n_sup_networks]``
        groups : numpy.ndarray, optional
            per window group assignment labels. Defaults to None.
            Shape: ``[n_samples,1]``
        return_dict : bool, optional
            Whether or not to return a dictionary with values for each group. Defaults to False.

        Returns
        -------
        score_results: numpy.ndarray or dict
            Array or dictionary of results either as the mean performance of all groups,
            the performance of all samples, or a dictionary of results for each group.
        """
        assert self.predictor_type == "logistic", "predict_type must be 'logistic' for auc_score() method."
        _, y_pred, _ = self.transform(X, aux)
        if groups is not None:
            auc_dict = {}
            for group in np.unique(groups):
                auc_list = []
                for sup_net in range(self.n_sup_networks):
                    auc = roc_auc_score(y[:, sup_net], y_pred[:, sup_net])
                    auc_list.append(auc)
                auc_dict[group] = auc_list
            if return_dict:
                score_results = auc_dict
            else:
                auc_array = np.vstack([auc_dict[key] for key in np.unique(groups)])
                score_results = np.mean(auc_array, axis=0)
        else:
            auc_list = []
            for sup_net in range(self.n_sup_networks):
                auc = roc_auc_score(y[:, sup_net], y_pred[:, sup_net])
                auc_list.append(auc)
            score_results = np.array(auc_list)
        return score_results
    
    def acme_score(self, aux, treatment=False, intercept_mask=None, avg_intercept=True, simulaltions=100):
        """
        Calculate the average causal mediation effect (or indirect effect) given the auxillary feature (treatment)
        ACME(t) = E[Y(t, M(1)) - Y(t, M(0))]

        Parameters
        ----------
        aux : numpy.ndarray
            Auxiliary Features
        treatment : bool, optional
            If True, calculate the ACME for the treatment group.
            Otherwise, calculate the ACME for the control group.
            Defaults to False.
        intercept_mask (torch.Tensor, optional): window specific intercept
            mask. Defaults to None.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept (bool, optional): Whether or not to average intercepts -
            used in evaluation. Defaults to True.
        simulaltions (int, optional): Number of simulations to use for estimating
            the ACME. Defaults to 100.
        
        Returns
        -------
        acme_mean : float
            Average causal mediation effect.
        acme_std : float
            Standard deviation of the average causal mediation effect.
        """
        assert len(aux.shape) == 2, "aux must be a 2D array."
        assert ((aux[:,0]==0) | (aux[:,0]==1)).all(), "The first column of aux must be binary."
        if not isinstance(aux, torch.Tensor):
            aux = torch.tensor(aux).float().to(self.device)
        else:
            aux = aux.float().to(self.device)
        n_samples = aux.shape[0]
        t0, t1 = torch.zeros(n_samples, 1).to(self.device), torch.ones(n_samples, 1).to(self.device)
        if aux.shape[1] > 1:
            t0 = torch.cat([t0, aux[:,1:]], dim=1)
            t1 = torch.cat([t1, aux[:,1:]], dim=1)
        if self.aug_aux_dim is not None:
            t0, t1 = self.aux_transform(t0), self.aux_transform(t1)
        acme_arr = []
        for _ in range(simulaltions):
            z0 = self.prior_dist.sample(*self.ivae.prior_params(t0)) if self.identifiable else self.prior_dist.sample(*self.ivae.prior_params(), size=n_samples)
            z1 = self.prior_dist.sample(*self.ivae.prior_params(t1)) if self.identifiable else self.prior_dist.sample(*self.ivae.prior_params(), size=n_samples)
            t = t1 if treatment else t0
            y_m0 = self.get_all_class_predictions(t, z0, intercept_mask=intercept_mask, avg_intercept=avg_intercept)
            y_m1 = self.get_all_class_predictions(t, z1, intercept_mask=intercept_mask, avg_intercept=avg_intercept)
            acme_arr.append((y_m1 - y_m0).mean().item())
        return np.mean(acme_arr), np.std(acme_arr)
    
    def ade_score(self, aux, treatment=False, intercept_mask=None, avg_intercept=True, simulaltions=100):
        """
        Calculate the average direct effect given the auxillary feature (treatment)
        ADE(t) = E[Y(1, M(t)) - Y(0, M(t))]

        Parameters
        ----------
        aux : numpy.ndarray
            Auxiliary Features
        treatment : bool, optional
            If True, calculate the ADE for the treatment group.
            Otherwise, calculate the ADE for the control group.
            Defaults to False.
        intercept_mask (torch.Tensor, optional): window specific intercept
            mask. Defaults to None.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept (bool, optional): Whether or not to average intercepts -
            used in evaluation. Defaults to True.
        simulaltions (int, optional): Number of simulations to use for estimating
            the ADE. Defaults to 100.
        
        Returns
        -------
        ade_mean : float
            Average direct effect.
        ade_std : float
            Standard deviation of the average direct effect.
        """
        assert len(aux.shape) == 2, "aux must be a 2D array."
        assert ((aux[:,0]==0) | (aux[:,0]==1)).all(), "The first column of aux must be binary."
        if not isinstance(aux, torch.Tensor):
            aux = torch.tensor(aux).float().to(self.device)
        else:
            aux = aux.float().to(self.device)
        n_samples = aux.shape[0]
        t0, t1 = torch.zeros(n_samples, 1).to(self.device), torch.ones(n_samples, 1).to(self.device)
        if aux.shape[1] > 1:
            t0 = torch.cat([t0, aux[:,1:]], dim=1)
            t1 = torch.cat([t1, aux[:,1:]], dim=1)
        if self.aug_aux_dim is not None:
            t0, t1 = self.aux_transform(t0), self.aux_transform(t1)
        ade_arr = []
        for _ in range(simulaltions):
            z0 = self.prior_dist.sample(*self.ivae.prior_params(t0)) if self.identifiable else self.prior_dist.sample(*self.ivae.prior_params(), size=n_samples)
            z1 = self.prior_dist.sample(*self.ivae.prior_params(t1)) if self.identifiable else self.prior_dist.sample(*self.ivae.prior_params(), size=n_samples)
            z = z1 if treatment else z0
            y_t0 = self.get_all_class_predictions(t0, z, intercept_mask=intercept_mask, avg_intercept=avg_intercept)
            y_t1 = self.get_all_class_predictions(t1, z, intercept_mask=intercept_mask, avg_intercept=avg_intercept)
            ade_arr.append((y_t1 - y_t0).mean().item())
        return np.mean(ade_arr), np.std(ade_arr)
    
    def ate_score(self, aux, intercept_mask=None, avg_intercept=True, simulaltions=100):
        """
        Calculate the average treatment effect (or total effect)
        ATE = E[Y(1, M(1)) - Y(0, M(0))]

        Parameters
        ----------
        aux : numpy.ndarray
            Auxiliary Features
        intercept_mask (torch.Tensor, optional): window specific intercept
            mask. Defaults to None.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept (bool, optional): Whether or not to average intercepts -
            used in evaluation. Defaults to True.
        simulaltions (int, optional): Number of simulations to use for estimating
            the ATE. Defaults to 100.
        
        Returns
        -------
        ate_mean : float
            Average treatment effect.
        ate_std : float
            Standard deviation of the average treatment effect.
        """
        assert len(aux.shape) == 2, "aux must be a 2D array."
        assert ((aux[:,0]==0) | (aux[:,0]==1)).all(), "The first column of aux must be binary."
        if not isinstance(aux, torch.Tensor):
            aux = torch.tensor(aux).float().to(self.device)
        else:
            aux = aux.float().to(self.device)
        n_samples = aux.shape[0]
        t0, t1 = torch.zeros(n_samples, 1).to(self.device), torch.ones(n_samples, 1).to(self.device)
        if aux.shape[1] > 1:
            t0 = torch.cat([t0, aux[:,1:]], dim=1)
            t1 = torch.cat([t1, aux[:,1:]], dim=1)
        if self.aug_aux_dim is not None:
            t0, t1 = self.aux_transform(t0), self.aux_transform(t1)
        ate_arr = []
        for _ in range(simulaltions):
            z0 = self.prior_dist.sample(*self.ivae.prior_params(t0)) if self.identifiable else self.prior_dist.sample(*self.ivae.prior_params(), size=n_samples)
            z1 = self.prior_dist.sample(*self.ivae.prior_params(t1)) if self.identifiable else self.prior_dist.sample(*self.ivae.prior_params(), size=n_samples)
            y_0 = self.get_all_class_predictions(t0, z0, intercept_mask=intercept_mask, avg_intercept=avg_intercept)
            y_1 = self.get_all_class_predictions(t1, z1, intercept_mask=intercept_mask, avg_intercept=avg_intercept)
            ate_arr.append((y_1 - y_0).mean().item())
        return np.mean(ate_arr), np.std(ate_arr)

if __name__ == "__main__":
    pass

###############################################################################
# MIT License

# Copyright (c) 2023 Ziyang Jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################