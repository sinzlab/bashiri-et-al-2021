from warnings import warn
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal, LowRankMultivariateNormal, MultivariateNormal

from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.nn_helpers import set_random_seed
from neuralpredictors.training import eval_state
from ..utility import set_random_seed, get_conditional_means, get_conditional_variances
from . import transforms
from .transforms import (
    Identity,
    SQRT,
    Anscombe,
    ELU,
    ELUF,
    Affine,
    Log,
    Flow,
    MeanTransfom,
    Exp,
    InvELU,
    Softplus,
    InvSoftplus,
)


class ZIF_Loglikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_transform, targets, rho, qs, means, C, psi_diag):

        uniform_masks = (targets < rho).detach()  # spike
        gaussian_masks = (targets >= rho).detach()  # slab

        # compute spike loglikelihood
        uniform_log_probs = torch.log(1 - qs) - torch.log(rho)
        uniform_log_probs = (uniform_log_probs * uniform_masks).sum(dim=1)

        # compute slab loglikelihood
        transformed_targets, logdets = sample_transform(targets, mask=gaussian_masks)

        original_space_nonzero_log_probs = []
        for mean, q, transformed_target, logdet, gaussian_mask in zip(
            means, qs, transformed_targets, logdets, gaussian_masks
        ):  # loop through the batch

            density_fn = LowRankMultivariateNormal(
                mean[gaussian_mask], C[:, gaussian_mask].T, psi_diag[gaussian_mask]
            )
            gaussian_log_prob = density_fn.log_prob(transformed_target[gaussian_mask])

            original_space_nonzero_log_prob = (
                gaussian_log_prob
                + logdet[gaussian_mask].sum()
                + torch.log(q[gaussian_mask]).sum()
            )

            original_space_nonzero_log_probs.append(original_space_nonzero_log_prob)

        original_space_nonzero_log_probs = torch.stack(original_space_nonzero_log_probs)

        loglikelihood = uniform_log_probs + original_space_nonzero_log_probs

        return loglikelihood


def get_learned_transforms(name, zero_threshold, n_dimensions=1):
    """
    The learned transformation here corresponds to that of FlowFA, with the
    only difference that the first affine ensures that the reverse-transformation
    yields above zero-threshold samples.
    """

    if name == "learned-mini":
        return [
            Affine(
                n_dimensions=n_dimensions,
                init_t=-zero_threshold,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),
            Log(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            Exp(n_dimensions=n_dimensions),
        ]

    elif name == "learned2":
        return [
            Affine(
                n_dimensions=n_dimensions,
                init_t=-zero_threshold,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),
            Log(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            Exp(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
        ]

    elif name == "learned6":
        return [
            Affine(
                n_dimensions=n_dimensions,
                init_t=-zero_threshold,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),
            InvELU(n_dimensions=n_dimensions, offset=1.0),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(
                n_dimensions=n_dimensions,
                init_t=1.0,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),  # go back to +ve support with elu+1
            Affine(
                n_dimensions=n_dimensions, only_positive_shift=True
            ),  # allow shifts but stay in the +ve support
            InvELU(
                n_dimensions=n_dimensions, offset=1.0
            ),  # go to continuous space with inverse elu+1
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(
                n_dimensions=n_dimensions,
                init_t=1.0,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),
            Affine(
                n_dimensions=n_dimensions,
                only_positive_shift=True,
                init_t=1.0,
            ),  # allow shifts but stay in the +ve support
            InvELU(
                n_dimensions=n_dimensions, offset=1.0
            ),  # go to continuous space with inverse elu+1
        ]


class ZeroInflatedFlowFA(nn.Module):
    def __init__(
        self,
        dataloaders,
        seed,
        image_model_fn,
        image_model_config,
        d_latent,
        use_avg_reg,
        latent_weights_sparsity_reg_lambda,
        sample_transform,
        per_neuron_samples_transform,
        zero_threshold,
        init_psi_diag_coef,
        init_C_coef,
        unit_variance_constraint,
    ):
        super().__init__()

        # set the random seed
        set_random_seed(seed)

        try:
            module_path, class_name = split_module_name(image_model_fn)
            model_fn = dynamic_import(module_path, class_name)
        except:
            raise ValueError("model function does not exist.")

        self.encoding_model = model_fn(dataloaders, seed, **image_model_config)
        self.d_latent = None if d_latent == 0 else d_latent
        self.use_avg_reg = use_avg_reg
        self.latent_weights_sparsity_reg_lambda = latent_weights_sparsity_reg_lambda
        self.unit_variance_constraint = unit_variance_constraint

        dataloaders = dataloaders["train"] if "train" in dataloaders else dataloaders
        temp_b = next(iter(list(dataloaders.values())[0]))._asdict()
        d_out = temp_b["targets"].shape[1]
        self.d_out = d_out

        if self.d_latent is not None:
            self._C = nn.Parameter(
                torch.rand(d_latent, d_out) * init_C_coef, requires_grad=False
            )

        self.logpsi_diag = nn.Parameter(
            torch.log(torch.ones(d_out) * init_psi_diag_coef), requires_grad=False
        )

        self.per_neuron_samples_transform = per_neuron_samples_transform
        n_dimensions = d_out if per_neuron_samples_transform else 1

        self.sample_transform = Flow(
            get_learned_transforms(
                sample_transform, zero_threshold, n_dimensions=n_dimensions
            )
        )
        self.register_buffer("zero_threshold", torch.tensor(zero_threshold))

    @property
    def psi_diag(self):
        return torch.exp(self.logpsi_diag)

    @property
    def C(self):
        if self.d_latent is not None:
            return self._C

        else:
            device = self.psi_diag.device
            return torch.zeros(1, self.d_out).to(device)

    @property
    def C_and_psi_diag(self):
        if self.unit_variance_constraint:
            sigma_diag = (self.C ** 2).sum(0) + self.psi_diag
            return self.C / sigma_diag.sqrt(), self.psi_diag / sigma_diag
        else:
            return self.C, self.psi_diag

    @property
    def sigma(self):
        C, psi_diag = self.C_and_psi_diag
        return C.T @ C + torch.diag(psi_diag)

    def forward(self, *args, data_key=None, return_all=False):
        mu, q = self.encoding_model(*args, data_key=data_key)
        return mu, q

    def log_likelihood(self, *batch, data_key=None, in_bits=False):

        mu, q = self.forward(*batch, data_key=data_key)
        C, psi_diag = self.C_and_psi_diag
        inputs, targets = batch[:2]

        loglikelihood = ZIF_Loglikelihood()(
            self.sample_transform,
            targets,
            self.zero_threshold,
            q,
            mu,
            C,
            psi_diag,
        )

        return loglikelihood / np.log(2.0) if in_bits else loglikelihood

    def loss(self, *batch, data_key=None, use_avg=False):
        agg_fn = torch.mean if use_avg else torch.sum
        loss = -self.log_likelihood(*batch, data_key=data_key)
        return agg_fn(loss)

    def regularizer(self, data_key=None):
        agg_fn = torch.mean if self.use_avg_reg else torch.sum

        reg = 0.0
        if self.d_latent is not None:
            latent_weights_sparsity_reg = (
                self.latent_weights_sparsity_reg_lambda
                * agg_fn(torch.norm(self.C, p=1, dim=1))
            )
            reg = reg + latent_weights_sparsity_reg

        return self.encoding_model.regularizer(data_key) + reg

    def apply_changes_while_training(self):
        if self.d_latent is not None:
            self._C.requires_grad_(True)
        self.logpsi_diag.requires_grad_(True)


def ziffa(
    dataloaders,
    seed,
    image_model_fn=None,
    image_model_config=None,
    d_latent=0,
    use_avg_reg=False,
    latent_weights_sparsity_reg_lambda=0.0,
    sample_transform=None,
    per_neuron_samples_transform=False,
    zero_threshold=None,
    init_psi_diag_coef=0.01,
    init_C_coef=0.1,
    unit_variance_constraint=False,
):

    if image_model_fn is None:
        raise ValueError("Please specify image-model function.")

    if image_model_config is None:
        raise ValueError(
            "Please specify the config of the image-model, excluding dataloaders and seed."
        )

    device = image_model_config.get("device", "cuda")
    set_random_seed(seed)
    return ZeroInflatedFlowFA(
        dataloaders,
        seed,
        image_model_fn=image_model_fn,
        image_model_config=image_model_config,
        d_latent=d_latent,
        use_avg_reg=use_avg_reg,
        latent_weights_sparsity_reg_lambda=latent_weights_sparsity_reg_lambda,
        sample_transform=sample_transform,
        per_neuron_samples_transform=per_neuron_samples_transform,
        zero_threshold=zero_threshold,
        init_psi_diag_coef=init_psi_diag_coef,
        init_C_coef=init_C_coef,
        unit_variance_constraint=unit_variance_constraint,
    ).to(device)