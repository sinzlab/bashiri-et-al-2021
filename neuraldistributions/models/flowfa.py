from warnings import warn
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import LowRankMultivariateNormal

from ..utility import set_random_seed
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.nn_helpers import set_random_seed
from neuralpredictors.training import eval_state
from . import transforms
from .transforms import (
    Identity,
    SQRT,
    Anscombe,
    ELU,
    InvELU,
    ELUF,
    Affine,
    Log,
    Flow,
    MeanTransfom,
    Exp,
)


def get_learned_transforms(name, n_dimensions=1):
    if name == "learned-mini":
        return [
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
            Log(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            Exp(n_dimensions=n_dimensions),
        ]

    elif name == "learned2":
        return [
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
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
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
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


def freeze_params(model):
    for name, param in model.named_parameters():
        param.requires_grad_(False)


def unfreeze_params(model):
    for name, param in model.named_parameters():
        param.requires_grad_(True)


class FlowFA(nn.Module):
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
        mean_transform,
        per_neuron_samples_transform,
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

        if sample_transform is None:
            self.sample_transform = getattr(transforms, "identity")(numpy=False)

        elif sample_transform in ["identity", "sqrt", "anscombe"] + [
            f"example{i+1}" for i in range(10)
        ]:
            self.sample_transform = getattr(transforms, sample_transform)(numpy=False)

        elif "learned" in sample_transform:
            self.sample_transform = Flow(
                get_learned_transforms(sample_transform, n_dimensions=n_dimensions)
            )

        else:
            raise ValueError("The passed sample_transform is not available.")

        if (mean_transform is None) or (mean_transform == "identity"):
            self.mean_transform = lambda x: x
        elif mean_transform == "learned":
            self.mean_transform = MeanTransfom(hidden_layers=2, hidden_features=10)
        elif mean_transform == "anscombe":
            self.mean_transform = lambda x: Anscombe.anscombe(x) - 1 / (4 * x.sqrt())
        else:
            raise ValueError(
                "At the moment only three options for sample transform are available: identity, learned and anscombe"
            )

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

        image_model_pred = self.encoding_model(*args, data_key=data_key) + 1e-8
        mu = self.mean_transform(image_model_pred)
        return (mu, self.sigma) if return_all else mu

    def predict_mean(self, *batch, data_key=None):
        return self.forward(*batch, data_key=data_key)

    def log_likelihood(self, *batch, data_key=None, in_bits=False):

        # get model predictions with the covariance matrix
        mu = self.forward(*batch, data_key=data_key)

        inputs, targets = batch[:2]
        transformed_targets, logdet = self.sample_transform(targets)

        C, psi_diag = self.C_and_psi_diag
        dist = LowRankMultivariateNormal(mu, C.T, psi_diag)
        loglikelihood = dist.log_prob(transformed_targets) + logdet.sum(dim=1)

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

    def evaluate(self):
        raise NotImplementedError()


def flowfa(
    dataloaders,
    seed,
    image_model_fn=None,
    image_model_config=None,
    d_latent=0,
    use_avg_reg=False,
    latent_weights_sparsity_reg_lambda=0.0,
    sample_transform=None,
    mean_transform=None,
    per_neuron_samples_transform=False,
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
    return FlowFA(
        dataloaders,
        seed,
        image_model_fn=image_model_fn,
        image_model_config=image_model_config,
        d_latent=d_latent,
        use_avg_reg=use_avg_reg,
        latent_weights_sparsity_reg_lambda=latent_weights_sparsity_reg_lambda,
        sample_transform=sample_transform,
        mean_transform=mean_transform,
        per_neuron_samples_transform=per_neuron_samples_transform,
        init_psi_diag_coef=init_psi_diag_coef,
        init_C_coef=init_C_coef,
        unit_variance_constraint=unit_variance_constraint,
    ).to(device)


class LearnableBaseDist(nn.Module):
    def __init__(
        self,
        d_out,
        d_latent,
        learn_mu=True,
        learn_sigma=False,
        init_psi_diag_coef=0.01,
        init_C_coef=0.1,
        unit_variance_constraint=False,
    ):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(d_out) * 0.0, requires_grad=learn_mu)
        self.logpsi_diag = nn.Parameter(
            torch.log(torch.ones(d_out) * init_psi_diag_coef), requires_grad=learn_sigma
        )
        self.C = nn.Parameter(
            torch.rand(d_latent, d_out) * init_C_coef, requires_grad=learn_sigma
        )
        self.unit_variance_constraint = unit_variance_constraint

    @property
    def psi_diag(self):
        return torch.exp(self.logpsi_diag)

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

    def log_prob(self, x):
        C, psi_diag = self.C_and_psi_diag
        return LowRankMultivariateNormal(self.mu, C.T, psi_diag).log_prob(x)

    def sample(self, n):
        C, psi_diag = self.C_and_psi_diag
        z = LowRankMultivariateNormal(self.mu, C.T, psi_diag).sample((n,))
        return z


class DensityEstimator(nn.Module):
    def __init__(
        self,
        dataloaders,
        seed,
        transform,
        d_latent,
        per_neuron_samples_transform,
        init_psi_diag_coef,
        init_C_coef,
        unit_variance_constraint,
    ):
        super().__init__()

        dataloaders = dataloaders["train"] if "train" in dataloaders else dataloaders
        temp_b = next(iter(dataloaders))._asdict()
        d_out = temp_b["samples"].shape[1]
        self.d_out = d_out

        self.per_neuron_samples_transform = per_neuron_samples_transform
        n_dimensions = d_out if per_neuron_samples_transform else 1

        if transform in ["identity", "sqrt", "anscombe"] + [
            f"example{i+1}" for i in range(10)
        ]:
            self.flow = getattr(transforms, transform)(numpy=False)
        else:
            self.flow = Flow(
                get_learned_transforms(transform, n_dimensions=n_dimensions)
            )

        self.dist = LearnableBaseDist(
            temp_b["samples"].shape[1],
            d_latent=d_latent,
            init_psi_diag_coef=init_psi_diag_coef,
            init_C_coef=init_C_coef,
            unit_variance_constraint=unit_variance_constraint,
        )

    def log_likelihood(self, *batch, in_bits=False):

        y = batch[0]
        x, logdet = self.flow(y)
        loglikelihood = self.dist.log_prob(x) + logdet.sum(dim=1)

        return loglikelihood / np.log(2.0) if in_bits else loglikelihood

    def loss(self, *batch):

        loss = -self.log_likelihood(*batch)
        return loss.mean()

    def apply_changes_while_training(self):
        self.dist.logpsi_diag.requires_grad_(True)
        self.dist.C.requires_grad_(True)

    def regularizer(self):
        return 0.0


def density_estimator(
    dataloaders,
    seed,
    transform,
    d_latent,
    per_neuron_samples_transform=False,
    init_psi_diag_coef=0.01,
    init_C_coef=0.1,
    unit_variance_constraint=False,
    device="cuda",
):
    return DensityEstimator(
        dataloaders,
        seed,
        transform,
        d_latent,
        init_psi_diag_coef=init_psi_diag_coef,
        init_C_coef=init_C_coef,
        unit_variance_constraint=unit_variance_constraint,
        per_neuron_samples_transform=per_neuron_samples_transform,
    ).to(device)
