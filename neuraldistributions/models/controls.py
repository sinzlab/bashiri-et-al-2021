import copy
import numpy as np
import torch
from torch import nn
from torch import distributions

from .flowfa import FlowFA
from neuralpredictors.layers.cores import SE2dCore
from neuralmetrics.training.losses import ZIGLoss
from neuralmetrics.utils import fitted_zig_mean, fitted_zig_variance
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from neuralmetrics.models.neuralnet.readouts import MultipleZIGReadout

from ..utility import set_random_seed


class Poisson(FlowFA):
    def __init__(
        self,
        dataloaders,
        seed,
        image_model_fn,
        image_model_config,
        use_avg_reg,
    ):
        super().__init__(
            dataloaders=dataloaders,
            seed=seed,
            image_model_fn=image_model_fn,
            image_model_config=image_model_config,
            use_avg_reg=use_avg_reg,
            d_latent=None,
            sample_transform=None,
            mean_transform=None,
            latent_weights_sparsity_reg_lambda=0,
            per_neuron_samples_transform=False,
            init_psi_diag_coef=0.01,
            init_C_coef=0.1,
            unit_variance_constraint=False,
        )

    def log_likelihood(self, *batch, data_key=None, in_bits=False):

        predictions = self.forward(*batch, data_key=data_key)
        targets = batch[1]
        loglikelihood = (
            distributions.Poisson(predictions).log_prob(torch.floor(targets)).sum(dim=1)
        )

        return loglikelihood / np.log(2.0) if in_bits else loglikelihood

    def loss(self, *batch, data_key=None, use_avg=False):
        agg_fn = torch.mean if use_avg else torch.sum

        predictions = self.forward(*batch, data_key=data_key)
        targets = batch[1]
        poisson_loss = nn.PoissonNLLLoss(log_input=False, reduction="none")
        loss = poisson_loss(predictions, targets).sum(dim=1)

        return agg_fn(loss)

    def forward(self, *batch, data_key=None):
        image_model_pred = self.encoding_model(batch[0], data_key=data_key)
        return image_model_pred

    def predict_mean(self, *batch, data_key=None):
        mu = self.forward(*batch, data_key=data_key)
        return mu


def poisson(
    dataloaders,
    seed,
    image_model_fn=None,
    image_model_config=None,
    use_avg_reg=False,
):

    if image_model_fn is None:
        raise ValueError("Please specify image-model function.")

    if image_model_config is None:
        raise ValueError(
            "Please specify the config of the image-model, excluding dataloaders and seed."
        )

    device = image_model_config.get("device", "cuda")
    set_random_seed(seed)
    return Poisson(
        dataloaders,
        seed,
        image_model_fn=image_model_fn,
        image_model_config=image_model_config,
        use_avg_reg=use_avg_reg,
    ).to(device)


class ZIG(FlowFA):
    def __init__(
        self,
        dataloaders,
        seed,
        image_model_fn,
        image_model_config,
        use_avg_reg,
    ):
        super().__init__(
            dataloaders=dataloaders,
            seed=seed,
            image_model_fn=image_model_fn,
            image_model_config=image_model_config,
            use_avg_reg=use_avg_reg,
            d_latent=None,
            sample_transform=None,
            mean_transform=None,
            latent_weights_sparsity_reg_lambda=0,
            per_neuron_samples_transform=False,
            init_psi_diag_coef=0.01,
            init_C_coef=0.1,
            unit_variance_constraint=False,
        )

        self.zig_loss = ZIGLoss()

    def forward(self, *batch, data_key=None):
        theta, k, loc, q = self.encoding_model(batch[0], data_key=data_key)
        return theta, k, loc, q

    def log_likelihood(self, *batch, data_key=None, in_bits=False):

        theta, k, loc, q = self.forward(*batch, data_key=data_key)
        targets = batch[1]

        loglikelihood = (-self.zig_loss(targets, theta, k, loc, q)).sum(dim=1)

        return loglikelihood / np.log(2.0) if in_bits else loglikelihood

    def predict_mean(self, *batch, data_key=None):
        theta, k, loc, q = self.forward(*batch, data_key=data_key)
        return fitted_zig_mean(theta, k, loc, q)

    def predict_variance(self, *batch, data_key=None):
        theta, k, loc, q = self.forward(*batch, data_key=data_key)
        return fitted_zig_variance(theta, k, loc, q)


def zig(
    dataloaders,
    seed,
    image_model_fn=None,
    image_model_config=None,
    use_avg_reg=False,
):

    if image_model_fn is None:
        raise ValueError("Please specify image-model function.")

    if image_model_config is None:
        raise ValueError(
            "Please specify the config of the image-model, excluding dataloaders and seed."
        )

    device = image_model_config.get("device", "cuda")
    set_random_seed(seed)
    return ZIG(
        dataloaders,
        seed,
        image_model_fn=image_model_fn,
        image_model_config=image_model_config,
        use_avg_reg=use_avg_reg,
    ).to(device)


class ZIFEncoder(nn.Module):
    def __init__(
        self,
        core,
        readout,
        eps=1e-12,
    ):
        super().__init__()
        self.core = core
        self.readout = readout
        self.eps = eps

    def q_nl(self, q):
        return torch.sigmoid(q) * 0.99999 + self.eps

    def forward(self, *args, data_key=None, detach_core=False, **kwargs):

        x = args[0]
        batch_size = x.shape[0]

        x = self.core(x)
        if detach_core:
            x = x.detach()

        if "sample" in kwargs:
            mean, q = self.readout(x, data_key=data_key, sample=kwargs["sample"])
        else:
            mean, q = self.readout(x, data_key=data_key)

        return mean, self.q_nl(q)

    def regularizer(self, data_key, detach_core=False):
        return int(
            not detach_core
        ) * self.core.regularizer() + self.readout.regularizer(data_key)


def zif_fullgaussian2d(
    dataloaders,
    seed,
    data_info=None,
    # core args
    hidden_channels=64,
    input_kern=9,
    hidden_kern=7,
    layers=4,
    gamma_input=6.3831,
    skip=0,
    bias=False,
    final_nonlinearity=True,
    momentum=0.9,
    pad_input=False,
    batch_norm=True,
    hidden_dilation=1,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    stack=-1,
    se_reduction=32,
    n_se_blocks=0,
    depth_separable=True,
    linear=False,
    # readout args
    init_mu_range=0.3,
    init_sigma=0.1,
    readout_bias=True,
    gamma_readout=0.0076,
    gauss_type="full",
    grid_mean_predictor=None,
    share_features=False,
    share_grid=False,
    share_transform=False,
    init_noise=1e-3,
    init_transform_scale=0.2,
    zero_threshold=None,
    eps=1.0e-12,
):

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")
        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {}
            for k, v in dataloaders.items():
                # real data
                try:
                    if v.dataset.neurons.animal_ids[0] != 0:
                        source_grids[k] = v.dataset.neurons.cell_motor_coordinates[
                            :, :input_dim
                        ]
                    # simulated data -> get random linear non-degenerate transform of true positions
                    else:
                        source_grid_true = v.dataset.neurons.center[:, :input_dim]
                        det = 0.0
                        loops = 0
                        grid_bias = np.random.rand(2) * 3
                        while det < 5.0 and loops < 100:
                            matrix = np.random.rand(2, 2) * 3
                            det = np.linalg.det(matrix)
                            loops += 1
                        assert det > 5.0, "Did not find a non-degenerate matrix"
                        source_grids[k] = np.add(
                            (matrix @ source_grid_true.T).T, grid_bias
                        )
                except FileNotFoundError:
                    print(
                        "Dataset type is not recognized to be from Baylor College of Medicine."
                    )
                    source_grids[k] = v.dataset.neurons.cell_motor_coordinates[
                        :, :input_dim
                    ]
        elif grid_mean_predictor_type == "shared":
            pass
        else:
            raise ValueError(
                "Grid mean predictor type {} not understood.".format(
                    grid_mean_predictor_type
                )
            )

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {
            k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()
        }
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(
                all_multi_unit_ids
            ), "All multi unit IDs must be present in all datasets"

    set_random_seed(seed)

    core = SE2dCore(
        input_channels=core_input_channels,
        hidden_channels=hidden_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        layers=layers,
        gamma_input=gamma_input,
        skip=skip,
        final_nonlinearity=final_nonlinearity,
        bias=bias,
        momentum=momentum,
        pad_input=pad_input,
        batch_norm=batch_norm,
        hidden_dilation=hidden_dilation,
        laplace_padding=laplace_padding,
        input_regularizer=input_regularizer,
        stack=stack,
        se_reduction=se_reduction,
        n_se_blocks=n_se_blocks,
        depth_separable=depth_separable,
        linear=linear,
    )

    readout = MultipleZIGReadout(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        init_mu_range=init_mu_range,
        bias=readout_bias,
        init_sigma=init_sigma,
        gamma_readout=gamma_readout,
        gauss_type=gauss_type,
        grid_mean_predictor=grid_mean_predictor,
        grid_mean_predictor_type=grid_mean_predictor_type,
        source_grids=source_grids,
        share_features=share_features,
        share_grid=share_grid,
        share_transform=share_transform,
        shared_match_ids=shared_match_ids,
        init_noise=init_noise,
        init_transform_scale=init_transform_scale,
        inferred_params_n=2,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0, keepdims=True).repeat(
                readout[key].bias.shape[0], 1
            )

    model = ZIFEncoder(
        core,
        readout,
        eps=eps,
    )

    return model