import copy

from neuralpredictors.layers.cores import SE2dCore
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from .encoders import ZIGEncoder
from .readouts import MultipleZIGReadout
import numpy as np
from warnings import warn


def zig_se2d_fullgaussian2d(
    dataloaders,
    seed,
    data_info=None,
    transfer_state_dict=None,
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
    grid_mean_predictor={
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 0,
        "hidden_features": 30,
        "final_tanh": True,
    },
    share_features=False,
    share_grid=False,
    share_transform=False,
    init_noise=1e-3,
    init_transform_scale=0.2,
    zero_thresholds=None,
    init_ks=None,
    theta_image_dependent=True,
    k_image_dependent=False,
    loc_image_dependent=False,
    q_image_dependent=True,
    offset=1.0e-6,
):
    """
    Model class of a SE2dCore and a Gaussian readout)
    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        share_grid: whether to share the grid between neurons. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        share_transform: whether to share the transform from the grid_mean_predictor between neurons. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        init_noise: noise for initialization of weights
        init_transform_scale: scale of the weights of the randomly intialized grid_mean_predictor network
        all other args: See Documentation of SE2dCore in neuralpredictors.layers.cores and
            FullGaussian2d in neuralpredictors.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    if zero_thresholds == "from dataset":
        zero_thresholds = {}
        for k, v in dataloaders["train"].items():
            zero_thresholds[k] = v.dataset.neurons.normalized_zero_thresholds

    if init_ks == "from dataset":
        init_ks = {}
        for k, v in dataloaders["train"].items():
            init_ks[k] = v.dataset.neurons.normalized_ks

    if transfer_state_dict is not None:
        print(
            "Transfer state_dict given. This will only have an effect in the bayesian hypersearch. See: TrainedModelBayesianTransfer "
        )

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
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
        inferred_params_n=theta_image_dependent
        + k_image_dependent
        + loc_image_dependent
        + q_image_dependent,
    )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0, keepdims=True).repeat(
                readout[key].bias.shape[0], 1
            )

    model = ZIGEncoder(
        core,
        readout,
        zero_thresholds=zero_thresholds,
        init_ks=init_ks,
        theta_image_dependent=theta_image_dependent,
        k_image_dependent=k_image_dependent,
        loc_image_dependent=loc_image_dependent,
        q_image_dependent=q_image_dependent,
        offset=offset,
    )

    return model