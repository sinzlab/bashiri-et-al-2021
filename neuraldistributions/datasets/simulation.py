from collections import namedtuple
import numpy as np
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from ..utility import get_dataloader, set_random_seed
from ..models import transforms


def gaussian(n_samples, mean, C, psi_diag, transform):
    """
    mean     -> shape: (n_neurons,)
    C        -> shape: (d_latent, n_neurons)
    psi_diag -> shape: (n_neurons,)
    transformation_fn: the transformation function applied on the observation (we apply the inverse on the Gaussian samples)
    """

    d_latent, n_neurons = C.shape
    z = np.random.randn(n_samples, d_latent)

    independent_noise = np.random.randn(n_samples, n_neurons) * np.sqrt(
        psi_diag[None, :]
    )
    samples = mean[None, :] + z @ C + independent_noise

    if transform == "anscombe":
        samples = np.abs(samples - np.sqrt(1.5)) + 1e-12 + np.sqrt(1.5)
        assert not (
            samples < np.sqrt(1.5)
        ).any(), "there are values smaller than sqrt(3/2)"

    transform_fn = getattr(transforms, transform)(numpy=True)
    return transform_fn.inv(samples)


def poisson(n_samples, mean, C, nonlinearity):
    """
    mean     -> shape: (n_neurons,)
    C        -> shape: (d_latent, n_neurons)
    nonlinearity -> \in {exp, softplus}
    """

    nonlinearities = {
        "exp": np.exp,
        "softplus": lambda x: np.log(1.0 + np.exp(x)),
    }

    mean = np.array(mean)
    C = np.array(C)

    d_latent, n_neurons = C.shape
    z = np.random.randn(n_samples, d_latent)

    return np.random.poisson(nonlinearities[nonlinearity](mean[None, :] + z @ C))


def poisson_cont(n_samples, mean, C, nonlinearity):
    """
    mean     -> shape: (n_neurons,)
    C        -> shape: (d_latent, n_neurons)
    nonlinearity -> \in {exp, softplus}
    """

    nonlinearities = {
        "exp": np.exp,
        "softplus": lambda x: np.log(1.0 + np.exp(x)),
    }

    mean = np.array(mean)
    C = np.array(C)

    d_latent, n_neurons = C.shape
    z = np.random.randn(n_samples, d_latent)
    eps = np.random.rand(n_samples, n_neurons)

    return np.random.poisson(nonlinearities[nonlinearity](mean[None, :] + z @ C)) + eps


def zig():
    raise NotImplementedError()


def simulation_loaders(
    seed,
    batch_size,
    shuffle_train=True,
    device="cuda",
    simulation_fn=None,
    simulation_config=None,
):

    try:
        module_path, class_name = split_module_name(simulation_fn)
        simulation_fn = dynamic_import(module_path, class_name)
    except:
        raise ValueError("simulation function does not exist.")

    set_random_seed(seed)
    train_loader = get_dataloader(
        simulation_fn(**simulation_config),
        names=["samples"],
        batch_size=int(batch_size),
        shuffle=shuffle_train,
        device=device,
    )
    test_loader = get_dataloader(
        simulation_fn(**simulation_config),
        names=["samples"],
        batch_size=int(batch_size),
        shuffle=False,
        device=device,
    )

    neurons = namedtuple("neurons", ["ids"])
    neurons.ids = np.arange(len(simulation_config["mean"]))

    train_loader.dataset.neurons = neurons
    test_loader.dataset.neurons = neurons

    return {"train": train_loader, "test": test_loader}
