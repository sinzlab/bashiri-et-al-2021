import numpy as np
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from .utility.ziffa_scoring_functions import (
    get_conditional_mean_torch,
    get_conditional_variance_torch,
)
from .utility import imread


def generate_one_sample(model, image_path, verbose=False):

    # prepare the input image
    input_image = imread(image_path, xres=64, yres=36)
    device = next(model.parameters()).device
    input_image = torch.from_numpy(input_image).to(device)

    # get the distribution params
    mu, q = model(input_image)
    C, psi_diag = model.C_and_psi_diag
    R = psi_diag.diag()
    sigma = model.sigma
    n_neurons = sigma.shape[0]
    zero_threshold = model.zero_threshold.item()

    # specify "zero" vs "non-zero" neurons
    uniform_samples = torch.rand(n_neurons).to(device)
    nonzero_neurons_idx = torch.where(uniform_samples < q[0])[0]
    zero_neurons_idx = torch.where(uniform_samples >= q[0])[0]

    # subselect the distribution parameters for the non-zero neurons
    mu_nz = mu[:, nonzero_neurons_idx]
    C_nz = C[:, nonzero_neurons_idx]
    R_nz = R[:, nonzero_neurons_idx][nonzero_neurons_idx, :]
    sigma_nz = sigma[:, nonzero_neurons_idx][nonzero_neurons_idx, :]

    # generate samples for the non-zero neurons
    joint_density_fn = MultivariateNormal(mu_nz, sigma_nz)
    gaussian_sample_nz = joint_density_fn.sample()

    """
    at this point we have gaussian samples for nan-zero neurons, and
    now we want to make sure that for these non-zero neurons we have 
    valid samples in the neuronal space.
    """

    iterations = 0
    while True:

        # find out valid samples (ok neurons) vs invalid samples (nan neurons) by inv-transforming the gaussian samples into neuronal space
        sample_temp = torch.zeros_like(q)
        sample_temp[:, nonzero_neurons_idx] = gaussian_sample_nz
        neural_sample_nz = model.sample_transform.inv(sample_temp)[
            :, nonzero_neurons_idx
        ]

        # get the index of nan and ok neurons
        valid_neurons_idx = torch.where(~neural_sample_nz.isnan())[1]
        invalid_neurons_idx = torch.where(neural_sample_nz.isnan())[1]

        if len(invalid_neurons_idx) == 0:
            if verbose:
                print(f"Iteration {iterations+1}: all neurons have valid sample.")
            break
        else:
            if verbose:
                print(
                    f"Iteration {iterations+1}: {len(invalid_neurons_idx)} neurons with invalid sample."
                )

        conditional_mean = np.array(
            [
                get_conditional_mean_torch(
                    ini, valid_neurons_idx, R_nz, C_nz, mu_nz, gaussian_sample_nz
                )
                for ini in invalid_neurons_idx
            ]
        ).T
        conditional_variance = np.array(
            [
                get_conditional_variance_torch(ini, valid_neurons_idx, R_nz, C_nz)
                for ini in invalid_neurons_idx
            ]
        ).T
        conditional_mu = torch.from_numpy(conditional_mean).to(device)
        conditional_sigma = torch.from_numpy(conditional_variance).to(device).diag()

        joint_density_fn = MultivariateNormal(conditional_mu, conditional_sigma)
        new_samples = joint_density_fn.sample()

        # For the invalid neurons, replace the old samples with the new samples
        gaussian_sample_nz[:, invalid_neurons_idx] = new_samples
        iterations += 1

    # create an array containing samples for zero and positive neurons
    nonzero_neurons_idx = nonzero_neurons_idx.cpu().data.numpy()
    zero_neurons_idx = zero_neurons_idx.cpu().data.numpy()

    neural_sample = torch.zeros_like(uniform_samples).cpu().data.numpy()
    neural_sample[nonzero_neurons_idx] = neural_sample_nz.cpu().data.numpy()
    neural_sample[zero_neurons_idx] = np.random.uniform(
        0, zero_threshold, size=len(zero_neurons_idx)
    )

    return neural_sample[None, :]


def generate_n_samples(model, image_path, n_samples=1, random_seed=42, verbose=False):
    # fix random seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    samples = []
    for _ in tqdm(range(n_samples)):
        samples.append(generate_one_sample(model, image_path, verbose=verbose))

    return np.concatenate(samples)