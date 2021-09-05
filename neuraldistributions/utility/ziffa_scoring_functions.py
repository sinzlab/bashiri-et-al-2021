from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Normal

from neuralpredictors.measures import corr
from .model_evaluation import spearman_corr, woodbury_torch


def get_conditional_mean_torch(ind, not_ind, R, C, means, samples):
    cov_mat = R + C.T @ C
    cov_others_others_inv = woodbury_torch(
        R[:, not_ind][not_ind, :], C[:, not_ind].T, C[:, not_ind]
    )
    cov_self_others = cov_mat[ind, not_ind]

    return (
        (
            means[:, ind]
            + cov_self_others @ cov_others_others_inv @ (samples - means)[:, not_ind].T
        )
        .cpu()
        .data.numpy()
    )


def get_conditional_variance_torch(ind, not_ind, R, C):
    cov_mat = R + C.T @ C
    cov_others_others_inv = woodbury_torch(
        R[:, not_ind][not_ind, :], C[:, not_ind].T, C[:, not_ind]
    )
    cov_self_others = cov_mat[ind, not_ind]

    return (
        (
            cov_mat[ind, ind]
            - cov_self_others @ cov_others_others_inv @ cov_self_others.T
        )
        .cpu()
        .data.numpy()
    )


def get_conditional_mean_and_variance(model, target, mu, R, C):

    gaussian_mask = target >= model.zero_threshold.item()
    transformed_target = model.sample_transform(target)[0]

    neuron_index = torch.arange(target.shape[1]).to(target.device)

    conditional_mean, conditional_variance = [], []
    for ind in tqdm(range(target.shape[1])):

        not_ind = torch.where(gaussian_mask & (neuron_index != ind))[1]

        cond_mean = get_conditional_mean_torch(
            ind, not_ind, R, C, mu, transformed_target
        ).item()
        cond_var = get_conditional_variance_torch(ind, not_ind, R, C).item()

        conditional_mean.append(cond_mean)
        conditional_variance.append(cond_var)

    conditional_mean = np.array(conditional_mean)
    conditional_variance = np.array(conditional_variance)

    return conditional_mean, conditional_variance


def get_conditional_non_zero_mean(
    ziffa_model, data_key, image, target, n_samples=1000000, batch_size=1000
):

    device = next(ziffa_model.parameters()).device

    C, psi_diag = ziffa_model.C_and_psi_diag
    R = psi_diag.diag()

    mu, q_ziffa = ziffa_model(torch.from_numpy(image).to(device), data_key=data_key)
    conditional_mean, conditional_variance = get_conditional_mean_and_variance(
        ziffa_model, target, mu, R, C
    )

    if (n_samples % batch_size) == 0:
        n_samples_across_iters = [batch_size] * (n_samples // batch_size)
    else:
        n_samples_across_iters = [batch_size] * (n_samples // batch_size) + [
            n_samples % batch_size
        ]

    sum_original_space = []
    n_valid_samples = 0.0
    for n in tqdm(n_samples_across_iters):
        samples_gaussian = Normal(
            torch.from_numpy(conditional_mean).to(device),
            torch.from_numpy(conditional_variance).to(device).sqrt(),
        ).sample((n,))

        _samples_original_space = (
            ziffa_model.sample_transform.inv(samples_gaussian).cpu().data.numpy()
        )

        sum_original_space.append(
            np.nansum(_samples_original_space, axis=0, keepdims=True)
        )
        n_valid_samples += (~np.isnan(_samples_original_space)).sum(axis=0)

    return (
        (np.concatenate(sum_original_space) / n_valid_samples).sum(axis=0),
        q_ziffa.cpu().data.numpy(),
    )


def single_trial_conditional_correlation_per_datakey_ziffa(
    dataloader, model_ziffa, data_key, n_samples=300000, batch_size=1000
):

    device = next(model_ziffa.parameters()).device

    # get images and responses
    test_images = np.concatenate([b[0].cpu().data.numpy() for b in dataloader])
    test_responses = np.concatenate([b[1].cpu().data.numpy() for b in dataloader])

    # get the  conditional mean
    predicted_conditional_means = []
    for image_idx, _ in enumerate(test_images):

        image = test_images[[image_idx]]
        target = torch.from_numpy(test_responses[[image_idx]]).to(device)
        non_zero_mean, q = get_conditional_non_zero_mean(
            model_ziffa,
            data_key,
            image,
            target,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        predicted_mean = (1 - q) * (
            model_ziffa.zero_threshold.item() / 2
        ) + q * non_zero_mean

        predicted_conditional_means.append(predicted_mean)

    predicted_conditional_means = np.vstack(predicted_conditional_means)

    # compute correlation
    return {
        "pearson": corr(test_responses, predicted_conditional_means, axis=0),
        "spearman": spearman_corr(test_responses, predicted_conditional_means, axis=0),
    }


def single_trial_conditional_correlation_ziffa(
    dataloaders, model_ziffa, corr_type="pearson", n_samples=300000, batch_size=1000
):

    model_ziffa.eval()

    correlations, data_keys = [], []
    for data_key, dataloader in dataloaders.items():
        correlation_dict = single_trial_conditional_correlation_per_datakey_ziffa(
            dataloader,
            model_ziffa,
            data_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        correlation = correlation_dict[corr_type]
        n_neurons = len(correlation)

        correlations.append(correlation)
        data_keys.extend([data_key] * n_neurons)

    return data_keys, np.hstack(correlations)
