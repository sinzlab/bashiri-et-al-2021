from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Normal

from neuralpredictors.measures import corr
from .model_evaluation import (
    get_conditional_means,
    get_conditional_variances,
    spearman_corr,
)


def get_conditional_means_and_variances_flowfa(
    dataloader, flowfa_model, data_key, use_torch=True
):

    C, psi_diag = flowfa_model.C_and_psi_diag

    R = psi_diag.diag().cpu().data.numpy()
    C = C.cpu().data.numpy()

    transformed_responses, predicted_means = [], []
    for b in dataloader:
        transformed_responses.append(
            flowfa_model.sample_transform(b[1])[0].cpu().data.numpy()
        )
        predicted_means.append(flowfa_model(*b, data_key=data_key).cpu().data.numpy())

    transformed_responses = np.concatenate(transformed_responses)
    predicted_means = np.concatenate(predicted_means)

    conditional_predicted_means = get_conditional_means(
        R, C, predicted_means, transformed_responses, use_torch=use_torch
    )

    conditional_variances = get_conditional_variances(R, C, use_torch=use_torch)

    return conditional_predicted_means, conditional_variances


def get_conditional_mean_original_space_per_image_flowfa(
    flowfa_model,
    conditional_mean,
    conditional_variance,
    n_samples=300000,
    batch_size=1000,
    use_torch=True,
):

    device = next(flowfa_model.parameters()).device

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
            flowfa_model.sample_transform.inv(samples_gaussian).cpu().data.numpy()
        )

        sum_original_space.append(
            np.nansum(_samples_original_space, axis=0, keepdims=True)
        )
        n_valid_samples += (~np.isnan(_samples_original_space)).sum(axis=0)

    return (np.concatenate(sum_original_space) / n_valid_samples).sum(axis=0)


def single_trial_conditional_correlation_per_datakey_flowfa(
    dataloader,
    model_flowfa,
    data_key,
    n_samples=300000,
    batch_size=1000,
    use_torch=True,
):

    # get the transformed targets, conditional means and conditional variances
    (
        conditional_means,
        conditional_variances,
    ) = get_conditional_means_and_variances_flowfa(
        dataloader, model_flowfa, data_key, use_torch=use_torch
    )

    # Estimate the mean in original space per conditional mean (variance is the same for all trials)
    predicted_conditional_means = []
    for conditional_mean in conditional_means:

        predicted_conditional_mean = (
            get_conditional_mean_original_space_per_image_flowfa(
                model_flowfa,
                conditional_mean,
                conditional_variances,
                n_samples=n_samples,
                batch_size=batch_size,
            )
        )
        predicted_conditional_means.append(predicted_conditional_mean)

    predicted_conditional_means = np.vstack(predicted_conditional_means)

    # get the test responses
    test_responses = np.concatenate([b[1].cpu().data.numpy() for b in dataloader])

    # compute correlation
    return {
        "pearson": corr(test_responses, predicted_conditional_means, axis=0),
        "spearman": spearman_corr(test_responses, predicted_conditional_means, axis=0),
    }


def single_trial_conditional_correlation_flowfa(
    dataloaders,
    model_flowfa,
    corr_type="pearson",
    n_samples=300000,
    batch_size=1000,
    use_torch=True,
):

    model_flowfa.eval()

    correlations, data_keys = [], []
    for data_key, dataloader in dataloaders.items():
        correlation_dict = single_trial_conditional_correlation_per_datakey_flowfa(
            dataloader,
            model_flowfa,
            data_key,
            n_samples=n_samples,
            batch_size=batch_size,
            use_torch=use_torch,
        )

        correlation = correlation_dict[corr_type]
        n_neurons = len(correlation)

        correlations.append(correlation)
        data_keys.extend([data_key] * n_neurons)

    return data_keys, np.hstack(correlations)