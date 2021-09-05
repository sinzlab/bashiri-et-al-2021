import warnings
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr


def woodbury_torch(A, U, V):
    k = V.shape[0]
    A_inv = torch.diag(1.0 / torch.diag(A))
    B_inv = torch.inverse(torch.eye(k).to(A.device) + V @ A_inv @ U)
    return A_inv - (A_inv @ U @ B_inv @ V @ A_inv)


def get_conditional_mean_torch(ind, R, C, means, samples):
    cov_mat = R + C.T @ C
    not_ind = ~np.isin(torch.arange(cov_mat.shape[0]), ind)
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


def get_conditional_variance_torch(ind, R, C):
    cov_mat = R + C.T @ C
    not_ind = ~np.isin(torch.arange(cov_mat.shape[0]), ind)
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


def woodbury_np(A, U, V):
    k = V.shape[0]
    A_inv = np.diag(1.0 / np.diag(A))
    B_inv = np.linalg.inv(np.eye(k) + V @ A_inv @ U)
    return A_inv - (A_inv @ U @ B_inv @ V @ A_inv)


def get_conditional_mean_np(ind, R, C, means, samples):
    cov_mat = R + C.T @ C
    not_ind = ~np.isin(np.arange(cov_mat.shape[0]), ind)
    cov_others_others_inv = woodbury_np(
        R[:, not_ind][not_ind, :], C[:, not_ind].T, C[:, not_ind]
    )
    cov_self_others = cov_mat[ind, not_ind]
    return (
        means[:, ind]
        + cov_self_others @ cov_others_others_inv @ (samples - means)[:, not_ind].T
    )


def get_conditional_variance_np(ind, R, C):
    cov_mat = R + C.T @ C
    not_ind = ~np.isin(np.arange(cov_mat.shape[0]), ind)
    cov_others_others_inv = woodbury_np(
        R[:, not_ind][not_ind, :], C[:, not_ind].T, C[:, not_ind]
    )
    cov_self_others = cov_mat[ind, not_ind]
    return (
        cov_mat[ind, ind] - cov_self_others @ cov_others_others_inv @ cov_self_others.T
    )


def get_conditional_means(R, C, means, samples, use_torch=False):
    if hasattr(tqdm, "_instances"):
        tqdm._instances.clear()

    cov_mat = R + C.T @ C
    neurons_n = cov_mat.shape[0]
    if use_torch:

        # check if the diagonal is not too low
        if (sum(np.diag(R) <= 0.01) / np.diag(R).shape[0]) >= 0.5:
            warnings.warn(
                "More than 50% of the diagonal entries are smaller than 0.01. The results may not be accurate enough."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tR, tC, tmeans, tsamples = (
            torch.from_numpy(R.astype(np.float32)).to(device),
            torch.from_numpy(C.astype(np.float32)).to(device),
            torch.from_numpy(means.astype(np.float32)).to(device),
            torch.from_numpy(samples.astype(np.float32)).to(device),
        )
        cond_mean = list(
            map(
                get_conditional_mean_torch,
                tqdm(np.arange(neurons_n)),
                [tR] * neurons_n,
                [tC] * neurons_n,
                [tmeans] * neurons_n,
                [tsamples] * neurons_n,
            )
        )

    else:
        cond_mean = list(
            map(
                get_conditional_mean_np,
                tqdm(np.arange(neurons_n)),
                [R] * neurons_n,
                [C] * neurons_n,
                [means] * neurons_n,
                [samples] * neurons_n,
            )
        )

    return np.array(cond_mean).T


def get_conditional_variances(R, C, use_torch=False):
    if hasattr(tqdm, "_instances"):
        tqdm._instances.clear()

    cov_mat = R + C.T @ C
    neurons_n = cov_mat.shape[0]
    if use_torch:

        # check if the diagonal is not too low
        if (sum(np.diag(R) <= 0.01) / np.diag(R).shape[0]) >= 0.5:
            warnings.warn(
                "More than 50% of the diagonal entries are smaller than 0.01. The results may not be accurate enough."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tR, tC = (
            torch.from_numpy(R.astype(np.float32)).to(device),
            torch.from_numpy(C.astype(np.float32)).to(device),
        )
        cond_variance = list(
            map(
                get_conditional_variance_torch,
                tqdm(np.arange(neurons_n)),
                [tR] * neurons_n,
                [tC] * neurons_n,
            )
        )

    else:
        cond_variance = list(
            map(
                get_conditional_variance_np,
                tqdm(np.arange(neurons_n)),
                [R] * neurons_n,
                [C] * neurons_n,
            )
        )

    return np.array(cond_variance).T


def perdim_spearman_corr(a, b):
    return spearmanr(a, b)[0]


def spearman_corr(aa, bb, axis=0):
    if axis == 0:
        return np.array(list(map(perdim_spearman_corr, tqdm(aa.T), bb.T)))
    else:
        return np.array(list(map(perdim_spearman_corr, aa, bb)))