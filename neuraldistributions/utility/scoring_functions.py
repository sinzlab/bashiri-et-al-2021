import numpy as np
import torch
from neuralpredictors.measures import corr

from .model_evaluation import (
    spearman_corr,
    get_conditional_means,
    get_conditional_variances,
)
from ..models.controls import Poisson, ZIG
from ..models.flowfa import FlowFA
from ..models.ziffa import ZeroInflatedFlowFA
from .flowfa_scoring_functions import single_trial_conditional_correlation_flowfa
from .ziffa_scoring_functions import single_trial_conditional_correlation_ziffa


def get_loglikelihood(
    dataloaders, model, per_unit=True, in_bits=False, has_datakey=True, **kwargs
):
    model.eval()

    if has_datakey:

        log_likelihoods, data_keys = [], []
        for data_key, dataloader in dataloaders.items():
            log_likelihood = np.concatenate(
                [
                    model.log_likelihood(*b, data_key=data_key, in_bits=in_bits)
                    .cpu()
                    .data.numpy()
                    for b in dataloader
                ]
            ).sum()

            n_neurons = len(dataloader.dataset.neurons.ids)
            log_likelihoods.append([log_likelihood] * n_neurons)
            data_keys.extend([data_key] * n_neurons)

        return data_keys, np.hstack(log_likelihoods)

    else:

        log_likelihood = np.concatenate(
            [
                model.log_likelihood(*b, in_bits=in_bits).cpu().data.numpy()
                for b in dataloaders
            ]
        ).sum()

        n_neurons = len(dataloaders.dataset.neurons.ids)
        log_likelihoods = np.array([log_likelihood] * n_neurons)
        data_keys = ["none"] * n_neurons

        return data_keys, log_likelihoods


class Correlation:
    def __init__(self, mode="single_trial"):
        self.mode = mode

    @staticmethod
    def single_trial(dataloaders, model, use_torch=True, corr_type="pearson", **kwargs):

        model.eval()

        resps, preds, data_keys = [], [], []
        for data_key, loader in dataloaders.items():

            transformed_responses, predicted_mean = [], []
            for b in loader:
                transformed_responses.append(
                    model.sample_transform(b[1])[0].cpu().data.numpy()
                )

                predicted_mean.append(
                    model.predict_mean(*b, data_key=data_key).cpu().data.numpy()
                )

            transformed_responses = np.concatenate(transformed_responses)
            predicted_mean = np.concatenate(predicted_mean)

            n_neurons = predicted_mean.shape[1]
            resps.append(transformed_responses)
            preds.append(predicted_mean)
            data_keys.extend([data_key] * n_neurons)

        if corr_type == "pearson":
            correlations = corr(np.hstack(preds), np.hstack(resps), axis=0)
        elif corr_type == "spearman":
            correlations = spearman_corr(np.hstack(preds), np.hstack(resps), axis=0)
        else:
            raise ValueError("the specific corr_type is not available.")

        return data_keys, correlations

    @staticmethod
    def single_trial_conditional_flowfa(
        dataloaders,
        model,
        corr_type="pearson",
        n_samples=300000,
        batch_size=1000,
        use_torch=True,
        **kwargs,
    ):
        assert isinstance(model, FlowFA)
        return single_trial_conditional_correlation_flowfa(
            dataloaders,
            model,
            corr_type=corr_type,
            n_samples=n_samples,
            batch_size=batch_size,
            use_torch=use_torch,
        )

    @staticmethod
    def single_trial_conditional_ziffa(
        dataloaders,
        model,
        corr_type="pearson",
        n_samples=300000,
        batch_size=1000,
        **kwargs,
    ):
        assert isinstance(model, ZeroInflatedFlowFA)
        return single_trial_conditional_correlation_ziffa(
            dataloaders,
            model,
            corr_type=corr_type,
            n_samples=n_samples,
            batch_size=batch_size,
        )

    @staticmethod
    def conditional_single_trial(
        dataloaders, model, per_unit=True, use_torch=True, corr_type="pearson", **kwargs
    ):

        model.eval()

        resps, cond_preds, data_keys = [], [], []
        for data_key, loader in dataloaders.items():

            C, psi_diag = model.C_and_psi_diag

            R = psi_diag.diag().cpu().data.numpy()
            C = C.cpu().data.numpy()

            transformed_responses, predicted_mean = [], []
            for b in loader:
                transformed_responses.append(
                    model.sample_transform(b[1])[0].cpu().data.numpy()
                )
                predicted_mean.append(
                    model.predict_mean(*b, data_key=data_key).cpu().data.numpy()
                )

            transformed_responses = np.concatenate(transformed_responses)
            predicted_mean = np.concatenate(predicted_mean)

            conditional_predicted_means = get_conditional_means(
                R, C, predicted_mean, transformed_responses, use_torch=use_torch
            )

            n_neurons = predicted_mean.shape[1]
            resps.append(transformed_responses)
            cond_preds.append(conditional_predicted_means)
            data_keys.extend([data_key] * n_neurons)

        if corr_type == "pearson":
            correlations = corr(np.hstack(cond_preds), np.hstack(resps), axis=0)
        elif corr_type == "spearman":
            correlations = spearman_corr(
                np.hstack(cond_preds), np.hstack(resps), axis=0
            )
        else:
            raise ValueError("the specific corr_type is not available.")

        return data_keys, correlations

    def __call__(self, dataloaders, model, per_unit=True, **kwargs):

        if self.mode == "single_trial":
            if isinstance(model, ZIG) or isinstance(model, Poisson):
                return self.single_trial(dataloaders, model, **kwargs)
            elif isinstance(model, FlowFA):
                return self.single_trial_conditional_flowfa(
                    dataloaders, model, **kwargs
                )
            elif isinstance(model, ZeroInflatedFlowFA):
                return self.single_trial_conditional_ziffa(dataloaders, model, **kwargs)
            else:
                raise ValueError(
                    "Function to compute correlation for this model does not exist."
                )

        else:
            raise NotImplementedError(
                f"{self.mode} is not implemented, yet. Wanna make a PR? ;)"
            )