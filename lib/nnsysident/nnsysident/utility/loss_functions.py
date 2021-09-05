import torch
from torch import nn
import warnings
import numpy as np


class AnscombeLoss(nn.Module):
    def __init__(self, avg=False, per_neuron=False):
        super().__init__()
        self.per_neuron = per_neuron
        self.avg = avg
        if self.avg:
            warnings.warn(
                "Anscombeloss is averaged per batch. It's recommended so use sum instead"
            )

    def forward(self, predictions, targets):
        agg_fn = torch.mean if self.avg else torch.sum

        # transform the responses and inferred firing rate
        target = self.anscombe(targets).detach()
        mu = self.anscombe(predictions) - 1 / (4 * predictions.sqrt())
        sigma = torch.eye(mu.shape[1]).to(mu.device)

        # MultivariateNormal(mu).log_prob(target) #, covariance_matrix=sigma
        # compute the loss
        SSE = torch.sum((target - mu) ** 2, dim=1)
        loss = -(
            -.5*SSE - (target.shape[1]/2) * np.log(2*np.pi)
            + target.shape[1] * np.log(2)
            - target.log().sum(dim=1)
        )

        return agg_fn(loss.view(-1, loss.shape[-1]), dim=0) if self.per_neuron else agg_fn(loss)

    @staticmethod
    def anscombe(x):
        return 2 * torch.sqrt(x + 3 / 8)
