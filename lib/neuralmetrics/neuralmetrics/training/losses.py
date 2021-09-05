import torch.nn as nn
import torch
from torch.distributions import Normal


class ZIGLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, targets, theta, k, loc, q):

        neurons_n = targets.shape[1]

        if loc.requires_grad:
            self.multi_clamp(loc, [0.0] * neurons_n, targets.max(dim=0)[0])

        zero_mask = targets <= loc
        nonzero_mask = targets > loc

        # spike loss

        # TODO: Discuss "- torch.log(loc)"
        spike_loss = torch.log(1 - q) - torch.log(loc)
        spike_loss = spike_loss * zero_mask

        # make sure loc is always smaller than the smallest non-zero response
        if loc.requires_grad:
            self.multi_clamp(
                loc,
                [0.0] * neurons_n,
                self.find_nonzero_min(targets * nonzero_mask) * 0.999,
            )

        # slab loss
        slab_loss = (
            torch.log(q)
            + (k - 1)
            * torch.log(targets * nonzero_mask - loc * nonzero_mask + 1.0 * zero_mask)
            - (targets * nonzero_mask - loc * nonzero_mask) / theta
            - k * torch.log(theta)
            - torch.lgamma(k)
        )
        slab_loss = slab_loss * nonzero_mask

        # total loss
        loss = spike_loss + slab_loss

        return -loss

    @staticmethod
    def multi_clamp(tensor, mins, maxs):
        for tensor_row in tensor:
            for tensorval, minval, maxval in zip(tensor_row, mins, maxs):
                tensorval.data.clamp_(minval, maxval)

    @staticmethod
    def find_nonzero_min(tensor, dim=0):
        tensor[tensor == 0.0] += tensor.max()
        return tensor.min(dim)[0]


class ZIGLoss_pn(nn.Module):
    def __init__(self, avg=False):
        super().__init__()
        self.avg = avg

    def forward(self, targets, theta, k, loc, q):
        agg_gn = torch.mean if self.avg else torch.sum

        # make sure loc is always smaller than the smallest non-zero response
        loc.data.clamp_(0, targets.max())

        # split zero and non-zero targets
        zero_targets = targets[targets < loc]
        nonzero_targets = targets[targets >= loc]

        # spike loss
        spike_loss = torch.log(1 - q)
        spike_loss = spike_loss * zero_targets.shape[0]
        spike_loss = agg_gn(spike_loss, dim=0)

        # make sure loc is always smaller than the smallest non-zero response
        loc.data.clamp_(0, nonzero_targets.min() - 1e-6)

        # slab loss
        slab_loss = (
            torch.log(q)
            + (k - 1) * torch.log(nonzero_targets - loc)
            - (nonzero_targets - loc) / theta
            - k * torch.log(theta)
            - torch.lgamma(k)
        )
        slab_loss = agg_gn(slab_loss, dim=0)

        # total loss
        loss = spike_loss + slab_loss

        return -loss