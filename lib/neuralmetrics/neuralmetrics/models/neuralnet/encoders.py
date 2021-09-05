import torch.nn as nn
import torch
import numpy as np
from warnings import warn

from ...utils import fitted_zig_mean, fitted_zig_variance

class ZIGEncoder(nn.Module):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        init_ks=None,
        theta_image_dependent=True,
        k_image_dependent=False,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
    ):

        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = offset
        self.zero_thresholds = zero_thresholds

        if not theta_image_dependent:
            self.logtheta = nn.ParameterDict(
                {
                    data_key: nn.Parameter(torch.zeros(1, ro.outdims))
                    for data_key, ro in self.readout.items()
                }
            )

        if not k_image_dependent:
            if isinstance(init_ks, dict):
                self.logk = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            torch.ones(1, ro.outdims) * init_ks[data_key]
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            elif init_ks is None:
                self.logk = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(torch.ones(1, ro.outdims) * 0.0)
                        for data_key, ro in self.readout.items()
                    }
                )
            else:
                raise ValueError(
                    "init_ks should either be of type {data_key: init_k_value} or None."
                )

        else:
            if init_ks is not None:
                warn(
                    "init_ks are set but will be ignored because k_image_dependent is True"
                )

        if not loc_image_dependent:
            if isinstance(zero_thresholds, dict):
                self.logloc = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            torch.ones(1, ro.outdims)
                            * np.log(zero_thresholds[data_key]),
                            requires_grad=False,
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            elif zero_thresholds is None:
                self.logloc = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            (torch.rand(1, ro.outdims) + 1) * np.log(0.1),
                            requires_grad=True,
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            else:
                raise ValueError(
                    "zero_thresholds should either be of type {data_key: zero_shreshold_value} or None."
                )

        else:
            if zero_thresholds is not None:
                warn(
                    "zero thresholds are set but will be ignored because loc_image_dependent is True"
                )

        if not q_image_dependent:
            self.q = nn.ParameterDict(
                {
                    data_key: nn.Parameter(torch.rand(1, ro.outdims) * 2 - 1)
                    for data_key, ro in self.readout.items()
                }
            )

    def theta_nl(self, logtheta):
        return torch.exp(logtheta) + self.offset

    def k_nl(self, logk):
        if self.zero_thresholds is not None:
            return torch.exp(logk) + self.offset
        else:
            return torch.exp(logk) + 1.1 + self.offset

    def loc_nl(self, logloc):
        loc = torch.exp(logloc)
        assert not torch.any(
            loc == 0.0
        ), "loc should not be zero! Because of numerical instability. Check the code!"
        return loc

    def q_nl(self, q):
        return torch.sigmoid(q) * 0.99999 + self.offset

    def forward(self, x, data_key=None, detach_core=False, **kwargs):
        if data_key is None:
            warn("data_key is not specified and set to None...")
        batch_size = x.shape[0]

        # get readout outputs
        x = self.core(x)
        if detach_core:
            x = x.detach()
        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"])
        else:
            x = self.readout(x, data_key=data_key)

        readout_out_idx = 0
        if "logtheta" in dir(self):
            logtheta = getattr(self, "logtheta")[data_key].repeat(batch_size, 1)
        else:
            logtheta = x[readout_out_idx]
            readout_out_idx += 1

        if "logk" in dir(self):
            logk = getattr(self, "logk")[data_key].repeat(batch_size, 1)
        else:
            logk = x[readout_out_idx]
            readout_out_idx += 1

        if "logloc" in dir(self):
            logloc = getattr(self, "logloc")[data_key].repeat(batch_size, 1)
        else:
            logloc = x[readout_out_idx]
            readout_out_idx += 1

        if "q" in dir(self):
            q = getattr(self, "q")[data_key].repeat(batch_size, 1)
        else:
            q = x[readout_out_idx]
            readout_out_idx += 1

        return (
            self.theta_nl(logtheta),
            self.k_nl(logk),
            self.loc_nl(logloc),
            self.q_nl(q),
        )

    def regularizer(self, data_key):
        return 0.

    def predict_mean(self, x, data_key=None):
        theta, k, loc, q = self.forward(x, data_key=data_key)
        return fitted_zig_mean(theta, k, loc, q)

    def predict_variance(self, x, data_key=None):
        theta, k, loc, q = self.forward(x, data_key=data_key)
        return fitted_zig_variance(theta, k, loc, q)