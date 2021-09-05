import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Identity(nn.Module):
    def __init__(self, n_dimensions=1):
        super().__init__()

    def inv(self, x):
        return x

    def forward(self, y):
        return y

    def log_abs_det_jacobian(self, y):
        return torch.zeros_like(y)


class Anscombe(nn.Module):
    def __init__(self, n_dimensions=1):
        super().__init__()

    def inv(self, x):
        """
        From Normal to Poisson
        """
        return (x / 2) ** 2 - 3 / 8

    @staticmethod
    def anscombe(x):
        return 2 * torch.sqrt(x + 3 / 8)

    def forward(self, y):
        """
        From Poisson to Normal
        """
        return self.anscombe(y)

    def log_abs_det_jacobian(self, y):
        return np.log(2) - torch.log(self.anscombe(y))


class SQRT(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x):
        return x ** 2

    def forward(self, y):
        return torch.sqrt(y)

    def log_abs_det_jacobian(self, y):
        return -np.log(2) - 0.5 * torch.log(y + self.eps)


class Affine(nn.Module):
    def __init__(
        self,
        n_dimensions=1,
        only_positive_shift=False,
        learn_t=True,
        learn_a=True,
        init_t=0.0,
        init_a=1.0,
        eps=1e-12,
    ):
        """
        Affine transformation layer with positively-constrained scale.
        """
        super().__init__()
        self._a = nn.Parameter(
            torch.zeros(1, n_dimensions) + init_a, requires_grad=learn_a
        )
        self._t = nn.Parameter(
            torch.zeros(1, n_dimensions) + init_t, requires_grad=learn_t
        )
        self.only_positive_shift = only_positive_shift
        self.eps = eps

    @property
    def a(self):
        self._a.data.clamp_(min=0.0)
        return self._a

    @property
    def t(self):
        if self.only_positive_shift:
            self._t.data.clamp_(min=0.0)
        return self._t

    def inv(self, x):
        return (x - self.t) / self.a

    def forward(self, y):
        return self.a * y + self.t

    def log_abs_det_jacobian(self, y):
        return torch.log(torch.abs(self.a) + self.eps)


class Log(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x):
        return torch.exp(x)

    def forward(self, y):
        return torch.log(y + self.eps)

    def log_abs_det_jacobian(self, y):
        return -torch.log(y + self.eps)


class Exp(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x):
        return torch.log(x + self.eps)

    def forward(self, y):
        return torch.exp(y)

    def log_abs_det_jacobian(self, y):
        return y


class ELU(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()

        # modify this in case you want this to be learnable
        self.register_buffer("alpha", torch.ones(1, n_dimensions) * 1.0)

        self.eps = eps

    def inv_elu(self, x):

        return torch.clamp_min(x, 0) + torch.clamp_max(
            torch.log(x / self.alpha + 1.0 + self.eps), 0
        )

    def inv(self, x):
        return self.inv_elu(x)

    def forward(self, y):
        return F.elu(y)

    def log_abs_det_jacobian(self, y):
        device = y.device
        return (y < 0.0).detach() * (torch.log(torch.abs(self.alpha)) + y)


class InvELU(nn.Module):
    def __init__(self, n_dimensions=1, alpha=1.0, offset=1.0, eps=1e-12):
        """
        Inverse ELU tranform. However, the offset is about the forward.
        """
        super().__init__()
        self.register_buffer("alpha", torch.ones(1, n_dimensions) * alpha)
        self.offset = offset
        self.eps = eps

    def inv(self, x):
        return torch.where(
            (x <= 0.0).detach(),
            self.alpha * (torch.exp(x) - 1.0) + self.offset,
            x + self.offset,
        )

    def forward(self, y):
        return torch.where(
            (y <= self.offset).detach(),
            torch.log((y - self.offset) / self.alpha + 1 + self.eps),
            y - self.offset,
        )

    def log_abs_det_jacobian(self, y):
        det_jacobian = 1 / (y - self.offset + self.alpha + self.eps)
        return (
            torch.log(torch.abs(det_jacobian) + self.eps) * (y <= self.offset).detach()
        )


class ELUF(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.register_buffer("alpha", torch.ones(1, n_dimensions) * 1.0)
        self.eps = eps

    def inv(self, y):
        device = y.device
        return torch.minimum(torch.zeros(1).to(device), -1.0 * y) + torch.maximum(
            torch.zeros(1).to(device), -1.0 * torch.log(y / self.alpha + 1.0 + self.eps)
        )

    def forward(self, x):
        device = x.device
        return torch.minimum(
            torch.zeros(1).to(device), self.alpha * (torch.exp(-1.0 * x) - 1.0)
        ) + torch.maximum(torch.zeros(1).to(device), -1.0 * x)

    def log_abs_det_jacobian(self, y):
        device = y.device
        return torch.minimum(torch.zeros(1).to(device), -y)


class Tanh(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.eps = 1e-12

    def inv(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x) + self.eps)

    def forward(self, y):
        return torch.tanh(y)

    def log_abs_det_jacobian(self, y):
        return torch.log((1 - torch.tanh(y) ** 2).abs() + self.eps)


class Softplus(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x):
        return torch.log(torch.exp(x) - 1 + self.eps)

    def forward(self, y):
        return torch.log(1 + torch.exp(y))

    def log_abs_det_jacobian(self, y):
        return -torch.log(torch.exp(-y) + 1)


class InvSoftplus(nn.Module):
    def __init__(self, n_dimensions=1, eps=1e-12):
        super().__init__()
        self.eps = eps

    def inv(self, x):
        return torch.log(1 + torch.exp(x))

    def forward(self, y):
        return torch.log(torch.exp(y) - 1 + self.eps)

    def log_abs_det_jacobian(self, y):
        return -torch.log(1 - torch.exp(-y) + self.eps)


class Flow(nn.Module):
    def __init__(self, transforms):
        """
        Direction is from responses (original space) into latent space (Gaussian space)/
        """
        super().__init__()
        self.layers = nn.ModuleList(transforms)

    def inv(self, z):
        x = z
        for l in reversed(self.layers):
            x = l.inv(x)

        return x

    def forward(self, value, mask=None):

        logdet = 0
        y = value
        for li, l in enumerate(self.layers):

            x = l(y)

            # logdet is expected to be dx/dy
            logdet = logdet + l.log_abs_det_jacobian(y)

            # This avoids making gradients nan
            if mask is not None:  # TODO: replace this with a backward hook
                x = torch.where(~mask, torch.zeros_like(x), x)
                logdet = torch.where(~mask, torch.zeros_like(logdet), logdet)

            y = x

        return x, logdet


class ELU1_NL(nn.Module):
    def __init__(self, flipped=False):
        super().__init__()
        self.flipped = flipped

    def forward(self, x):
        return F.elu(x * -1) + 1 if self.flipped else F.elu(x) + 1


class ELU_NL(nn.Module):
    def __init__(self, flipped=False):
        super().__init__()
        self.flipped = flipped

    def forward(self, x):
        return F.elu(x * -1) if self.flipped else F.elu(x)


class Tanh_NL(nn.Module):
    def __init__(self, flipped=False):
        super().__init__()
        self.flipped = flipped

    def forward(self, x):
        return torch.tanh(x * -1) if self.flipped else torch.tanh(x)


class MeanTransfom(nn.Module):
    def __init__(self, hidden_layers=4, hidden_features=10, final_nonlinearity=False):
        super().__init__()

        in_dim = 1
        out_dim = 1

        layers = [
            nn.Linear(
                in_dim,
                hidden_features if hidden_layers > 0 else out_dim,
            )
        ]

        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ELU(),  # (bool(i % 2)),
                    nn.Linear(
                        hidden_features,
                        hidden_features if i < hidden_layers - 1 else out_dim,
                    ),
                ]
            )

        if final_nonlinearity:
            layers.append(ELU1_NL())

        self.f = nn.Sequential(*layers)

    def forward(self, x):
        b, n = x.shape
        return self.f(x.reshape(-1, 1)).reshape(b, n)


## The implementaion used when having loga in the affine transform
# class TransformLayer(nn.Module):
#     def __init__(self, a=1.0, b=0.0, c=0.0, numpy=False):
#         super().__init__()
#         self.a = a
#         self.b = b
#         self.c = c
#         self.numpy = numpy
#         self.flow = Flow([Affine(), Log(), Affine(), Exp()])
#         self.flow.layers[0]._t.data = (
#             torch.tensor([c]).to(self.flow.layers[0]._t.device).to(torch.float32)
#         )
#         self.flow.layers[2].loga.data = (
#             torch.log(torch.tensor([a]))
#             .to(self.flow.layers[2].loga.device)
#             .to(torch.float32)
#         )
#         self.flow.layers[2]._t.data = (
#             torch.tensor([b]).to(self.flow.layers[2]._t.device).to(torch.float32)
#         )

#         self.requires_grad_(False)

#     def forward(self, y):
#         if self.numpy:
#             return np.exp(self.a * np.log(y + self.c) + self.b)
#         else:
#             return self.flow(y)

#     def inv(self, x):
#         if self.numpy:
#             return ((x ** 2) ** (1 / (2 * self.a))) / np.exp(self.b / self.a) - self.c
#         else:
#             return self.flow.inv(x)


class TransformLayer(nn.Module):
    def __init__(self, a=1.0, b=0.0, c=0.0, numpy=False):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.numpy = numpy
        self.flow = Flow([Affine(), Log(), Affine(), Exp()])
        self.flow.layers[0]._t.data = (
            torch.tensor([c]).to(self.flow.layers[0]._t.device).to(torch.float32)
        )
        self.flow.layers[2]._a.data = (
            torch.tensor([a]).to(self.flow.layers[2]._a.device).to(torch.float32)
        )
        self.flow.layers[2]._t.data = (
            torch.tensor([b]).to(self.flow.layers[2]._t.device).to(torch.float32)
        )

        self.requires_grad_(False)

    def forward(self, y):
        if self.numpy:
            return np.exp(self.a * np.log(y + self.c) + self.b)
        else:
            return self.flow(y)

    def inv(self, x):
        if self.numpy:
            return ((x ** 2) ** (1 / (2 * self.a))) / np.exp(self.b / self.a) - self.c
        else:
            return self.flow.inv(x)


identity = lambda numpy=True: TransformLayer(a=1.0, b=0.0, numpy=numpy)
sqrt = lambda numpy=True: TransformLayer(a=0.5, b=0.0, numpy=numpy)
anscombe = lambda numpy=True: TransformLayer(
    a=0.5, b=np.log(2.0), c=(3 / 8), numpy=numpy
)
example1 = lambda numpy=True: TransformLayer(a=1.78, b=-2.95, numpy=numpy)
example2 = lambda numpy=True: TransformLayer(a=0.25, b=0.59, numpy=numpy)
example3 = lambda numpy=True: TransformLayer(a=0.125, b=0.88, numpy=numpy)
example4 = lambda numpy=True: TransformLayer(a=1.78, b=-2.55, numpy=numpy)
example5 = lambda numpy=True: TransformLayer(a=0.46, b=0.49, numpy=numpy)
example6 = lambda numpy=True: TransformLayer(a=0.25, b=0.98, numpy=numpy)
example7 = lambda numpy=True: TransformLayer(a=0.125, b=1.27, numpy=numpy)
example8 = lambda numpy=True: TransformLayer(a=1.78, b=-2.24, numpy=numpy)
example9 = lambda numpy=True: TransformLayer(a=0.25, b=1.29, numpy=numpy)
example10 = lambda numpy=True: TransformLayer(a=0.125, b=1.58, numpy=numpy)
