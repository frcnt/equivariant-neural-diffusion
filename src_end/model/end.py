from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_sum, scatter_mean

from src_end.integrator.sde import euler_maruyama
from src_end.model.distributions import GaussianDistribution
from src_end.model.parameterization import EquivariantParameterization
from src_end.model.volatility import Volatility
from src_end.ops import time_derivative, scatter_center


class END(nn.Module):
    def __init__(self,
                 f: EquivariantParameterization,
                 r: EquivariantParameterization,
                 g: Volatility,
                 p_eps: GaussianDistribution,
                 p_z1: GaussianDistribution,
                 p_w: GaussianDistribution,
                 parameterization: Literal["x0", "direct"] = "x0",
                 zero_cog_score: bool = True,
                 which_loss: Literal["simple", "scatter"] = "simple"
                 ):
        super().__init__()

        self._f_transformation = f
        self.r = r
        self.g = g

        self.p_eps = p_eps
        self.p_z1 = p_z1
        self.p_w = p_w

        self.parameterization = parameterization
        self.zero_cog_score = zero_cog_score
        self.which_loss = which_loss

    def forward_transformation(self, t: torch.Tensor, *args, return_dt: bool = False):
        def f(t_: torch.Tensor):
            return self._f_transformation(t_, *args)

        if return_dt:
            return time_derivative(f, t)
        else:
            return self._f_transformation(t, *args)

    def loss_diffusion(self, t: torch.Tensor, batch: Batch | Data) -> dict[str, torch.Tensor]:
        h, pos, edge_index, index = batch.h, batch.pos, batch.edge_index, batch.batch
        context = getattr(batch, "context", None)

        g_h, g_pos = self.g(t)
        g_h, g_pos = g_h[index], g_pos[index]

        (z_h, z_pos), (target_drift_h, target_drift_pos) = self.target_backward_drift(t, h, pos, g_h, g_pos, edge_index,
                                                                                      index, context=context,
                                                                                      zero_cog_score=self.zero_cog_score)
        (approx_drift_h, approx_drift_pos) = self.approximate_backward_drift(t, z_h, z_pos, g_h, g_pos,
                                                                             edge_index,
                                                                             index, context=context,
                                                                             zero_cog_score=self.zero_cog_score)
        assert approx_drift_h.shape == target_drift_h.shape
        assert approx_drift_pos.shape == target_drift_pos.shape

        if self.which_loss == "scatter":
            loss_h = torch.mean(scatter_mean(((target_drift_h - approx_drift_h) / g_h) ** 2, dim=0, index=index))
            loss_pos = torch.mean(
                scatter_mean(((target_drift_pos - approx_drift_pos) / g_pos) ** 2, dim=0, index=index))
        else:
            loss_h = F.mse_loss(approx_drift_h / g_h, target_drift_h / g_h)
            loss_pos = F.mse_loss(approx_drift_pos / g_pos, target_drift_pos / g_pos)

        return {"h": loss_h,
                "pos": loss_pos}

    def target_backward_drift(self,
                              t: torch.Tensor,
                              h: torch.Tensor,
                              pos: torch.Tensor,
                              g_h: torch.Tensor,
                              g_pos: torch.Tensor,
                              edge_index: torch.Tensor,
                              index: torch.Tensor,
                              context: Optional[torch.Tensor] = None,
                              zero_cog_score: bool = True
                              ):

        eps_h, eps_pos = self.p_eps.sample(index)

        out_f = self.forward_transformation(t, h, pos,
                                            edge_index, index,
                                            context,
                                            return_dt=True)

        (mu_h, sigma_h, mu_pos, U_pos), (d_mu_h_dt, d_sigma_h_dt, d_mu_pos_dt, d_U_pos_dt) = out_f

        z_h = self.z_h(mu=mu_h, sigma=sigma_h, eps=eps_h)
        d_z_h_dt = self.z_h(mu=d_mu_h_dt, sigma=d_sigma_h_dt, eps=eps_h)

        z_pos = self.z_pos(mu=mu_pos, U_unc=U_pos, eps=eps_pos, index=index)
        d_z_pos_dt = self.z_pos(mu=d_mu_pos_dt, U_unc=d_U_pos_dt, eps=eps_pos, index=index)

        score_h = self.score_h(eps=eps_h, sigma=sigma_h)
        score_pos = self.score_pos(eps=eps_pos, U_unc=U_pos, index=index,
                                   zero_cog=zero_cog_score)

        target_drift_h = self.backward_drift(d_z_h_dt, score_h, g_h)
        target_drift_pos = self.backward_drift(d_z_pos_dt, score_pos, g_pos)

        return (z_h, z_pos), (target_drift_h, target_drift_pos)

    def approximate_backward_drift(self,
                                   t: torch.Tensor,
                                   z_h: torch.Tensor,
                                   z_pos: torch.Tensor,
                                   g_h: torch.Tensor,
                                   g_pos: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   index: torch.Tensor,
                                   context: Optional[torch.Tensor] = None,
                                   zero_cog_score: bool = True, ):
        if self.parameterization == "x0":
            h, pos = self.r.forward(t, z_h, z_pos, edge_index=edge_index, index=index, context=context)
            out_f = self.forward_transformation(t, h, pos,
                                                edge_index, index,
                                                context,
                                                return_dt=True)

            (mu_h, sigma_h, mu_pos, U_pos), (d_mu_h_dt, d_sigma_h_dt, d_mu_pos_dt, d_U_pos_dt) = out_f

            eps_h = self.solve_h(z_h, mu_h, sigma_h)
            eps_pos, (inv_U_unc, C) = self.solve_pos(U_unc=U_pos, pos=(z_pos - mu_pos), index=index)

            d_z_h_dt = self.z_h(mu=d_mu_h_dt, sigma=d_sigma_h_dt, eps=eps_h)
            d_z_pos_dt = self.z_pos(mu=d_mu_pos_dt, U_unc=d_U_pos_dt, eps=eps_pos, index=index)

            score_h = self.score_h(eps=eps_h, sigma=sigma_h)
            score_pos = self.score_pos(eps=eps_pos, U_unc=U_pos, inv_U_unc=inv_U_unc, C=C, index=index,
                                       zero_cog=zero_cog_score)

        elif self.parameterization == "direct":
            (d_z_h_dt, score_h), (d_z_pos_dt, score_pos) = self.r.forward(t, z_h, z_pos, edge_index=edge_index,
                                                                          index=index, context=context)
        else:
            raise NotImplementedError

        approx_drift_h = self.backward_drift(d_z_h_dt, score_h, g_h)
        approx_drift_pos = self.backward_drift(d_z_pos_dt, score_pos, g_pos)

        return approx_drift_h, approx_drift_pos

    @staticmethod
    def diffusion(g: torch.Tensor, w: torch.Tensor):
        return g * w

    @staticmethod
    def backward_drift(dz_dt: torch.Tensor, score: torch.Tensor, g: torch.Tensor):
        return dz_dt - .5 * g ** 2 * score

    def z_h(self, mu: torch.Tensor, sigma: torch.Tensor, eps: torch.Tensor):
        return mu + sigma * eps

    def score_h(self, eps: torch.Tensor, sigma: torch.Tensor):
        return -eps / sigma

    def solve_h(self, z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        return (z - mu) / sigma

    def z_pos(self, mu: torch.Tensor, U_unc: torch.Tensor, eps: torch.Tensor, index: torch.Tensor):
        return mu + scatter_center(torch.einsum("nij, nj -> ni", U_unc, eps), index=index)

    def score_pos(self, eps: torch.Tensor, U_unc: torch.Tensor, index: torch.Tensor,
                  inv_U_unc: Optional[torch.Tensor] = None,
                  C: Optional[torch.Tensor] = None,
                  zero_cog: bool = True):

        d = eps.shape[-1]

        if inv_U_unc is None or C is None:
            inv_U_unc = _inv_3x3(U_unc)
            V = scatter_sum(inv_U_unc, dim=0, index=index)  # NOTE avoids 1/n later on by summing instead of mean
            inv_V = _inv_3x3(V)
            C = inv_V[index] @ (inv_U_unc - torch.eye(d, device=eps.device)[None, ...])

        y = torch.einsum("nji, nj -> ni", inv_U_unc, eps)  # OBS: first index for getting inv_U_unc.T
        ybar = scatter_sum(y, dim=0, index=index)
        c = torch.einsum("nji, nj -> ni", C, ybar[index])  # OBS: first index for getting  B.T

        score = c - y

        if zero_cog:
            return scatter_center(score, index=index)  # project on zero_cog

        return score

    def solve_pos(self, U_unc: torch.Tensor, pos: torch.Tensor, index: torch.Tensor):
        d = pos.shape[-1]

        inv_U_unc = _inv_3x3(U_unc)
        V = scatter_sum(inv_U_unc, dim=0, index=index)  # NOTE avoids 1/n latter on by summing instead of mean
        inv_V = _inv_3x3(V)
        C = inv_V[index] @ (inv_U_unc - torch.eye(d, device=pos.device)[None, ...])

        c = scatter_sum(torch.einsum("nij, nj -> ni", C, pos), dim=0, index=index)[index]
        inv_pos = torch.einsum("nij, nj -> ni", inv_U_unc, (pos - c))

        return inv_pos, (inv_U_unc, C)

    def split_h_pos(self, h_pos):
        return torch.split(h_pos, [self.p_w.dim_h, self.p_w.dim_pos], dim=-1)

    @torch.no_grad()
    def sample(self,
               batch: Batch | Data,
               method: Literal["sde"] = "sde",
               **kwargs):
        _allowed_methods = ("sde",)

        edge_index, index = batch.edge_index, batch.batch

        z_h, z_pos = self.p_z1.sample(index)
        z = torch.cat([z_h, z_pos], dim=1)
        if method == "sde":
            sample = self.sample_sde(z, edge_index=edge_index, index=index, **kwargs)
        else:
            raise NotImplementedError(f"'method'={method} not in {_allowed_methods}")

        h, pos = self.split_h_pos(sample)

        return h, pos

    def sample_sde(self, zs, edge_index, index, **kwargs):
        def drift_diff(z_t, t, w):
            z_h_t, z_pos_t = self.split_h_pos(z_t)
            g_h, g_pos = self.g.forward(t)

            # drift
            drift_h, drift_pos = self.approximate_backward_drift(t,
                                                                 z_h_t, z_pos_t,
                                                                 g_h, g_pos,
                                                                 edge_index, index, zero_cog_score=True)
            drift = torch.cat([drift_h, drift_pos], dim=1)

            # max_diff
            w_h, w_pos = self.split_h_pos(w)
            diff_h, diff_pos = self.diffusion(g_h, w_h), self.diffusion(g_pos, w_pos)
            diff = torch.cat([diff_h, diff_pos], dim=1)

            return drift, diff

        def wiener():
            w_h, w_pos = self.p_w.sample(index)
            return torch.cat([w_h, w_pos], dim=1)

        return euler_maruyama(f=drift_diff, w=wiener, zs=zs, ts=1., tf=0., **kwargs)


def _det_3x3(t: torch.Tensor):
    assert t.ndim >= 2 and t.shape[-2:] == (3, 3)
    return (t[..., 0, 0] * t[..., 1, 1] * t[..., 2, 2] +
            t[..., 0, 1] * t[..., 1, 2] * t[..., 2, 0] +
            t[..., 1, 0] * t[..., 2, 1] * t[..., 0, 2] -
            t[..., 2, 0] * t[..., 1, 1] * t[..., 0, 2] -
            t[..., 0, 0] * t[..., 2, 1] * t[..., 1, 2] -
            t[..., 1, 0] * t[..., 0, 1] * t[..., 2, 2])


def _inv_3x3(t: torch.Tensor):
    assert t.ndim >= 2 and t.shape[-2:] == (3, 3)

    a = t[..., 0, 0]
    b = t[..., 0, 1]
    c = t[..., 0, 2]
    d = t[..., 1, 0]
    e = t[..., 1, 1]
    f = t[..., 1, 2]
    g = t[..., 2, 0]
    h = t[..., 2, 1]
    i = t[..., 2, 2]

    inv = torch.empty_like(t)
    inv[..., 0, 0] = e * i - f * h
    inv[..., 0, 1] = c * h - b * i
    inv[..., 0, 2] = b * f - c * e
    inv[..., 1, 0] = f * g - d * i
    inv[..., 1, 1] = a * i - c * g
    inv[..., 1, 2] = c * d - a * f
    inv[..., 2, 0] = d * h - e * g
    inv[..., 2, 1] = b * g - a * h
    inv[..., 2, 2] = a * e - b * d

    det = _det_3x3(t)

    return inv / det[..., None, None]
