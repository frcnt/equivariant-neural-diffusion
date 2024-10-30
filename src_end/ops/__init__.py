from typing import Tuple, Callable

import torch
from torch_scatter import scatter_mean


def scatter_center(pos: torch.Tensor,
                   index: torch.Tensor = None):
    return pos - scatter_mean(pos, index=index, dim=0)[index]


def center(pos: torch.Tensor):
    return pos - torch.mean(pos, dim=0)


def is_centered(pos: torch.Tensor,
                index: torch.Tensor,
                tol: float = 1e-3,
                debug: bool = True):
    com = scatter_mean(pos, index=index, dim=0)
    if debug:
        print("Debug is_centered:", torch.amax(torch.abs(com)))
    return torch.all(com < tol)


def jvp(
        f: Callable[[torch.Tensor], torch.Tensor | Tuple[torch.Tensor]],
        x: torch.Tensor,
        v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Jacobian-Vector-Product of function 'f' evaluated in 'x' with vector 'v'.
    """
    create_graph = torch.is_grad_enabled()
    out, prod = torch.autograd.functional.jvp(f, x, v, create_graph=create_graph)
    return out, prod


def time_derivative(
        f: Callable[[torch.Tensor], torch.Tensor | Tuple[torch.Tensor]],
        t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute time-derivative of function 'f' evaluated in 't'.
    NOTE: 'f' should be a function of time solely.
    """
    assert t.ndim == 2 and t.shape[1] == 1, f"t.shape={t.shape}"
    v = torch.ones_like(t)
    out, df_dt = jvp(f, t, v)
    return out, df_dt
