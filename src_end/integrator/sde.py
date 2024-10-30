from typing import Callable, Tuple

import torch
from tqdm import tqdm


@torch.no_grad()
def euler_maruyama(
        f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        w: Callable[[], torch.Tensor],
        zs: torch.Tensor,
        ts: float,
        tf: float,
        n_steps: int = 1000,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    device = zs.device

    ts = torch.linspace(ts, tf, n_steps + 1, device=device)

    ts = ts[:, None, None]

    zt = zs
    for i in tqdm(range(n_steps)):
        t = ts[i]
        dt = ts[i + 1] - t

        t = t.expand(len(zt), 1)  # one time step per node

        wt = w()

        drift, diff = f(zt, t, wt)

        zt = zt + drift * dt + diff * abs(dt) ** 0.5

    return zt
