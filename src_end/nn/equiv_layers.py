from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_add, scatter, scatter_softmax, scatter_mean

from src_end.nn.layers import DenseLayer


class EquivLayerNorm(nn.Module):
    def __init__(
            self,
            dims: Tuple[int, Optional[int]],
            eps: float = 1e-6,
            affine: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight_s = nn.Parameter(torch.Tensor(self.sdim))
            self.bias_s = nn.Parameter(torch.Tensor(self.sdim))
            # self.weight_v = nn.Parameter(torch.Tensor(self.vdim))
        else:
            self.register_parameter("weight_s", None)
            self.register_parameter("bias_s", None)
            # self.register_parameter("weight_v", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight_s.data.fill_(1.0)
            self.bias_s.data.fill_(0.0)
            # self.weight_v.data.fill_(1.0)

    def forward(self, x: dict, batch: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s, v = x.get("s"), x.get("v")
        batch_size = int(batch.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)

        s = s - smean[batch]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)  # .sqrt()
        sout = s / var[batch]

        if self.affine and self.weight_s is not None and self.bias_s is not None:
            sout = sout * self.weight_s + self.bias_s

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)  # .sqrt()
            vout = v / vmean[batch]
            # if self.affine and self.weight_v is not None:
        #     vout = vout * self.weight_v

        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims}, " f"affine={self.affine})"


class GatedEquivBlock(nn.Module):
    def __init__(
            self,
            in_dims: Tuple[int, int],
            out_dims: Tuple[int, Optional[int]],
            hs_dim: Optional[int] = None,
            hv_dim: Optional[int] = None,
            norm_eps: float = 1e-6,
            use_mlp: bool = False,
    ):
        super(GatedEquivBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.Wv0 = DenseLayer(self.vi, self.hv_dim + self.vo, bias=False)

        if not use_mlp:
            self.Ws = DenseLayer(self.hv_dim + self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(
                    self.hv_dim + self.si, self.si, bias=True, activation=nn.SiLU()
                ),
                DenseLayer(self.si, self.vo + self.so, bias=True),
            )
            if self.vo > 0:
                self.Wv1 = DenseLayer(self.vo, self.vo, bias=False)
            else:
                self.Wv1 = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        reset(self.Wv0)
        if self.use_mlp:
            if self.vo > 0:
                reset(self.Wv1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        s, v = x
        vv = self.Wv0(v)

        if self.vo > 0:
            vdot, v = vv.split([self.hv_dim, self.vo], dim=-1)
        else:
            vdot = vv

        vdot = torch.clamp(torch.pow(vdot, 2).sum(dim=1), min=self.norm_eps)  # .sqrt()

        s = torch.cat([s, vdot], dim=-1)
        s = self.Ws(s)
        if self.vo > 0:
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv1(v)

        return s, v


class EQGATLayer(MessagePassing):
    def __init__(
            self,
            in_dims: Tuple[int, Optional[int]],
            out_dims: Tuple[int, Optional[int]],
            edge_dim: int,
            d_dim: int = 1,
            eps: float = 1e-6,
            has_v_in: bool = False,
            use_mlp_update: bool = True,
            vector_aggr: str = "mean",
    ):
        super(EQGATLayer, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        assert edge_dim is not None

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in

        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.edge_net = nn.Sequential(
            DenseLayer(
                2 * self.si + edge_dim + d_dim + 1 + 2, self.si, bias=True,
                activation=nn.SiLU()
            ),
            DenseLayer(self.si, self.v_mul * self.vi + self.si, bias=True),
        )

        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(
            in_dims=(self.si, self.vi),
            hs_dim=self.si,
            hv_dim=self.vi,
            out_dims=(self.so, self.vo),
            norm_eps=eps,
            use_mlp=use_mlp_update,
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
            self,
            x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            edge_index: torch.Tensor,
            edge_attr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
    ):
        s, v, p = x
        d, a, r, e = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            p=p,
            edge_attr=(d, a, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        out = {"s": s, "v": v, "p": p, "e": e}
        return out

    def aggregate(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            dim_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ms_j, mv_j = inputs

        s = scatter_add(ms_j, index=index, dim=0, dim_size=dim_size)
        v = scatter(
            mv_j, index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size
        )

        return s, v

    def message(
            self,
            sa_i: torch.Tensor,
            sa_j: torch.Tensor,
            sb_j: torch.Tensor,
            va_i: torch.Tensor,
            va_j: torch.Tensor,
            vb_j: torch.Tensor,
            p_i: torch.Tensor,
            p_j: torch.Tensor,
            index: torch.Tensor,
            edge_attr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        d, a, r, e = edge_attr

        d_i, d_j = (
            torch.pow(p_i, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
            torch.pow(p_j, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
        )

        aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), d, a, e, d_i, d_j], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul * self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1 = vij0.chunk(2, dim=-1)
            nv0_j = r.unsqueeze(-1) * vij0
            nv_j = nv0_j + vij1 * vb_j
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            nv0_j = r.unsqueeze(-1) * vij0
            nv_j = nv0_j

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j

        return ns_j, nv_j


class EQGATBackbone(nn.Module):
    def __init__(
            self,
            hn_dim: tuple[int, int],
            edge_dim: int,
            d_dim: int = 1,
            num_layers: int = 5,
            vector_aggr: str = "mean",
            use_context: bool = False,
    ):
        super(EQGATBackbone, self).__init__()

        self.num_layers = num_layers

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim
        self.use_context = use_context

        layers = []

        for i in range(num_layers):
            layers.append(
                EQGATLayer(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    d_dim=d_dim,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                )
            )
        self.layers = nn.ModuleList(layers)

        if use_context:
            context_layers = [nn.Sequential(DenseLayer(self.sdim, self.sdim, activation=nn.SiLU()),
                                            DenseLayer(self.sdim, self.sdim, bias=False)) for _ in range(num_layers)]
            self.context_layers = nn.ModuleList(context_layers)

        self.norms = nn.ModuleList([EquivLayerNorm(dims=hn_dim) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(
            self,
            s: torch.Tensor,
            v: torch.Tensor,
            p: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor = None,
            context: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        # edge_attr (d, a, pairwise_vector, e)
        d, a, r, e = edge_attr
        for i in range(len(self.layers)):
            edge_index_in = edge_index
            edge_attr_in = (d, a, r, e)

            if self.use_context:
                assert isinstance(context, torch.Tensor)
                s = s + self.context_layers[i](context)

            s, v = self.norms[i].forward(x={"s": s, "v": v}, batch=index)
            out = self.layers[i].forward(
                x=(s, v, p),
                index=index,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
            )
            s, v, p = out["s"], out["v"], out["p"]

        out = {"s": s, "v": v, "p": p}

        return out
