"""
shapefun.py  —  mirrors shapefun.m

Shape functions and their parametric derivatives for Q4 and Q9 elements,
evaluated at all Gauss points.
"""

import torch
from gaussfun import gaussfun

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


def shapefun(mesh_type: str, ngp: int):
    """
    Evaluate shape functions and parametric derivatives at all Gauss points.

    Parameters
    ----------
    mesh_type : "Q4" or "Q9"
    ngp       : number of Gauss points (4 or 9)

    Returns
    -------
    N    : [ngp, n_nodes_el]   shape function values
    dNds : [ngp, n_nodes_el]   dN/ds  (parametric derivative)
    dNdt : [ngp, n_nodes_el]   dN/dt  (parametric derivative)
    """
    gp_locs, _ = gaussfun(ngp)   # [ngp, 2]

    if mesh_type == "Q4":
        return _shapefun_Q4(gp_locs)
    elif mesh_type == "Q9":
        return _shapefun_Q9(gp_locs)
    else:
        raise ValueError(f"shapefun: unsupported mesh_type={mesh_type}.")


# ── Q4 (4-node quadrilateral) ─────────────────────────────────────────────────

def _shapefun_Q4(gp_locs: torch.Tensor):
    """
    Q4 shape functions at all Gauss points.
    Node ordering (counter-clockwise):
        1: (-1,-1)   2: (+1,-1)   3: (+1,+1)   4: (-1,+1)

    Mirrors shapefun.m Q4 branch.
    """
    ngp = gp_locs.shape[0]
    N    = torch.zeros(ngp, 4, dtype=DTYPE, device=DEVICE)
    dNds = torch.zeros(ngp, 4, dtype=DTYPE, device=DEVICE)
    dNdt = torch.zeros(ngp, 4, dtype=DTYPE, device=DEVICE)

    for i in range(ngp):
        s = gp_locs[i, 0]
        t = gp_locs[i, 1]

        N[i, :] = torch.stack([
            (-1 + s) * (-1 + t),
            -(1 + s) * (-1 + t),
             (1 + s) * (1 + t),
            -(-1 + s) * (1 + t),
        ]) / 4.0

        dNds[i, :] = torch.stack([
            (-1 + t), (1 - t), (1 + t), (-1 - t)
        ]) / 4.0

        dNdt[i, :] = torch.stack([
            (-1 + s), (-1 - s), (1 + s), (1 - s)
        ]) / 4.0

    return N, dNds, dNdt


# ── Q9 (9-node quadrilateral) ─────────────────────────────────────────────────

def _shapefun_Q9(gp_locs: torch.Tensor):
    """
    Q9 shape functions at all Gauss points.
    Mirrors shapefun.m Q9 branch.
    """
    ngp = gp_locs.shape[0]
    N    = torch.zeros(ngp, 9, dtype=DTYPE, device=DEVICE)
    dNds = torch.zeros(ngp, 9, dtype=DTYPE, device=DEVICE)
    dNdt = torch.zeros(ngp, 9, dtype=DTYPE, device=DEVICE)

    for i in range(ngp):
        s = gp_locs[i, 0]
        t = gp_locs[i, 1]

        N[i, :] = torch.stack([
            s*(s-1)*t*(t-1)/4,   s*(s+1)*t*(t-1)/4,
            s*(s+1)*t*(t+1)/4,   s*(s-1)*t*(t+1)/4,
            -(s-1)*(s+1)*t*(t-1)/2, -s*(s+1)*(t-1)*(t+1)/2,
            -(s-1)*(s+1)*t*(t+1)/2, -s*(s-1)*(t-1)*(t+1)/2,
            (s-1)*(s+1)*(t-1)*(t+1),
        ])

        dNds[i, :] = torch.stack([
            ((-1+2*s)*(-1+t)*t)/4,   ((1+2*s)*(-1+t)*t)/4,
            ((1+2*s)*t*(1+t))/4,      ((-1+2*s)*t*(1+t))/4,
            -(s*(-1+t)*t),            -((1+2*s)*(-1+t**2))/2,
            -(s*t*(1+t)),             -((-1+2*s)*(-1+t**2))/2,
            2*s*(-1+t**2),
        ])

        dNdt[i, :] = torch.stack([
            ((-1+s)*s*(-1+2*t))/4,   (s*(1+s)*(-1+2*t))/4,
            (s*(1+s)*(1+2*t))/4,      ((-1+s)*s*(1+2*t))/4,
            -((-1+s**2)*(-1+2*t))/2, -(s*(1+s)*t),
            -((-1+s**2)*(1+2*t))/2,  -((-1+s)*s*t),
            2*(-1+s**2)*t,
        ])

    return N, dNds, dNdt
