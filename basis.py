"""
basis.py  —  mirrors basis.m

Precomputes reference-configuration FEM quantities for all elements
and all Gauss points.  Called once before training begins.

Outputs stored in FEMBasis dataclass:
    dN_dX   : [nel, ngp, n_nodes_el, 2]   dN_I / dX_j  in reference config
    detJ0   : [nel, ngp]                  det(J0) at each GP
    gp_wts  : [ngp]                       Gauss weights
"""

import torch
from dataclasses import dataclass
from gaussfun import gaussfun
from shapefun import shapefun

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


@dataclass
class FEMBasis:
    dN_dX:   torch.Tensor   # [nel, ngp, n_nodes_el, 2]
    detJ0:   torch.Tensor   # [nel, ngp]
    gp_wts:  torch.Tensor   # [ngp]
    gp_locs: torch.Tensor   # [ngp, 2]  parametric coordinates
    N:       torch.Tensor   # [ngp, n_nodes_el]  shape function values
    # element nodal coordinates (reference)
    coords_el: torch.Tensor  # [nel, n_nodes_el, 2]


def build_basis(nodes: torch.Tensor,
                conn:  torch.Tensor,
                mesh_type: str = "Q4",
                ngp: int = 4) -> FEMBasis:
    """
    Precompute dN/dX and detJ0 for all elements.  Mirrors basis.m.

    Parameters
    ----------
    nodes     : [n_nodes, 2]  reference nodal coordinates
    conn      : [nel, n_nodes_el]  element connectivity (0-based)
    mesh_type : "Q4" or "Q9"
    ngp       : number of Gauss points per element

    Returns
    -------
    FEMBasis dataclass
    """
    nel, n_nodes_el = conn.shape

    gp_locs, gp_wts = gaussfun(ngp)
    N, dNds_all, dNdt_all = shapefun(mesh_type, ngp)  # [ngp, n_nodes_el] each

    # Element nodal coordinates  [nel, n_nodes_el, 2]
    coords_el = nodes[conn]   # [nel, n_nodes_el, 2]

    dN_dX_out = torch.zeros(nel, ngp, n_nodes_el, 2, dtype=DTYPE, device=DEVICE)
    detJ0_out = torch.zeros(nel, ngp,               dtype=DTYPE, device=DEVICE)

    for igp in range(ngp):
        dNds = dNds_all[igp]   # [n_nodes_el]
        dNdt = dNdt_all[igp]   # [n_nodes_el]

        # Reference Jacobian J0 components for all elements at once
        # X_el : [nel, n_nodes_el, 2]
        # dx0ds = sum_I  dNds_I * X_I   -> [nel]
        dx0ds = (dNds.unsqueeze(0) * coords_el[:, :, 0]).sum(dim=1)  # [nel]
        dy0ds = (dNds.unsqueeze(0) * coords_el[:, :, 1]).sum(dim=1)  # [nel]
        dx0dt = (dNdt.unsqueeze(0) * coords_el[:, :, 0]).sum(dim=1)  # [nel]
        dy0dt = (dNdt.unsqueeze(0) * coords_el[:, :, 1]).sum(dim=1)  # [nel]

        detJ = dx0ds * dy0dt - dx0dt * dy0ds   # [nel]

        # dN/dX via inverse of J0  (Cramer's rule for 2×2)
        inv_detJ = 1.0 / detJ.clamp(min=1e-14)

        # dNdx_I =  (J0_22 * dNds_I - J0_21 * dNdt_I) / detJ
        # dNdy_I = (-J0_12 * dNds_I + J0_11 * dNdt_I) / detJ
        dNdx = ( dy0dt.unsqueeze(1) * dNds.unsqueeze(0)
                - dy0ds.unsqueeze(1) * dNdt.unsqueeze(0)) * inv_detJ.unsqueeze(1)
        dNdy = (-dx0dt.unsqueeze(1) * dNds.unsqueeze(0)
                + dx0ds.unsqueeze(1) * dNdt.unsqueeze(0)) * inv_detJ.unsqueeze(1)

        dN_dX_out[:, igp, :, 0] = dNdx   # [nel, n_nodes_el]
        dN_dX_out[:, igp, :, 1] = dNdy
        detJ0_out[:, igp]       = detJ

    return FEMBasis(
        dN_dX    = dN_dX_out,
        detJ0    = detJ0_out,
        gp_wts   = gp_wts,
        gp_locs  = gp_locs,
        N        = N,
        coords_el= coords_el,
    )
