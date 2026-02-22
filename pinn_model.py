"""
pinn_model.py  -  Neural network model and total energy assembly

Plays the role of element.m in the MATLAB code. Assembles the scalar
total incremental potential Pi used as the loss for the NN optimizer.
"""

import torch
import torch.nn as nn
from basis import FEMBasis
from initiate_fun import SimState
from mat_params import MaterialParams
from dirichlet import DirichletBC, apply_dirichlet
from time_int_euler_vec import time_int_euler_vec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


# =============================================================================
# Neural Network  (X -> u)
# =============================================================================

class DisplacementNet(nn.Module):
    """
    Fully-connected network: R^2 -> R^2
    Input  : [n_nodes, 2]   reference coordinates
    Output : [n_nodes, 2]   predicted displacements
    """
    def __init__(self, hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.net(X)


# =============================================================================
# Deformation gradient from nodal displacements  (mirrors def_grad.m)
# =============================================================================

def compute_F_all_gp(u, conn, fem):
    """
    Compute 3x3 deformation gradient F at all Gauss points.
    F = I + grad(u),  embedded into 3x3 (plane strain: F33 = 1).

    u    : [n_nodes, 2]
    conn : [nel, n_nodes_el]
    fem  : FEMBasis
    returns F3 : [nel, ngp, 3, 3]
    """
    u_el = u[conn]    # [nel, n_nodes_el, 2]

    # H_ij = sum_I  u_{I,i} * dN_I/dX_j
    # u_el  indices: e=element, I=node, i=displacement direction
    # dN_dX indices: e=element, n=gauss point, I=node, j=spatial direction
    H = torch.einsum("eIi, enIj -> enij", u_el, fem.dN_dX)

    I2 = torch.eye(2, dtype=DTYPE, device=DEVICE)
    F2 = I2 + H

    nel, ngp = F2.shape[:2]
    F3 = torch.zeros(nel, ngp, 3, 3, dtype=DTYPE, device=DEVICE)
    F3[..., :2, :2] = F2
    F3[...,  2,  2] = 1.0

    return F3


# =============================================================================
# Total incremental potential  Pi  (the loss)
# =============================================================================

def compute_Pi(u_pred, bc, conn, fem, state, mat, inp, f_ext):
    """
    Pi = integral_Omega0 [ Psi + D_ve + D_vp ] dV0  -  f_ext . u

    Returns: Pi, Psi_total, D_total, state_new
    """
    nel, ngp = state.nel, state.ngp

    # Enforce Dirichlet BCs
    u = apply_dirichlet(u_pred, bc)

    # Deformation gradient at all GPs
    F_gp = compute_F_all_gp(u, conn, fem)

    # No damage: g = 1, gp = 0
    g_ones  = torch.ones (ngp, dtype=DTYPE, device=DEVICE)
    gp_zero = torch.zeros(ngp, dtype=DTYPE, device=DEVICE)

    Psi_all  = torch.zeros(nel, ngp, dtype=DTYPE, device=DEVICE)
    D_ve_all = torch.zeros(nel, ngp, dtype=DTYPE, device=DEVICE)
    D_vp_all = torch.zeros(nel, ngp, dtype=DTYPE, device=DEVICE)

    F_i_new   = state.F_i.clone()
    F_e_new   = state.F_e.clone()
    F_p_new   = state.F_p.clone()
    J_new     = state.J.clone()
    e0_new    = state.e0.clone()
    yb_new    = state.yb_max.clone()
    sigma_new = state.sigma.clone()

    for iel in range(nel):
        (T_iel, yb_max_iel,
         _dsig, _dH,
         Fi_iel, Fe_iel, Fp_iel,
         J_iel, e0_iel,
         D_ve_iel, D_vp_iel) = time_int_euler_vec(
            F       = F_gp[iel],
            F_i     = state.F_i[iel],
            F_e     = state.F_e[iel],
            F_p     = state.F_p[iel],
            J       = state.J[iel],
            e0      = state.e0[iel],
            e_t     = state.e_t[iel],
            g       = g_ones,
            gp      = gp_zero,
            mat     = mat,
            inp     = inp,
            yb_max0 = state.yb_max[iel],
            iel     = iel,
        )

        Psi_all[iel]  = yb_max_iel
        D_ve_all[iel] = D_ve_iel
        D_vp_all[iel] = D_vp_iel

        with torch.no_grad():
            F_i_new[iel]   = Fi_iel
            F_e_new[iel]   = Fe_iel
            F_p_new[iel]   = Fp_iel
            J_new[iel]     = J_iel
            e0_new[iel]    = e0_iel
            yb_new[iel]    = yb_max_iel
            sigma_new[iel] = torch.stack([
                T_iel[:, 0, 0], T_iel[:, 1, 1], T_iel[:, 0, 1]
            ], dim=-1)

    # Spatial integration
    dv      = fem.detJ0 * fem.gp_wts.unsqueeze(0)
    W_gp    = Psi_all + D_ve_all + D_vp_all
    Pi_int  = (W_gp    * dv).sum()
    Psi_tot = (Psi_all * dv).sum()
    D_tot   = ((D_ve_all + D_vp_all) * dv).sum()

    Pi_ext  = (f_ext * u.flatten()).sum()
    Pi      = Pi_int - Pi_ext

    state_new          = state.detach_copy()
    state_new.F_i      = F_i_new.detach()
    state_new.F_e      = F_e_new.detach()
    state_new.F_p      = F_p_new.detach()
    state_new.J        = J_new.detach()
    state_new.e0       = e0_new.detach()
    state_new.yb_max   = yb_new.detach()
    state_new.sigma    = sigma_new.detach()

    return Pi, Psi_tot, D_tot, state_new


# =============================================================================
# Physical residual  dPi/du  (convergence criterion)
# =============================================================================

def compute_residual(u_pred, Pi, bc, ref_norm=None):
    """
    Compute normalised residual  ||dPi/du_free|| / ||r0||.

    For displacement-controlled loading f_ext = 0, so we normalise by the
    residual at the first iteration of each load step (r0). This mirrors
    the relative residual check in the MATLAB Newton-Raphson solver.

    Parameters
    ----------
    u_pred   : [n_nodes, 2]
    Pi       : scalar
    bc       : DirichletBC
    ref_norm : float or None
               None on the first call -> returns rel_res = 1.0
               Pass the returned res_norm on all later calls.

    Returns
    -------
    rel_res  : float   ||r_free|| / ref_norm
    res_norm : float   ||r_free||  (store as ref_norm on first call)
    grad_u   : [n_nodes, 2]
    """
    grad_u = torch.autograd.grad(
        outputs=Pi,
        inputs=u_pred,
        create_graph=False,
        retain_graph=True,
    )[0]

    grad_free = grad_u.clone()
    grad_free[~bc.free_mask] = 0.0

    res_norm = float(grad_free.norm())

    if ref_norm is None or ref_norm < 1e-14:
        rel_res = 1.0
    else:
        rel_res = res_norm / ref_norm

    return rel_res, res_norm, grad_u