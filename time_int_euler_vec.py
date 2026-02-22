"""
time_int_euler_vec.py  —  mirrors time_int_euler_vec.m

Forward-Euler time integration of the viscoelastic-viscoplastic
constitutive model at all Gauss points of one element.

All operations are differentiable via PyTorch autograd so that
∂Π/∂u can be computed by backpropagation through this function.

Key difference from MATLAB:
    Hard if/else branching → torch.where  (maintains autograd graph)
"""

import torch
from stress_vectorized import stress_vectorized, dev3
from mat_params import MaterialParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
EPS    = 1e-12


def time_int_euler_vec(F:     torch.Tensor,
                       F_i:   torch.Tensor,
                       F_e:   torch.Tensor,
                       F_p:   torch.Tensor,
                       J:     torch.Tensor,
                       e0:    torch.Tensor,
                       e_t:   torch.Tensor,
                       g:     torch.Tensor,
                       gp:    torch.Tensor,
                       mat:   MaterialParams,
                       inp,
                       yb_max0: torch.Tensor,
                       iel:   int) -> tuple:
    """
    Time integration for element `iel` at all Gauss points.

    Parameters  (all for one element, ngp Gauss points)
    ----------
    F       : [ngp, 3, 3]  total deformation gradient at t_{n+1}
    F_i     : [ngp, 3, 3]  intermittent factor at t_n   (frozen history)
    F_e     : [ngp, 3, 3]  elastic factor at t_n        (frozen history)
    F_p     : [ngp, 3, 3]  plastic factor at t_n        (frozen history)
    J       : [ngp]        Jacobian at t_n
    e0      : [ngp]        accumulated plastic strain at t_n
    e_t     : [ngp, 2]     strain magnitude history [t_{n-1}, t_n]
    g       : [ngp]        phase-field degradation   (= 1 when no damage)
    gp      : [ngp]        dg/dphi                  (= 0 when no damage)
    mat     : MaterialParams
    inp     : InputData    (provides dt, g0)
    yb_max0 : [ngp]        previous max stored energy
    iel     : int          element index (0-based)

    Returns
    -------
    T        : [ngp, 3, 3]  Kirchhoff stress at t_{n+1}
    yb_max   : [ngp]        updated max stored energy
    dsig_dphi: [3, ngp]     ∂σ/∂φ  (for phase-field coupling, ignored here)
    dH_deps  : [3, ngp]     stress components (for phase-field driving force)
    F_i_new  : [ngp, 3, 3]  updated F_i
    F_e_new  : [ngp, 3, 3]  updated F_e
    F_p_new  : [ngp, 3, 3]  updated F_p
    J_new    : [ngp]        Jacobian at t_{n+1}
    e0_new   : [ngp]        updated accumulated plastic strain
    D_ve     : [ngp]        viscoelastic dissipation increment
    D_vp     : [ngp]        viscoplastic dissipation increment
    """
    ngp      = F.shape[0]
    dt       = inp.dt
    g0       = inp.g0
    g_total  = (g + g0).unsqueeze(-1).unsqueeze(-1)   # [ngp, 1, 1]

    I3 = torch.eye(3, dtype=DTYPE, device=DEVICE).unsqueeze(0).expand(ngp, 3, 3)

    # ── 1. Viscoelastic update ─────────────────────────────────────────────────
    # Non-equilibrium stress at t_n from F_e (neq branch)
    Tneq, _  = stress_vectorized(F_e, J, mat, "neq", iel)   # [ngp, 3, 3]
    Tnep_P   = dev3(g_total * Tneq)                          # deviatoric part
    tau_ve   = (Tnep_P ** 2).sum(dim=(-2, -1)).clamp(min=0).sqrt()  # [ngp]

    # Eyring flow rate
    factor       = mat.dH / (mat.kb * mat.Temp)
    ratio        = (tau_ve / mat.tau_base).clamp(min=EPS)
    gamma_dot_v  = mat.gamma_dot_0 * torch.exp(factor * (ratio ** 0.657 - 1.0))

    # Viscoelastic dissipation increment
    D_ve = tau_ve * gamma_dot_v * dt                         # [ngp]

    # Update F_i  (differentiable: use torch.where instead of if/else)
    F_ie_t     = F_i @ F_e                                   # [ngp, 3, 3]
    F_e_inv    = torch.linalg.inv(F_e)
    scale_ve   = (gamma_dot_v / tau_ve.clamp(min=EPS)).unsqueeze(-1).unsqueeze(-1)
    f_v_flow   = scale_ve * Tnep_P
    dF_i       = F_e_inv @ f_v_flow @ F_ie_t * dt

    mask_ve    = (tau_ve > EPS).unsqueeze(-1).unsqueeze(-1).expand_as(F_i)
    F_i_new    = torch.where(mask_ve, F_i + dF_i, F_i)

    # ── 2. Viscoplastic update ─────────────────────────────────────────────────
    # Equilibrium stress at t_n from F_ie = F_i * F_e  (eq branch)
    Teq, _   = stress_vectorized(F_ie_t, J, mat, "eq", iel)  # [ngp, 3, 3]

    # Green-Lagrange strain magnitude at t_{n+1}
    FtF      = F.transpose(-2, -1) @ F
    ep       = 0.5 * (FtF - I3)
    e_t_dt   = (ep ** 2).sum(dim=(-2, -1)).clamp(min=0).sqrt()   # [ngp]

    TT       = dev3(g_total * (Teq + Tneq))
    tau_p    = (TT ** 2).sum(dim=(-2, -1)).clamp(min=0).sqrt()   # [ngp]

    # Plastic strain rate
    e_diff   = (e_t_dt - e0).clamp(min=0)
    edot     = e_t_dt / (dt + EPS)
    gdp_raw  = mat.ab * e_diff.clamp(min=EPS) ** (mat.bb - 1.0) * edot

    # Conditions for viscoplastic flow (torch.where keeps grad)
    mask_p     = tau_p > mat.sigma0
    mask_yield = (e_t_dt - e0 > 0) & mask_p

    gdp = torch.where(mask_yield, gdp_raw, torch.zeros_like(gdp_raw))

    # Dissipation increment
    D_vp = torch.where(mask_yield, tau_p * gdp * dt,
                       torch.zeros_like(tau_p))              # [ngp]

    # F_p update
    F_ie_t_inv  = torch.linalg.inv(F_ie_t)
    F_iep_t     = F_ie_t @ F_p
    scale_vp    = (gdp / tau_p.clamp(min=EPS)).unsqueeze(-1).unsqueeze(-1)
    dF_p        = F_ie_t_inv @ TT @ F_iep_t * dt * scale_vp

    mask_p_exp  = mask_yield.unsqueeze(-1).unsqueeze(-1).expand_as(F_p)
    F_p_new     = torch.where(mask_p_exp, F_p + dF_p, F_p)

    e0_new      = e_t_dt                                     # [ngp]

    # ── 3. Kinematics at t_{n+1} ──────────────────────────────────────────────
    J_new    = torch.linalg.det(F).clamp(min=EPS)            # [ngp]
    Fb_new   = J_new.pow(-1.0 / 3.0).unsqueeze(-1).unsqueeze(-1) * F
    F_ie_dt  = Fb_new @ torch.linalg.inv(F_p_new)
    F_e_new  = F_ie_dt @ torch.linalg.inv(F_i_new)

    # ── 4. Stress and energy at t_{n+1} ───────────────────────────────────────
    Teq_dt,  yb_eq  = stress_vectorized(F_ie_dt, J_new, mat, "eq",  iel)
    Tneq_dt, yb_neq = stress_vectorized(F_e_new, J_new, mat, "neq", iel)

    # Volumetric stress (equilibrium branch)
    Kv         = mat.Kv
    Jm1_J      = J_new - 1.0 / J_new.clamp(min=EPS)
    Teq_v_dt   = (0.5 * Kv * Jm1_J).unsqueeze(-1).unsqueeze(-1) * I3
    yb_eq_v    = 0.5 * Kv * (0.5 * (J_new ** 2 - 1.0) - torch.log(J_new.clamp(min=EPS)))

    # Base (compression): no volumetric energy contribution
    g_t  = (g + g0).unsqueeze(-1).unsqueeze(-1)
    T    = g_t * (Teq_dt + Tneq_dt) + Teq_v_dt
    yb   = yb_eq + yb_neq
    dH   = Teq_dt + Tneq_dt

    # Tension branch (J >= 1): add volumetric contribution
    mask_t = (J_new >= 1.0).unsqueeze(-1).unsqueeze(-1).expand_as(T)
    T      = torch.where(mask_t,
                         g_t * (Teq_dt + Tneq_dt + Teq_v_dt),
                         T)
    yb_mask = J_new >= 1.0
    yb      = torch.where(yb_mask, yb_eq + yb_eq_v + yb_neq, yb)
    dH      = torch.where(mask_t,
                          Teq_dt + Tneq_dt + Teq_v_dt,
                          dH)

    # ── 5. Phase-field coupling quantities (for future use) ───────────────────
    gp_row    = gp.unsqueeze(-1).unsqueeze(-1)      # [ngp, 1, 1]
    dsig_dphi = torch.stack([
        (gp_row * dH)[..., 0, 0],
        (gp_row * dH)[..., 1, 1],
        (gp_row * dH)[..., 0, 1],
    ], dim=0)                                        # [3, ngp]

    dH_deps   = torch.stack([
        dH[..., 0, 0],
        dH[..., 1, 1],
        dH[..., 0, 1],
    ], dim=0)                                        # [3, ngp]

    yb_max = torch.maximum(yb_max0, yb)              # [ngp]

    return (T, yb_max, dsig_dphi, dH_deps,
            F_i_new, F_e_new, F_p_new, J_new, e0_new,
            D_ve, D_vp)
