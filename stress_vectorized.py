"""
stress_vectorized.py  —  mirrors stress_vectorized.m

Computes the isochoric Kirchhoff stress and stored energy density
for the fiber-reinforced hyperelastic model (Mori-Tanaka homogenisation).

Works on batched tensors so the entire mesh can be processed at once.
All operations are differentiable via PyTorch autograd.

Branch:
    type == "eq"   -> mu = mu_i  (equilibrium / long-term network)
    type == "neq"  -> mu = mu_e  (non-equilibrium / viscous network)
"""

import torch
from mat_params import MaterialParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


# ── Tensor utilities ──────────────────────────────────────────────────────────

def dev3(A: torch.Tensor) -> torch.Tensor:
    """Deviatoric part of a batch of 3×3 tensors.  A : [..., 3, 3]"""
    tr = A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]
    I  = torch.eye(3, dtype=A.dtype, device=A.device)
    return A - (tr / 3.0).unsqueeze(-1).unsqueeze(-1) * I


# ── Fiber model functions (inline from stress_vectorized.m) ───────────────────

def _ffun(Ibar4: torch.Tensor, a1: float, a2: float, a3: float):
    """Nonlinear fiber stiffness f(Ibar4) and derivative fp."""
    exp_term = torch.exp(a3 * (Ibar4 - 1.0))
    f  = a1 + a2 * exp_term
    fp = a2 * a3 * exp_term
    return f, fp


def _ggfun(f: torch.Tensor, fp: torch.Tensor,
           vf: float, si: float):
    """Mori-Tanaka interaction g(f) and derivative gp."""
    denom = (1.0 - vf) * f + si + vf
    g     = ((1.0 + si * vf) * f + (1.0 - vf) * si) / denom
    gp    = (  (1.0 + si * vf) * fp * denom
             - (1.0 - vf) * fp * ((1.0 + si * vf) * f + (1.0 - vf) * si)
            ) / (denom * denom)
    return g, gp


# ── Main function ─────────────────────────────────────────────────────────────

def stress_vectorized(Fb:   torch.Tensor,
                      J:    torch.Tensor,
                      mat:  MaterialParams,
                      branch: str,
                      iel:  int) -> tuple:
    """
    Compute isochoric Kirchhoff stress T and stored energy density yb
    for element `iel` at all its Gauss points.

    Parameters
    ----------
    Fb     : [ngp, 3, 3]  isochoric deformation gradient  (J^{-1/3} * F)
    J      : [ngp]        Jacobian = det(F)
    mat    : MaterialParams
    branch : "eq" or "neq"
    iel    : element index (0-based)

    Returns
    -------
    T  : [ngp, 3, 3]   isochoric Kirchhoff stress
    yb : [ngp]         stored energy density
    """
    mu = mat.mu_i if branch == "eq" else mat.mu_e

    ngp      = Fb.shape[0]
    half_mu  = 0.5 * mu

    # Common kinematic quantities
    Cbar     = Fb.transpose(-2, -1) @ Fb          # [ngp, 3, 3]
    Bbar     = Fb @ Fb.transpose(-2, -1)          # [ngp, 3, 3]
    Bbar_dev = dev3(Bbar)                          # [ngp, 3, 3]
    Ibar1    = Cbar[..., 0, 0] + Cbar[..., 1, 1] + Cbar[..., 2, 2]  # [ngp]
    C2       = Cbar @ Cbar                         # [ngp, 3, 3]  (for Ibar5)

    invJ2    = (2.0 / J).unsqueeze(-1).unsqueeze(-1)   # [ngp, 1, 1]
    J23      = J.pow(2.0 / 3.0)                        # [ngp]

    I3       = torch.eye(3, dtype=DTYPE, device=DEVICE)

    n_fib    = int(mat.n_fiber_family[iel].item())
    T        = torch.zeros(ngp, 3, 3, dtype=DTYPE, device=DEVICE)
    yb       = torch.zeros(ngp,       dtype=DTYPE, device=DEVICE)

    for i in range(n_fib):
        a0_i = mat.a0[iel][:, i]          # [3] unit fiber direction
        vf_i = float(mat.vf[iel][i])
        vr_i = float(mat.vf_ratio[iel][i])
        vm_i = float(mat.vm[iel])

        # ── Pseudo-invariants ─────────────────────────────────────────────────
        # Ibar4 = a0' * Cbar * a0   (using pre-computed Cbar)
        a0t   = a0_i.unsqueeze(-1)                       # [3, 1]
        Ibar4 = (a0_i @ Cbar @ a0t).squeeze(-1).squeeze(-1)   # [ngp]  a0'*Cbar*a0
        Ibar5 = (a0_i @ C2   @ a0t).squeeze(-1).squeeze(-1)   # [ngp]  a0'*C^2*a0

        EPS = 1e-12
        Ibar4          = Ibar4.clamp(min=EPS)
        Ibar4_sqrt     = Ibar4.sqrt()
        Ibar4_sqrt_inv = 1.0 / Ibar4_sqrt
        Ibar4_inv      = 1.0 / Ibar4
        Ibar4_inv_1p5  = Ibar4_inv * Ibar4_sqrt_inv    # Ibar4^{-1.5}
        Ibar4_inv_2    = Ibar4_inv * Ibar4_inv          # Ibar4^{-2}
        Ibar4_sq       = Ibar4 * Ibar4                  # Ibar4^2

        # ── Fiber model ───────────────────────────────────────────────────────
        f,   fp  = _ffun(Ibar4, mat.a1, mat.a2, mat.a3)
        g1, gp1  = _ggfun(f, fp, vf_i, 1.0)
        g2, gp2  = _ggfun(f, fp, vf_i, 0.4)

        vm_vf_f  = vm_i + vf_i * f

        # ── W coefficients (stress weights) ───────────────────────────────────
        W1 = half_mu * g2

        term1 = vf_i * fp * (Ibar4 + 2.0 * Ibar4_sqrt_inv - 3.0)
        term2 = vm_vf_f  * (1.0 - Ibar4_inv_1p5)
        term3 = -g1 * (Ibar5 * Ibar4_inv_2 + 1.0)
        term4 =  g2 * (Ibar5 * Ibar4_inv_2 + Ibar4_inv_1p5)
        term5 = (Ibar5 - Ibar4_sq) * (0.5 * Ibar4_inv) * gp1
        term6 = 0.5 * (Ibar1 - (Ibar5 + 2.0 * Ibar4_sqrt) * Ibar4_inv) * gp2
        W4    = half_mu * (term1 + term2 + term3 + term4 + term5 + term6)
        W5    = 0.5 * Ibar4_inv * (g1 - g2) * mu

        # ── Fiber direction in current config ─────────────────────────────────
        sqrt_J23_I4 = (J23 * Ibar4).clamp(min=EPS).sqrt()             # [ngp]
        Fb_a0       = (Fb @ a0_i.unsqueeze(-1)).squeeze(-1)            # [ngp, 3]
        a           = Fb_a0 / sqrt_J23_I4.unsqueeze(-1)                # [ngp, 3]

        # ── Stress terms ──────────────────────────────────────────────────────
        T_t1 = W1.unsqueeze(-1).unsqueeze(-1) * Bbar_dev               # [ngp,3,3]

        aat  = a.unsqueeze(-1) @ a.unsqueeze(-2)                        # [ngp,3,3]
        T_t2 = (W4 * Ibar4).unsqueeze(-1).unsqueeze(-1) * (aat - I3 / 3.0)

        Bbar_a = (Bbar @ a.unsqueeze(-1)).squeeze(-1)                   # [ngp,3]
        aBat   = a.unsqueeze(-1)     @ Bbar_a.unsqueeze(-2)            # [ngp,3,3]
        Baat   = Bbar_a.unsqueeze(-1) @ a.unsqueeze(-2)                # [ngp,3,3]
        T_t3   = (W5 * Ibar4).unsqueeze(-1).unsqueeze(-1) * (
                  aBat + Baat
                  - (2.0 / 3.0) * Ibar5.unsqueeze(-1).unsqueeze(-1) * I3)

        T_i = invJ2 * (T_t1 + T_t2 + T_t3)

        # ── Energy terms ──────────────────────────────────────────────────────
        yb_t1 = half_mu * vm_vf_f * (Ibar4 + 2.0 * Ibar4_sqrt_inv - 3.0)
        yb_t2 = half_mu * g1      * (Ibar5 - Ibar4_sq) * Ibar4_inv
        yb_t3 = half_mu * g2      * (Ibar1 - (Ibar5 + 2.0 * Ibar4_sqrt) * Ibar4_inv)
        yb_i  = yb_t1 + yb_t2 + yb_t3

        # ── Accumulate (weighted by fiber volume fraction ratio) ──────────────
        T  = T  + vr_i * T_i
        yb = yb + vr_i * yb_i

    return T, yb
