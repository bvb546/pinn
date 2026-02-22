"""
mat_params.py  —  mirrors mat_params.m

Defines material parameters for the viscoelastic-viscoplastic model
with fiber reinforcement (Mori-Tanaka homogenisation).

All stresses in MPa, lengths in mm.
"""

import math
import torch
from dataclasses import dataclass, field
from typing import List
from inputdata import InputData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


@dataclass
class FiberLayer:
    """Properties for one fiber layer (maps to one range of elements)."""
    vf:       float              # fiber volume fraction
    alpha:    float              # orientation concentration parameter
    a_tensor: torch.Tensor       # [3,3] orientation tensor

    # Derived (filled by mat_params)
    n_fiber_family: int                  = 0
    a0:             torch.Tensor         = None   # [3, n_fam] unit fiber directions
    vf_ratio:       torch.Tensor         = None   # [n_fam] fraction of total vf
    vf_vec:         torch.Tensor         = None   # [n_fam] = vf * vf_ratio
    vm:             float                = 0.0    # matrix volume fraction = 1 - vf
    A:              torch.Tensor         = None   # [2,2] structural tensor


@dataclass
class MaterialParams:
    # ── Nanoparticle / amplification factors ─────────────────────────────────
    wnp:  float = 0.0    # nanoparticle weight fraction
    zita: float = 0.0    # moisture content

    # ── Viscoelasticity (Eyring) ──────────────────────────────────────────────
    gamma_dot_0: float = 1.0447e12
    dH:          float = 1.977e-19   # J
    Temp:        float = 296.0       # K
    Tref:        float = 296.0       # K  reference temperature
    tau_base:    float = 40.0        # MPa
    kb:          float = 1.3806e-23  # J/K

    # ── Viscoplasticity ───────────────────────────────────────────────────────
    ab:     float = field(init=False)
    bb:     float = 1.1
    sigma0: float = field(init=False)
    sigmad: float = field(init=False)

    # ── Elastic moduli (filled after amplification) ───────────────────────────
    mu_e: float = field(init=False)
    mu_i: float = field(init=False)
    Kv:   float = field(init=False)
    Gc:   float = field(init=False)
    lc:   float = 0.02    # length scale for phase-field

    # ── Fiber model ───────────────────────────────────────────────────────────
    a1: float = 9.0
    a2: float = 1.0
    a3: float = 1.0

    # ── Per-element fiber data (filled by mat_params) ─────────────────────────
    # These mirror input.n_fiber_family, input.a0, input.vf, input.vf_ratio, input.vm
    n_fiber_family: torch.Tensor = None   # [nel] int tensor
    a0:             list         = None   # list[nel]  each [3, n_fam]
    vf:             list         = None   # list[nel]  each [n_fam]
    vf_ratio:       list         = None   # list[nel]  each [n_fam]
    vm:             torch.Tensor = None   # [nel]

    def __post_init__(self):
        # Amplification factors (Xnp, Xw, X)  — mirrors mat_params.m
        ro_np = 3.0; ro_p = 1.2
        vp    = (self.wnp * ro_p /
                 (ro_np + self.wnp * ro_p - ro_np * self.wnp))
        Xnp   = 1.0 + 3.5 * vp + 18.0 * vp**2
        Xw    = 0.057 * (self.zita)**2 - 9.5 * self.zita + 1.0
        X     = Xnp * Xw

        alpha_e  = 2.0 - math.exp(0.01093 * (self.Temp - self.Tref))

        # Moduli
        self.mu_e = X * 790.0  * alpha_e
        self.mu_i = X * 760.0  * alpha_e
        self.Kv   = X * 1154.0 * alpha_e
        self.Gc   = 190e-3               # N/mm

        # Viscoplastic parameters
        self.ab     = 22.0 * self.zita + 0.8
        self.bb     = 1.1
        self.sigma0 = 30.0  * X
        self.sigmad = 85.0  * X


def mat_params(inp: InputData, mesh_info: dict) -> MaterialParams:
    """
    Build MaterialParams and assign fiber data to each element.
    Mirrors mat_params.m.

    mesh_info must contain:
        'nel'        : int
        'elem_layer' : [nel] int array — which layer each element belongs to
    """
    mat = MaterialParams()

    nel        = mesh_info["nel"]
    elem_layer = mesh_info["elem_layer"]    # [nel] numpy array, 1-based

    # ── Define fiber layers (mirrors the layer loop in mat_params.m) ──────────
    def make_layer(vf, alpha, a_mat_np):
        import numpy as np
        a_tensor = torch.tensor(a_mat_np, dtype=DTYPE, device=DEVICE)
        layer    = FiberLayer(vf=vf, alpha=alpha, a_tensor=a_tensor)

        # Eigen-decomposition to get fiber directions
        eigvals, eigvecs = torch.linalg.eigh(a_tensor)   # ascending order
        mask = eigvals > 1e-10
        n_fam = int(mask.sum().item())

        a0_raw = eigvecs[:, mask]                          # [3, n_fam]
        # Normalise each column
        norms = a0_raw.norm(dim=0, keepdim=True).clamp(min=1e-12)
        a0    = a0_raw / norms                             # [3, n_fam]

        ev_pos        = eigvals[mask]
        vf_ratio      = ev_pos / ev_pos.sum()              # [n_fam]
        vf_vec        = vf * vf_ratio                      # [n_fam]
        vm            = 1.0 - vf

        A_2d = torch.eye(2, dtype=DTYPE, device=DEVICE) + alpha * a_tensor[:2, :2]

        layer.n_fiber_family = n_fam
        layer.a0             = a0
        layer.vf_ratio       = vf_ratio
        layer.vf_vec         = vf_vec
        layer.vm             = vm
        layer.A              = A_2d
        return layer

    import numpy as np

    # Layer definitions — mirrors mat_params.m
    a0_mat = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]], dtype=np.float64)
    a2_mat = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]], dtype=np.float64)
    a3_mat = np.array([[0.5,-0.5, 0], [-0.5, 0.5, 0], [0, 0, 0]], dtype=np.float64)

    layers = [
        make_layer(0.5, 2.0, a0_mat),   # Layer 1: 0°
        make_layer(0.5, 2.0, a2_mat),   # Layer 2: 45°
        make_layer(0.5, 2.0, a3_mat),   # Layer 3: -45°
    ]

    # ── Assign fiber data element-by-element ──────────────────────────────────
    n_fiber_family_list = []
    a0_list             = []
    vf_list             = []
    vf_ratio_list       = []
    vm_list             = []

    for iel in range(nel):
        layer_idx = int(elem_layer[iel]) - 1   # convert 1-based to 0-based
        layer_idx = max(0, min(layer_idx, len(layers) - 1))
        lay       = layers[layer_idx]

        n_fiber_family_list.append(lay.n_fiber_family)
        a0_list.append(lay.a0)                 # [3, n_fam]
        vf_list.append(lay.vf_vec)             # [n_fam]
        vf_ratio_list.append(lay.vf_ratio)     # [n_fam]
        vm_list.append(lay.vm)

    mat.n_fiber_family = torch.tensor(n_fiber_family_list,
                                      dtype=torch.long, device=DEVICE)
    mat.a0             = a0_list
    mat.vf             = vf_list
    mat.vf_ratio       = vf_ratio_list
    mat.vm             = torch.tensor(vm_list, dtype=DTYPE, device=DEVICE)

    return mat
