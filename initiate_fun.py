"""
initiate_fun.py  —  mirrors initiate_fun.m

Initialises all history variables and simulation state variables
to their values at t = 0.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


@dataclass
class SimState:
    """
    Holds all history and state variables at the current time step t_n.
    Mirrors the var struct in the MATLAB code.

    Shape conventions (all on DEVICE):
        F_i, F_e, F_p  : [nel, ngp, 3, 3]
        J              : [nel, ngp]
        e0             : [nel, ngp]   accumulated plastic strain
        e_t            : [nel, ngp, 2]
        yb_max         : [nel, ngp]   maximum stored energy (crack driving force)
        sigma          : [nel, ngp, 3]  Cauchy/Kirchhoff stress components
    """
    nel: int
    ngp: int

    # ── Internal variable tensors ─────────────────────────────────────────────
    F_i:    torch.Tensor = field(default=None)
    F_e:    torch.Tensor = field(default=None)
    F_p:    torch.Tensor = field(default=None)
    J:      torch.Tensor = field(default=None)
    e0:     torch.Tensor = field(default=None)
    e_t:    torch.Tensor = field(default=None)
    yb_max: torch.Tensor = field(default=None)
    sigma:  torch.Tensor = field(default=None)
    h:      torch.Tensor = field(default=None)   # thickness (for 2D plane stress)

    # ── Simulation counters ───────────────────────────────────────────────────
    step:     int   = 1
    iter_u:   int   = 0
    converged: bool = False

    def __post_init__(self):
        nel, ngp = self.nel, self.ngp
        I3 = torch.eye(3, dtype=DTYPE, device=DEVICE)
        base = I3.unsqueeze(0).unsqueeze(0).expand(nel, ngp, 3, 3)

        self.F_i    = base.clone()
        self.F_e    = base.clone()
        self.F_p    = base.clone()
        self.J      = torch.ones (nel, ngp,    dtype=DTYPE, device=DEVICE)
        self.e0     = torch.zeros(nel, ngp,    dtype=DTYPE, device=DEVICE)
        self.e_t    = torch.zeros(nel, ngp, 2, dtype=DTYPE, device=DEVICE)
        self.yb_max = torch.zeros(nel, ngp,    dtype=DTYPE, device=DEVICE)
        self.sigma  = torch.zeros(nel, ngp, 3, dtype=DTYPE, device=DEVICE)
        self.h      = torch.ones (nel, ngp,    dtype=DTYPE, device=DEVICE)

    def detach_copy(self) -> "SimState":
        """
        Return a new SimState with all tensors detached from the autograd
        graph.  Used to freeze history variables during NN training.
        Mirrors the role of 'var0 = var' in the MATLAB Newton-Raphson loop.
        """
        s = object.__new__(SimState)
        s.nel      = self.nel
        s.ngp      = self.ngp
        s.step     = self.step
        s.iter_u   = self.iter_u
        s.converged= self.converged
        for attr in ("F_i", "F_e", "F_p", "J", "e0", "e_t", "yb_max",
                     "sigma", "h"):
            setattr(s, attr, getattr(self, attr).detach().clone())
        return s

    def update_from(self, other: "SimState"):
        """
        Copy updated tensors from 'other' into self after convergence.
        Mirrors  var = var0  after Newton convergence in MATLAB.
        """
        for attr in ("F_i", "F_e", "F_p", "J", "e0", "e_t", "yb_max",
                     "sigma", "h"):
            setattr(self, attr, getattr(other, attr).detach().clone())
        self.step     = other.step
        self.converged= other.converged


def initiate_fun(inp, mesh_info: dict) -> SimState:
    """
    Create and return the initial SimState.  Mirrors initiate_fun.m.

    Parameters
    ----------
    inp       : InputData
    mesh_info : dict returned by gmsh_read()
    """
    nel = mesh_info["nel"]
    ngp = inp.ngp

    state = SimState(nel=nel, ngp=ngp)

    # Thickness (uniform)
    state.h = inp.h0 * torch.ones(nel, ngp, dtype=DTYPE, device=DEVICE)

    return state
