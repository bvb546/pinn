"""
inputdata.py  -  mirrors inputdata_stag.m

Defines all simulation control parameters.
Material parameters are set separately in mat_params.py.
"""

from dataclasses import dataclass, field
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


@dataclass
class InputData:
    # Problem dimensions
    ndof: int = 3           # total DOF per node (2 displacement + 1 phase-field)
    dim:  int = 2           # spatial dimension

    # Mesh
    filename:  str = "mesh.msh"
    mesh_type: str = "Q4"
    ngp:       int = 4      # Gauss points per element
    numlayers: int = 3      # number of fiber layers

    # Loading
    h0:       float = 1.0          # thickness (mm)
    load_rate: float = 1.0 / 60.0  # mm/s
    load_inc:  float = 1e-3        # displacement increment per step (mm)
    disp_max:  float = 5.0         # maximum total displacement (mm)

    # Solver
    tol:       float = 1e-4   # convergence tolerance  ||r||/||r0|| < tol
    max_adam:  int   = 3000   # maximum Adam iterations per step
    max_lbfgs: int   = 300    # maximum L-BFGS iterations per step
    lr_adam:   float = 1e-3   # Adam learning rate

    # Neural network
    nn_hidden: int = 64
    nn_layers: int = 4

    # Phase-field (not active at this stage)
    g0:     float = 1e-5
    g_type: int   = 1

    # Plotting
    plot_interval: int = 1    # save contour plots every this many converged steps
                               # set to 0 to disable plotting

    # Device / precision
    device: torch.device = field(default_factory=lambda: DEVICE)
    dtype:  torch.dtype  = field(default_factory=lambda: DTYPE)

    # Derived (filled in __post_init__)
    dt: float = field(init=False)

    def __post_init__(self):
        self.dt = self.load_inc / self.load_rate


def get_input():
    """Return the InputData instance.  Mirrors inputdata_stag.m."""
    return InputData()