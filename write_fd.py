"""
write_fd.py  -  mirrors write_fd.m and store_data.m

Stores and writes force-displacement data across load steps.
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


@dataclass
class StoredVars:
    """Accumulates results across all load steps."""
    displacement : List[float] = field(default_factory=list)
    reaction     : List[float] = field(default_factory=list)
    load_step    : List[int]   = field(default_factory=list)
    iter_step    : List[int]   = field(default_factory=list)
    Pi_history   : List[float] = field(default_factory=list)
    Psi_history  : List[float] = field(default_factory=list)
    D_history    : List[float] = field(default_factory=list)
    time_history : List[float] = field(default_factory=list)


def compute_reaction(u_eval, Pi_val, mesh_info):
    """
    Compute the reaction force at the top (prescribed) nodes.

    At equilibrium, dPi/du at a prescribed DOF equals the reaction force
    that the boundary must exert to maintain that prescription.
    This is the same as extracting reactions from the assembled internal
    force vector in standard FEM.

    Units: Pi is in N*mm, u is in mm  ->  dPi/du is in N.

    Parameters
    ----------
    u_eval     : [n_nodes, 2]  displacement field (requires_grad=True)
    Pi_val     : scalar        total potential computed from u_eval
    mesh_info  : dict          contains 'top_nodes' (0-based node indices)

    Returns
    -------
    reaction : float   total reaction force in N (sum over top nodes, y-dir)
    """
    grad_u = torch.autograd.grad(
        outputs=Pi_val,
        inputs=u_eval,
        create_graph=False,
        retain_graph=True,
    )[0]                             # [n_nodes, 2]

    top_nodes = mesh_info["top_nodes"]   # 0-based indices of top edge nodes
    # Sum of y-direction internal forces at top nodes
    # Negative sign: reaction opposes the applied displacement direction
    reaction = -float(grad_u[top_nodes, 1].sum())

    return reaction


def store_data(stored, state, total_disp, reaction,
               step, n_iter, Pi, Psi, D, elapsed):
    """
    Store results from the current load step.

    Parameters
    ----------
    stored     : StoredVars
    state      : SimState (not used directly, kept for API consistency)
    total_disp : float   prescribed displacement at this step (mm)
    reaction   : float   reaction force (N) computed from autograd
    step       : int
    n_iter     : int
    Pi, Psi, D : float
    elapsed    : float   wall time (s)
    """
    stored.displacement.append(total_disp)
    stored.reaction.append(reaction)
    stored.load_step.append(step)
    stored.iter_step.append(n_iter)
    stored.Pi_history.append(Pi)
    stored.Psi_history.append(Psi)
    stored.D_history.append(D)
    stored.time_history.append(elapsed)
    return stored


def write_fd(stored, output_dir="output"):
    """Write force-displacement data to CSV.  Mirrors write_fd.m."""
    Path(output_dir).mkdir(exist_ok=True)
    fd_file = Path(output_dir) / "force_displacement.csv"

    with open(fd_file, "w") as f:
        f.write("step,displacement_mm,reaction_N,n_iter,"
                "Pi,Psi,D_dissipation,wall_time_s\n")
        for i, step in enumerate(stored.load_step):
            f.write(
                str(step) + "," +
                "{:.8f}".format(stored.displacement[i]) + "," +
                "{:.6f}".format(stored.reaction[i]) + "," +
                str(stored.iter_step[i]) + "," +
                "{:.6e}".format(stored.Pi_history[i]) + "," +
                "{:.6e}".format(stored.Psi_history[i]) + "," +
                "{:.6e}".format(stored.D_history[i]) + "," +
                "{:.2f}".format(stored.time_history[i]) + "\n"
            )

    print("  Force-displacement data written to " + str(fd_file), flush=True)