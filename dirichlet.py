"""
dirichlet.py  —  mirrors dirichlet.m

Enforces Dirichlet (essential) boundary conditions by direct substitution
into the NN-predicted displacement field.

In FEM this is done by modifying rows/columns of the stiffness matrix.
Here it is done by overwriting the NN output at constrained nodes before
computing the deformation gradient and the energy — giving EXACT enforcement
with zero approximation error.
"""

import torch
from dataclasses import dataclass, field
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


@dataclass
class DirichletBC:
    """
    Stores all Dirichlet BC information for the current load step.

    node_ids : [n_bc]  0-based node indices
    dof_ids  : [n_bc]  0 = x-displacement, 1 = y-displacement
    values   : [n_bc]  prescribed displacement values
    free_mask: [n_nodes, 2]  True where DOF is free (for residual computation)
    """
    node_ids:  torch.Tensor
    dof_ids:   torch.Tensor
    values:    torch.Tensor
    free_mask: torch.Tensor   # [n_nodes, 2]


def build_dirichlet(mesh_info: dict,
                    load_step_disp: float,
                    n_nodes: int) -> DirichletBC:
    """
    Build the DirichletBC object for the current load step.
    Mirrors the BC setup in gmsh_read_sn.m and the Newton-Raphson driver.

    Parameters
    ----------
    mesh_info       : dict from gmsh_read()
    load_step_disp  : current total applied displacement (mm)
    n_nodes         : total number of nodes

    Returns
    -------
    DirichletBC
    """
    node_ids_list = []
    dof_ids_list  = []
    values_list   = []

    # ── Fixed DOFs: bottom edge (dof_ebc) ─────────────────────────────────────
    # dof_ebc contains flat 0-based DOF indices  (node*2 + dof)
    for flat_dof in mesh_info["dof_ebc"]:
        node = flat_dof // 2
        dof  = flat_dof %  2
        node_ids_list.append(node)
        dof_ids_list.append(dof)
        values_list.append(0.0)

    # ── Prescribed DOFs: top edge (dof_disp) — uy = load_step_disp ────────────
    for flat_dof in mesh_info["dof_disp"]:
        node = flat_dof // 2
        dof  = flat_dof %  2
        node_ids_list.append(node)
        dof_ids_list.append(dof)
        values_list.append(load_step_disp)

    node_ids = torch.tensor(node_ids_list, dtype=torch.long,  device=DEVICE)
    dof_ids  = torch.tensor(dof_ids_list,  dtype=torch.long,  device=DEVICE)
    values   = torch.tensor(values_list,   dtype=DTYPE,       device=DEVICE)

    # ── Free DOF mask ─────────────────────────────────────────────────────────
    free_mask = torch.ones(n_nodes, 2, dtype=torch.bool, device=DEVICE)
    for k in range(len(node_ids_list)):
        free_mask[node_ids_list[k], dof_ids_list[k]] = False

    return DirichletBC(node_ids=node_ids, dof_ids=dof_ids,
                       values=values, free_mask=free_mask)


def apply_dirichlet(u_pred: torch.Tensor, bc: DirichletBC) -> torch.Tensor:
    """
    Enforce Dirichlet BCs by direct substitution into the NN output.

    u_pred : [n_nodes, 2]  raw NN prediction
    returns: [n_nodes, 2]  with constrained DOFs set to prescribed values

    The clone() ensures the original NN output is not modified in-place,
    preserving the autograd graph for backpropagation.
    """
    u = u_pred.clone()
    u[bc.node_ids, bc.dof_ids] = bc.values
    return u
