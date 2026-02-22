"""
plotfun.py  -  mirrors plotfun.m

Saves stress and strain contour plots to output/plots/ every
plot_interval converged load steps.

Fields plotted
--------------
Stress (Kirchhoff, MPa):
    S11  -  sigma_xx
    S22  -  sigma_yy
    S12  -  sigma_xy
    Svm  -  von Mises equivalent stress

Strain (Green-Lagrange, dimensionless):
    E11  -  epsilon_xx
    E22  -  epsilon_yy
    E12  -  epsilon_xy  (engineering shear * 0.5)
    Eeq  -  equivalent (von Mises) strain

Displacement:
    U1   -  x-displacement (mm)
    U2   -  y-displacement (mm)

Strategy: average each field over the 4 Gauss points of each element,
then do a nodal average by accumulating element contributions weighted
by the number of elements sharing each node.  This gives smooth contours
from Q4 data without requiring superconvergent extrapolation.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe on HPC
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


# =============================================================================
# GP-to-node averaging
# =============================================================================

def gp_to_nodes(field_el, conn, n_nodes):
    """
    Average a per-element scalar field to nodes.

    Parameters
    ----------
    field_el : [nel]      element-averaged scalar values
    conn     : [nel, 4]   element connectivity (0-based, numpy)
    n_nodes  : int

    Returns
    -------
    field_nodes : [n_nodes]  node-averaged values
    """
    field_el = np.asarray(field_el)
    conn     = np.asarray(conn)
    nel      = conn.shape[0]

    node_sum   = np.zeros(n_nodes, dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.float64)

    for iel in range(nel):
        for inode in conn[iel]:
            node_sum[inode]   += field_el[iel]
            node_count[inode] += 1.0

    # Avoid division by zero for isolated nodes
    mask = node_count > 0
    node_sum[mask] /= node_count[mask]

    return node_sum


# =============================================================================
# Compute element-averaged stress and strain
# =============================================================================

def compute_fields(state, u_sol, conn_np, fem, n_nodes):
    """
    Compute element-averaged stress and strain fields, then extrapolate
    to nodes.

    Parameters
    ----------
    state    : SimState      contains state.sigma [nel, ngp, 3]
    u_sol    : [n_nodes, 2]  converged displacement (torch or numpy)
    conn_np  : [nel, 4]      numpy connectivity (0-based)
    fem      : FEMBasis
    n_nodes  : int

    Returns
    -------
    dict of nodal arrays, each [n_nodes]:
        S11, S22, S12, Svm, E11, E22, E12, Eeq, U1, U2
    """
    # --- Stress (average over GPs) ---
    # state.sigma : [nel, ngp, 3]  components [S11, S22, S12]
    sigma_np = state.sigma.cpu().numpy()   # [nel, ngp, 3]
    S11_el = sigma_np[:, :, 0].mean(axis=1)   # [nel]
    S22_el = sigma_np[:, :, 1].mean(axis=1)
    S12_el = sigma_np[:, :, 2].mean(axis=1)
    Svm_el = np.sqrt(S11_el**2 - S11_el*S22_el + S22_el**2 + 3.0*S12_el**2)

    # --- Strain from displacement gradient ---
    # Use the same FEM basis to compute F = I + grad(u) at each GP,
    # then E = 0.5*(F^T F - I), averaged over GPs.
    u_np   = u_sol.cpu().numpy() if torch.is_tensor(u_sol) else np.asarray(u_sol)
    dN_dX  = fem.dN_dX.cpu().numpy()  # [nel, ngp, 4, 2]
    nel, ngp = dN_dX.shape[:2]

    E11_el = np.zeros(nel)
    E22_el = np.zeros(nel)
    E12_el = np.zeros(nel)

    for iel in range(nel):
        conn_el = conn_np[iel]           # [4]
        u_el    = u_np[conn_el]          # [4, 2]

        E11_gp = 0.0; E22_gp = 0.0; E12_gp = 0.0

        for igp in range(ngp):
            dNdX = dN_dX[iel, igp]      # [4, 2]

            # Displacement gradient  H_ij = sum_I u_{I,i} * dN_I/dX_j
            H = u_el.T @ dNdX           # [2, 2]

            # Deformation gradient F = I + H
            F = np.eye(2) + H

            # Green-Lagrange strain E = 0.5*(F^T F - I)
            C = F.T @ F
            E = 0.5 * (C - np.eye(2))

            E11_gp += E[0, 0]
            E22_gp += E[1, 1]
            E12_gp += E[0, 1]   # E12 = 0.5 * engineering shear

        E11_el[iel] = E11_gp / ngp
        E22_el[iel] = E22_gp / ngp
        E12_el[iel] = E12_gp / ngp

    # von Mises equivalent strain
    Eeq_el = np.sqrt(
        (2.0/3.0) * (E11_el**2 + E22_el**2 + E11_el*E22_el + 3.0*E12_el**2)
    )

    # --- Nodal averaging ---
    fields = {}
    for name, data in [("S11", S11_el), ("S22", S22_el),
                        ("S12", S12_el), ("Svm", Svm_el),
                        ("E11", E11_el), ("E22", E22_el),
                        ("E12", E12_el), ("Eeq", Eeq_el)]:
        fields[name] = gp_to_nodes(data, conn_np, n_nodes)

    # Displacement (already nodal)
    fields["U1"] = u_np[:, 0]
    fields["U2"] = u_np[:, 1]

    return fields


# =============================================================================
# Single contour plot
# =============================================================================

def _plot_contour(nodes, triangles, values, title, label, filepath, deformed=None):
    """
    Save one contour plot.

    nodes     : [n_nodes, 2]  reference coordinates
    triangles : [n_tri, 3]    triangulation (each Q4 split into 2 triangles)
    values    : [n_nodes]     nodal scalar field
    deformed  : [n_nodes, 2]  deformed coordinates (optional, for overlay)
    """
    fig, ax = plt.subplots(figsize=(5, 6))

    # Plot on deformed or reference configuration
    plot_nodes = deformed if deformed is not None else nodes
    tri = mtri.Triangulation(plot_nodes[:, 0], plot_nodes[:, 1], triangles)

    levels = 20
    cf = ax.tricontourf(tri, values, levels=levels, cmap="jet")
    plt.colorbar(cf, ax=ax, label=label)

    # Overlay mesh edges (thin, reference config)
    ax.triplot(mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles),
               color="k", linewidth=0.2, alpha=0.3)

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


# =============================================================================
# Main plotting function  (call from main.py)
# =============================================================================

def plotfun(state, u_sol, nodes_np, conn_np, fem, step, total_disp,
            output_dir="output"):
    """
    Compute and save all stress and strain contour plots for this step.

    Parameters
    ----------
    state      : SimState
    u_sol      : [n_nodes, 2]  converged displacement (torch tensor)
    nodes_np   : [n_nodes, 2]  reference coordinates (numpy)
    conn_np    : [nel, 4]      element connectivity  (numpy, 0-based)
    fem        : FEMBasis
    step       : int           current load step number
    total_disp : float         applied displacement (mm), used in titles
    output_dir : str
    """
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_nodes = nodes_np.shape[0]
    u_np    = u_sol.cpu().numpy() if torch.is_tensor(u_sol) else np.asarray(u_sol)

    # Deformed coordinates
    deformed = nodes_np + u_np

    # Build triangulation: split each Q4 into 2 triangles
    # Q4 node order: 0-1-2-3 (counter-clockwise)
    # Triangle 1: 0-1-2,  Triangle 2: 0-2-3
    triangles = []
    for el in conn_np:
        triangles.append([el[0], el[1], el[2]])
        triangles.append([el[0], el[2], el[3]])
    triangles = np.array(triangles, dtype=np.int64)

    # Compute all fields
    fields = compute_fields(state, u_sol, conn_np, fem, n_nodes)

    step_str = str(step).zfill(4)
    disp_str = "{:.4f}".format(total_disp)
    prefix   = str(plot_dir) + "/step" + step_str + "_"

    # Plot definitions:  (field_key, title, colorbar_label, filename_suffix)
    plots = [
        # Stress
        ("S11", "S11 - sigma_xx  [step " + step_str + ", u=" + disp_str + " mm]",
         "S11 (MPa)", "S11.png"),
        ("S22", "S22 - sigma_yy  [step " + step_str + ", u=" + disp_str + " mm]",
         "S22 (MPa)", "S22.png"),
        ("S12", "S12 - sigma_xy  [step " + step_str + ", u=" + disp_str + " mm]",
         "S12 (MPa)", "S12.png"),
        ("Svm", "von Mises stress  [step " + step_str + ", u=" + disp_str + " mm]",
         "Svm (MPa)", "Svm.png"),
        # Strain
        ("E11", "E11 - epsilon_xx  [step " + step_str + ", u=" + disp_str + " mm]",
         "E11 (-)", "E11.png"),
        ("E22", "E22 - epsilon_yy  [step " + step_str + ", u=" + disp_str + " mm]",
         "E22 (-)", "E22.png"),
        ("E12", "E12 - epsilon_xy  [step " + step_str + ", u=" + disp_str + " mm]",
         "E12 (-)", "E12.png"),
        ("Eeq", "Equivalent strain  [step " + step_str + ", u=" + disp_str + " mm]",
         "Eeq (-)", "Eeq.png"),
        # Displacement
        ("U1",  "U1 - x-displacement  [step " + step_str + "]",
         "U1 (mm)", "U1.png"),
        ("U2",  "U2 - y-displacement  [step " + step_str + "]",
         "U2 (mm)", "U2.png"),
    ]

    saved = []
    for key, title, label, fname in plots:
        fpath = prefix + fname
        _plot_contour(
            nodes     = nodes_np,
            triangles = triangles,
            values    = fields[key],
            title     = title,
            label     = label,
            filepath  = fpath,
            deformed  = deformed,   # plot on deformed mesh
        )
        saved.append(fpath)

    print("  Plots saved: output/plots/step" + step_str + "_*.png  (" +
          str(len(saved)) + " files)", flush=True)
