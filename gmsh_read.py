"""
gmsh_read.py  —  mirrors gmsh_read_sn.m

Reads a Gmsh .msh (version 2 ASCII) file and returns mesh data as a dict.

Keys returned
-------------
nodes       : np.ndarray  [n_nodes, 2]   reference coordinates (x, y)
conn        : np.ndarray  [nel, 4]       element connectivity, 0-based
nel         : int
n_nodes     : int
dof_ebc     : list[int]   DOF indices fixed to zero  (bottom edge, tension BC)
dof_disp    : list[int]   DOF indices where displacement is applied (top edge)
elem_layer  : np.ndarray  [nel] int, 1-based layer index for each element
"""

import numpy as np
from pathlib import Path


def gmsh_read(filename: str, mesh_type: str = "Q4",
              numlayers: int = 3) -> dict:
    """
    Read a Gmsh 2.x ASCII mesh file.

    Parameters
    ----------
    filename  : path to the .msh file
    mesh_type : "Q4" (4-node quad) or "Q9" (9-node quad)
    numlayers : number of material layers (used for element-layer assignment)
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {filename}")

    lines = path.read_text().splitlines()

    # ── Parse nodes ──────────────────────────────────────────────────────────
    idx_nodes_start = _find_section(lines, "$Nodes") + 1
    n_nodes = int(lines[idx_nodes_start])
    nodes   = np.zeros((n_nodes, 2), dtype=np.float64)

    for i in range(n_nodes):
        parts = lines[idx_nodes_start + 1 + i].split()
        nodes[i, 0] = float(parts[1])   # x
        nodes[i, 1] = float(parts[2])   # y

    # ── Parse elements ────────────────────────────────────────────────────────
    idx_elem_start = _find_section(lines, "$Elements") + 1
    n_elem_total   = int(lines[idx_elem_start])

    # Gmsh element type codes:  3 = 4-node quad (Q4),  10 = 9-node quad (Q9)
    target_type = 3 if mesh_type == "Q4" else 10
    n_nodes_el  = 4 if mesh_type == "Q4" else 9

    conn_list = []
    for i in range(n_elem_total):
        parts    = lines[idx_elem_start + 1 + i].split()
        etype    = int(parts[1])
        n_tags   = int(parts[2])
        node_ids = [int(p) - 1 for p in parts[3 + n_tags:]]  # convert to 0-based
        if etype == target_type and len(node_ids) == n_nodes_el:
            conn_list.append(node_ids[:n_nodes_el])

    conn = np.array(conn_list, dtype=np.int64)   # [nel, 4]
    nel  = conn.shape[0]

    # ── Boundary condition DOFs ───────────────────────────────────────────────
    # Mirrors gmsh_read_sn.m  (tension loading)
    ymax = nodes[:, 1].max()
    ymin = nodes[:, 1].min()
    tol  = 1e-4

    bottom_nodes = np.where(nodes[:, 1] < ymin + tol)[0]  # 0-based
    top_nodes    = np.where(nodes[:, 1] > ymax - tol)[0]  # 0-based

    # dof_ebc:  bottom edge — fix both ux and uy  (1-based DOF indices)
    # DOF numbering: node i → DOFs 2i+1 (x), 2i+2 (y)  in 1-based
    dof_ebc = []
    for n in bottom_nodes:
        dof_ebc.append(2 * n)       # ux  (0-based flat index: node*2 + 0)
        dof_ebc.append(2 * n + 1)   # uy

    # dof_disp: top edge — prescribed uy displacement (0-based)
    dof_disp = [2 * n + 1 for n in top_nodes]

    # ── Element-to-layer assignment ───────────────────────────────────────────
    # Mirrors gmsh_read_sn.m: assign by centroid x-coordinate
    centroids_x = np.array([
        nodes[conn[i], 0].mean() for i in range(nel)
    ])
    xmin       = nodes[:, 0].min()
    xmax_coord = nodes[:, 0].max()
    layer_width = (xmax_coord - xmin) / numlayers if numlayers > 0 else 1.0

    elem_layer = np.ceil((centroids_x - xmin) / layer_width).astype(int)
    elem_layer = np.clip(elem_layer, 1, numlayers)

    # Print summary (mirrors MATLAB fprintf output)
    print(f"\nMesh read: {filename}")
    print(f"  Nodes:    {n_nodes}")
    print(f"  Elements: {nel}  ({mesh_type})")
    print(f"  Bottom BC nodes: {len(bottom_nodes)}")
    print(f"  Top disp nodes:  {len(top_nodes)}")
    print(f"\nElement-to-layer assignment:")
    for layer in range(1, numlayers + 1):
        count = int((elem_layer == layer).sum())
        print(f"  Layer {layer}: {count} elements")

    return {
        "nodes":       nodes,          # [n_nodes, 2]
        "conn":        conn,           # [nel, 4]  0-based
        "nel":         nel,
        "n_nodes":     n_nodes,
        "dof_ebc":     dof_ebc,        # list of 0-based flat DOF indices (fixed)
        "dof_disp":    dof_disp,       # list of 0-based flat DOF indices (prescribed)
        "elem_layer":  elem_layer,     # [nel] 1-based layer index
        "bottom_nodes": bottom_nodes,  # [n_bc] 0-based node indices
        "top_nodes":    top_nodes,     # [n_top] 0-based node indices
    }


def _find_section(lines, tag: str) -> int:
    """Return the line index of the given Gmsh section tag."""
    for i, line in enumerate(lines):
        if line.strip() == tag:
            return i
    raise ValueError(f"Section '{tag}' not found in mesh file.")
