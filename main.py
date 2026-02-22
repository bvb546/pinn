"""
main.py  -  mirrors main_stag.m

Top-level driver for the PINN large-deformation viscoelastic-viscoplastic
solver.

Usage:
    python -u main.py
"""

import torch
import time
import numpy as np
from pathlib import Path

from inputdata           import get_input
from gmsh_read           import gmsh_read
from mat_params          import mat_params
from basis               import build_basis
from initiate_fun        import initiate_fun
from pinn_model          import DisplacementNet, compute_Pi
from dirichlet           import build_dirichlet
from newton_raphson_pinn import newton_raphson_pinn
from write_fd            import StoredVars, store_data, write_fd, compute_reaction
from plotfun             import plotfun

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


def log(msg=""):
    print(msg, flush=True)


def main():
    log("=" * 65)
    log("  PINN Large-Deformation Viscoelastic-Viscoplastic Solver")
    log("=" * 65)
    log("  Device : " + str(DEVICE))
    log()

    # 1. Input data
    inp = get_input()
    log("[inputdata]  load_inc=" + str(inp.load_inc) + " mm" +
        "  dt=" + str(round(inp.dt, 4)) + " s" +
        "  tol=" + str(inp.tol) +
        "  plot_interval=" + str(inp.plot_interval))

    # 2. Read mesh
    mesh_info = gmsh_read(inp.filename,
                          mesh_type=inp.mesh_type,
                          numlayers=inp.numlayers)

    nel     = mesh_info["nel"]
    n_nodes = mesh_info["n_nodes"]

    nodes_np = mesh_info["nodes"]                                          # numpy [n_nodes, 2]
    conn_np  = mesh_info["conn"]                                           # numpy [nel, 4]
    nodes    = torch.tensor(nodes_np, dtype=DTYPE, device=DEVICE)
    conn     = torch.tensor(conn_np,  dtype=torch.long, device=DEVICE)

    log()
    log("[mesh]  " + str(n_nodes) + " nodes, " + str(nel) +
        " elements (" + inp.mesh_type + ")")

    # 3. Material parameters
    mat = mat_params(inp, mesh_info)
    log()
    log("[mat_params]" +
        "  mu_e="   + str(round(mat.mu_e,   1)) + " MPa" +
        "  mu_i="   + str(round(mat.mu_i,   1)) + " MPa" +
        "  Kv="     + str(round(mat.Kv,     1)) + " MPa" +
        "  sigma0=" + str(round(mat.sigma0, 1)) + " MPa")

    # 4. Precompute FEM basis
    fem = build_basis(nodes, conn, mesh_type=inp.mesh_type, ngp=inp.ngp)
    log()
    log("[basis]  dN/dX precomputed for " +
        str(nel) + "x" + str(inp.ngp) + " Gauss points")

    # 5. Initialise state
    state = initiate_fun(inp, mesh_info)
    log()
    log("[initiate_fun]  History variables initialised" +
        "  (nel=" + str(nel) + ", ngp=" + str(inp.ngp) + ")")

    # 6. Neural network
    net = DisplacementNet(hidden=inp.nn_hidden,
                          n_layers=inp.nn_layers).to(DEVICE)
    n_params = sum(p.numel() for p in net.parameters())
    log()
    log("[NN]  " + str(n_params) + " trainable parameters" +
        "  (hidden=" + str(inp.nn_hidden) +
        ", layers=" + str(inp.nn_layers) + ")")

    # External force vector (zero for displacement control)
    f_ext = torch.zeros(n_nodes * 2, dtype=DTYPE, device=DEVICE)

    Path("output/plots").mkdir(parents=True, exist_ok=True)

    # 7. Load stepping loop
    stored     = StoredVars()
    step       = 1
    total_disp = 0.0

    log()
    log("=" * 65)
    log("  Beginning load stepping")
    log("=" * 65)

    while True:
        log()
        log("-" * 65)
        log("  Step " + str(step) +
            "  |  disp = " +
            str(round(total_disp + inp.load_inc, 5)) + " mm")
        log("-" * 65)

        total_disp += inp.load_inc

        # a. Build Dirichlet BCs
        bc = build_dirichlet(mesh_info,
                             load_step_disp=total_disp,
                             n_nodes=n_nodes)

        # b. Run PINN optimizer
        t0 = time.time()

        state_new, u_sol, converged, n_iter = newton_raphson_pinn(
            net     = net,
            nodes   = nodes,
            conn    = conn,
            fem     = fem,
            state   = state,
            mat     = mat,
            inp     = inp,
            bc      = bc,
            f_ext   = f_ext,
            verbose = True,
        )

        elapsed = time.time() - t0
        log("  Wall time: " + str(round(elapsed, 1)) +
            " s  |  Iterations: " + str(n_iter))

        if converged:
            # Accept step and update state
            state.update_from(state_new)
            state.step = step

            # Evaluate Pi and reaction force via autograd
            u_eval = net(nodes)
            u_eval.requires_grad_(True)
            Pi_val, Psi_val, D_val, _ = compute_Pi(
                u_eval, bc, conn, fem, state, mat, inp, f_ext)

            reaction = compute_reaction(u_eval, Pi_val, mesh_info)

            # Store and write force-displacement data
            stored = store_data(
                stored, state,
                total_disp = total_disp,
                reaction   = reaction,
                step       = step,
                n_iter     = n_iter,
                Pi         = float(Pi_val),
                Psi        = float(Psi_val),
                D          = float(D_val),
                elapsed    = elapsed,
            )
            write_fd(stored)

            log()
            log("  Step " + str(step) + " summary:")
            log("    Prescribed disp : " + str(round(total_disp, 5)) + " mm")
            log("    Reaction force  : " + str(round(reaction, 4)) + " N")
            log("    Pi              : " + "{:.4e}".format(float(Pi_val)))
            log("    Psi             : " + "{:.4e}".format(float(Psi_val)))
            log("    D (dissip.)     : " + "{:.4e}".format(float(D_val)))

            # c. Save contour plots every plot_interval steps
            if inp.plot_interval > 0 and step % inp.plot_interval == 0:
                log()
                log("  Saving contour plots for step " + str(step) + " ...")
                plotfun(
                    state      = state,
                    u_sol      = u_sol,
                    nodes_np   = nodes_np,
                    conn_np    = conn_np,
                    fem        = fem,
                    step       = step,
                    total_disp = total_disp,
                    output_dir = "output",
                )

            step += 1

        else:
            log("  WARNING: Non-convergence at step " + str(step) +
                ". Reducing load increment.")
            total_disp   -= inp.load_inc
            inp.load_inc /= 2.0
            inp.dt        = inp.load_inc / inp.load_rate
            if inp.load_inc < 1e-8:
                log("  ERROR: Load increment too small. Stopping.")
                break

        if total_disp >= inp.disp_max:
            log()
            log("  Maximum displacement " +
                str(inp.disp_max) + " mm reached. Done.")
            break

    write_fd(stored)
    log()
    log("=" * 65)
    log("  Simulation complete.")
    log("  Results in: output/force_displacement.csv")
    log("  Plots in:   output/plots/")
    log("=" * 65)


if __name__ == "__main__":
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(42)
    main()