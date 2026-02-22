"""
newton_raphson_pinn.py  -  mirrors newton_raphson_disp.m

Two-phase optimizer: Adam then L-BFGS, minimising the total incremental
potential Pi.

Convergence criterion:  ||r|| / ||r0|| < tol
where r0 is the residual at the first Adam iteration of each load step.
This mirrors the relative residual check in the MATLAB Newton-Raphson solver
and works correctly for displacement-controlled loading where f_ext = 0.
"""

import torch
from pinn_model import DisplacementNet, compute_Pi, compute_residual
from initiate_fun import SimState
from mat_params import MaterialParams
from basis import FEMBasis
from dirichlet import DirichletBC
from inputdata import InputData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64


def newton_raphson_pinn(net, nodes, conn, fem, state, mat, inp,
                        bc, f_ext, verbose=True):
    """
    Run the PINN optimizer for one load step.

    Returns: state_new, u_sol, converged, n_iter
    """
    tol      = inp.tol
    ref_norm = None    # set after the very first residual evaluation

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # -------------------------------------------------------------------------
    # Phase 1: Adam
    # -------------------------------------------------------------------------
    opt_adam  = torch.optim.Adam(net.parameters(), lr=inp.lr_adam)
    state_new = state.detach_copy()
    iter_adam = inp.max_adam

    log("  [Adam]")

    for it in range(inp.max_adam):
        opt_adam.zero_grad()

        u_pred = net(nodes)
        u_pred.retain_grad()

        Pi, Psi, D, state_cand = compute_Pi(
            u_pred, bc, conn, fem, state, mat, inp, f_ext)

        Pi.backward(retain_graph=True)

        if it % 100 == 0:
            rel_res, res_norm, _ = compute_residual(u_pred, Pi, bc, ref_norm)

            # Capture reference norm at the very first evaluation
            if ref_norm is None:
                ref_norm = res_norm
                rel_res  = 1.0

            log("    Adam " + str(it).rjust(5) +
                " | Pi=" + "{:+.4e}".format(float(Pi)) +
                "  Psi=" + "{:.4e}".format(float(Psi)) +
                "  D=" + "{:.4e}".format(float(D)) +
                "  ||r||/||r0||=" + "{:.4e}".format(rel_res) +
                "  ||r||=" + "{:.4e}".format(res_norm))

            # Switch to L-BFGS once residual has dropped sufficiently
            if rel_res < tol * 10:
                log("    -> Switching to L-BFGS at Adam iter " + str(it))
                state_new = state_cand
                iter_adam = it
                break

        opt_adam.step()
    else:
        # Adam ran to max_adam without switching
        u_pred = net(nodes)
        u_pred.retain_grad()
        Pi, Psi, D, state_new = compute_Pi(
            u_pred, bc, conn, fem, state, mat, inp, f_ext)
        Pi.backward(retain_graph=True)
        _, res_norm, _ = compute_residual(u_pred, Pi, bc, ref_norm)
        if ref_norm is None:
            ref_norm = res_norm

    # -------------------------------------------------------------------------
    # Phase 2: L-BFGS
    # -------------------------------------------------------------------------
    opt_lbfgs = torch.optim.LBFGS(
        net.parameters(),
        max_iter         = 20,
        history_size     = 50,
        line_search_fn   = "strong_wolfe",
        tolerance_grad   = 1e-9,
        tolerance_change = 1e-12,
    )

    log("  [L-BFGS]")

    converged       = False
    Pi_prev         = float("inf")
    state_new_lbfgs = state_new
    n_lbfgs         = 0

    for it in range(inp.max_lbfgs):

        def closure():
            opt_lbfgs.zero_grad()
            u_c = net(nodes)
            Pi_c, _, _, _ = compute_Pi(
                u_c, bc, conn, fem, state, mat, inp, f_ext)
            Pi_c.backward()
            return Pi_c

        opt_lbfgs.step(closure)
        n_lbfgs += 1

        u_pred = net(nodes)
        u_pred.retain_grad()
        Pi, Psi, D, state_cand = compute_Pi(
            u_pred, bc, conn, fem, state, mat, inp, f_ext)
        Pi.backward(retain_graph=True)

        rel_res, res_norm, _ = compute_residual(u_pred, Pi, bc, ref_norm)
        dPi_rel = abs(float(Pi) - Pi_prev) / (abs(Pi_prev) + 1e-12)
        Pi_prev = float(Pi)

        if it % 10 == 0:
            log("    LBFGS " + str(it).rjust(4) +
                " | Pi=" + "{:+.4e}".format(float(Pi)) +
                "  Psi=" + "{:.4e}".format(float(Psi)) +
                "  D=" + "{:.4e}".format(float(D)) +
                "  ||r||/||r0||=" + "{:.4e}".format(rel_res) +
                "  ||r||=" + "{:.4e}".format(res_norm) +
                "  dPi=" + "{:.2e}".format(dPi_rel))

        state_new_lbfgs = state_cand

        if rel_res < tol and dPi_rel < tol * 1e-2:
            log("  Converged: L-BFGS iter " + str(it) +
                "  ||r||/||r0||=" + "{:.2e}".format(rel_res) +
                "  dPi=" + "{:.2e}".format(dPi_rel))
            converged = True
            break

    if not converged:
        log("  WARNING: Not converged after " + str(n_lbfgs) +
            " L-BFGS iters.  ||r||/||r0||=" + "{:.4e}".format(rel_res))

    with torch.no_grad():
        u_sol = net(nodes)

    n_iter_total = iter_adam + n_lbfgs
    return state_new_lbfgs, u_sol.detach(), converged, n_iter_total