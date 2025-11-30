# -*- coding: utf-8 -*-
"""
ECSW hyper-reduction operator construction for the fluidic pinball (Re = 30)
===========================================================================

This script builds the tensors and reduced operators needed for an ECSW
hyper-reduced ROM of the Navier–Stokes nonlinear term, following:

    - Start from:
        * Mass matrix M
        * POD basis v (from PODmodes_30.npz)
        * Snapshots X (from Snap_30.npz)
        * Tensorised "interpolation" matrices at Gauss points:
              PHIu, PHIv, dPHIudx, dPHIudy, dPHIvdx, dPHIvdy
        * Quadrature weights Wgauss (from Wgauss.txt)

    - Build, at Gauss level, the nonlinear integrand terms for u and v:
          n_u = u ∂x u + v ∂y u
          n_v = u ∂x v + v ∂y v

      both:
          • in reduced (POD) space, and
          • in full-order space (for error checking).

    - Assemble the least-squares systems for the ECSW weights for u and v:
          lhs_u x_u ≈ rhs_u, lhs_v x_v ≈ rhs_v    with x ≥ 0

      where the columns of lhs_* are contributions of each Gauss point.

    - Solve the NNLS problems with a greedy active-set strategy to get
      sparse weight vectors x_sparse_u, x_sparse_v.

    - Restrict PHI matrices to the selected Gauss points and precompute
      the hyper-reduced "transition" matrices.

    - Save all relevant ECSW structures in a NumPy .npz file:
          hr_freefem_pinball_30.npz
"""

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import diags

import funcIO as io  # rename to funcIO if you keep the old name


# =============================================================================
# 1. Greedy NNLS helper
# =============================================================================
def greedy_nnls(A: np.ndarray,
                b: np.ndarray,
                max_active: int = 200,
                verbose: bool = False) -> np.ndarray:
    """
    Greedy active-set wrapper around scipy.optimize.nnls.

    Parameters
    ----------
    A : (m, n) ndarray
        Design matrix (each column = candidate Gauss point contribution).
    b : (m,) ndarray
        Right-hand side.
    max_active : int
        Maximum number of active (non-zero) entries allowed in x.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    x : (n,) ndarray
        Non-negative solution with at most max_active non-zero entries.
    """
    m, n = A.shape
    x = np.zeros(n)
    residual = b.copy()
    active_set: list[int] = []
    inactive_set = list(range(n))  # not used explicitly, but kept for clarity

    for k in range(max_active):
        # 1) Correlation with residual
        correlations = A.T @ residual

        # Do not reconsider already active indices
        correlations[active_set] = -np.inf

        # 2) Add index with largest positive correlation
        j = int(np.argmax(correlations))
        if correlations[j] <= 0:
            if verbose:
                print("No more positive correlations, stopping early.")
            break

        active_set.append(j)
        A_active = A[:, active_set]

        # 3) Solve NNLS restricted to active set
        x_active, _ = nnls(A_active, b, maxiter=A_active.shape[1] * 100)

        # 4) Update x and residual
        x[:] = 0.0
        x[active_set] = x_active
        residual = b - A @ x
        err = np.linalg.norm(residual)

        if verbose:
            print(f"Step {k+1:3d}: selected index {j}, residual = {err:.3e}")

        # NOTE: stopping criterion is intentionally loose (1e2).
        # For a tighter fit, use e.g. 1e-2 or 1e-4 depending on your scale.
        if err < 1e-4:
            if verbose:
                print("Residual is small enough, stopping.")
            break

    return x


# =============================================================================
# 2. Helper to read FreeFEM real[int] vectors (like Wgauss.txt)
# =============================================================================
def readfield(name: str) -> np.ndarray:
    """
    Read a FreeFEM real[int] vector file with format:

        line 1 : integer N (number of entries)
        next lines: values written 5 per line (except last).

    This matches how FreeFEM prints a real[int] via 'ofstream << array'.

    Parameters
    ----------
    name : str
        Filename (e.g. "Wgauss.txt").

    Returns
    -------
    state : (N,) ndarray of float
    """
    with open(name, "r") as f:
        num = int(f.readline().strip())

    state = np.zeros(num, dtype=float)

    with open(name, "r") as f:
        _ = f.readline()  # skip header
        for j in range(num // 5):
            columns = f.readline().strip().split()
            for k in range(5):
                state[j * 5 + k] = float(columns[k])

        line = f.readline()
        if line:
            columns = line.strip().split()
            for k in range(num % 5):
                state[(num // 5) * 5 + k] = float(columns[k])

    return state


# =============================================================================
# 3. Main ECSW pipeline
# =============================================================================
def main() -> None:
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    p = 20               # number of POD modes used in the ROM
    max_active = 200     # max number of retained Gauss points per component

    # -------------------------------------------------------------------------
    # 3.1 Load mass matrix, snapshots, POD basis
    # -------------------------------------------------------------------------
    print("Loading mass matrix M ...")
    mass = io.ReadMatrix("M").tocsr()

    print("Loading snapshot matrix Snap_30 ...")
    X_full = io.ReadSnap("Snap_30")        # shape: (ndof, n_snapshots)
    # Discard first snapshots close to baseflow if desired
    X = X_full[:, 100:]
    n_snap = X.shape[1]

    print("Loading POD modes ...")
    POD = np.load("PODmodes_30.npz")
    v_full = POD["v"]                       # (ndof x r_full)
    v = v_full[:, :p]                       # restrict to first p modes

    # -------------------------------------------------------------------------
    # 3.2 Load hyper-reduction (tensorisation) matrices
    # -------------------------------------------------------------------------
    print("Loading tensorised PHI matrices ...")
    PHIu = io.ReadMatrix("PHIu").tocsr()
    PHIv = io.ReadMatrix("PHIv").tocsr()
    dPHIudx = io.ReadMatrix("dPHIudx").tocsr()
    dPHIudy = io.ReadMatrix("dPHIudy").tocsr()
    dPHIvdx = io.ReadMatrix("dPHIvdx").tocsr()
    dPHIvdy = io.ReadMatrix("dPHIvdy").tocsr()

    # Project interpolation matrices onto POD basis
    PHIu_pod = PHIu @ v
    PHIv_pod = PHIv @ v
    dPHIudx_pod = dPHIudx @ v
    dPHIudy_pod = dPHIudy @ v
    dPHIvdx_pod = dPHIvdx @ v
    dPHIvdy_pod = dPHIvdy @ v

    # "Test" side: vᵀ * PHIᵀ
    pod_t_PHIu_t = v.T @ PHIu.T
    pod_t_PHIv_t = v.T @ PHIv.T

    # Quadrature weights at Gauss points
    Wgauss = readfield("Wgauss.txt")   # shape: (N_gauss,)

    # -------------------------------------------------------------------------
    # 3.3 Project snapshots in POD space and assemble Gauss-level terms
    # -------------------------------------------------------------------------
    print("Assembling Gauss-level nonlinear terms ...")

    # POD coefficients of snapshots: a(t) = Vᵀ M X
    proj_states = v.T @ (mass @ X)  # shape: (p, n_snap)

    n_pod = p
    N_gauss = PHIu_pod.shape[0]
    N_snapshots = proj_states.shape[1]

    print(f"  PHIu_pod shape:      {PHIu_pod.shape}")
    print(f"  proj_states shape:   {proj_states.shape}")

    # Evaluate u, v and derivatives at Gauss points in reduced representation
    a_gauss_u = PHIu_pod @ proj_states       # (N_gauss x n_snap)
    a_gauss_v = PHIv_pod @ proj_states
    a_gauss_dxu = dPHIudx_pod @ proj_states
    a_gauss_dyu = dPHIudy_pod @ proj_states
    a_gauss_dxv = dPHIvdx_pod @ proj_states
    a_gauss_dyv = dPHIvdy_pod @ proj_states

    # Same in full-order space (for "true" reference)
    gauss_u = PHIu @ X
    gauss_v = PHIv @ X
    gauss_dxu = dPHIudx @ X
    gauss_dyu = dPHIudy @ X
    gauss_dxv = dPHIvdx @ X
    gauss_dyv = dPHIvdy @ X

    # Nonlinear integrands at Gauss points (reduced vs full)
    # n_u = u ∂x u + v ∂y u
    # n_v = u ∂x v + v ∂y v
    nn_gauss_u = a_gauss_u * a_gauss_dxu + a_gauss_v * a_gauss_dyu
    nn_gauss_v = a_gauss_u * a_gauss_dxv + a_gauss_v * a_gauss_dyv

    nn_gauss_true_u = gauss_u * gauss_dxu + gauss_v * gauss_dyu
    nn_gauss_true_v = gauss_u * gauss_dxv + gauss_v * gauss_dyv

    # Multiply by quadrature weights
    nn_gauss_u_times_omega = Wgauss[:, None] * nn_gauss_u
    nn_gauss_v_times_omega = Wgauss[:, None] * nn_gauss_v

    nn_gauss_true_u_times_omega = Wgauss[:, None] * nn_gauss_true_u
    nn_gauss_true_v_times_omega = Wgauss[:, None] * nn_gauss_true_v

    print(f"  nn_gauss_u_times_omega shape: {nn_gauss_u_times_omega.shape}")

    # ROM right-hand sides (reduced integrals) for each snapshot:
    # rhs_u(k,t) = - Σ_g ω_g (Φᵤᵀ v)_k(g) * n_u(g,t)
    rhs_u_rectangle = pod_t_PHIu_t @ nn_gauss_u_times_omega        # (p x n_snap)
    rhs_v_rectangle = pod_t_PHIv_t @ nn_gauss_v_times_omega

    rhs_true_u_rectangle = pod_t_PHIu_t @ nn_gauss_true_u_times_omega
    rhs_true_v_rectangle = pod_t_PHIv_t @ nn_gauss_true_v_times_omega

    # Flatten in time & mode to form a big least-squares system
    rhs_u_flat = -rhs_u_rectangle.T.reshape(-1)
    rhs_v_flat = -rhs_v_rectangle.T.reshape(-1)

    rhs_true_u_flat = -rhs_true_u_rectangle.T.reshape(-1)
    rhs_true_v_flat = -rhs_true_v_rectangle.T.reshape(-1)

    print(f"  rhs_u_flat shape: {rhs_u_flat.shape}")

    # Build design matrices (one column per Gauss point)
    # lhs_u(j,t,g) = - (vᵀ Φᵤᵀ)_j(g) * n_u(g,t)
    lhs_u_all = -pod_t_PHIu_t[:, :, None] * nn_gauss_u[None, :, :]  # (p, N_gauss, n_snap)
    lhs_v_all = -pod_t_PHIv_t[:, :, None] * nn_gauss_v[None, :, :]  # (p, N_gauss, n_snap)

    # Reshape to (p * n_snap, N_gauss)
    lhs_u = lhs_u_all.transpose(2, 0, 1).reshape(-1, N_gauss)
    lhs_v = lhs_v_all.transpose(2, 0, 1).reshape(-1, N_gauss)

    print(f"  lhs_u shape (rows, gauss): {lhs_u.shape}")

    # -------------------------------------------------------------------------
    # 3.4 Solve NNLS for u and v ECSW weights
    # -------------------------------------------------------------------------
    print(f"\nComputing ECSW weights with p={p} POD modes, max_active={max_active}\n")

    print("  => Solving NNLS for u-component")
    x_sparse_u = greedy_nnls(lhs_u, rhs_u_flat, max_active=max_active, verbose=True)

    residual_u = lhs_u @ x_sparse_u - rhs_u_flat
    err_u = np.linalg.norm(residual_u)

    residual_u_true = lhs_u @ x_sparse_u - rhs_true_u_flat
    err_u_true = np.linalg.norm(residual_u_true)

    print(f"  Selected {np.count_nonzero(x_sparse_u)} Gauss points (u) with residual {err_u:.3e}")
    print(f"  Residual vs full nonlinearity (u): {err_u_true:.3e}\n")

    print("  => Solving NNLS for v-component")
    x_sparse_v = greedy_nnls(lhs_v, rhs_v_flat, max_active=max_active, verbose=True)

    residual_v = lhs_v @ x_sparse_v - rhs_v_flat
    err_v = np.linalg.norm(residual_v)

    residual_v_true = lhs_v @ x_sparse_v - rhs_true_v_flat
    err_v_true = np.linalg.norm(residual_v_true)

    print(f"  Selected {np.count_nonzero(x_sparse_v)} Gauss points (v) with residual {err_v:.3e}")
    print(f"  Residual vs full nonlinearity (v): {err_v_true:.3e}\n")

    # -------------------------------------------------------------------------
    # 3.5 Build hyper-reduced matrices for u and v
    # -------------------------------------------------------------------------
    print("Building hyper-reduced transition matrices ...")

    # --- u-component ---
    lhs_u_indices = np.nonzero(x_sparse_u)[0]
    w_sparse_u = x_sparse_u[lhs_u_indices]
    Wtilde_u = diags(w_sparse_u)     # diagonal ECSW weights (sparse)

    PHIu_red_u = PHIu[lhs_u_indices, :]
    PHIv_red_u = PHIv[lhs_u_indices, :]
    dPHIudx_red_u = dPHIudx[lhs_u_indices, :]
    dPHIudy_red_u = dPHIudy[lhs_u_indices, :]

    PHIu_pod_red_u = PHIu_red_u @ v
    PHIv_pod_red_u = PHIv_red_u @ v
    dPHIudx_pod_red_u = dPHIudx_red_u @ v
    dPHIudy_pod_red_u = dPHIudy_red_u @ v

    pod_t_PHIu_t_Wu_red_u = v.T @ PHIu_red_u.T @ Wtilde_u   # (p x N_red_u)

    print(f"  U-side hyper-reduced PHI size: {PHIu_pod_red_u.shape}")
    print(f"  U-side test matrix size:       {pod_t_PHIu_t_Wu_red_u.shape}")

    # --- v-component ---
    lhs_v_indices = np.nonzero(x_sparse_v)[0]
    w_sparse_v = x_sparse_v[lhs_v_indices]
    Wtilde_v = diags(w_sparse_v)

    PHIu_red_v = PHIu[lhs_v_indices, :]
    PHIv_red_v = PHIv[lhs_v_indices, :]
    dPHIvdx_red_v = dPHIvdx[lhs_v_indices, :]
    dPHIvdy_red_v = dPHIvdy[lhs_v_indices, :]

    PHIu_pod_red_v = PHIu_red_v @ v
    PHIv_pod_red_v = PHIv_red_v @ v
    dPHIvdx_pod_red_v = dPHIvdx_red_v @ v
    dPHIvdy_pod_red_v = dPHIvdy_red_v @ v

    pod_t_PHIv_t_Wu_red_v = v.T @ PHIv_red_v.T @ Wtilde_v

    print(f"  V-side hyper-reduced PHI size: {PHIu_pod_red_v.shape}")
    print(f"  V-side test matrix size:       {pod_t_PHIv_t_Wu_red_v.shape}")

    # -------------------------------------------------------------------------
    # 3.6 Save everything to NumPy .npz
    # -------------------------------------------------------------------------
    print("\nSaving ECSW data to hr_freefem_pinball_30.npz ...")

    np.savez(
        "hr_freefem_pinball_30.npz",
        # ECSW weights and indices
        x_sparse_u=x_sparse_u,
        x_sparse_v=x_sparse_v,
        lhs_u_indices=lhs_u_indices,
        lhs_v_indices=lhs_v_indices,
        w_sparse_u=w_sparse_u,
        w_sparse_v=w_sparse_v,
        # Hyper-reduced POD-side matrices (u)
        PHIu_pod_red_u=PHIu_pod_red_u,
        PHIv_pod_red_u=PHIv_pod_red_u,
        dPHIudx_pod_red_u=dPHIudx_pod_red_u,
        dPHIudy_pod_red_u=dPHIudy_pod_red_u,
        pod_t_PHIu_t_Wu_red_u=pod_t_PHIu_t_Wu_red_u,
        # Hyper-reduced POD-side matrices (v)
        PHIu_pod_red_v=PHIu_pod_red_v,
        PHIv_pod_red_v=PHIv_pod_red_v,
        dPHIvdx_pod_red_v=dPHIvdx_pod_red_v,
        dPHIvdy_pod_red_v=dPHIvdy_pod_red_v,
        pod_t_PHIv_t_Wu_red_v=pod_t_PHIv_t_Wu_red_v,
    )

    print("Done.")


if __name__ == "__main__":
    main()
