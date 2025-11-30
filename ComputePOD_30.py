"""
Build POD modes for the fluidic pinball (Re = 30)
=================================================

This script:
    - loads the mass matrix M from FreeFEM (converted by freefem_io_utils.WriteMatrix)
    - loads the snapshot matrix Snap_30 (num_dofs x num_snapshots)
    - computes POD modes using the method of snapshots:
          C = Xᵀ M X
          C = V Σ Vᵀ
          Φ = X V Σ^{-1/2}
    - saves:
          PODmodes_30.npz  (v = modes, s = singular values)
    - produces:
          pinball_30_pod_eigenvals.png    (eigenvalue decay)
          pinball_30_PODcoefs_dynamics.png (first POD coefficients vs snapshot index)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

import funcIO as io

def main() -> None:
    # ---------------------------------------------------------------------
    # 1. Load mass matrix and snapshot matrix
    # ---------------------------------------------------------------------
    print("Loading mass matrix M ...")
    M_coo = io.ReadMatrix("M")
    M = M_coo.tocsr()

    print("Loading snapshot matrix Snap_30 ...")
    X = io.ReadSnap("Snap_30")     # shape: (num_dofs, num_snapshots)

    num_dofs, n_snap = X.shape
    print(f"Snapshot matrix size: {num_dofs} DOFs x {n_snap} snapshots")

    # ---------------------------------------------------------------------
    # 2. Compute POD via method of snapshots
    # ---------------------------------------------------------------------
    print("Computing POD modes (method of snapshots) ...")

    # Maximum number of POD modes (you can tune this)
    p_max = 100
    p = min(p_max, n_snap)

    # Correlation matrix C = Xᵀ M X  (n_snap x n_snap)
    C = X.T @ (M @ X)

    # SVD of the correlation matrix: C = V Σ Vᵀ
    #   v1: eigenvectors  (n_snap x n_snap)
    #   s : singular values (eigenvalues λ_i of the correlation matrix)
    v1, s, _ = la.svd(C)
    v1 = v1[:, :p]
    s = s[:p]

    # ---------------------------------------------------------------------
    # 3. Energy content of POD modes
    # ---------------------------------------------------------------------
    print("Energy content of the first POD modes:")
    temp_sum = 0.0
    full_sum = np.sum(s)

    for i, e in enumerate(s, start=1):
        temp_sum += 100.0 * e / full_sum
        print(f"  Mode {i:3d}: cumulative energy = {temp_sum:8.4f} %")

    # ---------------------------------------------------------------------
    # 4. Plot eigenvalue decay
    # ---------------------------------------------------------------------
    eigvals_fig = Path("pinball_30_pod_eigenvals.png")
    plt.figure(figsize=(15, 7))
    plt.semilogy(s, linestyle="--", marker="o", label="Base flow")
    plt.xlabel("POD mode index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(eigvals_fig, dpi=200)
    plt.close()
    print(f"Saved eigenvalue decay to {eigvals_fig}")

    # ---------------------------------------------------------------------
    # 5. Build POD modes: Φ = X V Σ^{-1/2}
    # ---------------------------------------------------------------------
    # Avoid division by zero in case of tiny singular values
    s_safe = np.where(s > 0, s, 1.0)
    Sigma_minus_half = np.diag(s_safe ** (-0.5))

    # POD modes (num_dofs x p)
    v = X @ (v1 @ Sigma_minus_half)

    pod_file = Path("PODmodes_30.npz")
    np.savez(pod_file, v=v, s=s)
    print(f"Saved POD modes and singular values to {pod_file}")

    # ---------------------------------------------------------------------
    # 6. Plot POD temporal coefficients for first 3–4 modes
    # ---------------------------------------------------------------------
    # Coefficients a_i(t_j) = <ϕ_i, x_j>_M
    # Here: A = Vᵀ M X   (p x n_snap)
    num_coeffs_to_plot = min(4, p)
    A = v[:, :num_coeffs_to_plot].T @ (M @ X)  # shape: (num_coeffs_to_plot, n_snap)

    coef_fig = Path("pinball_30_PODcoefs_dynamics.png")
    plt.figure()
    for i in range(num_coeffs_to_plot):
        plt.plot(A[i], label=f"Mode {i+1}")
    plt.xlabel("Snapshot index")
    plt.ylabel("POD coefficient (a_i)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(coef_fig, dpi=200)
    plt.close()
    print(f"Saved POD coefficient dynamics to {coef_fig}")

if __name__ == "__main__":
    main()
