# -*- coding: utf-8 -*-
"""
Compare POD ROM vs ECSW hyper-reduced ROM (Re = 30) and export PyTorch operators
================================================================================

This script:

1. Loads:
   - Mass matrix M
   - Snapshot matrix Snap_30
   - POD modes (PODmodes_30.npz)
   - ROM operators (ROM_30.npz: A, N)
   - ECSW hyper-reduced operators (hr_freefem_pinball_30.npz)

2. Builds and simulates:
   - Standard cubic POD ROM: dy/dt = A_p y + f_N(y)
   - ECSW hyper-reduced ROM: dy/dt = A_p y + f_ECSW(y)

3. Compares:
   - POD coefficients (FOM projection vs ROM vs ECSW ROM) for the first 3 modes

4. Saves:
   - A PyTorch dictionary "hr_freefem_pinball_30.pt" with all ECSW operators
     and reduced matrices needed for GPU code.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import funcIO as io  # rename to funcIO if needed

# =============================================================================
# 1. Nonlinear RHS helpers
# =============================================================================
def dydt_nonlinear(y: np.ndarray, Np: np.ndarray) -> np.ndarray:
    """
    Full cubic nonlinear term for standard ROM:

        f_i(y) = yᵀ Np[:,:,i] y

    Parameters
    ----------
    y : (p,) ndarray
        Reduced state.
    Np : (p, p, p) ndarray
        Cubic tensor of nonlinear coefficients.

    Returns
    -------
    rhs : (p,) ndarray
    """
    p = y.shape[0]
    rhs = np.zeros_like(y)
    for i in range(p):
        rhs[i] = y.T @ (Np[:, :, i] @ y)
    return rhs


def compute_hyperreduced_convection_rhs(
    a: np.ndarray,
    PHIu_pod_red_u: np.ndarray,
    PHIv_pod_red_u: np.ndarray,
    dPHIudx_pod_red_u: np.ndarray,
    dPHIudy_pod_red_u: np.ndarray,
    pod_t_PHIu_t_Wu_red_u: np.ndarray,
    PHIu_pod_red_v: np.ndarray,
    PHIv_pod_red_v: np.ndarray,
    dPHIvdx_pod_red_v: np.ndarray,
    dPHIvdy_pod_red_v: np.ndarray,
    pod_t_PHIv_t_Wu_red_v: np.ndarray,
) -> np.ndarray:
    """
    ECSW hyper-reduced nonlinear term for the ROM.

    For each reduced state a (POD coefficients), we compute:

        n_u(g) = u(g) ∂x u(g) + v(g) ∂y u(g)
        n_v(g) = u(g) ∂x v(g) + v(g) ∂y v(g)

    on the selected Gauss points, then apply:

        f_ECSW(a) = - (Vᵀ Φᵤᵀ W̃ᵤ n_u + Vᵀ Φᵥᵀ W̃ᵥ n_v)

    where all the pre-multiplications by Vᵀ and W̃ are baked into
    pod_t_PHI*_t_W*_red_* matrices.

    Parameters
    ----------
    a : (p,) ndarray
        Reduced state (POD coefficients).
    PHI*_pod_red_* : arrays
        Hyper-reduced POD-side interpolation matrices, restricted to
        ECSW Gauss points.
    pod_t_PHI*_t_W*_red_* : arrays
        Hyper-reduced test-side matrices (Vᵀ Φᵀ W̃).

    Returns
    -------
    rhs : (p,) ndarray
        Hyper-reduced nonlinear contribution.
    """

    # --- u-equation nonlinear term ---
    uu = PHIu_pod_red_u @ a         # u at selected Gauss points
    dxuu = dPHIudx_pod_red_u @ a    # ∂x u
    vu = PHIv_pod_red_u @ a         # v at Gauss points
    dyuu = dPHIudy_pod_red_u @ a    # ∂y u

    fu = uu * dxuu + vu * dyuu      # n_u(g)

    # --- v-equation nonlinear term ---
    uv = PHIu_pod_red_v @ a
    dxvv = dPHIvdx_pod_red_v @ a
    vv = PHIv_pod_red_v @ a
    dyvv = dPHIvdy_pod_red_v @ a

    fv = uv * dxvv + vv * dyvv      # n_v(g)

    # Test-side (already includes Vᵀ and weights W̃)
    return - (pod_t_PHIu_t_Wu_red_u @ fu + pod_t_PHIv_t_Wu_red_v @ fv)


# =============================================================================
# 2. Main comparison and export script
# =============================================================================
def main() -> None:
    # -------------------------------------------------------------------------
    # 2.1 Settings
    # -------------------------------------------------------------------------
    p = 20               # ROM dimension (number of POD modes)
    start_snap = 100     # snapshot index at which we start (to skip transients)

    # FOM time stepping parameters (must match FreeFEM / snapshot generation)
    dt_FOM = 0.0075       # FOM time step used in time_march_eigenmode1_Re30.edp
    step = 20            # snapshots taken every 'step' time steps
    snapshot_dt = dt_FOM * step  # physical time between snapshots

    # ROM time step (can be smaller than snapshot spacing for accuracy)
    dt_rom = 0.0075

    # -------------------------------------------------------------------------
    # 2.2 Load FOM data and POD basis
    # -------------------------------------------------------------------------
    print("Loading mass matrix M ...")
    M = io.ReadMatrix("M").tocsr()

    print("Loading snapshot matrix Snap_30 ...")
    X_full = io.ReadSnap("Snap_30")        # shape: (ndof, n_snapshots)
    ndof, n_snapshots = X_full.shape

    print("Loading POD modes ...")
    POD = np.load("PODmodes_30.npz")
    v_full = POD["v"]                       # (ndof x r_full)
    r_full = v_full.shape[1]
    p = min(p, r_full)
    v = v_full[:, :p]

    # Trim snapshots (start_snap to end)
    X = X_full[:, start_snap:]
    n_snaps_trim = X.shape[1]

    # FOM time vector for trimmed snapshots
    t_FOM = (start_snap + np.arange(n_snaps_trim)) * snapshot_dt

    # FOM projection in POD space: a(t) = Vᵀ M X
    print("Projecting FOM snapshots onto POD modes ...")
    MX = M @ X
    yy = v.T @ MX            # shape: (p x n_snaps_trim)

    # -------------------------------------------------------------------------
    # 2.3 Define ROM time grids
    # -------------------------------------------------------------------------
    T0 = t_FOM[0]
    Tend = t_FOM[-1]

    n_steps_rom = int(np.round((Tend - T0) / dt_rom)) + 1
    t_ROM = T0 + np.arange(n_steps_rom) * dt_rom

    print(f"FOM window: [{T0:.3f}, {Tend:.3f}] with {n_snaps_trim} snapshots")
    print(f"ROM window: [{t_ROM[0]:.3f}, {t_ROM[-1]:.3f}] with {n_steps_rom} steps")

    # -------------------------------------------------------------------------
    # 2.4 Load ROM operators and ECSW operators
    # -------------------------------------------------------------------------
    print("Loading ROM operators (A, N) ...")
    ROM = np.load("ROM_30.npz")
    A_full = ROM["A"]
    N_full = ROM["N"]

    Ap = A_full[:p, :p]
    Np = N_full[:p, :p, :p]

    print("Loading ECSW operators ...")
    HR = np.load("hr_freefem_pinball_30.npz", allow_pickle=True)

    PHIu_pod_red_u = HR["PHIu_pod_red_u"]
    PHIv_pod_red_u = HR["PHIv_pod_red_u"]
    dPHIudx_pod_red_u = HR["dPHIudx_pod_red_u"]
    dPHIudy_pod_red_u = HR["dPHIudy_pod_red_u"]
    pod_t_PHIu_t_Wu_red_u = HR["pod_t_PHIu_t_Wu_red_u"]

    PHIu_pod_red_v = HR["PHIu_pod_red_v"]
    PHIv_pod_red_v = HR["PHIv_pod_red_v"]
    dPHIvdx_pod_red_v = HR["dPHIvdx_pod_red_v"]
    dPHIvdy_pod_red_v = HR["dPHIvdy_pod_red_v"]
    pod_t_PHIv_t_Wu_red_v = HR["pod_t_PHIv_t_Wu_red_v"]

    x_sparse_u = HR["x_sparse_u"]
    x_sparse_v = HR["x_sparse_v"]

    n_u_pts = int(np.count_nonzero(x_sparse_u))
    n_v_pts = int(np.count_nonzero(x_sparse_v))

    print(f"ECSW: {n_u_pts} Gauss points for u, {n_v_pts} for v")

    # -------------------------------------------------------------------------
    # 2.5 Standard ROM simulation (full cubic tensor)
    # -------------------------------------------------------------------------
    print("\nSimulating standard cubic ROM ...")

    y = np.zeros((p, n_steps_rom))
    # initial condition from first trimmed snapshot
    y[:, 0] = yy[:, 0]

    I_p = np.eye(p)

    total_start = time.time()

    # Step 0: 1st-order scheme
    step_start = time.time()
    nonlin = dydt_nonlinear(y[:, 0], Np)
    step_mid = time.time()
    print(f"Step 0 ROM nonlin took {step_mid - step_start:.6e} s")

    y[:, 1] = np.linalg.solve(I_p - dt_rom * Ap, y[:, 0] + dt_rom * nonlin)

    step_end = time.time()
    print(f"Step 0 ROM complete took {step_end - step_start:.6e} s")

    # Remaining steps: BDF2-like
    for j in range(2, n_steps_rom):
        non_lin_prev = nonlin
        nonlin = dydt_nonlinear(y[:, j - 1], Np)

        rhs = 4.0 * y[:, j - 1] - y[:, j - 2] \
              + 4.0 * dt_rom * nonlin - 2.0 * dt_rom * non_lin_prev

        y[:, j] = np.linalg.solve(3.0 * I_p - 2.0 * dt_rom * Ap, rhs)

    total_end = time.time()
    print(f"Total ROM simulation time: {total_end - total_start:.6e} s")

    # -------------------------------------------------------------------------
    # 2.6 ECSW hyper-reduced ROM simulation
    # -------------------------------------------------------------------------
    print("\nSimulating ECSW hyper-reduced ROM ...")

    y2 = np.zeros((p, n_steps_rom))
    y2[:, 0] = yy[:, 0]

    total_start = time.time()

    # Step 0
    step_start = time.time()
    nonlin = compute_hyperreduced_convection_rhs(
        y2[:, 0],
        PHIu_pod_red_u,
        PHIv_pod_red_u,
        dPHIudx_pod_red_u,
        dPHIudy_pod_red_u,
        pod_t_PHIu_t_Wu_red_u,
        PHIu_pod_red_v,
        PHIv_pod_red_v,
        dPHIvdx_pod_red_v,
        dPHIvdy_pod_red_v,
        pod_t_PHIv_t_Wu_red_v,
    )
    step_mid = time.time()
    print(f"Step 0 HR nonlin took {step_mid - step_start:.6e} s")

    y2[:, 1] = np.linalg.solve(I_p - dt_rom * Ap, y2[:, 0] + dt_rom * nonlin)

    step_end = time.time()
    print(f"Step 0 HR complete took {step_end - step_start:.6e} s")

    # Remaining steps: BDF2-like
    for j in range(2, n_steps_rom):
        non_lin_prev = nonlin
        nonlin = compute_hyperreduced_convection_rhs(
            y2[:, j - 1],
            PHIu_pod_red_u,
            PHIv_pod_red_u,
            dPHIudx_pod_red_u,
            dPHIudy_pod_red_u,
            pod_t_PHIu_t_Wu_red_u,
            PHIu_pod_red_v,
            PHIv_pod_red_v,
            dPHIvdx_pod_red_v,
            dPHIvdy_pod_red_v,
            pod_t_PHIv_t_Wu_red_v,
        )

        rhs = 4.0 * y2[:, j - 1] - y2[:, j - 2] \
              + 4.0 * dt_rom * nonlin - 2.0 * dt_rom * non_lin_prev

        y2[:, j] = np.linalg.solve(3.0 * I_p - 2.0 * dt_rom * Ap, rhs)

    total_end = time.time()
    print(f"Total HR simulation time: {total_end - total_start:.6e} s")

    # -------------------------------------------------------------------------
    # 2.7 Align ROM outputs with FOM times for plotting
    # -------------------------------------------------------------------------
    # Sample ROM at snapshot times for visual comparison
    rom_indices = np.round((t_FOM - T0) / dt_rom).astype(int)
    rom_indices = np.clip(rom_indices, 0, n_steps_rom - 1)

    y_sampled = y[:, rom_indices]
    y2_sampled = y2[:, rom_indices]

    # -------------------------------------------------------------------------
    # 2.8 Plot comparison: FOM projection vs ROM vs ECSW ROM
    # -------------------------------------------------------------------------
    print("\nPlotting POD coefficients: FOM vs ROM vs ECSW ROM ...")

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    modes = [0, 1, 2]
    labels = [
        "FOM projected",
        f"POD ROM ({p} modes)",
        f"ECSW ROM ({p} modes, {n_u_pts} u-pts, {n_v_pts} v-pts)",
    ]

    for i, mode in enumerate(modes):
        axs[i].plot(t_FOM, yy[mode, :], color="black", label=labels[0])
        axs[i].plot(t_FOM, y_sampled[mode, :], color="green", label=labels[1])
        axs[i].plot(t_FOM, y2_sampled[mode, :], color="red", label=labels[2])
        axs[i].set_ylabel(f"Mode {mode+1}")
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time")
    fig.suptitle("POD coefficients: FOM vs POD ROM vs ECSW-ROM", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("HR_pinball_30_HR_comparison.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # 2.9 Save PyTorch data for GPU work
    # -------------------------------------------------------------------------
    print("\nSaving PyTorch ECSW operator package to hr_freefem_pinball_30.pt ...")

    hr_freefem_pinball_30 = {
        "PHIu_pod_red_u": torch.tensor(PHIu_pod_red_u, dtype=torch.float32),
        "PHIv_pod_red_u": torch.tensor(PHIv_pod_red_u, dtype=torch.float32),
        "dPHIudx_pod_red_u": torch.tensor(dPHIudx_pod_red_u, dtype=torch.float32),
        "dPHIudy_pod_red_u": torch.tensor(dPHIudy_pod_red_u, dtype=torch.float32),
        "pod_t_PHIu_t_Wu_red_u": torch.tensor(
            pod_t_PHIu_t_Wu_red_u, dtype=torch.float32
        ),
        "PHIu_pod_red_v": torch.tensor(PHIu_pod_red_v, dtype=torch.float32),
        "PHIv_pod_red_v": torch.tensor(PHIv_pod_red_v, dtype=torch.float32),
        "dPHIvdx_pod_red_v": torch.tensor(dPHIvdx_pod_red_v, dtype=torch.float32),
        "dPHIvdy_pod_red_v": torch.tensor(dPHIvdy_pod_red_v, dtype=torch.float32),
        "pod_t_PHIv_t_Wu_red_v": torch.tensor(
            pod_t_PHIv_t_Wu_red_v, dtype=torch.float32
        ),
        "Ap": torch.tensor(Ap, dtype=torch.float32),
        "dt_rom": float(dt_rom),
        "x_sparse_u": torch.tensor(x_sparse_u, dtype=torch.float32),
        "x_sparse_v": torch.tensor(x_sparse_v, dtype=torch.float32),
    }

    torch.save(hr_freefem_pinball_30, "hr_freefem_pinball_30.pt")

    print("Done.\n")

if __name__ == "__main__":
    main()
