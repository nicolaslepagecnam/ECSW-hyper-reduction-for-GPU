# -*- coding: utf-8 -*-
"""
ROM simulation and PyTorch export for the fluidic pinball (Re = 30)
===================================================================

This script:
    1. Loads:
        - Mass matrix M
        - ROM operators (A, N) from ROM_30.npz
        - POD modes from PODmodes_30.npz
        - FOM snapshots from Snap_30.npz
        - Drag / lift vectors from FreeFEM

    2. Builds a reduced-order model (ROM) of dimension p:
        dy/dt = A_p y + f_nl(y),   with f_nl cubic (from N_p)

    3. Runs a ROM time integration and compares:
        - POD coefficients (FOM vs ROM)
        - Drag / lift coefficients (FOM vs ROM)

    4. Saves:
        - A PyTorch dataset of FOM snapshots
        - A PyTorch "ROM package" (modes, M, N, A, drag/lift) for later use
          in PyTorch models.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

import funcIO as io   # your I/O helper module


# ---------------------------------------------------------------------------#
# 1. Nonlinear reduced RHS
# ---------------------------------------------------------------------------#
def dydt_nonlinear(y: np.ndarray, Np: np.ndarray) -> np.ndarray:
    """
    Compute nonlinear term for reduced-order model.

    Parameters
    ----------
    y : (p,) ndarray
        Current reduced state vector.
    Np : (p, p, p) ndarray
        Reduced nonlinear tensor N_p[j,k,i] such that
            f_i(y) = yᵀ Np[:,:,i] y

    Returns
    -------
    rhs : (p,) ndarray
        Nonlinear contribution dy/dt|_nl.
    """
    p = y.shape[0]
    rhs = np.zeros_like(y)
    for i in range(p):
        rhs[i] = y.T @ (Np[:, :, i] @ y)
    return rhs


def main() -> None:
    # -----------------------------------------------------------------------
    # 2. Global parameters (must match your FOM simulation)
    # -----------------------------------------------------------------------
    num_snap = 20000       # total number of time steps in FreeFEM simulation
    dt = 0.0075;             # FOM time step
    step = 20              # snapshot interval in time steps
    snapshot_dt = dt * step  # physical time between snapshots

    # Dimension of the ROM (number of POD modes)
    p = 20

    # Starting snapshot index (skip very early transients close to baseflow)
    start_snap = 100

    # -----------------------------------------------------------------------
    # 3. Load matrices, ROM operators, modes, snapshots
    # -----------------------------------------------------------------------
    print("Loading mass matrix M ...")
    M = io.ReadMatrix("M")        # SciPy sparse COO matrix

    print("Loading ROM operators (A, N) ...")
    ROM = np.load("ROM_30.npz")
    A_full = ROM["A"]   # (r_full x r_full)
    N_full = ROM["N"]   # (r_full x r_full x r_full)

    print("Loading POD modes ...")
    POD = np.load("PODmodes_30.npz")
    v_full = POD["v"]   # (ndof x r_full)
    s = POD["s"]

    # Keep first p modes for the ROM
    r_full = v_full.shape[1]
    p = min(p, r_full)
    v = v_full[:, :p]
    print(f"Using p = {p} POD modes out of r_full = {r_full}")

    # Reduced operators
    Ap = A_full[:p, :p]
    Np = N_full[:p, :p, :p]

    print("Loading FOM snapshots ...")
    X_full = io.ReadSnap("Snap_30")   # shape: (ndof, num_snapshots)
    ndof, n_snaps = X_full.shape
    print(f"Snapshot matrix size: {ndof} DOFs x {n_snaps} snapshots")

    # Full time vector for FOM (all snapshots)
    t_full = np.arange(n_snaps) * snapshot_dt

    # -----------------------------------------------------------------------
    # 4. Trim snapshots to a window starting at `start_snap`
    # -----------------------------------------------------------------------
    X = X_full[:, start_snap:]   # (ndof x n_snaps_trim)
    n_snaps_trim = X.shape[1]
    t_FOM = t_full[start_snap:]

    # Time grid for ROM integration: same [t0, t_end], smaller dt_rom if desired
    dt_rom = dt   # can be chosen differently; here we keep it simple
    T0 = t_FOM[0]
    Tend = t_FOM[-1]
    n_steps_rom = int(np.round((Tend - T0) / dt_rom)) + 1
    t_ROM = T0 + np.arange(n_steps_rom) * dt_rom

    print(f"FOM time window: [{T0:.3f}, {Tend:.3f}] with {n_snaps_trim} snapshots")
    print(f"ROM time window: [{t_ROM[0]:.3f}, {t_ROM[-1]:.3f}] with {n_steps_rom} steps")

    # -----------------------------------------------------------------------
    # 5. Ground-truth projection in POD space
    # -----------------------------------------------------------------------
    print("Projecting snapshots onto POD modes ...")
    # M is sparse (coo); M @ X uses matrix-vector over snapshots
    MX = M @ X   # (ndof x n_snaps_trim)
    state_proj = v.T @ MX  # (p x n_snaps_trim)

    # -----------------------------------------------------------------------
    # 6. ROM integration
    # -----------------------------------------------------------------------
    print("Integrating ROM ...")

    rom_pred = np.zeros((p, n_steps_rom))
    # Initial condition: POD coefficients of first trimmed snapshot
    rom_pred[:, 0] = state_proj[:, 0]

    # Time-stepping: first step = 1st-order, then BDF2-type scheme.
    I_p = np.eye(p)

    # 1st step (BDF1 / explicit-nonlinear)
    y0 = rom_pred[:, 0]
    rhs0 = dydt_nonlinear(y0, Np)
    rom_pred[:, 1] = np.linalg.solve(I_p - dt_rom * Ap,
                                     y0 + dt_rom * rhs0)

    # Subsequent steps (BDF2-like)
    for j in range(2, n_steps_rom):
        y_nm1 = rom_pred[:, j - 1]   # y^{n-1}
        y_nm2 = rom_pred[:, j - 2]   # y^{n-2}
        f_nm1 = dydt_nonlinear(y_nm1, Np)
        f_nm2 = dydt_nonlinear(y_nm2, Np)

        rhs = 4.0 * y_nm1 - y_nm2 + 4.0 * dt_rom * f_nm1 - 2.0 * dt_rom * f_nm2
        rom_pred[:, j] = np.linalg.solve(3.0 * I_p - 2.0 * dt_rom * Ap, rhs)

    # -----------------------------------------------------------------------
    # 7. Drag / lift: FOM vs ROM
    # -----------------------------------------------------------------------
    print("Loading drag/lift vectors ...")
    # NOTE: adapt filenames if needed; here we assume Re=30
    dragvec = io.readvect("DragVec_30.txt")  # shape (ndof,)
    liftvec = io.readvect("LiftVec_30.txt")  # shape (ndof,)

    # Reduced drag/lift vectors in POD space (using first 100 modes for safety)
    r_draglift = min(100, r_full)
    v_draglift = v_full[:, :r_draglift]

    red_dragvec_full = dragvec @ v_draglift     # (r_draglift,)
    red_liftvec_full = liftvec @ v_draglift     # (r_draglift,)

    # True drag/lift from FOM states
    drag_true = dragvec @ X        # (n_snaps_trim,)
    lift_true = liftvec @ X        # (n_snaps_trim,)

    # Match ROM sampling to FOM snapshots for plotting
    rom_stride = int(round(snapshot_dt / dt_rom))  # e.g. 0.5 / 0.025 = 20
    rom_indices = np.arange(0, n_steps_rom, rom_stride)
    rom_indices = rom_indices[rom_indices < n_steps_rom]

    drag_rom = red_dragvec_full[:p] @ rom_pred[:, rom_indices]  # (len(rom_indices),)
    lift_rom = red_liftvec_full[:p] @ rom_pred[:, rom_indices]  # (len(rom_indices),)
    t_ROM_sampled = t_ROM[rom_indices]

    # -----------------------------------------------------------------------
    # 8. Plot: POD coefficients and drag/lift
    # -----------------------------------------------------------------------
    print("Plotting ROM vs FOM ...")

    fig = plt.figure(figsize=(25, 15))
    gs = GridSpec(6, 2, width_ratios=[1, 1])

    # Left column: POD coefficients of the first 3 modes
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(t_FOM, state_proj[0, :], label="FOM mode 1")
    ax1.plot(t_ROM_sampled, rom_pred[0, rom_indices], label="ROM mode 1")
    ax1.set_title("POD Mode 1 Evolution")
    ax1.set_xlabel("Time")
    ax1.legend()

    ax2 = fig.add_subplot(gs[2:4, 0])
    ax2.plot(t_FOM, state_proj[1, :], label="FOM mode 2")
    ax2.plot(t_ROM_sampled, rom_pred[1, rom_indices], label="ROM mode 2")
    ax2.set_title("POD Mode 2 Evolution")
    ax2.set_xlabel("Time")
    ax2.legend()

    ax3 = fig.add_subplot(gs[4:6, 0])
    ax3.plot(t_FOM, state_proj[2, :], label="FOM mode 3")
    ax3.plot(t_ROM_sampled, rom_pred[2, rom_indices], label="ROM mode 3")
    ax3.set_title("POD Mode 3 Evolution")
    ax3.set_xlabel("Time")
    ax3.legend()

    # Right column: drag and lift
    ax4 = fig.add_subplot(gs[0:3, 1])
    ax4.plot(t_FOM, drag_true, label="Cd FOM")
    ax4.plot(t_ROM_sampled, drag_rom, label="Cd ROM")
    ax4.set_title("Drag Coefficient Evolution")
    ax4.set_xlabel("Time")
    ax4.legend()

    ax5 = fig.add_subplot(gs[3:6, 1])
    ax5.plot(t_FOM, lift_true, label="Cl FOM")
    ax5.plot(t_ROM_sampled, lift_rom, label="Cl ROM")
    ax5.set_title("Lift Coefficient Evolution")
    ax5.set_xlabel("Time")
    ax5.legend()

    plt.tight_layout()
    plt.savefig("pinball_ROM_30.png", dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # 9. Save PyTorch dataset and ROM package
    # -----------------------------------------------------------------------
    print("Saving PyTorch dataset and ROM package ...")

    # Convert mass matrix to a PyTorch sparse tensor
    M_coo = M.tocoo()
    indices = np.vstack((M_coo.row, M_coo.col))
    mass_sparse_tensor = torch.sparse_coo_tensor(
        torch.as_tensor(indices, dtype=torch.long),
        torch.as_tensor(M_coo.data, dtype=torch.float32),
        size=M_coo.shape,
    )

    # Full FOM dataset (snapshots + time)
    freefem_data = {
        "states": torch.as_tensor(X_full, dtype=torch.float32),  # all snapshots
        "t": torch.as_tensor(t_full, dtype=torch.float32),
    }
    torch.save(freefem_data, "dataset_freefem_pinball_30.pt")

    # ROM-related objects
    freefem_pod = {
        "modes": torch.as_tensor(POD["v"], dtype=torch.float32),   # full POD basis
        "M": mass_sparse_tensor,                                   # mass matrix
        "N": torch.as_tensor(Np, dtype=torch.float32),             # reduced N tensor
        "redmat": torch.as_tensor(Ap, dtype=torch.float32),        # reduced A
        "dt": float(dt_rom),
        "dragvec": torch.as_tensor(red_dragvec_full[:p], dtype=torch.float32),
        "liftvec": torch.as_tensor(red_liftvec_full[:p], dtype=torch.float32),
    }
    torch.save(freefem_pod, "pod_freefem_pinball_30.pt")

    print("Done.")


if __name__ == "__main__":
    main()
