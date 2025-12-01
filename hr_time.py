"""
Benchmark: regular quadratic tensor contraction vs ECSW-style hyper-reduced
convection term in PyTorch, with timing and visualization.

Outputs:
- hr_time_1.png : Regular vs hyper-reduced (n-scaling)
- hr_time_2.png : Hyper-reduced scaling in n and m
- hr_time_3.txt : Crossover report (n where hyper < regular, for each m)
- hr_time_4.png : 3D surface of hyper-reduced times
"""

# =============================================================================
# Optional: Set single-thread BLAS for reproducible timings
# (uncomment BEFORE importing numpy/torch, then restart Python)
# =============================================================================
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D plotting)


# =============================================================================
# Torch runtime and timing utilities
# =============================================================================

# For stability and reproducibility
torch.set_num_threads(1)          # pin to 1 CPU thread
torch.set_num_interop_threads(1)
torch.manual_seed(0)

device = torch.device("cpu")      # use CPU here; change to "cuda" if desired
dtype = torch.float64             # float64 to match NumPy defaults


def _sync() -> None:
    """Synchronize CUDA (if used) to get accurate timings."""
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()


def stable_time(fn, number: int = 10, repeats: int = 11) -> float:
    """
    Time a callable using median-of-repeats, with GC disabled and CUDA sync.

    Parameters
    ----------
    fn : callable
        Function with no arguments to time.
    number : int
        Number of calls per repeat.
    repeats : int
        Number of repeats; median is taken.

    Returns
    -------
    float
        Median time per call (seconds).
    """
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        # Warm-up
        fn()
        _sync()

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(number):
                fn()
            _sync()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        return np.median(times) / number
    finally:
        if gc_was_enabled:
            gc.enable()


# =============================================================================
# Baseline: regular quadratic tensor contraction with einsum
# =============================================================================
# nonlin_new = torch.einsum("ijk,mi,mj->mk", N_tensor, output_new, output_new)
# We set batch size B = 1 to match the single state-vector 'a' in hyper-reduction.

n_min, n_max = 5, 250
xs = np.arange(n_min, n_max + 1)          # n = 5..250  (length 246)
results_regular = np.zeros(xs.size, dtype=np.float64)

for idx, n in enumerate(xs):
    B = 1  # single state
    # Shapes: N_tensor [n, n, n], output_new [B, n] -> result [B, n]
    N_tensor = torch.rand((n, n, n), device=device, dtype=dtype)
    output_new = torch.rand((B, n), device=device, dtype=dtype)

    def run_regular():
        return torch.einsum("ijk,mi,mj->mk", N_tensor, output_new, output_new)

    t_reg = stable_time(run_regular, number=100, repeats=7)
    results_regular[idx] = t_reg

    print(f"[Regular] n={n:3d}  t={t_reg:.3e} s")


# =============================================================================
# Hyper-reduced convection RHS
# =============================================================================
# Formula (ECSW-style):
#   f_hyper(a) = - (pod_t_PHIu_t_Wu_red_u @ fu + pod_t_PHIv_t_Wu_red_v @ fv)
# with fu, fv computed from PHI*, dPHI* and coefficients a.


def compute_hyperreduced_convection_rhs_torch(
    a: torch.Tensor,
    PHIu_pod_red_u: torch.Tensor,
    PHIv_pod_red_u: torch.Tensor,
    dPHIudx_pod_red_u: torch.Tensor,
    dPHIudy_pod_red_u: torch.Tensor,
    pod_t_PHIu_t_Wu_red_u: torch.Tensor,
    PHIu_pod_red_v: torch.Tensor,
    PHIv_pod_red_v: torch.Tensor,
    dPHIvdx_pod_red_v: torch.Tensor,
    dPHIvdy_pod_red_v: torch.Tensor,
    pod_t_PHIv_t_Wu_red_v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute hyper-reduced convection RHS for a given reduced state 'a'.

    All PHI*, dPHI* have shape (m, n), 'a' has shape (n,),
    pod_t_* have shape (n, m). Returns vector of shape (n,).
    """
    # u-equation contribution
    uu = PHIu_pod_red_u @ a
    dxuu = dPHIudx_pod_red_u @ a
    vu = PHIv_pod_red_u @ a
    dyuu = dPHIudy_pod_red_u @ a
    fu = uu * dxuu + vu * dyuu

    # v-equation contribution
    uv = PHIu_pod_red_v @ a
    dxvv = dPHIvdx_pod_red_v @ a
    vv = PHIv_pod_red_v @ a
    dyvv = dPHIvdy_pod_red_v @ a
    fv = uv * dxvv + vv * dyvv

    return -(
        pod_t_PHIu_t_Wu_red_u @ fu
        + pod_t_PHIv_t_Wu_red_v @ fv
    )


m_values = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results_hyper = np.zeros((len(m_values), xs.size), dtype=np.float64)

for i, m in enumerate(m_values):
    for idx, n in enumerate(xs):
        # Generate random inputs with correct shapes
        a = torch.rand((n,), device=device, dtype=dtype)

        PHIu_pod_red_u = torch.rand((m, n), device=device, dtype=dtype)
        PHIv_pod_red_u = torch.rand((m, n), device=device, dtype=dtype)
        dPHIudx_pod_red_u = torch.rand((m, n), device=device, dtype=dtype)
        dPHIudy_pod_red_u = torch.rand((m, n), device=device, dtype=dtype)
        pod_t_PHIu_t_Wu_red_u = torch.rand((n, m), device=device, dtype=dtype)

        PHIu_pod_red_v = torch.rand((m, n), device=device, dtype=dtype)
        PHIv_pod_red_v = torch.rand((m, n), device=device, dtype=dtype)
        dPHIvdx_pod_red_v = torch.rand((m, n), device=device, dtype=dtype)
        dPHIvdy_pod_red_v = torch.rand((m, n), device=device, dtype=dtype)
        pod_t_PHIv_t_Wu_red_v = torch.rand((n, m), device=device, dtype=dtype)

        def run_hyper():
            return compute_hyperreduced_convection_rhs_torch(
                a,
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

        # Heuristic: same timing parameters as regular case
        t_h = stable_time(run_hyper, number=100, repeats=7)
        results_hyper[i, idx] = t_h
        print(f"[Hyper]  m={m:4d} n={n:3d}  t={t_h:.3e} s")


# =============================================================================
# Plots
# =============================================================================

# -------------------------------------------------------------------------
# Plot 1: n vs time, regular vs all hyper curves
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6), dpi=160)
plt.plot(xs, results_regular, label="Regular (einsum)")
for i, m in enumerate(m_values):
    plt.plot(xs, results_hyper[i, :], label=f"Hyper (m={m})")

plt.xlabel("Number of Modes (n)")
plt.ylabel("Time per Evaluation (s)")
plt.title("Regular vs Hyper-Reduced: n-scaling")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("hr_time_1.png")
plt.close()


# -------------------------------------------------------------------------
# Plot 2: side-by-side hyper-only views
#   Left : n vs time for each m
#   Right: m vs time for selected n
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)

# Left subplot: n vs time for each m
ax_left = axes[0]
for i, m in enumerate(m_values):
    ax_left.plot(xs, results_hyper[i, :], label=f"m={m}")
ax_left.set_xlabel("Number of Modes (n)")
ax_left.set_ylabel("Time per Evaluation (s)")
ax_left.set_title("Hyper Reduction: n-scaling (by m)")
ax_left.grid(True, which="both")
ax_left.legend()

# Right subplot: m vs time for selected n
ax_right = axes[1]
selected_n = [10, 50, 100, 150, 200, 250]
m_vals = np.array(m_values)

for n_fix in selected_n:
    # index in results arrays: n = n_fix corresponds to xs.index(n_fix)
    idx_fix = np.where(xs == n_fix)[0]
    if idx_fix.size == 0:
        continue
    y = results_hyper[:, idx_fix[0]]
    ax_right.plot(m_vals, y, marker="o", label=f"n={n_fix}")

ax_right.set_xlabel("m (Gauss points)")
ax_right.set_ylabel("Time per Evaluation (s)")
ax_right.set_title("Hyper Reduction: m-scaling (by fixed n)")
ax_right.grid(True, which="both")
ax_right.legend()

plt.tight_layout()
plt.savefig("hr_time_2.png")
plt.close()


# =============================================================================
# Crossover report: for each m, smallest n where hyper < regular
# =============================================================================
def crossover_n(xs_arr: np.ndarray,
                regular: np.ndarray,
                hyper_row: np.ndarray):
    """
    Return smallest n in xs_arr such that hyper_row(n) < regular(n).

    Parameters
    ----------
    xs_arr : np.ndarray
        Array of n values (same length as regular/hyper_row).
    regular : np.ndarray
        Regular timings for each n.
    hyper_row : np.ndarray
        Hyper timings for each n, for fixed m.

    Returns
    -------
    int or None
        First n where hyper_row < regular, or None if no crossover.
    """
    diff = hyper_row - regular
    idx = np.where(diff < 0)[0]
    return int(xs_arr[idx[0]]) if idx.size > 0 else None


lines = []
lines.append("Crossover points (smallest n where Hyper(m) < Regular):\n")
for i, m in enumerate(m_values):
    n_cross = crossover_n(xs, results_regular, results_hyper[i, :])
    if n_cross is None:
        lines.append(f"m={m:>5}: no crossover up to n={xs[-1]}\n")
    else:
        lines.append(f"m={m:>5}: n* = {n_cross}\n")

report = "".join(lines)
print(report)
with open("hr_time_3.txt", "w", encoding="utf-8") as f:
    f.write(report)


# =============================================================================
# Plot 3: 3D surface of hyper times (Z) vs X = n and Y = m
# =============================================================================
M, N = results_hyper.shape  # M = len(m_values), N = len(xs)
X = np.tile(xs, (M, 1))                     # shape (M, N)
Y = np.tile(np.array(m_values)[:, None], (1, N))  # shape (M, N)
Z = results_hyper                           # shape (M, N)

fig = plt.figure(figsize=(9, 6), dpi=160)
ax3d = fig.add_subplot(111, projection="3d")

surf = ax3d.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
ax3d.set_xlabel("n (modes)")
ax3d.set_ylabel("m (Gauss points)")
ax3d.set_zlabel("Time (s)")
ax3d.set_title("Hyper Reduction Time Surface")

# Use integer ticks for m
ax3d.yaxis.set_major_formatter(ScalarFormatter())
fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1, label="Time (s)")

plt.tight_layout()
plt.savefig("hr_time_4.png")
plt.close()
