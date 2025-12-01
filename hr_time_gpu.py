"""
GPU benchmark: quadratic tensor contraction via einsum vs
ECSW-style hyper-reduced RHS implemented as a small NN.

Outputs:
- hr_vs_rom_time_median.png : log–log plot of median runtime vs n
"""

import statistics
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

# Change "cuda:2" to "cuda" or "cuda:0" if needed on your machine.
device = torch.device("cuda:2")
dtype = torch.float32

BATCH = 256        # batch size for input vectors
REPEATS = 100      # calls per timing batch
TRIALS = 21        # number of batches; we take the median of their averages
M_LIST = [10, 50, 100, 200, 500, 1000]  # number of Gauss points (per component)

# Optional: keep matmul behavior consistent
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("high")

torch.manual_seed(0)


# =============================================================================
# Utilities
# =============================================================================

def median_and_iqr(vals: Iterable[float]) -> Tuple[float, float]:
    """
    Compute median and relative IQR (interquartile range) of a sequence.

    Parameters
    ----------
    vals : iterable of float
        Values to summarize.

    Returns
    -------
    median : float
        Median of the values.
    iqr_percent : float
        IQR expressed as a percentage of the median.
    """
    vals = list(vals)
    med = statistics.median(vals)
    q1 = statistics.quantiles(vals, n=4)[0]  # 25th percentile
    q3 = statistics.quantiles(vals, n=4)[2]  # 75th percentile
    iqr_percent = 100.0 * (q3 - q1) / med if med != 0.0 else 0.0
    return med, iqr_percent


# Warm up CUDA context
_ = (torch.randn(1, device=device) @ torch.randn(1, device=device))
torch.cuda.synchronize()


# =============================================================================
# Benchmark 1: einsum-based quadratic convection term
# =============================================================================

ns = []
einsum_med = []   # seconds per call (median of TRIALS)
einsum_iqr = []   # relative IQR (%) across batch averages

for n in range(1, 251):
    # N_tensor: (n, n, n)
    # v      : (BATCH, n)
    N_tensor = torch.randn(n, n, n, device=device, dtype=dtype)
    v = torch.randn(BATCH, n, device=device, dtype=dtype)

    # One quick warmup for kernels/allocator
    _ = torch.einsum("ijk,mi,mj->mk", N_tensor, v, v)
    torch.cuda.synchronize()

    batch_avgs = []
    for _ in range(TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _rep in range(REPEATS):
            _ = torch.einsum("ijk,mi,mj->mk", N_tensor, v, v)
        end.record()

        torch.cuda.synchronize()
        # elapsed_time returns milliseconds for the whole batch
        batch_ms = start.elapsed_time(end)
        batch_avgs.append((batch_ms / 1000.0) / REPEATS)  # seconds per call

    med, iqr = median_and_iqr(batch_avgs)
    ns.append(n)
    einsum_med.append(med)
    einsum_iqr.append(iqr)

    print(f"[einsum] n={n:3d}  median={med * 1e6:8.2f} µs  IQR={iqr:8.2f} %")


# =============================================================================
# Benchmark 2: ECSW-style hyper-reduced RHS implemented as NN
# =============================================================================

class HRRHS(torch.nn.Module):
    """
    Hyper-reduced RHS as a small linear network.

    Parameters
    ----------
    p : int
        Dimension of the reduced state (number of modes).
    Ng_u : int
        Number of Gauss points for the u-component.
    Ng_v : int
        Number of Gauss points for the v-component.
    device : torch.device
        Device on which to allocate buffers.
    dtype : torch.dtype
        Data type for all buffers.
    """

    def __init__(self, p: int, Ng_u: int, Ng_v: int,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.Ng_u = Ng_u
        self.Ng_v = Ng_v

        # Shapes: (4*Ng, p) so that linear(a, W) : (B, p) -> (B, 4*Ng)
        self.register_buffer(
            "W_all_u",
            torch.randn(4 * Ng_u, p, device=device, dtype=dtype),
        )
        self.register_buffer(
            "W_all_v",
            torch.randn(4 * Ng_v, p, device=device, dtype=dtype),
        )

        # P_proj: (p, Ng_u + Ng_v); linear(concat, P) : (B, Ng_u+Ng_v) -> (B, p)
        self.register_buffer(
            "P_proj",
            torch.randn(p, Ng_u + Ng_v, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def rhs(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyper-reduced RHS for a batch of reduced states.

        Parameters
        ----------
        a : torch.Tensor
            Input of shape (BATCH, p).

        Returns
        -------
        torch.Tensor
            Output RHS of shape (BATCH, p).
        """
        # u-equation part
        zu = F.linear(a, self.W_all_u)                  # (B, 4*Ng_u)
        uu, vu, dxuu, dyuu = zu.split(self.Ng_u, dim=1)  # each (B, Ng_u)
        fu = uu.mul(dxuu)
        fu = torch.addcmul(fu, vu, dyuu)                # fu = uu*dxuu + vu*dyuu

        # v-equation part
        zv = F.linear(a, self.W_all_v)                  # (B, 4*Ng_v)
        uv, vv, dxvv, dyvv = zv.split(self.Ng_v, dim=1)  # each (B, Ng_v)
        fv = uv.mul(dxvv)
        fv = torch.addcmul(fv, vv, dyvv)                # fv = uv*dxvv + vv*dyvv

        concat = torch.cat([fu, fv], dim=1)             # (B, Ng_u + Ng_v)
        rhs = F.linear(concat, self.P_proj)             # (B, p)

        return -rhs


rhs_med = {m: [] for m in M_LIST}  # seconds per call
rhs_iqr = {m: [] for m in M_LIST}  # relative IQR (%)

for n in range(1, 251):
    p = n
    a = torch.randn(BATCH, p, device=device, dtype=dtype)

    for m in M_LIST:
        model = HRRHS(p=p, Ng_u=m, Ng_v=m,
                      device=device, dtype=dtype).to(device)

        # Short warmup
        for _ in range(5):
            _ = model.rhs(a)
        torch.cuda.synchronize()

        batch_avgs = []
        for _ in range(TRIALS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _rep in range(REPEATS):
                _ = model.rhs(a)
            end.record()

            torch.cuda.synchronize()
            batch_ms = start.elapsed_time(end)
            batch_avgs.append((batch_ms / 1000.0) / REPEATS)

        med, iqr = median_and_iqr(batch_avgs)
        rhs_med[m].append(med)
        rhs_iqr[m].append(iqr)

        print(
            f"[rhs]    n={n:3d}  m={m:4d}  "
            f"median={med * 1e6:9.2f} µs  IQR={iqr:8.2f} %"
        )


# =============================================================================
# Plot: log–log runtime vs n
# =============================================================================

plt.figure(figsize=(9, 6))

# Einsum curve (median)
plt.loglog(ns, [t * 1e6 for t in einsum_med],
           label="einsum (median)", linewidth=2)

# One curve per m for the RHS
for m in M_LIST:
    plt.loglog(ns, [t * 1e6 for t in rhs_med[m]],
               label=f"rhs m={m} (median)", linewidth=1)

plt.xlabel("Number of modes n")
plt.ylabel("Time per call (µs, median of batches)")
plt.title("GPU runtime (median-of-batches): einsum vs rhs on cuda:2")
plt.legend(ncol=2)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("hr_vs_rom_time_median.png", dpi=220)
plt.close()

print("Saved plot as hr_vs_rom_time_median.png")
