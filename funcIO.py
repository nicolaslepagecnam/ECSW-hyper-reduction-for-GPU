
"""
Utility functions to read/write matrices, vectors and snapshots
from FreeFEM++ text outputs.

Original author: S. Sipp
Modified & documented by: N. Lepage
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import sparse as sp

def WriteMatrix(name: str) -> None:
    """
    Read a FreeFEM sparse matrix stored in a text file `name + ".txt"`
    and save it as a compressed NumPy archive `name + ".npz"`.

    The expected text format is:
        - line 1: header (ignored)
        - line 2: header (ignored)
        - line 3: "dim_n dim_m nbre"
        - next nbre lines: "i j a_ij"

    Parameters
    ----------
    name : str
        Base filename (without extension).
    """
    txt_path = Path(name + ".txt")
    if not txt_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {txt_path}")

    with txt_path.open("r") as f:
        # Skip first two headers
        f.readline()
        f.readline()

        header = f.readline().strip()
        columns = header.split()
        dim_n = int(columns[0])   # Number of rows
        dim_m = int(columns[1])   # Number of columns
        nbre = int(columns[2])    # Number of non-zero entries

        print(f"Reading matrix: {dim_n} x {dim_m}, nnz = {nbre}")

        ii = np.zeros(nbre, dtype=int)
        jj = np.zeros(nbre, dtype=int)
        aa = np.zeros(nbre, dtype=float)

        for icount in range(nbre):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"Unexpected end of file after {icount} entries "
                    f"(expected {nbre}) in {txt_path}"
                )
            parts = line.strip().split()
            ii[icount] = int(parts[0])
            jj[icount] = int(parts[1])
            aa[icount] = float(parts[2])

    # Save data and dimensions
    np.savez(name + ".npz", ii=ii, jj=jj, aa=aa, dim_n=dim_n, dim_m=dim_m)


def ReadMatrix(name: str) -> sp.coo_matrix:
    """
    Load a sparse matrix previously saved by WriteMatrix(`name`).

    Parameters
    ----------
    name : str
        Base filename (without extension).

    Returns
    -------
    mat : scipy.sparse.coo_matrix
        Sparse matrix reconstructed from the stored triplets.
    """
    npzfile = np.load(name + ".npz")
    ii = npzfile["ii"]
    jj = npzfile["jj"]
    aa = npzfile["aa"]
    dim_n = int(npzfile["dim_n"])
    dim_m = int(npzfile["dim_m"])

    mat = sp.coo_matrix((aa, (ii, jj)), shape=(dim_n, dim_m))
    return mat


def readvect(readname: str) -> np.ndarray:
    """
    Read a FreeFEM vector file with format:

        line 1: integer N (number of entries)
        following lines: values written 5 per line (except last line)

    Parameters
    ----------
    readname : str
        Full filename (e.g. "sol_100.txt").

    Returns
    -------
    state : np.ndarray
        1D array of length N with the vector values.
    """
    path = Path(readname)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    # Read number of entries
    with path.open("r") as f:
        header1 = f.readline().strip()
        num = int(header1)

    state = np.zeros(num, dtype=float)

    with path.open("r") as f:
        # Skip header
        _ = f.readline()
        # Full blocks of 5 values
        for j in range(num // 5):
            line = f.readline()
            columns = line.strip().split()
            for k in range(5):
                state[j * 5 + k] = float(columns[k])
        # Remaining values
        line = f.readline()
        if line:
            columns = line.strip().split()
            for k in range(num % 5):
                state[(num // 5) * 5 + k] = float(columns[k])

    return state


def WriteSnaps(numts: int,
               step: int,
               inputfolder: str,
               outputname: str) -> None:
    """
    Collect time snapshots from FreeFEM files 'sol_<t>.txt'
    and store them as a 2D NumPy array in `outputname + ".npz"`.

    Parameters
    ----------
    numts : int
        Total number of time steps in the simulation.
    step : int
        Sampling step between snapshots (e.g. 20).
    inputfolder : str
        Folder containing the FreeFEM snapshot files. Must end with '/' or '\\'.
    outputname : str
        Base name for the output .npz file.
    """
    inputfolder = str(inputfolder)
    if not inputfolder.endswith("/") and not inputfolder.endswith("\\"):
        inputfolder = inputfolder + "/"

    nbre = int(numts / step)

    # Read size from one snapshot (here assumed to be sol_100.txt)
    # You may want to generalise this if needed.
    ref_file = Path(inputfolder) / "sol_100.txt"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference snapshot not found: {ref_file}")

    with ref_file.open("r") as f:
        num = int(f.readline().strip())

    state = np.zeros((num, nbre + 1), dtype=float)

    for i in range(1, nbre + 1):
        print(f"Reading snapshot {i}/{nbre}")
        snap_file = Path(inputfolder) / f"sol_{i * step}.txt"
        if not snap_file.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_file}")

        with snap_file.open("r") as f:
            _ = f.readline()  # skip size
            # Full blocks of 5
            for j in range(num // 5):
                line = f.readline()
                columns = line.strip().split()
                for k in range(5):
                    state[j * 5 + k, i] = float(columns[k])
            # Remaining values
            line = f.readline()
            if line:
                columns = line.strip().split()
                for k in range(num % 5):
                    state[(num // 5) * 5 + k, i] = float(columns[k])

    np.savez(outputname + ".npz", state)


def ReadSnap(name: str) -> np.ndarray:
    """
    Read snapshots saved by WriteSnaps.

    Parameters
    ----------
    name : str
        Base filename (without extension).

    Returns
    -------
    sn : np.ndarray
        2D array [num_dofs, num_snapshots+1].
    """
    npzfile = np.load(name + ".npz")
    sn = npzfile["arr_0"]
    return sn


def ReadStruct() -> np.ndarray:
    """
    Read an integer structure vector from 'StructVect.txt' with format:

        line 1: integer N
        following lines: values, 5 per line (except last)

    Returns
    -------
    state : np.ndarray
        1D array of length N with integer entries.
    """
    path = Path("StructVect.txt")
    if not path.exists():
        raise FileNotFoundError(f"StructVect.txt not found in {path.resolve().parent}")

    print("Reading structure from StructVect.txt")

    with path.open("r") as f:
        num = int(f.readline().strip())
        state = np.zeros(num, dtype=int)

        for j in range(num // 5):
            line = f.readline()
            columns = line.strip().split()
            for k in range(5):
                state[j * 5 + k] = int(float(columns[k]))  # original code used float

        line = f.readline()
        if line:
            columns = line.strip().split()
            for k in range(num % 5):
                state[(num // 5) * 5 + k] = int(float(columns[k]))

    return state


def ReadData() -> Tuple[int, float, int]:
    """
    Read metadata from 'data.txt' with format:

        line 1: integer num   (number of time steps or samples)
        line 2: float dt      (time step)
        line 3: integer step  (sampling step)

    Returns
    -------
    num : int
    dt : float
    step : int
    """
    path = Path("data.txt")
    if not path.exists():
        raise FileNotFoundError(f"data.txt not found in {path.resolve().parent}")

    with path.open("r") as f:
        num = int(f.readline().strip())
        dt = float(f.readline().strip())
        step = int(f.readline().strip())

    return num, dt, step
