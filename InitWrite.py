
"""
EXPORT FREEFEM OPERATORS AND SNAPSHOTS TO NUMPY FORMAT
------------------------------------------------------

This script:
    - loads operator matrices exported from FreeFEM (A, As, M, DerX, DerY, PHI*, etc.)
    - converts them to sparse SciPy matrices via freefem_io_utils.WriteMatrix()
    - loads snapshots (sol_t.txt files) into a big matrix Snap

This prepares all offline data needed for:
    * POD
    * ECSW greedy selection
    * GPU hyper-reduced operator build
    * Training neural network correctors (if used)
"""

import funcIO as io

# ================================================================
# 1) Regular FOM / ROM operators
# ================================================================

print("\n=== Importing standard operators ===")

io.WriteMatrix('M')          # Mass matrix
io.WriteMatrix('DerX')       # ∂/∂x operator
io.WriteMatrix('DerY')       # ∂/∂y operator

io.WriteMatrix('A_30')       # Linearised operator A
io.WriteMatrix('As_30')      # Shifted operator As


# ================================================================
# 2) Hyper-reduction tensorised operators (PHI matrices)
# ================================================================

print("\n=== Importing tensorised PHI matrices ===")

io.WriteMatrix('PHIu')      
io.WriteMatrix('PHIv')
io.WriteMatrix('dPHIudx')
io.WriteMatrix('dPHIudy')
io.WriteMatrix('dPHIvdx')
io.WriteMatrix('dPHIvdy')


# ================================================================
# 3) Snapshot matrix for POD
# ================================================================

print("\n=== Building snapshot matrix ===")
print("Reading snapshots from sol_30/ ...")

io.WriteSnaps(
    numts = 20000,
    step = 20,
    inputfolder = 'sol_30/',
    outputname = "Snap_30"
)

print("\nAll matrices successfully converted to .npz format.\n")
