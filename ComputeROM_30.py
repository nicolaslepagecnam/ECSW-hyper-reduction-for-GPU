# -*- coding: utf-8 -*-
"""
Build reduced-order model (ROM) operators for the fluidic pinball (Re = 30)
===========================================================================

This script:
    - loads the mass matrix M and linearised operator A_30 from FreeFEM exports
    - loads POD modes (v) from PODmodes_30.npz
    - projects A_30 onto the POD basis to obtain the reduced matrix A
    - builds the cubic nonlinearity tensor N for the quadratic convective term:
          N[j,k,i] = Σ_n v_n,i * ( u_j * ∂_x v_k + v_j * ∂_y v_k )_n
      where (u_j, v_j) is the j-th POD mode, and (∂_x v_k, ∂_y v_k) are its
      spatial derivatives.

Outputs:
    - ROM_30.npz with:
        * A : (r x r) reduced linear operator
        * N : (r x r x r) reduced nonlinear convection tensor
"""

import numpy as np
import scipy

import funcIO as io

def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load matrices and POD basis
    # -------------------------------------------------------------------------
    print("Loading mass matrix M ...")
    mass = io.ReadMatrix("M")   # currently not used explicitly, but kept for consistency

    print("Loading full linearised operator A_30 ...")
    lns = io.ReadMatrix("A_30")  # full-order linearised NS operator

    print("Loading POD modes ...")
    POD = np.load("PODmodes_30.npz")
    v_full = POD["v"]
    s = POD["s"]

    # Number of POD modes to keep for the ROM
    r = 100
    r = min(r, v_full.shape[1])
    v = v_full[:, :r]  # (ndof x r)

    print(f"Using r = {r} POD modes")

    # -------------------------------------------------------------------------
    # 2. Build reduced linear operator A = Vᵀ A V
    # -------------------------------------------------------------------------
    print("Projecting linear operator onto POD basis ...")
    # Note: lns is sparse; v is dense
    A = v.T @ (lns @ v)   # shape: (r, r)

    # -------------------------------------------------------------------------
    # 3. Preprocessing for nonlinear term
    # -------------------------------------------------------------------------
    print("Preprocessing for cubic nonlinearities ...")

    print("  Loading derivative operators DerX, DerY ...")
    ddx = io.ReadMatrix("DerX")   # ∂/∂x operator
    ddy = io.ReadMatrix("DerY")   # ∂/∂y operator

    print("  Computing spatial derivatives of POD modes ...")
    # ddx, ddy are sparse; v is dense
    dvdx = ddx @ v   # shape: (ndof, r)
    dvdy = ddy @ v   # shape: (ndof, r)

    # Structure vector indicating the position of (u, v, p) components
    print("  Loading structure vector (component layout) ...")
    svect = io.ReadStruct()  # length = ndof, entries tell if index is 'u' component

    ndof = v.shape[0]
    assert svect.size == ndof, "StructVect size must match number of DOFs"

    # uu and vv are "lifted" velocity vectors used in the nonlinear term:
    #   For each mode j,
    #       u_j = v[.., j] on u DOF indices
    #       v_j = v[.., j] on v DOF indices
    #   and we build:
    #       uu = (-u_j, -u_j, 0)
    #       vv = (-v_j, -v_j, 0)
    uu = np.zeros_like(v)
    vv = np.zeros_like(v)

    print("  Building uu and vv (velocity-only components) ...")
    for ii in range(svect.size):
        if svect[ii] == 1:  # 'u' component
            # u location
            uu[ii, :] = -v[ii, :]
            uu[ii + 1, :] = -v[ii, :]  # second 'u' index (duplicate)
            # v location (immediately after a u component by construction)
            vv[ii, :]     = -v[ii + 1, :]
            vv[ii + 1, :] = -v[ii + 1, :]

    # -------------------------------------------------------------------------
    # 4. Build cubic nonlinearity tensor N
    # -------------------------------------------------------------------------
    print("Generating cubic nonlinearities N[j,k,i] ...")
    r = v.shape[1]
    N = np.zeros((r, r, r), dtype=float)

    # triple loop:
    # N[j,k,i] = Σ_n v[n,i] * ( uu[n,j]*dvdx[n,k] + vv[n,j]*dvdy[n,k] )
    print('Generate cubic nonlinearities')
    for j in range(0,v.shape[1]):
        print('N(',j,',l,m)')    
        for k in range(0,v.shape[1]):
            temp = uu[:,j]*dvdx[:,k]+vv[:,j]*dvdy[:,k] # with arrays, * refers to component-wise product
            for i in range(0,v.shape[1]):
                ww=v[:,i]*temp
                N[j,k,i]=np.sum(ww)

    # -------------------------------------------------------------------------
    # 5. Save ROM operators
    # -------------------------------------------------------------------------
    print("Saving ROM operators to ROM_30.npz ...")
    np.savez("ROM_30.npz", A=A, N=N)

    print("Done. Reduced operators saved.")

if __name__ == "__main__":
    main()
