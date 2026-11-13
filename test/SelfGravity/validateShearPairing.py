#!/usr/bin/env python3
######################################################################################
# Idefix MHD astrophysical code
#
# Source file test/SelfGravity/validateShearPairing.py
#
# Last modified : 11/2026
#
# Licensed under CeCILL 2.1 License, see COPYING for more information
######################################################################################

"""
Standalone numerical validation of the shear remap / Green's function pairing
used by the shearing-box spectral self-gravity solvers.

The shear-periodic Laplacian is diagonal in the eigenbasis
    exp(i(K_m x + ky y)),   K_m = 2*pi*m/Lx - St*ky,   St = S*Lx*t/Lx,
so the kx-shift slope of the spectral kernel MUST equal the shift slope
applied to the field by the remap. This script builds an exact solution from
shear-periodic eigenmodes and checks all combinations of remap shift slope
(ramp = residual, sawtooth = wrapped total, full = unwrapped total, fold)
against all kernel slopes, using exact spectral shifts (no interpolation).

Expected results (verified):
  - before the first wrap (S*Lx*t < Ly): ramp == sawtooth == full, and any
    of them paired with the matching kernel slope reproduces the exact
    solution to machine precision;
  - after the first wrap: only (ramp, ddy-kernel) and (full, St-kernel)
    remain exact; the (sawtooth, ddy-kernel) pairing used historically by
    the Idefix solvers produces O(1) errors, which motivated the fix of
    wrapping the remap time at the shear period (ShearRemap::WrappedTime).

Run with:  python3 validateShearPairing.py   (requires numpy)
"""

import numpy as np

# ---------------------------------------------------------------
# Ground truth: exact shear-periodic eigenmode solution of the
# 2D Poisson analog  laplacian(phi) = rho  with kernel G = -1/k^2.
# Eigenmodes: exp(i(K_m x + ky y)),  K_m = 2*pi*m/Lx - St*ky,
# with St = S*Lx*t/Lx (total, unwrapped shear slope).
# ---------------------------------------------------------------
Lx = Ly = 1.0
Nx = Ny = 64
S = 1.0
grav = 1.0


def eigen_K(m, ky, St):
    return 2 * np.pi * m / Lx - St * ky


def run(t, modes):
    St = S * Lx * t / Lx
    ddy = (S * Lx * t) % Ly / Lx  # residual (wrapped) slope
    mInt = int(np.floor(S * Lx * t / Ly))  # number of completed wraps
    x = (np.arange(Nx) + 0.5) * Lx / Nx
    y = (np.arange(Ny) + 0.5) * Ly / Ny
    X, Y = np.meshgrid(x, y, indexing="ij")

    # exact phi and rho (sampled eigenmodes)
    phi_ex = np.zeros((Nx, Ny))
    rho = np.zeros((Nx, Ny))
    for m, n, A, ph in modes:
        ky = 2 * np.pi * n / Ly
        K = eigen_K(m, ky, St)
        c = np.cos(K * X + ky * Y + ph)
        phi_ex += A * c
        rho += -A * (K * K + ky * ky) * c  # laplacian(phi) = rho

    def shift(field, delta):  # g(y)=f(y+delta(x)): exact spectral
        f = np.fft.fft(field, axis=1)
        kyv = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
        return np.real(
            np.fft.ifft(f * np.exp(1j * kyv[None, :] * delta[:, None]), axis=1)
        )

    def pipeline(sigma_mode, kslope):
        if sigma_mode == "ramp":
            delta = ddy * x
        elif sigma_mode == "saw":
            delta = np.mod(S * x * t, Ly)
        elif sigma_mode == "full":
            delta = S * x * t
        elif sigma_mode == "fold":
            delta = S * x * t - ddy * x
        rs = shift(rho, delta)
        rF = np.fft.fft2(rs)
        kxv = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
        kyv = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
        KX, KY = np.meshgrid(kxv, kyv, indexing="ij")
        keff2 = (KX - kslope * KY) ** 2 + KY**2
        G = np.where(keff2 > 0, -1.0 / keff2, 0.0)
        pF = rF * G
        ps = np.real(np.fft.ifft2(pF))
        phi = shift(ps, -delta)
        return phi

    scale = np.max(np.abs(phi_ex))
    res = {}
    for sm in ["ramp", "saw", "full", "fold"]:
        for kn, ks in [
            ("ddy", -ddy),
            ("pddy", +ddy),
            ("St", -St),
            ("pSt", +St),
            ("0", 0.0),
        ]:
            err = np.max(np.abs(pipeline(sm, ks) - phi_ex)) / scale
            res[(sm, kn)] = err
    print(f"--- t={t}  (m={mInt}, ddy={ddy:.2f}, St={St:.2f}) ---")
    for sm in ["ramp", "saw", "full", "fold"]:
        line = "  " + sm + ": "
        for kn in ["ddy", "pddy", "St", "pSt", "0"]:
            line += f"{kn}={res[(sm, kn)]:.1e}  "
        print(line)


modes = [
    (1, 1, 1.0, 0.3),
    (2, -1, 0.5, 1.1),
    (3, 2, 0.3, 2.0),
    (-1, -1, 1.0, -0.3),
    (-2, 1, 0.5, -1.1),
    (-3, -2, 0.3, -2.0),
]
for t in [0.3, 1.3, 2.3]:
    run(t, modes)
