#!/usr/bin/env python3
######################################################################################
# Idefix MHD astrophysical code
#
# Source file test/SelfGravity/ShearingWave/python/testidefix.py
#
# Last modified : 11/2026
#
# Copyright(C) by :
# - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2025)
# - Steven Rendon Restrepo <srendon@aip.de> (AIP - 2026)
# and other code contributors
#
# Licensed under CeCILL 2.1 License, see COPYING for more information
######################################################################################

"""
Validation of the 2D shearing-box self-gravity solver on the self-gravitating
linear shearing wave of Paardekooper (2012, MNRAS 421, 3286, Sect. 4.2).

A single adiabatic shearing wave with zero potential-vorticity perturbation
evolves in a razor-thin self-gravitating sheet (Q = 1, zero smoothing). Its
complex amplitude W(t) = Sigma1/Sigma0 must follow the linear shearing-wave
equation (eq. 89 of the paper, with xi1 = 0):

    W'' - 2 q Om (kx ky / k^2) W'
        + [ k^2 cs^2 + 2 (2-q) Om^2 - 2 pi G Sigma0 k
            - 2 q (2-q) Om^2 ky^2 / k^2 ] W = 0,

with kx(t) = kx0 + q Om ky t. The amplitude is extracted from the simulation
by projecting Sigma - Sigma0 onto the shearing plane wave exp(-i theta(t)).
"""

import glob
import os
import sys

sys.path.append(os.getenv("IDEFIX_DIR"))

import numpy as np
from scipy.integrate import solve_ivp

from pytools.vtk_io import readVTK

# Parameters of the problem (must match idefix.ini / setup.cpp)
q, Om, G = 1.5, 1.0, 1.0
Sigma0 = 0.025
Q = 1.0
cs = Q * np.pi * G * Sigma0 / Om  # kappa = Omega
kx0, ky = -4.0 * np.pi, 2.0 * np.pi
A0 = 5.0e-4


def shearing_amplitude(V):
    """Complex amplitude of the shearing wave, projected on the plane wave."""
    t = V.t[0]
    x = np.squeeze(V.x).astype(np.float64)
    y = np.squeeze(V.y).astype(np.float64)
    rho = np.array(V.data["RHO"])
    if rho.ndim == 3:
        rho = rho[:, :, 0]
    kx = kx0 + q * Om * ky * t
    theta = np.exp(-1j * (kx * x[:, None] + ky * y[None, :]))
    return 2.0 * np.mean((rho - Sigma0) * theta) / Sigma0


# Collect all vtk outputs
files = sorted(glob.glob("../data.*.vtk"))
if len(files) == 0:
    print("FAILURE!")
    print("No vtk output found")
    sys.exit(1)


# Reference solution of the linear shearing-wave equation (eq. 89, xi1 = 0)
def kxt(t):
    return kx0 + q * Om * ky * t


def eq89(t, y):
    k2 = kxt(t) ** 2 + ky**2
    k = np.sqrt(k2)
    a = -2.0 * q * Om * kxt(t) * ky / k2
    b = (
        k2 * cs**2
        + 2.0 * (2.0 - q) * Om**2
        - 2.0 * np.pi * G * Sigma0 * k
        - 2.0 * q * (2.0 - q) * Om**2 * ky**2 / k2
    )
    return [y[1], -a * y[1] - b * y[0]]


times, wsim = [], []
for f in files:
    V = readVTK(f)
    times.append(V.t[0])
    wsim.append(shearing_amplitude(V))
times = np.array(times)
wsim = np.array(wsim)

sol = solve_ivp(eq89, [0.0, times[-1]], [A0, 0.0], t_eval=times, rtol=1e-10, atol=1e-12)
wref = sol.y[0]

# Error normalised by the peak amplitude of the reference solution.
# The residual error is dominated by the linear-interpolation shear remap
# acting on an increasingly oscillatory wave (kx(t) grows linearly with
# time), and converges to second order with resolution: ~1.6e-1 at
# 128^2, ~4e-2 at 256^2 (the reference amplitude is reproduced to better
# than 1%). The tolerance below is therefore set for the 128^2 reference
# configuration of Paardekooper (2012); it still catches any regression
# of the solver (a broken remap/kernel pairing yields O(1) errors).
scale = np.max(np.abs(wref))
error = np.max(np.abs(wsim - wref)) / scale
print(f"t = {times[-1]:.2f} Omega^-1, peak |W| = {scale:.4e}, error = {error:.3e}")
if error < 2.0e-1:
    print("SUCCESS!")
    sys.exit(0)
else:
    print("FAILURE!")
    sys.exit(1)
