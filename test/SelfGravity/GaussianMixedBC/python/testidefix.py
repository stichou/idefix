#!/usr/bin/env python3
######################################################################################
# Idefix MHD astrophysical code
#
# Source file test/SelfGravity/GaussianMixedBC/python/testidefix.py
#
# Last modified : 11/2026
#
# Copyright(C) by :
# - Steven Rendon Restrepo <srendon@aip.de> (AIP - 2026)
# and other code contributors
#
# Licensed under CeCILL 2.1 License, see COPYING for more information
######################################################################################
# -*- coding: utf-8 -*-
"""
Validation of the VGF-HybridBC solver on the 3D mixed-boundary-conditions
benchmark (Rendon Restrepo & Gressel 2026, Sect. 3.2). The density
    rho = rho0 + A cos(kx x) cos(ky y) exp(-0.5 (z/H)^2)
(with a positive background slab rho0, so that the Riemann solver can
operate) must induce, after one cycle, the analytic velocity
    v = a t with
    a_x =  4 pi G A kx sin(kx x) cos(ky y) S(z)
    a_y =  4 pi G A ky cos(kx x) sin(ky y) S(z)
    a_z = -4 pi G A cos(kx x) cos(ky y) S'(z) - 4 pi G rho0 z
where S(z) = 2 a cosh(k z) + F+(z) + F-(z) is given in eq. (22) of the paper.
"""

import os
import sys

sys.path.append(os.getenv("IDEFIX_DIR"))

import numpy as np
from scipy.special import erf

from pytools.vtk_io import readVTK

# Arguments of the problem
gravCst = 1.0  # grav cst G=1
H = 1.0  # vertical scale height
rho0 = 1.0  # background slab density
A = 0.1  # amplitude of the benchmark perturbation

# Collect the final vtk file
V = readVTK("../data.0001.vtk")
t = V.t[0]

x = np.squeeze(V.x).astype(np.float64)
y = np.squeeze(V.y).astype(np.float64)
z = np.squeeze(V.z).astype(np.float64)
L = x[-1] - x[0] + (x[1] - x[0])
kx = 2.0 * np.pi / L
ky = 2.0 * np.pi / L
k = np.sqrt(kx**2 + ky**2)
Lz = z[-1] - z[0] + (z[1] - z[0])


def S(zz):
    Fp = (
        np.sqrt(np.pi / 2)
        * H
        / (2 * k)
        * np.exp((k * H / 2) * (k * H + 2 * zz / H))
        * erf((k * H + zz / H) / np.sqrt(2))
    )
    Fm = (
        np.sqrt(np.pi / 2)
        * H
        / (2 * k)
        * np.exp((k * H / 2) * (k * H - 2 * zz / H))
        * erf((k * H - zz / H) / np.sqrt(2))
    )
    a = (
        -np.sqrt(np.pi / 2)
        * H
        / (2 * k)
        * np.exp((k * H) ** 2 / 2)
        * erf((k * H + Lz / (2 * H)) / np.sqrt(2))
    )
    return 2 * a * np.cosh(k * zz) + Fp + Fm


X, Y, Z = np.meshgrid(x - 0.5 * L, y - 0.5 * L, z - 0.5 * L, indexing="ij")

ax = 4.0 * np.pi * gravCst * A * kx * np.sin(kx * X) * np.cos(ky * Y) * S(Z)
ay = 4.0 * np.pi * gravCst * A * ky * np.cos(kx * X) * np.sin(ky * Y) * S(Z)
h = 1e-5
dSdz = (S(Z + h) - S(Z - h)) / (2 * h)
az = (
    -4.0 * np.pi * gravCst * A * np.cos(kx * X) * np.cos(ky * Y) * dSdz
    - 4.0 * np.pi * gravCst * rho0 * Z
)

# The two edge cells at each vertical boundary are affected by the
# truncation of the density and by the ghost-cell extrapolation of the
# potential, as reported in the paper; exclude them from the comparison.
error = 0.0
for name, ana in [("VX1", ax), ("VX2", ay), ("VX3", az)]:
    v = np.array(V.data[name])
    err = np.abs(v - ana * t)[:, :, 2:-2]
    scale = np.max(np.abs(ana * t))
    error = max(error, np.max(err) / scale)

print("Error=%e" % error)
if error < 5e-3:
    print("SUCCESS!")
    sys.exit(0)
else:
    print("FAILURE!")
    sys.exit(1)
