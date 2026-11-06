#!/usr/bin/env python3
######################################################################################
# Idefix MHD astrophysical code
#
# Source file test/SelfGravity/VerticalGaussian/python/testidefix.py
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
# -*- coding: utf-8 -*-
"""
Validation of the VGF-HybridBC solver on the 1D vertical Gaussian benchmark
(Rendon Restrepo & Gressel 2026, Sect. 3.1). The density
rho(z) = exp(-0.5 (z-z0)^2) is truncated at the vertical boundaries, and the
induced vertical velocity after one cycle must match the analytic force
    a_z(z) = -4 pi G [ sqrt(pi/2) erf((z-z0)/sqrt(2)) + a ]
    a      = -0.5 sqrt(pi/2) [ erf((-Lz/2-z0)/sqrt(2)) + erf((Lz/2-z0)/sqrt(2)) ]
The offset z0 is recovered from the initial density profile, so the same
script validates both the centred (z0=0) and shifted (z0=2) variants.
"""

import os
import sys

sys.path.append(os.getenv("IDEFIX_DIR"))

import numpy as np
from scipy.special import erf

from pytools.vtk_io import readVTK

# Arguments of the problem
gravCst = 1.0  # grav cst G=1

# Collect the initial and final vtk files
V0 = readVTK("../data.0000.vtk")
V1 = readVTK("../data.0001.vtk")

z = np.squeeze(V0.z).astype(np.float64)
zc = 0.5 * (z[0] + z[-1])  # centre of the vertical domain
zz = z - zc
Lz = z[-1] - z[0] + (z[1] - z[0])

# Recover the Gaussian offset z0 from the initial density profile:
# ln(rho) = -0.5 z^2 + z0 z - 0.5 z0^2 (away from the truncated edges)
rho0 = np.array(V0.data["RHO"]).mean(axis=(0, 1))
mask = rho0 > 0.01
coeffs = np.polyfit(zz[mask], np.log(rho0[mask]), 2)
z0 = coeffs[1]
if abs(coeffs[0] + 0.5) > 1e-2:
    print("FAILURE!")
    print("Initial density is not a unit-width Gaussian")
    sys.exit(1)

# Analytic vertical force
a = (
    -0.5
    * np.sqrt(np.pi / 2)
    * (erf((-0.5 * Lz - z0) / np.sqrt(2)) + erf((0.5 * Lz - z0) / np.sqrt(2)))
)
az = -4.0 * np.pi * gravCst * (np.sqrt(np.pi / 2) * erf((zz - z0) / np.sqrt(2)) + a)

# Numerical velocity after one cycle (mean over the horizontal directions)
t = V1.t[0]
vz = np.array(V1.data["VX3"]).mean(axis=(0, 1))

# The two edge cells at each vertical boundary are affected by the
# truncation of the density and by the ghost-cell extrapolation of the
# potential, as reported in the paper; exclude them from the comparison.
err = np.abs(vz - az * t)
scale = np.max(np.abs(az * t))
error = np.max(err[2:-2]) / scale

print(f"z0 = {z0:.6f}")
print("Error=%e" % error)
if error < 5e-3:
    print("SUCCESS!")
    sys.exit(0)
else:
    print("FAILURE!")
    sys.exit(1)
