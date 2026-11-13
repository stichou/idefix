#!/usr/bin/env python3
######################################################################################
# Idefix MHD astrophysical code
#
# Source file test/SelfGravity/ShearingWave/testme.py
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

@author: srendon
"""

import os
import sys

sys.path.append(os.getenv("IDEFIX_DIR"))

import pytools.idfx_test as tst


def testMe(test):
    # The shearing box spectral solvers require heFFTe (and therefore MPI at build time)
    test.cmake.append("Idefix_HEFFTE=ON")
    test.configure()
    test.compile()
    inifiles = ["idefix.ini"]

    # loop on all the ini files for this test
    for ini in inifiles:
        test.run(inputFile=ini)
        test.standardTest()


test = tst.idfxTest(__file__)

# if no decomposition is specified, use that one
if not test.dec:
    test.dec = ["2", "1"]

if not test.all:
    if test.check:
        test.standardTest()
    else:
        testMe(test)
else:
    test.noplot = True
    test.vectPot = False
    test.single = False
    test.reconstruction = 2
    test.mpi = False
    testMe(test)
    test.mpi = True
    testMe(test)
