// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file test/SelfGravity/VerticalGaussian/setup.cpp
//
// Last modified : 11/2026
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2025)
// - Steven Rendon Restrepo <srendon@aip.de> (AIP - 2026)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include "setup.hpp"

// 1D vertical Gaussian benchmark for the VGF-HybridBC solver
// (Rendon Restrepo & Gressel 2026, Sect. 3.1):
//   rho(z) = exp(-0.5 (z-z0)^2) for z in [-Lz/2, Lz/2], zero outside
// The vertical offset z0 is read from the [Setup] block.

real z0Glob;
real zcGlob;

// Default constructor
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  z0Glob = input.GetOrSet<real>("Setup","z0",0,2.0);
  zcGlob = 0.5*(grid.xend[KDIR]+grid.xbeg[KDIR]);
}

// Initialisation routine
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  IdefixHostArray1D<real> x3 = d.x[KDIR];
  const real prs0 = 1.0;
  for(int k = 0; k < d.np_tot[KDIR]; k++) {
    for(int j = 0; j < d.np_tot[JDIR]; j++) {
      for(int i = 0; i < d.np_tot[IDIR]; i++) {
        const real zz = x3(k) - zcGlob;
        d.Vc(RHO,k,j,i) = exp(-0.5*pow(zz-z0Glob,2.0));
        d.Vc(VX1,k,j,i) = 0.0;
        d.Vc(VX2,k,j,i) = 0.0;
        d.Vc(VX3,k,j,i) = 0.0;
        d.Vc(PRS,k,j,i) = prs0;
      }
    }
  }
  d.SyncToDevice();
}
