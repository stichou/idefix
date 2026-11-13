// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file test/SelfGravity/ShearingWave/setup.cpp
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

// Self-gravitating linear shearing wave (Paardekooper 2012, MNRAS 421, 3286,
// Sect. 4.2): a single adiabatic shearing wave
//   Sigma1/Sigma0 = A cos(kx(t) x + ky y),   kx(t) = kx0 + q*Omega*ky*t,
// evolved in a razor-thin self-gravitating shearing sheet with zero
// smoothing (delta = 0), compared against the linear theory of
// Paardekooper (2012), eq. (89) (see python/testidefix.py).
//
// The initial condition is an exact linear eigenstate with zero potential
// vorticity perturbation (xi1 = 0) and zero initial amplitude derivative:
//   Sigma = Sigma0 (1 + A cos(theta)),
//   p     = p0 (Sigma/Sigma0)^gamma         (adiabatic: p1 = cs^2 Sigma1),
//   vx    = -(2-q) Omega A (ky/k^2) sin(theta),
//   vy    = -q Omega x + (2-q) Omega A (kx0/k^2) sin(theta),
// with theta = kx0 x + ky y, cs = Q pi G Sigma0 (Toomre Q, kappa = Omega)
// and p0 = cs^2 Sigma0 / gamma.

// Default constructor
static real Omega;
static real shearParameter;   // sbS = -q*Omega
static real q;
static real gravCst;
static real gammaIdeal;
static real Sigma0;
static real Q;
static real amplitude;
static real mkx0, mky;         // initial wave numbers in units of 2*pi/L

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  // Tidal force +2 q Omega^2 x of the shearing sheet (the Coriolis force
  // is added by the rotation module; Fargo is not used here).
  IdefixArray1D<real> x = data.x[IDIR];
  real Om = Omega;
  real qq = q;
  idefix_for("BodyForce", data.beg[KDIR], data.end[KDIR],
                            data.beg[JDIR], data.end[JDIR],
                            data.beg[IDIR], data.end[IDIR],
      KOKKOS_LAMBDA (int k, int j, int i) {
        force(IDIR,k,j,i) = 2.0*qq*Om*Om*x(i);
        force(JDIR,k,j,i) = ZERO_F;
      });
  idfx::popRegion();
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  Omega = input.Get<real>("Hydro","rotation",0);
  shearParameter = input.Get<real>("Hydro","shearingBox",0);
  q = -shearParameter/Omega;
  gravCst = input.Get<real>("Gravity","gravCst",0);
  gammaIdeal = data.hydro->eos->GetGamma();
  Sigma0 = input.Get<real>("Setup","sigma0",0);
  Q = input.Get<real>("Setup","Q",0);
  amplitude = input.Get<real>("Setup","amplitude",0);
  mkx0 = input.Get<real>("Setup","kx0",0);
  mky = input.Get<real>("Setup","ky",0);

  data.gravity->EnrollBodyForce(BodyForce);
}

// Initialisation routine
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];

  const real Lx = data.mygrid->xend[IDIR]-data.mygrid->xbeg[IDIR];
  const real Ly = data.mygrid->xend[JDIR]-data.mygrid->xbeg[JDIR];
  const real kx0 = mkx0*2.0*M_PI/Lx;
  const real ky = mky*2.0*M_PI/Ly;
  const real k2 = kx0*kx0 + ky*ky;

  // Adiabatic sound speed from the Toomre parameter (kappa = Omega)
  const real cs = Q*M_PI*gravCst*Sigma0/Omega;
  const real prs0 = cs*cs*Sigma0/gammaIdeal;

  // Velocity amplitudes of the zero-potential-vorticity eigenstate
  const real uAmp = (2.0-q)*Omega*amplitude*ky/k2;
  const real vAmp = (2.0-q)*Omega*amplitude*kx0/k2;

  for(int k = 0; k < d.np_tot[KDIR]; k++) {
    for(int j = 0; j < d.np_tot[JDIR]; j++) {
      for(int i = 0; i < d.np_tot[IDIR]; i++) {
        const real xx = x1(i);
        const real yy = x2(j);
        const real theta = kx0*xx + ky*yy;
        d.Vc(RHO,k,j,i) = Sigma0*(1.0 + amplitude*cos(theta));
        d.Vc(PRS,k,j,i) = prs0*pow(d.Vc(RHO,k,j,i)/Sigma0, gammaIdeal);
        d.Vc(VX1,k,j,i) = -uAmp*sin(theta);
        d.Vc(VX2,k,j,i) = shearParameter*xx + vAmp*sin(theta);
      }
    }
  }
  d.SyncToDevice();
}
