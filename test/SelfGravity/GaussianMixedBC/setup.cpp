#include "setup.hpp"

// 3D benchmark with mixed boundary conditions for the VGF-HybridBC solver
// (Rendon Restrepo & Gressel 2026, Sect. 3.2):
//   rho(x,y,z) = rho0 + A cos(kx x) cos(ky y) exp(-0.5 (z/H)^2)
// with kx = ky = 2 pi / L, on a cubic box of size L.
// The constant background rho0 keeps the density positive (the pure
// benchmark of the paper is signed and cannot be evolved with a
// Riemann solver); its analytic contribution to the force is the
// truncated-slab solution a_z = -4 pi G rho0 z.

// Default constructor
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
}

// Initialisation routine
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  IdefixHostArray1D<real> x1 = d.x[IDIR];
  IdefixHostArray1D<real> x2 = d.x[JDIR];
  IdefixHostArray1D<real> x3 = d.x[KDIR];
  const real L = data.mygrid->xend[IDIR]-data.mygrid->xbeg[IDIR];
  const real kx = 2.0*M_PI/L;
  const real ky = 2.0*M_PI/L;
  const real H = 1.0;
  const real rho0 = 1.0;
  const real A = 0.1;
  const real prs0 = 1.0;
  for(int k = 0; k < d.np_tot[KDIR]; k++) {
    for(int j = 0; j < d.np_tot[JDIR]; j++) {
      for(int i = 0; i < d.np_tot[IDIR]; i++) {
        const real xx = x1(i) - 0.5*L;
        const real yy = x2(j) - 0.5*L;
        const real zz = x3(k) - 0.5*L;
        d.Vc(RHO,k,j,i) = rho0 + A*cos(kx*xx)*cos(ky*yy)*exp(-0.5*pow(zz/H,2.0));
        d.Vc(VX1,k,j,i) = 0.0;
        d.Vc(VX2,k,j,i) = 0.0;
        d.Vc(VX3,k,j,i) = 0.0;
        d.Vc(PRS,k,j,i) = prs0;
      }
    }
  }
  d.SyncToDevice();
}
