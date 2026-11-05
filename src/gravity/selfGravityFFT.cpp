// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityFFT.cpp
//
// Last modified : 11/2026
//
// Copyright(C) by :
// - Steven Rendon Restrepo <srendon@aip.de> (AIP - 2026)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifdef WITH_HEFFTE

#include <cmath>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "idefix.hpp"
#include "selfGravityFFT.hpp"
#include "heffteFft.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "grid.hpp"

void SelfGravityFFT::Init(Input &input, DataBlock *datain) {
  idfx::pushRegion("SelfGravityFFT::Init");
  this->data = datain;
  Grid *grid = data->mygrid;
  CheckCompatibility();

  // Work arrays on the active (ghost-free) local domain
  rho = IdefixArray3D<real>("SGFFT_Density", data->np_int[KDIR],
                                           data->np_int[JDIR],
                                           data->np_int[IDIR]);
  phiPacked = IdefixArray3D<real>("SGFFT_Potential", data->np_int[KDIR],
                                                data->np_int[JDIR],
                                                data->np_int[IDIR]);
  phi = IdefixArray3D<real>("SGFFT_Phi", data->np_tot[KDIR],
                                       data->np_tot[JDIR],
                                       data->np_tot[IDIR]);

  // Global and local descriptions of the runtime domain decomposition (i,j,k order)
  worldSize = {grid->np_int[IDIR], grid->np_int[JDIR], grid->np_int[KDIR]};
  std::array<int,3> localSize = {data->np_int[IDIR], data->np_int[JDIR], data->np_int[KDIR]};
  std::array<int,3> localOffset = {data->gbeg[IDIR] - data->nghost[IDIR],
                                   data->gbeg[JDIR] - data->nghost[JDIR],
                                   data->gbeg[KDIR] - data->nghost[KDIR]};

  fft = std::make_unique<HeffteFft>(worldSize, localSize, localOffset);

  outLow = fft->getOutboxLow();
  outSize = fft->getOutboxSize();
  rhoF = IdefixArray3D<complex>("SGFFT_rhoHat", outSize[2],
                                              outSize[1],
                                              outSize[0]);
  phiF = IdefixArray3D<complex>("SGFFT_phiHat", outSize[2],
                                              outSize[1],
                                              outSize[0]);

  #ifdef WITH_MPI
  std::vector<int> mapVars;
  mapVars.push_back(0);
  this->mpi.Init(data->mygrid, mapVars, data->nghost,
                 data->np_int, data->lbound, data->rbound, false);
  #endif

  idfx::popRegion();
}

void SelfGravityFFT::CheckCompatibility() {
  #if GEOMETRY != CARTESIAN
  IDEFIX_ERROR("SelfGravityFFT supports CARTESIAN geometry only.");
  #endif
  Grid *grid = data->mygrid;
  for(int dir = 0; dir < DIMENSIONS; dir++) {
    if(grid->lbound[dir] != periodic || grid->rbound[dir] != periodic) {
      IDEFIX_ERROR("SelfGravityFFT requires periodic boundary "
                   "conditions in all active dimensions.");
    }
  }
  if(grid->nproc[KDIR] > 1) {
    IDEFIX_ERROR("SelfGravityFFT currently requires no domain decomposition "
                 "along X3: use a slab (-dec n 1 1) or pencil (-dec n m 1) "
                 "decomposition.");
  }
}

void SelfGravityFFT::ShowConfig() {
  idfx::cout << "SelfGravityFFT: Using the heFFTe-based FFT Poisson solver "
             << "(periodic Cartesian, decomposition dictated by -dec)." << std::endl;
  if(skipSelfGravity > 1) {
    idfx::cout << "SelfGravityFFT: self-gravity field will be updated every "
               << skipSelfGravity << " cycles." << std::endl;
  }
}

void SelfGravityFFT::EnforcePeriodic(int dir, BoundarySide side, IdefixArray3D<real> &array) {
  idfx::pushRegion("SelfGravityFFT::EnforcePeriodic");

  IdefixArray3D<real> localVar = array;

  const int nxi = data->np_int[IDIR];
  const int nxj = data->np_int[JDIR];
  const int nxk = data->np_int[KDIR];

  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];
  const int kghost = data->nghost[KDIR];

  const int ibeg = (dir == IDIR) ? side*(ighost+nxi) : 0;
  const int iend = (dir == IDIR) ? ighost + side*(ighost+nxi) : data->np_tot[IDIR];
  const int jbeg = (dir == JDIR) ? side*(jghost+nxj) : 0;
  const int jend = (dir == JDIR) ? jghost + side*(jghost+nxj) : data->np_tot[JDIR];
  const int kbeg = (dir == KDIR) ? side*(kghost+nxk) : 0;
  const int kend = (dir == KDIR) ? kghost + side*(kghost+nxk) : data->np_tot[KDIR];

  // Periodicity across MPI boundaries is already enforced by the Mpi exchanges
  if(data->mygrid->nproc[dir] == 1) {
    idefix_for("SGFFT_BoundaryPeriodic", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
            int iref, jref, kref;
            if(dir==IDIR)
              iref = ighost + (i+ighost*(nxi-1))%nxi;
            else
              iref = i;
            if(dir==JDIR)
              jref = jghost + (j+jghost*(nxj-1))%nxj;
            else
              jref = j;
            if(dir==KDIR)
              kref = kghost + (k+kghost*(nxk-1))%nxk;
            else
              kref = k;

            localVar(k,j,i) = localVar(kref,jref,iref);
    });
  }
  idfx::popRegion();
}

void SelfGravityFFT::SetBoundaries(IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityFFT::SetBoundaries");

  #ifdef WITH_MPI
  IdefixArray4D<real> arr4D = IdefixArray4D<real> (arr.data(), 1, data->np_tot[KDIR],
                                                    data->np_tot[JDIR],
                                                    data->np_tot[IDIR]);
  #endif

  for(int dir = 0 ; dir < DIMENSIONS ; dir++) {
    #ifdef WITH_MPI
    if(data->mygrid->nproc[dir]>1) {
      switch(dir) {
        case 0:
          this->mpi.ExchangeX1(arr4D);
          break;
        case 1:
          this->mpi.ExchangeX2(arr4D);
          break;
        case 2:
          this->mpi.ExchangeX3(arr4D);
          break;
      }
    }
    #endif

    EnforcePeriodic(dir, left, arr);
    EnforcePeriodic(dir, right, arr);
  }

  idfx::popRegion();
}

void SelfGravityFFT::SolvePoisson() {
  idfx::pushRegion("SelfGravityFFT::SolvePoisson");
  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  Grid *grid = data->mygrid;

  // Pack the active density into the contiguous input box
  IdefixArray4D<real> Vc = data->hydro->Vc;
  IdefixArray3D<real> rhoLocal = rho;
  const int ibeg = data->beg[IDIR];
  const int jbeg = data->beg[JDIR];
  const int kbeg = data->beg[KDIR];
  idefix_for("SGFFT_PackRho", data->beg[KDIR], data->end[KDIR],
                              data->beg[JDIR], data->end[JDIR],
                              data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      rhoLocal(k - kbeg, j - jbeg, i - ibeg) = Vc(RHO, k, j, i);
    });

  fft->R2C(rho, rhoF);

  // Inverse Poisson equation in Fourier space: phiHat = -rhoHat / k^2
  IdefixArray3D<complex> rhoFLocal = rhoF;
  IdefixArray3D<complex> phiFLocal = phiF;
  const std::array<int,3> low = outLow;
  const std::array<int,3> nloc = outSize;
  const std::array<int,3> nGlob = worldSize;
  const std::array<real,3> Lx = {grid->xend[IDIR]-grid->xbeg[IDIR],
                                  grid->xend[JDIR]-grid->xbeg[JDIR],
                                  grid->xend[KDIR]-grid->xbeg[KDIR]};
  idefix_for("SGFFT_PoissonFFT", 0, nloc[2], 0, nloc[1], 0, nloc[0],
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int gi = low[0] + i;
        const int gj = low[1] + j;
        const int gk = low[2] + k;
        real k2 = 0.0;
        // Global mode number along each direction, inactive directions carry k=0
        if(nGlob[0] > 1) {
          const int n = (2*gi <= nGlob[0]) ? gi : nGlob[0] - gi;
          k2 += (2.0*M_PI*n/Lx[0]) * (2.0*M_PI*n/Lx[0]);
        }
        if(nGlob[1] > 1) {
          const int n = (2*gj <= nGlob[1]) ? gj : nGlob[1] - gj;
          k2 += (2.0*M_PI*n/Lx[1]) * (2.0*M_PI*n/Lx[1]);
        }
        if(nGlob[2] > 1) {
          const int n = (2*gk <= nGlob[2]) ? gk : nGlob[2] - gk;
          k2 += (2.0*M_PI*n/Lx[2]) * (2.0*M_PI*n/Lx[2]);
        }
        const real inv_k2 = (k2 > 0.0) ? 1.0/k2 : 0.0;
        phiFLocal(k,j,i) = -rhoFLocal(k,j,i) * inv_k2;
      });

  fft->C2R(phiF, phiPacked);

  // Unpack the potential and fill the ghost cells
  IdefixArray3D<real> phiPackedLocal = phiPacked;
  IdefixArray3D<real> phiLocal = phi;
  const int ibegPhi = data->beg[IDIR];
  const int jbegPhi = data->beg[JDIR];
  const int kbegPhi = data->beg[KDIR];
  idefix_for("SGFFT_UnpackPhi", data->beg[KDIR], data->end[KDIR],
                                data->beg[JDIR], data->end[JDIR],
                                data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      phiLocal(k, j, i) = phiPackedLocal(k - kbegPhi, j - jbegPhi, i - ibegPhi);
    });

  SetBoundaries(phi);

  elapsedTime += timer.seconds();

  idfx::popRegion();
}

void SelfGravityFFT::AddSelfGravityPotential(IdefixArray3D<real> &phiP) {
  idfx::pushRegion("SelfGravityFFT::AddSelfGravityPotential");
  IdefixArray3D<real> localPot = phiP;
  IdefixArray3D<real> pot = this->phi;
  real gravCst = this->data->gravity->gravCst;

  idefix_for("SGFFT_AddSelfGravityPotential", 0, data->np_tot[KDIR],
                                             0, data->np_tot[JDIR],
                                             0, data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      localPot(k,j,i) += 4.*M_PI*gravCst * pot(k, j, i);
    });
  idfx::popRegion();
}

void SelfGravityFFT::EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) {
  (void) myFunc;
  IDEFIX_ERROR("SelfGravityFFT only supports periodic boundaries; "
               "userdef boundaries are not supported.");
}

#endif // WITH_HEFFTE
