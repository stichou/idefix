// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityShearingBox2D.cpp
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
#include <sstream>
#include <string>
#include <vector>

#include "idefix.hpp"
#include "selfGravityShearingBox2D.hpp"
#include "heffteFft.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "grid.hpp"

namespace {
// Scaled complementary error function exp(q^2) erfc(q), overflow-safe on device
KOKKOS_INLINE_FUNCTION real ErfcX(real q) {
  if(q*q > 600.0) {
    return (1.0/(q*1.772453850905516027))*(1.0 - 0.5/(q*q));
  }
  return exp(q*q)*erfc(q);
}
}

void SelfGravityShearingBox2D::Init(Input &input, DataBlock *datain) {
  idfx::pushRegion("SelfGravityShearingBox2D::Init");
  this->data = datain;
  Grid *grid = data->mygrid;
  CheckCompatibility();

  // Work arrays on the active (ghost-free) local domain
  rho = IdefixArray3D<real>("SGSB2D_Density", data->np_int[KDIR],
                                            data->np_int[JDIR],
                                            data->np_int[IDIR]);
  phiPacked = IdefixArray3D<real>("SGSB2D_Potential", data->np_int[KDIR],
                                                 data->np_int[JDIR],
                                                 data->np_int[IDIR]);
  phi = IdefixArray3D<real>("SGSB2D_Phi", data->np_tot[KDIR],
                                        data->np_tot[JDIR],
                                        data->np_tot[IDIR]);

  // Kernel parameters
  std::string strKernel = input.GetOrSet<std::string>("SelfGravity","kernel",0,"razor");
  if(strKernel == "razor" || strKernel == "razor-thin") {
    kernelType = RAZOR;
  } else if(strKernel == "bessel") {
    kernelType = BESSEL;
  } else {
    std::stringstream msg;
    msg << "SelfGravityShearingBox2D: Unknown kernel \"" << strKernel << "\". "
        << "Use \"razor\" or \"bessel\".";
    IDEFIX_ERROR(msg);
  }
  scaleHeight = input.Get<real>("SelfGravity","scaleHeight",0);
  if(scaleHeight <= 0.0) {
    IDEFIX_ERROR("SelfGravityShearingBox2D: scaleHeight must be strictly positive.");
  }
  if(kernelType == RAZOR) {
    lambda = input.Get<real>("SelfGravity","lambda",0);
    if(lambda < 0.0) {
      IDEFIX_ERROR("SelfGravityShearingBox2D: lambda must be positive.");
    }
    if(lambda >= scaleHeight) {
      IDEFIX_ERROR("SelfGravityShearingBox2D: lambda must be smaller than the "
                   "disc scale height (scaleHeight).");
    }
  } else {
    eta = input.Get<real>("SelfGravity","eta",0);
    if(eta <= 0.0) {
      IDEFIX_ERROR("SelfGravityShearingBox2D: eta must be strictly positive.");
    }
  }

  // Global and local descriptions of the runtime domain decomposition (i,j,k order)
  worldSize = {grid->np_int[IDIR], grid->np_int[JDIR], grid->np_int[KDIR]};
  std::array<int,3> localSize = {data->np_int[IDIR], data->np_int[JDIR], data->np_int[KDIR]};
  std::array<int,3> localOffset = {data->gbeg[IDIR] - data->nghost[IDIR],
                                   data->gbeg[JDIR] - data->nghost[JDIR],
                                   data->gbeg[KDIR] - data->nghost[KDIR]};

  fft = std::make_unique<HeffteFft>(worldSize, localSize, localOffset);

  outLow = fft->getOutboxLow();
  outSize = fft->getOutboxSize();
  rhoF = IdefixArray3D<complex>("SGSB2D_rhoHat", outSize[2],
                                              outSize[1],
                                              outSize[0]);
  phiF = IdefixArray3D<complex>("SGSB2D_phiHat", outSize[2],
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

void SelfGravityShearingBox2D::CheckCompatibility() {
  #if GEOMETRY != CARTESIAN
  IDEFIX_ERROR("SelfGravityShearingBox2D supports CARTESIAN geometry only.");
  #endif
  #if DIMENSIONS != 2
  IDEFIX_ERROR("SelfGravityShearingBox2D requires a two-dimensional setup "
               "(DIMENSIONS = 2).");
  #endif
  Grid *grid = data->mygrid;
  for(int dir = 0; dir < DIMENSIONS; dir++) {
    if(grid->lbound[dir] != periodic || grid->rbound[dir] != periodic) {
      IDEFIX_ERROR("SelfGravityShearingBox2D requires periodic boundary "
                   "conditions in the shearing box plane.");
    }
  }
  if(grid->nproc[IDIR] > 1 && grid->nproc[JDIR] > 1) {
    IDEFIX_ERROR("SelfGravityShearingBox2D requires a slab decomposition: "
                 "use -dec n 1 or -dec 1 n.");
  }
}

void SelfGravityShearingBox2D::ShowConfig() {
  idfx::cout << "SelfGravityShearingBox2D: Using the heFFTe-based 2D shearing box "
             << "spectral solver (slab decomposition)." << std::endl;
  if(kernelType == RAZOR) {
    idfx::cout << "SelfGravityShearingBox2D: razor-thin disc kernel with lambda = "
               << lambda << " (H = " << scaleHeight << ")." << std::endl;
  } else {
    idfx::cout << "SelfGravityShearingBox2D: Bessel kernel with eta = " << eta
               << " (H = " << scaleHeight << ")." << std::endl;
  }
  if(skipSelfGravity > 1) {
    idfx::cout << "SelfGravityShearingBox2D: self-gravity field will be updated every "
               << skipSelfGravity << " cycles." << std::endl;
  }
}

void SelfGravityShearingBox2D::EnforcePeriodic(int dir, BoundarySide side,
                                               IdefixArray3D<real> &array) {
  idfx::pushRegion("SelfGravityShearingBox2D::EnforcePeriodic");

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
    idefix_for("SGSB2D_BoundaryPeriodic", kbeg, kend, jbeg, jend, ibeg, iend,
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

void SelfGravityShearingBox2D::SetBoundaries(IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox2D::SetBoundaries");

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

void SelfGravityShearingBox2D::SolvePoisson() {
  idfx::pushRegion("SelfGravityShearingBox2D::SolvePoisson");
  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  Grid *grid = data->mygrid;

  // Pack the active density into the contiguous input box
  IdefixArray4D<real> Vc = data->hydro->Vc;
  IdefixArray3D<real> rhoLocal = rho;
  const int ibeg = data->beg[IDIR];
  const int jbeg = data->beg[JDIR];
  const int kbeg = data->beg[KDIR];
  idefix_for("SGSB2D_PackRho", data->beg[KDIR], data->end[KDIR],
                              data->beg[JDIR], data->end[JDIR],
                              data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      rhoLocal(k - kbeg, j - jbeg, i - ibeg) = Vc(RHO, k, j, i);
    });

  fft->R2C(rho, rhoF);

  // Apply the self-gravity kernel in Fourier space:
  //   razor-thin disc: phiHat = -2 pi G exp(-lambda k) rhoHat / k
  //   Bessel potential: phiHat = -2 pi G exp((eta k H)^2) erfc(eta k H) rhoHat / k
  IdefixArray3D<complex> rhoFLocal = rhoF;
  IdefixArray3D<complex> phiFLocal = phiF;
  const std::array<int,3> low = outLow;
  const std::array<int,3> nloc = outSize;
  const std::array<int,3> nGlob = worldSize;
  const std::array<real,3> Lx = {grid->xend[IDIR]-grid->xbeg[IDIR],
                                  grid->xend[JDIR]-grid->xbeg[JDIR],
                                  grid->xend[KDIR]-grid->xbeg[KDIR]};
  const real gravCst = data->gravity->gravCst;
  const real lambdaLoc = this->lambda;
  const real etaLoc = this->eta;
  const real hLoc = this->scaleHeight;
  const KernelType kernelLoc = this->kernelType;
  const real TWO_PI = 2.0*M_PI;
  idefix_for("SGSB2D_KernelFFT", 0, nloc[2], 0, nloc[1], 0, nloc[0],
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int gi = low[0] + i;
        const int gj = low[1] + j;
        real k2 = 0.0;
        // Global mode number along each direction, inactive directions carry k=0
        if(nGlob[0] > 1) {
          const int n = (2*gi <= nGlob[0]) ? gi : nGlob[0] - gi;
          k2 += (TWO_PI*n/Lx[0]) * (TWO_PI*n/Lx[0]);
        }
        if(nGlob[1] > 1) {
          const int n = (2*gj <= nGlob[1]) ? gj : nGlob[1] - gj;
          k2 += (TWO_PI*n/Lx[1]) * (TWO_PI*n/Lx[1]);
        }
        real kern = 0.0;
        if(k2 > 0.0) {
          const real k = sqrt(k2);
          const real inv_k = 1.0/k;
          if(kernelLoc == RAZOR) {
            kern = TWO_PI*gravCst*exp(-lambdaLoc*k)*inv_k;
          } else {
            kern = TWO_PI*gravCst*inv_k*ErfcX(etaLoc*k*hLoc);
          }
        }
        phiFLocal(k,j,i) = -rhoFLocal(k,j,i) * kern;
      });

  fft->C2R(phiF, phiPacked);

  // Unpack the potential and fill the ghost cells
  IdefixArray3D<real> phiPackedLocal = phiPacked;
  IdefixArray3D<real> phiLocal = phi;
  const int ibegPhi = data->beg[IDIR];
  const int jbegPhi = data->beg[JDIR];
  const int kbegPhi = data->beg[KDIR];
  idefix_for("SGSB2D_UnpackPhi", data->beg[KDIR], data->end[KDIR],
                                data->beg[JDIR], data->end[JDIR],
                                data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      phiLocal(k, j, i) = phiPackedLocal(k - kbegPhi, j - jbegPhi, i - ibegPhi);
    });

  SetBoundaries(phi);

  elapsedTime += timer.seconds();

  idfx::popRegion();
}

void SelfGravityShearingBox2D::AddSelfGravityPotential(IdefixArray3D<real> &phiP) {
  idfx::pushRegion("SelfGravityShearingBox2D::AddSelfGravityPotential");
  // The 2 pi G factor of the kernels is already included in the potential
  IdefixArray3D<real> localPot = phiP;
  IdefixArray3D<real> pot = this->phi;

  idefix_for("SGSB2D_AddSelfGravityPotential", 0, data->np_tot[KDIR],
                                               0, data->np_tot[JDIR],
                                               0, data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      localPot(k,j,i) += pot(k, j, i);
    });
  idfx::popRegion();
}

void SelfGravityShearingBox2D::EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) {
  (void) myFunc;
  IDEFIX_ERROR("SelfGravityShearingBox2D only supports periodic boundaries; "
               "userdef boundaries are not supported.");
}

#endif // WITH_HEFFTE
