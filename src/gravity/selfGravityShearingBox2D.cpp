// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityShearingBox2D.cpp
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
  sbS = data->hydro->sbS;
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
  remapScratch = IdefixArray3D<real>("SGSB2D_Remap", data->np_int[KDIR],
                                                 data->np_int[JDIR],
                                                 data->np_int[IDIR]);
  ghostScratch = IdefixArray3D<real>("SGSB2D_GhostScratch", data->np_tot[KDIR],
                                               data->np_tot[JDIR],
                                               2*data->nghost[IDIR]);

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
  if(!data->hydro->haveShearingBox) {
    IDEFIX_ERROR("SelfGravityShearingBox2D requires the shearing box module: "
                 "define [Hydro] shearingBox in the input file "
                 "(use 0 for a non-shearing setup).");
  }
  if(sbS != 0.0) {
    // A non-zero shear makes the problem shear-periodic in x
    for(int dir = 0; dir < DIMENSIONS; dir++) {
      const bool lOk = (grid->lbound[dir] == periodic || grid->lbound[dir] == shearingbox);
      const bool rOk = (grid->rbound[dir] == periodic || grid->rbound[dir] == shearingbox);
      if(!lOk || !rOk) {
        IDEFIX_ERROR("SelfGravityShearingBox2D requires periodic or shearing box "
                     "boundary conditions in the shearing box plane.");
      }
    }
  } else {
    for(int dir = 0; dir < DIMENSIONS; dir++) {
      if(grid->lbound[dir] != periodic || grid->rbound[dir] != periodic) {
        IDEFIX_ERROR("SelfGravityShearingBox2D requires periodic boundary "
                     "conditions in the shearing box plane.");
      }
    }
  }
  if(grid->nproc[IDIR] > 1 && grid->nproc[JDIR] > 1) {
    IDEFIX_ERROR("SelfGravityShearingBox2D requires a slab decomposition: "
                 "use -dec n 1 or -dec 1 n.");
  }
  if(sbS != 0.0 && grid->nproc[JDIR] > 1) {
    IDEFIX_ERROR("SelfGravityShearingBox2D with a non-zero shear requires no domain "
                  "decomposition along X2: use -dec n 1.");
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

void SelfGravityShearingBox2D::EnforceShearingBoxPhi(real t, int dir, BoundarySide side,
                                                      IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox2D::EnforceShearingBoxPhi");
  if(dir != IDIR) {
    IDEFIX_ERROR("Shearing box boundaries can only be applied along the X1 direction");
  }

  IdefixArray3D<real> localVar = arr;

  const int nxi = data->np_int[IDIR];
  const int nxj = data->np_int[JDIR];
  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];

  const int istart = side*(ighost+nxi);
  const int iend = ighost + side*(ighost+nxi);

  const real Lx = data->mygrid->xend[IDIR] - data->mygrid->xbeg[IDIR];
  const real Ly = data->mygrid->xend[JDIR] - data->mygrid->xbeg[JDIR];
  const real dy = Ly/nxj;

  // Offset of the shear-periodic boundary, following the hydro Boundary module
  const int sign = 2*side-1;
  const real dL = ShearRemap::PositiveMod(sign*sbS*Lx*t, Ly);

  IdefixArray3D<real> scratch = ghostScratch;
  idefix_for("SGSB2D_ShearingBoxPhiCopy", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
                                            0, ighost,
      KOKKOS_LAMBDA (int k, int j, int q) {
        scratch(k,j,q) = localVar(k,j,istart+q);
      });
  idefix_for("SGSB2D_ShearingBoxPhi", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
                                           istart, iend,
      KOKKOS_LAMBDA (int k, int j, int i) {
        // sample the periodic-filled neighbour state at y_j - dL
        const real pos = j - dL/dy;
        const int j0 = static_cast<int>(Kokkos::floor(pos));
        const real frac = pos - j0;
        const int ja = jghost + (((j0 - jghost) % nxj + nxj) % nxj);
        const int jb = jghost + (((j0+1 - jghost) % nxj + nxj) % nxj);
        const int q = i - istart;
        localVar(k,j,i) = (1.0-frac)*scratch(k,ja,q) + frac*scratch(k,jb,q);
      });
  idfx::popRegion();
}

void SelfGravityShearingBox2D::SetBoundaries(IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox2D::SetBoundaries");

  #ifdef WITH_MPI
  IdefixArray4D<real> arr4D = IdefixArray4D<real> (arr.data(), 1, data->np_tot[KDIR],
                                                    data->np_tot[JDIR],
                                                    data->np_tot[IDIR]);
  #endif

  const real t = data->t;
  const bool haveShear = (sbS != 0.0);
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

    if(dir == IDIR && haveShear) {
      // Periodicity (or MPI exchange) first, then the shear offset.
      // The shear-periodic condition only applies at the domain edges:
      // internal MPI boundaries are continuous and must not be shifted.
      if(data->mygrid->nproc[dir] == 1) {
        EnforcePeriodic(dir, left, arr);
        EnforcePeriodic(dir, right, arr);
      }
      if(data->lbound[IDIR] != internal) EnforceShearingBoxPhi(t, dir, left, arr);
      if(data->rbound[IDIR] != internal) EnforceShearingBoxPhi(t, dir, right, arr);
    } else {
      EnforcePeriodic(dir, left, arr);
      EnforcePeriodic(dir, right, arr);
    }
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

  // Forward shear remap: map the density to the fully periodic frame
  // (each x-column is shifted along y by -S x t)
  IdefixArray3D<real> fftInput = rho;
  if(sbS != 0.0) {
    const real Ly = grid->xend[JDIR]-grid->xbeg[JDIR];
    const real dy = Ly/grid->np_int[JDIR];
    const int iOffset = data->gbeg[IDIR];  // grid->x is ghost-padded
    ShearRemap::ShiftAlongJ("SGSB2D_RemapFwd", rho, remapScratch, grid->x[IDIR],
                            +1.0, sbS, data->t, Ly, dy, iOffset,
                            data->np_int[KDIR], data->np_int[JDIR],
                            data->np_int[IDIR]);
    fftInput = remapScratch;
  }

  fft->R2C(fftInput, rhoF);

  // Effective in-plane wavevector: kx -> kx + (Delta_y0 / Lx) ky,
  // with Delta_y0 = mod(S Lx t, Ly) (eq. 3 of Rendon Restrepo & Gressel 2026)
  const real ddy = (sbS != 0.0)
                   ? ShearRemap::PositiveMod(sbS*(grid->xend[IDIR]-grid->xbeg[IDIR])*data->t,
                                             grid->xend[JDIR]-grid->xbeg[JDIR])
                     /(grid->xend[IDIR]-grid->xbeg[IDIR])
                   : 0.0;

  // Apply the self-gravity kernel in Fourier space:
  //   razor-thin disc: phiHat = -2 pi G exp(-lambda k) rhoHat / k
  //   Bessel potential: phiHat = -2 pi G exp((eta k H)^2) erfc(eta k H) rhoHat / k
  IdefixArray3D<complex> rhoFLocal = rhoF;
  IdefixArray3D<complex> phiFLocal = phiF;
  const std::array<int,3> nloc = outSize;
  const int low0 = outLow[0];
  const int low1 = outLow[1];
  const int nGlob0 = worldSize[0];
  const int nGlob1 = worldSize[1];
  const real Lx0 = grid->xend[IDIR]-grid->xbeg[IDIR];
  const real Lx1 = grid->xend[JDIR]-grid->xbeg[JDIR];
  const real gravCst = data->gravity->gravCst;
  const real lambdaLoc = this->lambda;
  const real etaLoc = this->eta;
  const real hLoc = this->scaleHeight;
  const KernelType kernelLoc = this->kernelType;
  const real TWO_PI = 2.0*M_PI;
  idefix_for("SGSB2D_KernelFFT", 0, nloc[2], 0, nloc[1], 0, nloc[0],
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int gi = low0 + i;
        const int gj = low1 + j;
        real kx = 0.0;
        real ky = 0.0;
        // Global mode number along each direction, inactive directions carry k=0
        if(nGlob0 > 1) {
          const int n = (2*gi <= nGlob0) ? gi : nGlob0 - gi;
          kx = TWO_PI*n/Lx0;
        }
        if(nGlob1 > 1) {
          // signed y mode number: the ddy correction is odd in ky
          const int n = (2*gj <= nGlob1) ? gj : gj - nGlob1;
          ky = TWO_PI*n/Lx1;
        }
        const real kxeff = kx - ddy*ky;
        const real k2 = kxeff*kxeff + ky*ky;
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

  // Inverse shear remap: map the potential back to the shearing frame
  IdefixArray3D<real> fftOutput = phiPacked;
  if(sbS != 0.0) {
    const real Ly = grid->xend[JDIR]-grid->xbeg[JDIR];
    const real dy = Ly/grid->np_int[JDIR];
    const int iOffset = data->gbeg[IDIR];  // grid->x is ghost-padded
    ShearRemap::ShiftAlongJ("SGSB2D_RemapBwd", phiPacked, remapScratch, grid->x[IDIR],
                            -1.0, sbS, data->t, Ly, dy, iOffset,
                            data->np_int[KDIR], data->np_int[JDIR],
                            data->np_int[IDIR]);
    fftOutput = remapScratch;
  }

  // Unpack the potential and fill the ghost cells
  IdefixArray3D<real> phiPackedLocal = fftOutput;
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
