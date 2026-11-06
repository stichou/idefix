// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityShearingBox3D.cpp
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

// Implementation of the VGF-HybridBC spectral Poisson solver
// (Rendon Restrepo & Gressel 2026, A&A 706, A65, arXiv:2510.10070),
// for shearing boxes with periodic boundaries in the x-y plane and
// vacuum (free-space) boundary conditions in the vertical direction.
// The y / kx shear shift is ignored at this stage.

#ifdef WITH_HEFFTE

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "idefix.hpp"
#include "selfGravityShearingBox3D.hpp"
#include "heffteFft.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"
#include "grid.hpp"

namespace {
// Vertical zero-padding factor required by the VGF-HybridBC method:
// the oscillatory nature of the Green's function and the aperiodic
// convolution in z demand quadrupling the vertical domain.
constexpr int VGF_PADDING = 4;
}

void SelfGravityShearingBox3D::Init(Input &input, DataBlock *datain) {
  idfx::pushRegion("SelfGravityShearingBox3D::Init");
  this->data = datain;
  Grid *grid = data->mygrid;
  CheckCompatibility();

  // Work arrays on the active (ghost-free) local domain, vertically padded
  rho = IdefixArray3D<real>("SGSB3D_Density", VGF_PADDING*data->np_int[KDIR],
                                            data->np_int[JDIR],
                                            data->np_int[IDIR]);
  phiPacked = IdefixArray3D<real>("SGSB3D_Potential", VGF_PADDING*data->np_int[KDIR],
                                                 data->np_int[JDIR],
                                                 data->np_int[IDIR]);
  phi = IdefixArray3D<real>("SGSB3D_Phi", data->np_tot[KDIR],
                                        data->np_tot[JDIR],
                                        data->np_tot[IDIR]);

  // Global and local descriptions of the padded FFT domain (i,j,k order)
  worldSize = {grid->np_int[IDIR], grid->np_int[JDIR], grid->np_int[KDIR]};
  worldSizePad = {grid->np_int[IDIR], grid->np_int[JDIR],
                  VGF_PADDING*grid->np_int[KDIR]};
  std::array<int,3> localSize = {data->np_int[IDIR], data->np_int[JDIR],
                                  VGF_PADDING*data->np_int[KDIR]};
  std::array<int,3> localOffset = {data->gbeg[IDIR] - data->nghost[IDIR],
                                   data->gbeg[JDIR] - data->nghost[JDIR],
                                   0};

  fft = std::make_unique<HeffteFft>(worldSizePad, localSize, localOffset);

  outLow = fft->getOutboxLow();
  outSize = fft->getOutboxSize();
  rhoF = IdefixArray3D<complex>("SGSB3D_rhoHat", outSize[2],
                                              outSize[1],
                                              outSize[0]);
  phiF = IdefixArray3D<complex>("SGSB3D_phiHat", outSize[2],
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

void SelfGravityShearingBox3D::CheckCompatibility() {
  #if GEOMETRY != CARTESIAN
  IDEFIX_ERROR("SelfGravityShearingBox3D supports CARTESIAN geometry only.");
  #endif
  #if DIMENSIONS != 3
  IDEFIX_ERROR("SelfGravityShearingBox3D requires a three-dimensional setup "
               "(DIMENSIONS = 3).");
  #endif
  Grid *grid = data->mygrid;
  for(int dir = IDIR; dir <= JDIR; dir++) {
    if(grid->lbound[dir] != periodic || grid->rbound[dir] != periodic) {
      IDEFIX_ERROR("SelfGravityShearingBox3D requires periodic boundary "
                   "conditions in the shearing box plane (X1 and X2).");
    }
  }
  if(grid->lbound[KDIR] == periodic || grid->rbound[KDIR] == periodic) {
    IDEFIX_ERROR("SelfGravityShearingBox3D imposes vacuum boundary conditions "
                 "in the vertical direction; fully periodic setups should use "
                 "the FFT solver instead.");
  }
  if(grid->nproc[KDIR] > 1) {
    IDEFIX_ERROR("SelfGravityShearingBox3D requires no domain decomposition "
                 "along X3: use a slab (-dec n 1 1) or pencil (-dec n m 1) "
                 "decomposition.");
  }
}

void SelfGravityShearingBox3D::ShowConfig() {
  idfx::cout << "SelfGravityShearingBox3D: Using the VGF-HybridBC spectral solver "
             << "(vacuum vertical BCs, vertical padding x"
             << VGF_PADDING << ", alpha = " << alpha << ")." << std::endl;
  if(skipSelfGravity > 1) {
    idfx::cout << "SelfGravityShearingBox3D: self-gravity field will be updated every "
               << skipSelfGravity << " cycles." << std::endl;
  }
}

void SelfGravityShearingBox3D::EnforcePeriodic(int dir, BoundarySide side,
                                               IdefixArray3D<real> &array) {
  idfx::pushRegion("SelfGravityShearingBox3D::EnforcePeriodic");

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
    idefix_for("SGSB3D_BoundaryPeriodic", kbeg, kend, jbeg, jend, ibeg, iend,
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

void SelfGravityShearingBox3D::ExtrapolateVerticalGhost(IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox3D::ExtrapolateVerticalGhost");

  IdefixArray3D<real> localVar = arr;
  const int kbeg = data->beg[KDIR];
  const int kend = data->end[KDIR] - 1;
  const int kghost = data->nghost[KDIR];

  idefix_for("SGSB3D_VerticalGhost", 0, data->np_tot[JDIR], 0, data->np_tot[IDIR],
      KOKKOS_LAMBDA (int j, int i) {
        const real gradDown = localVar(kbeg+1,j,i) - localVar(kbeg,j,i);
        const real gradUp = localVar(kend,j,i) - localVar(kend-1,j,i);
        for(int m = 1; m <= kghost; m++) {
          localVar(kbeg-m,j,i) = localVar(kbeg,j,i) - m*gradDown;
          localVar(kend+m,j,i) = localVar(kend,j,i) + m*gradUp;
        }
      });
  idfx::popRegion();
}

void SelfGravityShearingBox3D::SetBoundaries(IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox3D::SetBoundaries");

  #ifdef WITH_MPI
  IdefixArray4D<real> arr4D = IdefixArray4D<real> (arr.data(), 1, data->np_tot[KDIR],
                                                    data->np_tot[JDIR],
                                                    data->np_tot[IDIR]);
  #endif

  for(int dir = IDIR; dir <= JDIR; dir++) {
    #ifdef WITH_MPI
    if(data->mygrid->nproc[dir]>1) {
      switch(dir) {
        case 0:
          this->mpi.ExchangeX1(arr4D);
          break;
        case 1:
          this->mpi.ExchangeX2(arr4D);
          break;
      }
    }
    #endif

    EnforcePeriodic(dir, left, arr);
    EnforcePeriodic(dir, right, arr);
  }

  ExtrapolateVerticalGhost(arr);

  idfx::popRegion();
}

void SelfGravityShearingBox3D::SolvePoisson() {
  idfx::pushRegion("SelfGravityShearingBox3D::SolvePoisson");
  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  Grid *grid = data->mygrid;

  // Pack the active density into the first quarter of the padded input box
  IdefixArray4D<real> Vc = data->hydro->Vc;
  IdefixArray3D<real> rhoLocal = rho;
  const int ibeg = data->beg[IDIR];
  const int jbeg = data->beg[JDIR];
  const int kbeg = data->beg[KDIR];
  const int nk = data->np_int[KDIR];
  const int nkPad = VGF_PADDING*nk;
  idefix_for("SGSB3D_PackRho", 0, nkPad,
                              data->beg[JDIR], data->end[JDIR],
                              data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      rhoLocal(k, j - jbeg, i - ibeg) = (k < nk) ? Vc(RHO, k + kbeg, j, i) : 0.0;
    });

  fft->R2C(rho, rhoF);

  // Apply the VGF-HybridBC Green's function in Fourier space
  // (eq. 14 of Rendon Restrepo & Gressel 2026), with L = alpha Lz:
  //   G = L^2/2                                     if k = 0, kz = 0
  //   G = [kz L sin(kz L) + cos(kz L) - 1] / kz^2    if k = 0
  //   G = -[e^{-kL}((kz/k) sin(kz L) - cos(kz L)) + 1] / (k^2+kz^2)  otherwise
  // and phiHat = 4 pi G_cst * G * rhoHat
  IdefixArray3D<complex> rhoFLocal = rhoF;
  IdefixArray3D<complex> phiFLocal = phiF;
  const std::array<int,3> low = outLow;
  const std::array<int,3> nloc = outSize;
  const std::array<int,3> nGlob = worldSizePad;
  const std::array<real,3> Lx = {grid->xend[IDIR]-grid->xbeg[IDIR],
                                  grid->xend[JDIR]-grid->xbeg[JDIR],
                                  grid->xend[KDIR]-grid->xbeg[KDIR]};
  const real gravCst = data->gravity->gravCst;
  const real L = this->alpha * Lx[KDIR];
  const real TWO_PI = 2.0*M_PI;
  const real PI4G = 4.0*M_PI*gravCst;
  idefix_for("SGSB3D_KernelFFT", 0, nloc[2], 0, nloc[1], 0, nloc[0],
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int gi = low[0] + i;
        const int gj = low[1] + j;
        const int gk = low[2] + k;
        // In-plane wave numbers (periodic dimensions)
        const int ni = (2*gi <= nGlob[0]) ? gi : nGlob[0] - gi;
        const int nj = (2*gj <= nGlob[1]) ? gj : nGlob[1] - gj;
        const real kx = TWO_PI*ni/Lx[0];
        const real ky = TWO_PI*nj/Lx[1];
        const real k2d = kx*kx + ky*ky;
        // Vertical wave number of the padded domain
        const int nkp = (2*gk <= nGlob[2]) ? gk : nGlob[2] - gk;
        const real kz = TWO_PI*nkp/(VGF_PADDING*Lx[2]);

        real Gk = 0.0;
        if(k2d == 0.0 && kz == 0.0) {
          Gk = 0.5*L*L;
        } else if(k2d == 0.0) {
          Gk = (kz*L*sin(kz*L) + cos(kz*L) - 1.0)/(kz*kz);
        } else {
          const real k = sqrt(k2d);
          Gk = -(exp(-k*L)*((kz/k)*sin(kz*L) - cos(kz*L)) + 1.0)/(k2d + kz*kz);
        }
        phiFLocal(k,j,i) = rhoFLocal(k,j,i) * (PI4G*Gk);
      });

  fft->C2R(phiF, phiPacked);

  // Unpack the potential (first quarter of the padded box) and fill the ghosts
  IdefixArray3D<real> phiPackedLocal = phiPacked;
  IdefixArray3D<real> phiLocal = phi;
  const int ibegPhi = data->beg[IDIR];
  const int jbegPhi = data->beg[JDIR];
  const int kbegPhi = data->beg[KDIR];
  idefix_for("SGSB3D_UnpackPhi", data->beg[KDIR], data->end[KDIR],
                                data->beg[JDIR], data->end[JDIR],
                                data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      phiLocal(k, j, i) = phiPackedLocal(k - kbegPhi, j - jbegPhi, i - ibegPhi);
    });

  SetBoundaries(phi);

  elapsedTime += timer.seconds();

  idfx::popRegion();
}

void SelfGravityShearingBox3D::AddSelfGravityPotential(IdefixArray3D<real> &phiP) {
  idfx::pushRegion("SelfGravityShearingBox3D::AddSelfGravityPotential");
  // The 4 pi G factor of the VGF-HybridBC kernel is already included
  IdefixArray3D<real> localPot = phiP;
  IdefixArray3D<real> pot = this->phi;

  idefix_for("SGSB3D_AddSelfGravityPotential", 0, data->np_tot[KDIR],
                                               0, data->np_tot[JDIR],
                                               0, data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      localPot(k,j,i) += pot(k, j, i);
    });
  idfx::popRegion();
}

void SelfGravityShearingBox3D::EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) {
  (void) myFunc;
  IDEFIX_ERROR("SelfGravityShearingBox3D only supports periodic boundaries in the "
               "shearing box plane; userdef boundaries are not supported.");
}

#endif // WITH_HEFFTE
