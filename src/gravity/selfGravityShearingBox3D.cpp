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
// Vertical zero-padding factor required by the VGF-HybridBC method when the
// Green's function is applied spectrally (eq. 14 sampled on the padded kz
// grid): the oscillatory nature of the kernel and the aperiodic convolution
// in z demand quadrupling the vertical domain.
constexpr int VGF_PADDING_SPECTRAL = 4;
// Padding of the analytic-kernel variant: the closed-form real-space Green's
// function is compactly supported in |z| < alpha*Lz, and only lags
// |z| <= Lz-dz occur in the convolution, so the doubled grid is exact.
constexpr int VGF_PADDING_ANALYTIC = 2;
}  // namespace

void SelfGravityShearingBox3D::Init(Input &input, DataBlock *datain) {
  idfx::pushRegion("SelfGravityShearingBox3D::Init");
  this->data = datain;
  Grid *grid = data->mygrid;
  sbS = data->hydro->sbS;
  CheckCompatibility();

  // Two equivalent formulations of the VGF-HybridBC Green's function:
  // - "spectral": eq. 14 of Rendon Restrepo & Gressel 2026, sampled at the
  //   discrete kz modes of the quadrupled grid (historical behaviour).
  // - "analytic": the closed-form real-space kernel
  //     g_k(z) = -exp(-k|z|)/(2k) for |z| < L, 0 otherwise   (k > 0)
  //     g_0(z) = |z|/2 for |z| < L, 0 otherwise
  //   whose continuum Fourier transform is exactly eq. 14. Being compactly
  //   supported in |z| < L = alpha*Lz, it is applied on the doubled grid
  //   through its exact z-DFT (closed form, see SolvePoisson), which halves
  //   the FFT volume and the memory footprint of the solver.
  std::string strKernel = input.GetOrSet<std::string>("SelfGravity","kernel",0,"spectral");
  if(strKernel == "spectral") {
    useAnalyticKernel = false;
    vgfPadding = VGF_PADDING_SPECTRAL;
  } else if(strKernel == "analytic") {
    useAnalyticKernel = true;
    vgfPadding = VGF_PADDING_ANALYTIC;
  } else {
    std::stringstream msg;
    msg << "SelfGravityShearingBox3D: unknown [SelfGravity] kernel \"" << strKernel
        << "\". Use \"spectral\" or \"analytic\".";
    IDEFIX_ERROR(msg);
  }

  // Work arrays on the active (ghost-free) local domain, vertically padded
  rho = IdefixArray3D<real>("SGSB3D_Density", vgfPadding*data->np_int[KDIR],
                                            data->np_int[JDIR],
                                            data->np_int[IDIR]);
  phiPacked = IdefixArray3D<real>("SGSB3D_Potential", vgfPadding*data->np_int[KDIR],
                                                 data->np_int[JDIR],
                                                 data->np_int[IDIR]);
  phi = IdefixArray3D<real>("SGSB3D_Phi", data->np_tot[KDIR],
                                        data->np_tot[JDIR],
                                        data->np_tot[IDIR]);
  remapScratch = IdefixArray3D<real>("SGSB3D_Remap", vgfPadding*data->np_int[KDIR],
                                                 data->np_int[JDIR],
                                                 data->np_int[IDIR]);
  ghostScratch = IdefixArray3D<real>("SGSB3D_GhostScratch", data->np_tot[KDIR],
                                               data->np_tot[JDIR],
                                               2*data->nghost[IDIR]);

  // Global and local descriptions of the padded FFT domain (i,j,k order)
  worldSize = {grid->np_int[IDIR], grid->np_int[JDIR], grid->np_int[KDIR]};
  worldSizePad = {grid->np_int[IDIR], grid->np_int[JDIR],
                  vgfPadding*grid->np_int[KDIR]};
  std::array<int,3> localSize = {data->np_int[IDIR], data->np_int[JDIR],
                                  vgfPadding*data->np_int[KDIR]};
  std::array<int,3> localOffset = {data->gbeg[IDIR] - data->nghost[IDIR],
                                   data->gbeg[JDIR] - data->nghost[JDIR],
                                   0};

  // r2c along z: heFFTe transforms the r2c direction first, so the FFT runs
  // z -> y -> x. With the x/y directions already local (recommended -dec), the
  // z and y transforms are transpose-free and the complex output is
  // distributed in x-pencils, leaving a single reshape per transform (the
  // x-pencil gather). The complex spectrum is halved along z.
  fft = std::make_unique<HeffteFft>(worldSizePad, localSize, localOffset, 2);

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
  if(!data->hydro->haveShearingBox) {
    IDEFIX_ERROR("SelfGravityShearingBox3D requires the shearing box module: "
                 "define [Hydro] shearingBox in the input file "
                 "(use 0 for a non-shearing setup).");
  }
  if(sbS != 0.0) {
    // A non-zero shear makes the problem shear-periodic in x
    for(int dir = IDIR; dir <= JDIR; dir++) {
      const bool lOk = (grid->lbound[dir] == periodic || grid->lbound[dir] == shearingbox);
      const bool rOk = (grid->rbound[dir] == periodic || grid->rbound[dir] == shearingbox);
      if(!lOk || !rOk) {
        IDEFIX_ERROR("SelfGravityShearingBox3D requires periodic or shearing box "
                     "boundary conditions in the shearing box plane (X1 and X2).");
      }
    }
    if(grid->nproc[JDIR] > 1) {
      IDEFIX_ERROR("SelfGravityShearingBox3D with a non-zero shear requires no domain "
                   "decomposition along X2: use -dec n 1 1.");
    }
  } else {
    for(int dir = IDIR; dir <= JDIR; dir++) {
      if(grid->lbound[dir] != periodic || grid->rbound[dir] != periodic) {
        IDEFIX_ERROR("SelfGravityShearingBox3D requires periodic boundary "
                     "conditions in the shearing box plane (X1 and X2).");
      }
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
             << vgfPadding << ", alpha = " << alpha << ", kernel = "
             << (useAnalyticKernel ? "analytic" : "spectral") << ")." << std::endl;
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

void SelfGravityShearingBox3D::EnforceShearingBoxPhi(real t, int dir, BoundarySide side,
                                                      IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox3D::EnforceShearingBoxPhi");
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
  idefix_for("SGSB3D_ShearingBoxPhiCopy", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
                                            0, ighost,
      KOKKOS_LAMBDA (int k, int j, int q) {
        scratch(k,j,q) = localVar(k,j,istart+q);
      });
  idefix_for("SGSB3D_ShearingBoxPhi", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR],
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

void SelfGravityShearingBox3D::SetBoundaries(IdefixArray3D<real> &arr) {
  idfx::pushRegion("SelfGravityShearingBox3D::SetBoundaries");

  #ifdef WITH_MPI
  IdefixArray4D<real> arr4D = IdefixArray4D<real> (arr.data(), 1, data->np_tot[KDIR],
                                                    data->np_tot[JDIR],
                                                    data->np_tot[IDIR]);
  #endif

  const real t = data->t;
  const bool haveShear = (sbS != 0.0);
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

  ExtrapolateVerticalGhost(arr);

  idfx::popRegion();
}

void SelfGravityShearingBox3D::SolvePoisson() {
  idfx::pushRegion("SelfGravityShearingBox3D::SolvePoisson");
  Kokkos::Timer timer;
  elapsedTime -= timer.seconds();

  Grid *grid = data->mygrid;

  // Pack the active density into the lower (active) part of the padded input box
  IdefixArray4D<real> Vc = data->hydro->Vc;
  IdefixArray3D<real> rhoLocal = rho;
  const int ibeg = data->beg[IDIR];
  const int jbeg = data->beg[JDIR];
  const int kbeg = data->beg[KDIR];
  const int nk = data->np_int[KDIR];
  const int nkPad = vgfPadding*nk;
  idefix_for("SGSB3D_PackRho", 0, nkPad,
                              data->beg[JDIR], data->end[JDIR],
                              data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      rhoLocal(k, j - jbeg, i - ibeg) = (k < nk) ? Vc(RHO, k + kbeg, j, i) : 0.0;
    });

  // Forward shear remap: map the density to the fully periodic frame at
  // the nearest fully periodic point in time (each x-column is shifted
  // along y by the residual shear accumulated since the last completed
  // wrap, see ShearRemap::ShiftAlongJ)
  IdefixArray3D<real> fftInput = rho;
  if(sbS != 0.0) {
    const real Lx = grid->xend[IDIR]-grid->xbeg[IDIR];
    const real Ly = grid->xend[JDIR]-grid->xbeg[JDIR];
    const real dy = Ly/grid->np_int[JDIR];
    const int iOffset = data->gbeg[IDIR];  // grid->x is ghost-padded
    ShearRemap::ShiftAlongJ("SGSB3D_RemapFwd", rho, remapScratch, grid->x[IDIR],
                            +1.0, sbS, data->t, Lx, Ly, dy, iOffset,
                            vgfPadding*data->np_int[KDIR], data->np_int[JDIR],
                            data->np_int[IDIR]);
    fftInput = remapScratch;
  }

  fft->R2C(fftInput, rhoF);

  // Effective in-plane wavevector: kx -> kx - ddy*ky, with the residual
  // shear slope
  //   ddy = S*Om * mod(t, Ly/(S*Om*Lx)) = mod(S*Om*Lx*t, Ly)/Lx
  // (eq. 3 of Rendon Restrepo & Gressel 2026), i.e. the shear accumulated
  // since the last completed wrap. This MUST equal the shift slope applied
  // by the remap (ShearRemap::ShiftAlongJ): the remap reindexes the
  // shear-periodic eigenmodes K_m = 2*pi*m/Lx - (S*Om*t)*ky onto the regular
  // kx grid, and the Laplacian is only diagonal in that eigenbasis.
  const real ddy = (sbS != 0.0)
                   ? sbS*ShearRemap::WrappedTime(data->t, sbS,
                                                 grid->xend[IDIR]-grid->xbeg[IDIR],
                                                 grid->xend[JDIR]-grid->xbeg[JDIR])
                   : 0.0;

  // Apply the VGF-HybridBC Green's function in Fourier space
  // (eq. 14 of Rendon Restrepo & Gressel 2026), with L = alpha Lz:
  //   G = L^2/2                                     if k = 0, kz = 0
  //   G = [kz L sin(kz L) + cos(kz L) - 1] / kz^2    if k = 0
  //   G = -[e^{-kL}((kz/k) sin(kz L) - cos(kz L)) + 1] / (k^2+kz^2)  otherwise
  // and phiHat = 4 pi G_cst * G * rhoHat
  IdefixArray3D<complex> rhoFLocal = rhoF;
  IdefixArray3D<complex> phiFLocal = phiF;
  const std::array<int,3> nloc = outSize;
  const int low0 = outLow[0];
  const int low1 = outLow[1];
  const int low2 = outLow[2];
  const int nGlob0 = worldSizePad[0];
  const int nGlob1 = worldSizePad[1];
  const real Lx0 = grid->xend[IDIR]-grid->xbeg[IDIR];
  const real Lx1 = grid->xend[JDIR]-grid->xbeg[JDIR];
  const real Lx2 = grid->xend[KDIR]-grid->xbeg[KDIR];
  const real gravCst = data->gravity->gravCst;
  const real L = this->alpha * Lx2;
  const real TWO_PI = 2.0*M_PI;
  const real PI4G = 4.0*M_PI*gravCst;
  if(useAnalyticKernel) {
    // Analytic kernel (doubled grid): apply the exact z-DFT of the sampled
    // real-space Green's function, symmetrically packed on the padded grid,
    //   k2[q] = dz*g_k(q*dz)  for q = 0..Nz-1,  k2[2Nz-q] = k2[q],
    // whose cyclic convolution with the packed density is the exact discrete
    // aperiodic convolution (g_k is compactly supported in |z| < L, and only
    // lags |z| <= Lz-dz occur). For k > 0, g_k(q*dz) = -dz*exp(-k*q*dz)/(2k)
    // is geometric in q, so its cosine transform sums in closed form:
    //   Ghat(kz) = -dz/(2k) * (1 + 2*Re[ (z - z^Nz)/(1 - z) ]),
    // with z = exp(-k*dz - i*theta), theta = 2*pi*kz/(2*Nz) and
    // z^Nz = exp(-k*Lz)*(-1)^kz. For k = 0, g_0(z) = |z|/2 on |z| < L and the
    // (single) kx = ky = 0 mode is summed directly.
    // The shear enters exactly as in the spectral kernel, through the
    // shifted in-plane wave number kx -> kx - ddy*ky.
    const int nkz = nk;               // global Nz (no decomposition along z)
    const int nPad = vgfPadding*nkz;  // = 2*Nz
    const real dz = Lx2/nkz;
    idefix_for("SGSB3D_KernelAnalytic", 0, nloc[2], 0, nloc[1], 0, nloc[0],
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int gi = low0 + i;
        const int gj = low1 + j;
        const int gk = low2 + k;
        const int ni = (2*gi <= nGlob0) ? gi : gi - nGlob0;
        const int njs = (2*gj <= nGlob1) ? gj : gj - nGlob1;
        const real ky = TWO_PI*njs/Lx1;
        const real kx = TWO_PI*ni/Lx0 - ddy*ky;
        const real k2d = kx*kx + ky*ky;
        const real theta = TWO_PI*gk/nPad;

        real Gk = 0.0;
        if(k2d == 0.0) {
          real s = 0.0;
          for(int q = 1; q < nkz; q++) s += q*cos(theta*q);
          Gk = dz*dz*s;
        } else {
          const real kmod = sqrt(k2d);
          const real w = exp(-kmod*dz);
          const real cq = w*cos(theta);
          const real sq = w*sin(theta);
          const real wN = exp(-kmod*Lx2)*((gk % 2 == 0) ? 1.0 : -1.0);
          const real numRe = cq - wN;
          const real denRe = 1.0 - cq;
          const real den = denRe*denRe + sq*sq;
          const real Sre = (numRe*denRe - sq*sq)/den;
          Gk = -dz/(2.0*kmod)*(1.0 + 2.0*Sre);
        }
        phiFLocal(k,j,i) = rhoFLocal(k,j,i) * (PI4G*Gk);
      });
  } else {
  idefix_for("SGSB3D_KernelFFT", 0, nloc[2], 0, nloc[1], 0, nloc[0],
      KOKKOS_LAMBDA(int k, int j, int i) {
        const int gi = low0 + i;
        const int gj = low1 + j;
        const int gk = low2 + k;
        // In-plane wave numbers (periodic dimensions). The x spectrum is full:
        // use signed mode numbers so that the ddy shear shift is applied to
        // both signs of kx.
        const int ni = (2*gi <= nGlob0) ? gi : gi - nGlob0;
        const int njs = (2*gj <= nGlob1) ? gj : gj - nGlob1;
        const real ky = TWO_PI*njs/Lx1;
        const real kx = TWO_PI*ni/Lx0 - ddy*ky;
        const real k2d = kx*kx + ky*ky;
        // Vertical wave number of the padded domain. The z spectrum is halved
        // (r2c along z): gk spans the non-negative modes only, and the
        // negative modes are reconstructed by the c2r via Hermitian symmetry
        // (the Green's function is even in kz).
        const real kz = TWO_PI*gk/(vgfPadding*Lx2);

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
  }

  fft->C2R(phiF, phiPacked);

  // Inverse shear remap: map the potential back to the shearing frame
  IdefixArray3D<real> fftOutput = phiPacked;
  if(sbS != 0.0) {
    const real Lx = grid->xend[IDIR]-grid->xbeg[IDIR];
    const real Ly = grid->xend[JDIR]-grid->xbeg[JDIR];
    const real dy = Ly/grid->np_int[JDIR];
    const int iOffset = data->gbeg[IDIR];  // grid->x is ghost-padded
    ShearRemap::ShiftAlongJ("SGSB3D_RemapBwd", phiPacked, remapScratch, grid->x[IDIR],
                            -1.0, sbS, data->t, Lx, Ly, dy, iOffset,
                            vgfPadding*data->np_int[KDIR], data->np_int[JDIR],
                            data->np_int[IDIR]);
    fftOutput = remapScratch;
  }

  // Unpack the potential (lower part of the padded box) and fill the ghosts
  IdefixArray3D<real> phiPackedLocal = fftOutput;
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
