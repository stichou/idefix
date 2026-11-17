// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityShearingBox3D.hpp
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

#ifndef GRAVITY_SELFGRAVITYSHEARINGBOX3D_HPP_
#define GRAVITY_SELFGRAVITYSHEARINGBOX3D_HPP_

#include <array>
#include <memory>

#include "selfGravity.hpp"
#include "heffteFft.hpp"
#include "localFft.hpp"
#include "shearRemap.hpp"
#include "spectralTranspose.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

class SelfGravityShearingBox3D final : public SelfGravity {
 public:
  void Init(Input &, DataBlock *) override;
  void ShowConfig() override;
  void SolvePoisson() override;
  void AddSelfGravityPotential(IdefixArray3D<real> &) override;
  void EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) override;

  void EnforcePeriodic(int dir, BoundarySide side, IdefixArray3D<real> &);
  void EnforceShearingBoxPhi(real t, int dir, BoundarySide side, IdefixArray3D<real> &);
  void ExtrapolateVerticalGhost(IdefixArray3D<real> &);

 private:
  void CheckCompatibility();
  void SetBoundaries(IdefixArray3D<real> &);
  void SolvePoissonCompact();

  std::unique_ptr<HeffteFft> fft;

  // Compact (SAFI) pipeline: local batched transforms + a single pencil
  // transpose (the only communicating stage). All arrays are compact:
  // the vertical padding exists only in the pencil buffer rhoXP.
  std::unique_ptr<LocalFft> fftY;      ///< y r2c on the local slab
  std::unique_ptr<LocalFft> fftYinv;   ///< y c2r on the local slab
  std::unique_ptr<LocalFft> fftX;       ///< x c2c on the pencil
  std::unique_ptr<LocalFft> fftZ;       ///< z c2c on the padded pencil
  std::unique_ptr<SpectralTranspose> transpose;   ///< slab <-> pencil P1
  std::unique_ptr<SpectralTranspose> transpose2;  ///< pencil P1 <-> P2 (general case)
  IdefixArray3D<real> rhoCompact;   ///< (nz, ny, nxLoc) packed density
  IdefixArray3D<complex> rhoY;     ///< (nz, ny/2+1, nxLoc) y-spectral slab
  IdefixArray3D<complex> rhoX;     ///< pencil P1: (zLoc, kyLoc, nx), x full
  IdefixArray3D<complex> rhoXP;    ///< pencil P2: (vgfPadding*nz, ky2Loc, x2Loc), z full
  // Pencil P1 (after the first transpose): full x, slice of the (ky, kz)
  // space. In the simple case (nproc <= Ny/2+1) the split is along ky only
  // and P1 = P2.
  int kyOffset{0};  ///< global index of the first ky mode of the P1 slice
  int kyLocal{0};   ///< number of ky modes of the P1 slice
  int zOffset{0};   ///< global index of the first z plane of the P1 slice
  int zLocal{0};    ///< number of z planes of the P1 slice
  // Pencil P2 (transform stage of z): full z, slice of the (kx, ky) space.
  // Exists for any rank count; in the simple case P2 = P1.
  int x2Offset{0};   ///< global index of the first kx mode of the P2 slice
  int x2Local{0};    ///< number of kx modes of the P2 slice
  int ky2Offset{0};  ///< global index of the first ky mode of the P2 slice
  int ky2Local{0};   ///< number of ky modes of the P2 slice

  // Vertical zero-padding factor of the FFT grid. With the spectral kernel
  // (eq. 14 sampled on the padded kz grid) the VGF-HybridBC method requires
  // quadrupling (4). With the analytic kernel (closed-form real-space Green's
  // function, compactly supported in |z| < alpha*Lz), the cyclic convolution
  // on the doubled grid (2) is exact, since only lags |z| <= Lz-dz occur.
  int vgfPadding{4};
  bool useAnalyticKernel{false};

  // FFT pipeline of the solver:
  // - "padded": the historical pipeline, one distributed r2c/c2r transform
  //   of the vertically padded grid (z-first ordering).
  // - "compact": the SAFI pipeline, operating on the unpadded grid: local
  //   y-r2c, exact spectral unshear (SAFI phase), x transform after a single
  //   compact pencil transpose, local z transform on the padded columns, and
  //   the symmetric backward chain. All transform stages are local; the only
  //   communication is the transpose of the compact (unpadded) array.
  bool useCompactPipeline{false};

  // Packed local active arrays (no ghost cells) following the input box layout.
  // The vertical direction is zero-padded by a factor vgfPadding (x4 for the
  // spectral kernel, x2 for the analytic kernel).
  IdefixArray3D<real> rho;
  IdefixArray3D<real> phiPacked;
  IdefixArray3D<real> remapScratch;  ///< work array for the shear remap
  IdefixArray3D<real> ghostScratch;  ///< work array for the shear-periodic ghost fill
  // Potential with ghost cells
  IdefixArray3D<real> phi;
  // Local complex spectral coefficients, following the output box layout
  IdefixArray3D<complex> rhoF;
  IdefixArray3D<complex> phiF;

  std::array<int,3> worldSize{1,1,1};   // global active sizes, (i,j,k) order
  std::array<int,3> worldSizePad{1,1,1};  // global FFT sizes (z padded)
  std::array<int,3> outLow{0,0,0};      // local complex box, (i,j,k) order
  std::array<int,3> outSize{1,1,1};     // local complex box, (i,j,k) order

  real alpha{1.1};  ///< enclosure parameter of the VGF closure relation
  real sbS{0.0};    ///< shear rate (from [Hydro] shearingBox)

  #ifdef WITH_MPI
  Mpi mpi;
  #endif
};

#endif // GRAVITY_SELFGRAVITYSHEARINGBOX3D_HPP_
