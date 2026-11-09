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
#include "shearRemap.hpp"

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

  std::unique_ptr<HeffteFft> fft;

  // Vertical zero-padding factor of the FFT grid. With the spectral kernel
  // (eq. 14 sampled on the padded kz grid) the VGF-HybridBC method requires
  // quadrupling (4). With the analytic kernel (closed-form real-space Green's
  // function, compactly supported in |z| < alpha*Lz), the cyclic convolution
  // on the doubled grid (2) is exact, since only lags |z| <= Lz-dz occur.
  int vgfPadding{4};
  bool useAnalyticKernel{false};

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
