// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityShearingBox2D.hpp
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

#ifndef GRAVITY_SELFGRAVITYSHEARINGBOX2D_HPP_
#define GRAVITY_SELFGRAVITYSHEARINGBOX2D_HPP_

#include <array>
#include <memory>

#include "selfGravity.hpp"
#include "heffteFft.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

class SelfGravityShearingBox2D final : public SelfGravity {
 public:
  enum KernelType { RAZOR, BESSEL };

  void Init(Input &, DataBlock *) override;
  void ShowConfig() override;
  void SolvePoisson() override;
  void AddSelfGravityPotential(IdefixArray3D<real> &) override;
  void EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) override;

 private:
  void CheckCompatibility();
  void SetBoundaries(IdefixArray3D<real> &);
  void EnforcePeriodic(int dir, BoundarySide side, IdefixArray3D<real> &);

  std::unique_ptr<HeffteFft> fft;

  // Packed local active arrays (no ghost cells), following the input box layout
  IdefixArray3D<real> rho;
  IdefixArray3D<real> phiPacked;
  // Potential with ghost cells
  IdefixArray3D<real> phi;
  // Local complex spectral coefficients, following the output box layout
  IdefixArray3D<complex> rhoF;
  IdefixArray3D<complex> phiF;

  std::array<int,3> worldSize{1,1,1};  // global active sizes, (i,j,k) order
  std::array<int,3> outLow{0,0,0};      // local complex box, (i,j,k) order
  std::array<int,3> outSize{1,1,1};     // local complex box, (i,j,k) order

  KernelType kernelType{RAZOR};
  real lambda{0.0};       ///< softening length of the razor-thin disc kernel
  real eta{1.0};          ///< vertical smoothing parameter of the Bessel kernel
  real scaleHeight{0.0};  ///< disc scale height H

  #ifdef WITH_MPI
  Mpi mpi;
  #endif
};

#endif // GRAVITY_SELFGRAVITYSHEARINGBOX2D_HPP_
