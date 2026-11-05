// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravityIterative.hpp
//
// Last modified : 10/2023
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2021 - 2023)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef GRAVITY_SELFGRAVITYITERATIVE_HPP_
#define GRAVITY_SELFGRAVITYITERATIVE_HPP_

#include <memory>
#include <vector>

#include "selfGravity.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "fluid_defs.hpp"
#include "iterativesolver.hpp"
#include "laplacian.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

// Forward class hydro declaration
class DataBlock;

class SelfGravityIterative final : public SelfGravity {
 public:
  void Init(Input &, DataBlock *) override;
  void ShowConfig() override;
  void SolvePoisson() override;
  void AddSelfGravityPotential(IdefixArray3D<real> &) override;
  void EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) override;

  void InitSolver(); // (Re)initialisation of the solver for a given density distribution

  void SubstractMeanDensity();  // Compute and substract the average input density

  std::unique_ptr<IterativeSolver<Laplacian>> iterativeSolver;

  // The linear operator involved in Poisson equation
  std::unique_ptr<Laplacian> laplacian;

 private:
  IdefixArray3D<real> potential;  // Gravitational potential
  IdefixArray3D<real> density;  // Density
  real dt;  // CFL timestep

  // Local potential array size
  std::array<int,3> np_tot;

  std::array<Laplacian::LaplacianBoundaryType,3> lbound;  // Boundary condition to the left
  std::array<Laplacian::LaplacianBoundaryType,3> rbound;  // Boundary condition to the right

  bool isPeriodic;
  bool havePreconditioner{false};
  GravitySolver solver; // The solver  used to solve Poisson
};

#endif // GRAVITY_SELFGRAVITYITERATIVE_HPP_
