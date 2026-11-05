// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravity.hpp
//
// Last modified : 10/2023
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2021 - 2023)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef GRAVITY_SELFGRAVITY_HPP_
#define GRAVITY_SELFGRAVITY_HPP_

#include <memory>
#include <vector>

#include "idefix.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "laplacian.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

// Forward class hydro declaration
class DataBlock;

class SelfGravity {
 public:
  enum GravitySolver {JACOBI, BICGSTAB, PBICGSTAB, PCG, CG, PMINRES, MINRES};

  virtual ~SelfGravity() = default;

  virtual void Init(Input &, DataBlock *) = 0;  // Initialisation of the solver
  virtual void ShowConfig() = 0;                // display current configuration
  virtual void SolvePoisson() = 0;               // Solve Poisson equation
  virtual void AddSelfGravityPotential(IdefixArray3D<real> &) = 0;
  virtual void EnrollUserDefBoundary(Laplacian::UserDefBoundaryFunc myFunc) = 0;

  static std::unique_ptr<SelfGravity> Create(Input &, DataBlock *);

  real currentError{0};       // last error of the iterative solver
  int nsteps{0};              // # of steps of the latest iteration
  double elapsedTime{0.0};     // time spent solving self gravity

  // Whether we should skip self-gravity computation every n steps
  int skipSelfGravity{1};

 protected:
  DataBlock *data{nullptr};  // My parent data object
};

#endif // GRAVITY_SELFGRAVITY_HPP_
