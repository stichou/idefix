// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/selfGravity.cpp
//
// Last modified : 10/2023
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2021 - 2023)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <memory>
#include <string>

#include "selfGravity.hpp"
#include "selfGravityIterative.hpp"
#ifdef WITH_HEFFTE
#include "selfGravityFFT.hpp"
#include "selfGravityShearingBox2D.hpp"
#endif

std::unique_ptr<SelfGravity> SelfGravity::Create(Input &input, DataBlock *data) {
  std::string strSolver = input.GetOrSet<std::string>("SelfGravity","solver",0,"BICGSTAB");

  std::unique_ptr<SelfGravity> ptr;
  if(strSolver == "FFTShearingBox2D") {
    #ifdef WITH_HEFFTE
    ptr = std::make_unique<SelfGravityShearingBox2D>();
    #else
    IDEFIX_ERROR("[SelfGravity]: FFTShearingBox2D solver requested but "
                 "Idefix was not compiled with heFFTe support (Idefix_HEFFTE).");
    #endif
  } else if(strSolver == "FFT" || strSolver == "fft") {
    #ifdef WITH_HEFFTE
    ptr = std::make_unique<SelfGravityFFT>();
    #else
    IDEFIX_ERROR("[SelfGravity]: FFT solver requested but "
                 "Idefix was not compiled with heFFTe support (Idefix_HEFFTE).");
    #endif
  } else if(strSolver == "Jacobi" || strSolver == "BICGSTAB" || strSolver == "PBICGSTAB"
     || strSolver == "CG" || strSolver == "PCG" || strSolver == "MINRES"
     || strSolver == "PMINRES") {
    ptr = std::make_unique<SelfGravityIterative>();
  } else {
    std::stringstream msg;
    msg << "SelfGravity: Unknown solver \"" << strSolver << "\". "
        << "Use \"Jacobi\", \"BICGSTAB\", \"PBICGSTAB\", \"CG\", \"PCG\", "
        << "\"MINRES\", \"PMINRES\", \"FFT\" or \"FFTShearingBox2D\".";
    IDEFIX_ERROR(msg);
  }
  return ptr;
}
