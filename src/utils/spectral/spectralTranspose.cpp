// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/utils/spectral/spectralTranspose.cpp
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

#include <algorithm>
#include <vector>

#include "spectralTranspose.hpp"

#ifdef WITH_HEFFTE

#include "heffte.h"

#ifdef WITH_MPI
#include "mpi.hpp"
#else
#include <mpi.h>
#endif

namespace {
// heFFTe reshapes operate on std::complex<real>; Kokkos::complex<real>
// is layout-compatible (checked in heffteFft.cpp).
using CmplxT = std::complex<real>;
} // namespace

SpectralTranspose::SpectralTranspose(std::array<int,3> globalSize,
                                     std::array<int,3> inSize, std::array<int,3> inOffset,
                                     std::array<int,3> outSize, std::array<int,3> outOffset) {
  idfx::pushRegion("SpectralTranspose::SpectralTranspose");
  inCount = inSize[0]*inSize[1]*inSize[2];
  outCount = outSize[0]*outSize[1]*outSize[2];

  // Gather the boxes of all ranks: 6 integers per rank and direction
  const int rootSix[6] = {inOffset[0], inOffset[1], inOffset[2],
                          inSize[0], inSize[1], inSize[2]};
  const int rootOut[6] = {outOffset[0], outOffset[1], outOffset[2],
                          outSize[0], outSize[1], outSize[2]};
  std::vector<int> allIn(6*idfx::psize), allOut(6*idfx::psize);
  MPI_Allgather(rootSix, 6, MPI_INT, allIn.data(), 6, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(rootOut, 6, MPI_INT, allOut.data(), 6, MPI_INT, MPI_COMM_WORLD);

  std::vector<heffte::box3d<>> inBoxes, outBoxes;
  for(int r = 0; r < idfx::psize; r++) {
    inBoxes.push_back(heffte::box3d<>(
        {allIn[6*r+0], allIn[6*r+1], allIn[6*r+2]},
        {allIn[6*r+0]+allIn[6*r+3]-1, allIn[6*r+1]+allIn[6*r+4]-1, allIn[6*r+2]+allIn[6*r+5]-1}));
    outBoxes.push_back(heffte::box3d<>(
        {allOut[6*r+0], allOut[6*r+1], allOut[6*r+2]},
        {allOut[6*r+0]+allOut[6*r+3]-1, allOut[6*r+1]+allOut[6*r+4]-1,
         allOut[6*r+2]+allOut[6*r+5]-1}));
  }

  const heffte::plan_options options = heffte::default_options<HeffteBackendTag>();
  forward = heffte::make_reshape3d<HeffteBackendTag>(device.stream(), inBoxes,
                                                      outBoxes, MPI_COMM_WORLD, options);
  backward = heffte::make_reshape3d<HeffteBackendTag>(device.stream(), outBoxes,
                                                      inBoxes, MPI_COMM_WORLD, options);

  // Workspace shared by the reshapes (largest of the two)
  size_t ws = 0;
  if(forward) {
    ws = std::max(ws, forward->size_workspace());
  }
  if(backward) {
    ws = std::max(ws, backward->size_workspace());
  }
  if(ws > 0) {
    workspace = IdefixArray1D<complex>("SpectralTransposeWorkspace", ws);
  }
  idfx::popRegion();
}

void SpectralTranspose::Forward(const IdefixArray3D<complex> &in,
                                const IdefixArray3D<complex> &out) {
  idfx::pushRegion("SpectralTranspose::Forward");
  if(forward) {
    forward->apply(1, reinterpret_cast<CmplxT const*>(in.data()),
                   reinterpret_cast<CmplxT*>(out.data()),
                   reinterpret_cast<CmplxT*>(workspace.data()));
  } else {
    // Degenerate case: identical decompositions
    Kokkos::deep_copy(out, in);
  }
  idfx::popRegion();
}

void SpectralTranspose::Backward(const IdefixArray3D<complex> &in,
                                 const IdefixArray3D<complex> &out) {
  idfx::pushRegion("SpectralTranspose::Backward");
  if(backward) {
    backward->apply(1, reinterpret_cast<CmplxT const*>(in.data()),
                     reinterpret_cast<CmplxT*>(out.data()),
                     reinterpret_cast<CmplxT*>(workspace.data()));
  } else {
    // Degenerate case: identical decompositions
    Kokkos::deep_copy(out, in);
  }
  idfx::popRegion();
}

#endif // WITH_HEFFTE
