// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/utils/spectral/heffteFft.cpp
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

#include <memory>
#include <type_traits>
#include <vector>

#include "heffteFft.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#else
#include <mpi.h>
namespace {
// heFFTe always links against MPI, so make sure it is initialized even when
// Idefix itself is built without MPI.
bool InitMpiOnce() {
  int flag = 0;
  MPI_Initialized(&flag);
  if(!flag) MPI_Init(nullptr, nullptr);
  return true;
}
}
#endif

static_assert(sizeof(complex) == sizeof(std::complex<real>),
              "Kokkos::complex<real> must be layout-compatible with std::complex<real>");

HeffteFft::HeffteFft(std::array<int,3> worldSize,
                     std::array<int,3> localSize,
                     std::array<int,3> localOffset,
                     int r2cDirection) {
  idfx::pushRegion("HeffteFft::HeffteFft");
#ifndef WITH_MPI
  static bool mpiInitialized [[maybe_unused]] = InitMpiOnce();
#endif

  const heffte::box3d<> realWorld({0, 0, 0}, {worldSize[0]-1, worldSize[1]-1, worldSize[2]-1});
  const heffte::box3d<> cplxWorld = realWorld.r2c(r2cDirection);

  const heffte::box3d<> inbox({localOffset[0], localOffset[1], localOffset[2]},
                              {localOffset[0] + localSize[0] - 1,
                               localOffset[1] + localSize[1] - 1,
                               localOffset[2] + localSize[2] - 1});

  // The Fourier-space distribution is chosen by heFFTe for the complex world box,
  // independently of the input decomposition dictated by -dec.
  std::array<int,3> outProcGrid;
  if(r2cDirection == 2) {
    // x-pencils: split along (j,k) only, so that each rank owns the full x
    // extent of the complex world. heFFTe's plan then ends the forward
    // transform with the x-FFT on exactly this pencil distribution, which
    // allows the final reshape to be skipped.
    int py = 1;
    for(int p = 1; p*p <= idfx::psize; p++) {
      if(idfx::psize % p == 0) py = p;
    }
    py = idfx::psize / py;
    outProcGrid = {1, py, idfx::psize/py};
  } else {
    outProcGrid = heffte::proc_setup_min_surface(cplxWorld, idfx::psize);
  }
  const std::vector<heffte::box3d<>> outboxes = heffte::split_world(cplxWorld, outProcGrid);
  const heffte::box3d<> &localOutbox = outboxes[idfx::prank];
  outLow = {localOutbox.low[0], localOutbox.low[1], localOutbox.low[2]};
  outSize = {localOutbox.size[0], localOutbox.size[1], localOutbox.size[2]};

  worldSizeCplx = {cplxWorld.size[0], cplxWorld.size[1], cplxWorld.size[2]};

  fft = std::make_unique<heffte::fft3d_r2c<HeffteBackendTag>>(inbox, localOutbox, r2cDirection,
                                                              MPI_COMM_WORLD);

  idfx::popRegion();
}

void HeffteFft::R2C(const IdefixArray3D<real> &in, const IdefixArray3D<complex> &out) {
  idfx::pushRegion("HeffteFft::R2C");
  fft->forward(in.data(), reinterpret_cast<std::complex<real>*>(out.data()));
  idfx::popRegion();
}

void HeffteFft::C2R(const IdefixArray3D<complex> &in, const IdefixArray3D<real> &out) {
  idfx::pushRegion("HeffteFft::C2R");
  fft->backward(reinterpret_cast<const std::complex<real>*>(in.data()), out.data(),
                heffte::scale::full);
  idfx::popRegion();
}

std::array<int,3> HeffteFft::getOutboxLow() const {
  return outLow;
}

std::array<int,3> HeffteFft::getOutboxSize() const {
  return outSize;
}

std::array<int,3> HeffteFft::getWorldSizeCplx() const {
  return worldSizeCplx;
}

#endif // WITH_HEFFTE
