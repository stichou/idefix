// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/utils/spectral/localFft.cpp
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

#include "localFft.hpp"

#ifdef WITH_HEFFTE

#include <memory>
#include <type_traits>

#include "heffte.h"

namespace {

// The heFFTe one-dimensional executors operate on std::complex<real>;
// Kokkos::complex<real> is layout-compatible (checked in heffteFft.cpp).
using CmplxT = std::complex<real>;

CmplxT *asStd(IdefixArray3D<complex> &a) { return reinterpret_cast<CmplxT*>(a.data()); }
const CmplxT *asStd(const IdefixArray3D<complex> &a) {
  return reinterpret_cast<const CmplxT*>(a.data());
}

} // namespace

LocalFft::LocalFft(std::array<int,3> boxSize, int axisIn, LocalFftKind kindIn)
    : kind(kindIn), axis(axisIn) {
  idfx::pushRegion("LocalFft::LocalFft");
  if(axis < 0 || axis > 2) {
    IDEFIX_ERROR("LocalFft: the transform axis must be 0 (i), 1 (j) or 2 (k).");
  }
  for(int dir = 0; dir < 3; dir++) {
    if(boxSize[dir] < 1) {
      IDEFIX_ERROR("LocalFft: all box dimensions must be at least 1.");
    }
  }
  // heFFTe box in (i, j, k) order: dimension 0 (i) is stored contiguously,
  // which matches the flattening of IdefixArray3D<T>(nk, nj, ni).
  const heffte::box3d<> box({0, 0, 0}, {boxSize[0]-1, boxSize[1]-1, boxSize[2]-1});
  inCount = box.count();
  if(kind == LocalFftKind::R2C) {
    outSize = box.r2c(axis).size;
    outCount = box.r2c(axis).count();
    exec = std::unique_ptr<heffte::executor_base>(
        new typename heffte::one_dim_backend<HeffteBackendTag>::executor_r2c(
            device.stream(), box, axis));
  } else if(kind == LocalFftKind::C2R) {
    // The executor_r2c constructor always takes the REAL box and derives
    // both the real and complex layouts from it; for C2R the input array
    // has the size of box.r2c(axis).
    outSize = box.size;
    outCount = box.count();
    inCount = box.r2c(axis).count();
    exec = std::unique_ptr<heffte::executor_base>(
        new typename heffte::one_dim_backend<HeffteBackendTag>::executor_r2c(
            device.stream(), box, axis));
  } else {
    outSize = box.size;
    outCount = box.count();
    exec = std::unique_ptr<heffte::executor_base>(
        new typename heffte::one_dim_backend<HeffteBackendTag>::executor(
            device.stream(), box, axis));
  }
  // Some GPU backends (e.g. cuFFT r2c) require a workspace; CPU backends
  // declare none. heFFTe's own fft3d always allocates it, so do the same.
  if(exec->workspace_size() > 0) {
    workspace = IdefixArray1D<complex>("LocalFftWorkspace", exec->workspace_size());
  }

  idfx::popRegion();
}

void LocalFft::Forward(const IdefixArray3D<complex> &in,
                       const IdefixArray3D<complex> &out) {
  idfx::pushRegion("LocalFft::Forward(C2C)");
  if(kind != LocalFftKind::C2C) {
    IDEFIX_ERROR("LocalFft: Forward(C2C) requires a C2C transform.");
  }
  if(in.data() != out.data()) {
    Kokkos::deep_copy(out, in);
  }
  exec->forward(asStd(const_cast<IdefixArray3D<complex>&>(out)),
                reinterpret_cast<CmplxT*>(workspace.data()));
  idfx::popRegion();
}

void LocalFft::Backward(const IdefixArray3D<complex> &in,
                        const IdefixArray3D<complex> &out, real scale) {
  idfx::pushRegion("LocalFft::Backward(C2C)");
  if(kind != LocalFftKind::C2C) {
    IDEFIX_ERROR("LocalFft: Backward(C2C) requires a C2C transform.");
  }
  if(in.data() != out.data()) {
    Kokkos::deep_copy(out, in);
  }
  exec->backward(asStd(const_cast<IdefixArray3D<complex>&>(out)),
                 reinterpret_cast<CmplxT*>(workspace.data()));
  // Apply the scaling
  idefix_for("LocalFft::Scale", 0, outSize[2], 0, outSize[1], 0, outSize[0],
    KOKKOS_LAMBDA(int k, int j, int i) {
      out(k,j,i) *= scale;
    });
  idfx::popRegion();
}

void LocalFft::Forward(const IdefixArray3D<real> &in,
                       const IdefixArray3D<complex> &out) {
  idfx::pushRegion("LocalFft::Forward(R2C)");
  if(kind != LocalFftKind::R2C) {
    IDEFIX_ERROR("LocalFft: Forward(R2C) requires an R2C transform.");
  }
  exec->forward(in.data(), asStd(const_cast<IdefixArray3D<complex>&>(out)),
                reinterpret_cast<CmplxT*>(workspace.data()));
  idfx::popRegion();
}

void LocalFft::Backward(const IdefixArray3D<complex> &in,
                        const IdefixArray3D<real> &out, real scale) {
  idfx::pushRegion("LocalFft::Backward(C2R)");
  if(kind != LocalFftKind::C2R) {
    IDEFIX_ERROR("LocalFft: Backward(C2R) requires a C2R transform.");
  }
  exec->backward(asStd(const_cast<IdefixArray3D<complex>&>(in)), out.data(),
                 reinterpret_cast<CmplxT*>(workspace.data()));
  // Apply the scaling
  const std::array<int,3> os = outSize;
  idefix_for("LocalFft::Scale", 0, os[2], 0, os[1], 0, os[0],
    KOKKOS_LAMBDA(int k, int j, int i) {
      out(k,j,i) *= scale;
    });
  idfx::popRegion();
}

#endif // WITH_HEFFTE
