// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/utils/spectral/heffteFft.hpp
//
// Last modified : 11/2026
//
// Copyright(C) by :
// - Steven Rendon Restrepo <srendon@aip.de> (AIP - 2026)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef UTILS_SPECTRAL_HEFFTEFFT_HPP_
#define UTILS_SPECTRAL_HEFFTEFFT_HPP_

#include <array>
#include <memory>

#include "idefix.hpp"
#include "real_types.hpp"

#ifdef WITH_HEFFTE
#include "heffte.h"

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using HeffteBackendTag = heffte::backend::default_backend<heffte::tag::gpu>::type;
#else
using HeffteBackendTag = heffte::backend::default_backend<heffte::tag::cpu>::type;
#endif

class HeffteFft {
 public:
  // worldSize/localSize/localOffset are in (i,j,k) order (i = fastest dimension),
  // describing the global active grid and this rank's position in it,
  // as dictated by the runtime domain decomposition (-dec).
  HeffteFft(std::array<int,3> worldSize,
            std::array<int,3> localSize,
            std::array<int,3> localOffset);

  // in/out must be contiguous rank-3 arrays matching the real input box
  // (localSize) and the complex output box (getOutbox()), respectively.
  void R2C(const IdefixArray3D<real> &in, const IdefixArray3D<complex> &out);
  void C2R(const IdefixArray3D<complex> &in, const IdefixArray3D<real> &out);

  // Local box in the (i,j,k)-ordered global complex world grid
  std::array<int,3> getOutboxLow() const;
  std::array<int,3> getOutboxSize() const;
  // Global complex world size (i,j,k), first dimension reduced to floor(n/2)+1
  std::array<int,3> getWorldSizeCplx() const;

 private:
  std::unique_ptr<heffte::fft3d_r2c<HeffteBackendTag>> fft;
  std::array<int,3> outLow{0,0,0};
  std::array<int,3> outSize{1,1,1};
  std::array<int,3> worldSizeCplx{1,1,1};
};

#endif // WITH_HEFFTE
#endif // UTILS_SPECTRAL_HEFFTEFFT_HPP_
