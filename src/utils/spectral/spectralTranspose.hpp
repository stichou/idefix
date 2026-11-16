// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/utils/spectral/spectralTranspose.hpp
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

// Redistribution (transpose) of one complex field between two box
// decompositions of the same global index space, using heFFTe's
// reshape3d (the same machinery as the internal reshapes of
// heffte::fft3d).
//
// This is the ONLY communicating stage of the compact self-gravity
// pipeline: the slab (x_local, ky, z) -> pencil (x_full, ky-slice,
// z-slice) redistribution that makes the x direction local before the
// x transform.
//
// Boxes follow the Idefix array convention: a box (ni, nj, nk)
// describes an IdefixArray3D<T>(nk, nj, ni) with i stored
// contiguously. Each rank describes its own input and output box in
// the global index space; the boxes are gathered across ranks
// internally.

#ifndef UTILS_SPECTRAL_SPECTRALTRANSPOSE_HPP_
#define UTILS_SPECTRAL_SPECTRALTRANSPOSE_HPP_

#include <array>
#include <memory>
#include <vector>

#include "heffteFft.hpp"
#include "idefix.hpp"

#ifdef WITH_HEFFTE

class SpectralTranspose {
 public:
  // Redistribute a complex field between the input decomposition
  // (inSize, inOffset) and the output decomposition (outSize, outOffset)
  // of the global box (n0, n1, n2), given in (i, j, k) order.
  // Each rank passes its own boxes; they are gathered across ranks.
  // The union of the boxes must cover the global box in both cases.
  SpectralTranspose(std::array<int,3> globalSize,
                    std::array<int,3> inSize, std::array<int,3> inOffset,
                    std::array<int,3> outSize, std::array<int,3> outOffset);

  // Forward redistribution: `in` follows the input decomposition,
  // `out` the output decomposition.
  void Forward(const IdefixArray3D<complex> &in, const IdefixArray3D<complex> &out);
  // Backward redistribution: `in` follows the output decomposition,
  // `out` the input decomposition (exact inverse of Forward).
  void Backward(const IdefixArray3D<complex> &in, const IdefixArray3D<complex> &out);

 private:
  // heFFTe device instance: owns the (GPU) stream passed to the reshapes
  heffte::backend::device_instance<
      typename heffte::backend::buffer_traits<HeffteBackendTag>::location> device;

  std::unique_ptr<heffte::reshape3d_base<int>> forward;    // in -> out
  std::unique_ptr<heffte::reshape3d_base<int>> backward;   // out -> in
  IdefixArray1D<complex> workspace;                     // shared by both reshapes
  int inCount;                                          // entries of an input array
  int outCount;                                         // entries of an output array
};

#endif // WITH_HEFFTE

#endif // UTILS_SPECTRAL_SPECTRALTRANSPOSE_HPP_
