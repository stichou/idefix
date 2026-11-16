// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/utils/spectral/localFft.hpp
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

// Batched one-dimensional FFTs on a LOCAL (per-rank) box, wrapping the
// one-dimensional executors of heFFTe (the same backend as HeffteFft:
// stock, FFTW3, MKL, oneMKL, cuFFT or rocFFT depending on the build).
//
// This is the building block of the compact self-gravity pipeline, where
// every transform stage is local (y and z are never decomposed, and x is
// local after the single pencil transpose); only the transpose itself
// requires communication (see spectralTranspose.hpp).
//
// The local box follows the Idefix array convention: an array
// IdefixArray3D<T>(nk, nj, ni) is described by the box (ni, nj, nk),
// with the i direction stored contiguously (heFFTe dimension 0).
// The transform axis is 0 (i/x), 1 (j/y) or 2 (k/z).
//
// The transforms are UNSCALED in the forward direction (heFFTe
// convention); Backward() takes an explicit scaling factor so that the
// caller controls the normalisation of the composed pipeline.

#ifndef UTILS_SPECTRAL_LOCALFFT_HPP_
#define UTILS_SPECTRAL_LOCALFFT_HPP_

#include <array>
#include <memory>

#include "heffteFft.hpp"
#include "idefix.hpp"

#ifdef WITH_HEFFTE

// Kind of one-dimensional transform
enum class LocalFftKind {
  C2C,   // complex to complex
  R2C,   // real to complex (the transformed axis is halved: n -> n/2+1)
  C2R    // complex to real (the input axis has n/2+1 entries)
};

class LocalFft {
 public:
  // Construct a batched 1D transform along `axis` (0 = i, 1 = j, 2 = k)
  // of the local box `boxSize` given in (ni, nj, nk) order.
  LocalFft(std::array<int,3> boxSize, int axis, LocalFftKind kind);

  // Size of the output box (for R2C, the transformed axis is n/2+1)
  std::array<int,3> GetOutSize() const { return outSize; }
  // Number of entries of the output array
  int GetOutCount() const { return outCount; }

  // Complex-to-complex forward transform (unscaled). May operate in place
  // (in and out may be the same array).
  void Forward(const IdefixArray3D<complex> &in, const IdefixArray3D<complex> &out);
  // Complex-to-complex backward transform, scaled by `scale`
  // (out = scale * sum ... ; use scale = 1/n for the inverse transform).
  // May operate in place.
  void Backward(const IdefixArray3D<complex> &in, const IdefixArray3D<complex> &out,
                real scale);

  // Real-to-complex forward transform (unscaled); `out` has the size
  // returned by GetOutSize().
  void Forward(const IdefixArray3D<real> &in, const IdefixArray3D<complex> &out);
  // Complex-to-real backward transform, scaled by `scale`.
  void Backward(const IdefixArray3D<complex> &in, const IdefixArray3D<real> &out,
                real scale);

 private:
  // heFFTe device instance: owns the (GPU) stream passed to the executors
  heffte::backend::device_instance<
      typename heffte::backend::buffer_traits<HeffteBackendTag>::location> device;

  std::unique_ptr<heffte::executor_base> exec;   // C2C / R2C / C2R executor
  LocalFftKind kind;
  int axis;
  int inCount;                        // number of entries of the input array
  int outCount;                      // number of entries of the output array
  std::array<int,3> outSize;
};

#endif // WITH_HEFFTE

#endif // UTILS_SPECTRAL_LOCALFFT_HPP_
