// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/gravity/shearRemap.hpp
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

// Shearing-box remap utilities shared by the spectral self-gravity solvers
// (Rendon Restrepo & Gressel 2026, Sect. 2.1). The remap shifts each x-column
// of a packed (ghost-free) array along y by the shear offset, mapping the
// shear-periodic state to a fully periodic one (and back).

#ifndef GRAVITY_SHEARREMAP_HPP_
#define GRAVITY_SHEARREMAP_HPP_

#include <string>

#include "idefix.hpp"

namespace ShearRemap {

// a modulo b with a positive result
KOKKOS_INLINE_FUNCTION real PositiveMod(real a, real b) {
  return a - b*Kokkos::floor(a/b);
}

// Cyclic linear interpolation of src along j at position j + offset(i),
// where the per-column offset (in cells) derives from the shear offset
//   delta(i) = PositiveMod(sign*S*xglob(i)*t, Ly) / dy
// dst(k,j,i) = (1-eps) src(k,ja,i) + eps src(k,jb,i)
template <typename SrcView, typename DstView, typename XView>
void ShiftAlongJ(const std::string &name,
                 const SrcView &src, const DstView &dst, const XView &xglob,
                 real sign, real S, real t, real Ly, real dy,
                 int iOffset, int nk, int ny, int nx) {
  idefix_for(name.c_str(), 0, nk, 0, ny, 0, nx,
    KOKKOS_LAMBDA(int k, int j, int i) {
      const real offset = PositiveMod(sign*S*xglob(iOffset+i)*t, Ly)/dy;
      const real pos = j + offset;
      const int j0 = static_cast<int>(Kokkos::floor(pos));
      const real eps = pos - j0;
      const int ja = ((j0 % ny) + ny) % ny;
      const int jb = (((j0+1) % ny) + ny) % ny;
      dst(k,j,i) = (1.0-eps)*src(k,ja,i) + eps*src(k,jb,i);
    });
}

} // namespace ShearRemap

#endif // GRAVITY_SHEARREMAP_HPP_
