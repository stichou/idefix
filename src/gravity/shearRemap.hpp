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
// shear-periodic state to the fully periodic state at the nearest "fully
// periodic point in time" (and back).

#ifndef GRAVITY_SHEARREMAP_HPP_
#define GRAVITY_SHEARREMAP_HPP_

#include <cmath>
#include <string>

#include "idefix.hpp"

namespace ShearRemap {

// a modulo b with a positive result
KOKKOS_INLINE_FUNCTION real PositiveMod(real a, real b) {
  return a - b*Kokkos::floor(a/b);
}

// Time wrapped at the shear period Ly/(S*Lx) (with S the shear rate, i.e.
// S*Omega in the usual shearing-box notation): the time elapsed since the
// last completed shear wrap, i.e. the residual accumulated shear.
// Mapping the state to this fully periodic point in time is the only remap
// consistent with the ddy-shifted spectral Green's function, see ShiftAlongJ.
inline real WrappedTime(real t, real S, real Lx, real Ly) {
  if(S*Lx == 0.0) return t;
  return PositiveMod(t, Ly/std::fabs(S*Lx));
}

// Cyclic linear interpolation of src along j at position j + offset(i),
// where the per-column offset (in cells) derives from the residual shear
//   delta(i) = PositiveMod(sign*S*xglob(i)*tp, Ly) / dy,
//   tp = WrappedTime(t, S, Lx, Ly),
// i.e. each x-column is shifted along y by the shear accumulated since the
// last completed wrap (the residual ramp delta(x) = ddy*x, with
// ddy = S*tp = mod(S*Lx*t, Ly)/Lx), and dst samples src at j + delta.
//   dst(k,j,i) = (1-eps) src(k,ja,i) + eps src(k,jb,i)
//
// IMPORTANT: the shift slope applied here (ddy) must equal the kx-shift
// slope used by the spectral Green's function. The remap reindexes the
// shear-periodic eigenmodes exp(i(K_m x + ky y)), K_m = 2*pi*m/Lx - (S*t)*ky,
// onto the regular kx grid, and the Laplacian is only diagonal in that
// eigenbasis, so the kernel must be evaluated at the shifted wavenumber
// kx + ddy*ky. Shifting by the TOTAL wrapped displacement mod(S*x*t, Ly)
// instead (which is pointwise identical to the full unwrapped shift
// S*x*t) is inconsistent with the ddy-shifted kernel as soon as the shear
// exceeds one domain (S*Lx*t > Ly): the two slopes then differ by
// m*(Ly/Lx), an integer number of kx-grid spacings, and the potential is
// computed at the wrong eigenmode wavenumber for every ky != 0 mode.
// This mirrors the reference implementation (orbAdv_PPP in NIRVANA, which
// maps to the fully periodic point in time using SAFI, paired with the
// ddy-shifted kernel of gravitySpectral.c).
template <typename SrcView, typename DstView, typename XView>
void ShiftAlongJ(const std::string &name,
                 const SrcView &src, const DstView &dst, const XView &xglob,
                 real sign, real S, real t, real Lx, real Ly, real dy,
                 int iOffset, int nk, int ny, int nx) {
  // Residual time since the last completed shear wrap: only the residual
  // shift is consistent with the ddy-shifted spectral kernel.
  const real tp = WrappedTime(t, S, Lx, Ly);
  idefix_for(name.c_str(), 0, nk, 0, ny, 0, nx,
    KOKKOS_LAMBDA(int k, int j, int i) {
      const real offset = PositiveMod(sign*S*xglob(iOffset+i)*tp, Ly)/dy;
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
