#ifndef REMORA_GLS_H_
#define REMORA_GLS_H_

#include <AMReX_Array4.H>
#include <AMReX_Algorithm.H>

namespace gls {
/**
 * Third-order upstream turbulence flux in ROMS gls_prestep.F and gls_corstep.F.
 * @param[in] q             turbulence field and its time component
 * @param[in] mask          face mask in the advection direction
 * @param[in] i,j,k,n       face index and source component
 * @param[in] dir           horizontal direction (0 or 1)
 * @param[in] transport     mean volume transport of the adjacent vertical layers
 * @param[in] lo,hi         cell-centered domain bounds in dir on this level
 * @param[in] periodic      whether dir is periodic
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real upstream_flux (amrex::Array4<amrex::Real const> const& q,
                           amrex::Array4<amrex::Real const> const& mask,
                           int i, int j, int k, int n, int dir,
                           amrex::Real transport, int lo, int hi, bool periodic) noexcept
{
    using amrex::Real;
    const int di = dir == 0;
    const int dj = dir == 1;
    const int ci = i - (transport > Real(0) ? di : 0);
    const int cj = j - (transport > Real(0) ? dj : 0);
    const int c = dir == 0 ? ci : cj;
    Real curvature = Real(0);
    // ROMS copies the edge gradient to the next exterior face, so its curvature is zero.
    if (periodic || (c != lo-1 && c != hi+1)) {
        const Real left = (q(ci,cj,k,n) - q(ci-di,cj-dj,k,n)) * mask(ci,cj,0);
        const Real right = (q(ci+di,cj+dj,k,n) - q(ci,cj,k,n)) * mask(ci+di,cj+dj,0);
        curvature = right - left;
    }
    return transport * Real(0.5) * (q(i-di,j-dj,k,n) + q(i,j,k,n) -
                                     (Real(1)/Real(3)) * curvature);
}

/** Positive-part stability limiter in ROMS gls_corstep.F; inactive below Ghcri. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real limit_Gh (amrex::Real Gh, amrex::Real Gh0, amrex::Real Ghcri,
                      amrex::Real Ghmin) noexcept
{
    if (Gh > Ghcri) {
        const amrex::Real delta = Gh - Ghcri;
        Gh -= delta * delta / (Gh + Gh0 - amrex::Real(2)*Ghcri);
    }
    return amrex::max(Gh, Ghmin);
}
}
#endif
