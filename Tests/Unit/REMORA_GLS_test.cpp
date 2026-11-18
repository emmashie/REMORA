#include <REMORA_GLS.H>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

using amrex::Real;

int main ()
{
    int failures = 0;
    auto check = [&] (const char* name, Real actual, Real expected) {
        const Real tol = Real(128) * std::numeric_limits<Real>::epsilon() *
                         amrex::max(Real(1), std::abs(expected));
        if (!std::isfinite(actual) || std::abs(actual-expected) > tol) {
            std::printf("%s: got %.17g, expected %.17g\n", name,
                        double(actual), double(expected));
            ++failures;
        }
    };
    // At face 2 the four cell values are 1, 2, 4, 8. Expanding the ROMS stencil gives
    // (-1 + 5*2 + 2*4)/6 for positive flow, (2*2 + 5*4 - 8)/6 for negative flow.
    std::array<Real, 16> values, masks;
    const Real profile[] = {1, 2, 4, 8};
    for (int dir = 0; dir < 2; ++dir) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) { values[i+4*j] = profile[dir == 0 ? i : j]; }
        }
        masks.fill(Real(1));
        amrex::Array4<Real> q(values.data(), {0,0,0}, {4,4,1}, 1);
        amrex::Array4<Real> mask(masks.data(), {0,0,0}, {4,4,1}, 1);
        const int i = dir == 0 ? 2 : 0;
        const int j = dir == 1 ? 2 : 0;
        auto flux = [&] (Real u, int lo, int hi, bool periodic) {
            return gls::upstream_flux(q, mask, i,j,0,0,dir,u,lo,hi,periodic);
        };
        check("positive flow", flux(2,0,3,true), Real(17)/Real(3));
        check("negative flow", flux(-2,0,3,true), -Real(16)/Real(3));
        check("zero flow", flux(0,0,3,true), 0);
        check("exterior west/south curvature", flux(2,2,5,false), 6);
        check("exterior east/north curvature", flux(-2,-2,1,false), -6);
        masks.fill(Real(0));
        check("masked gradients", flux(2,0,3,true), 6);
        masks.fill(Real(1));
        mask(i,j,0) = 0;
        check("masked center gradient, positive", flux(2,0,3,true), Real(19)/Real(3));
        check("masked center gradient, negative", flux(-2,0,3,true), -Real(14)/Real(3));
    }
    // Canuto A: below the transition the limiter is the identity, including at the
    // inactive denominator's zero. At Gh0 the positive-part correction subtracts 0.00145.
    const Real Gh0 = Real(0.0329), Ghcri = Real(0.03), Ghmin = Real(-0.28);
    check("Gh lower bound", gls::limit_Gh(Real(-1),Gh0,Ghcri,Ghmin), Ghmin);
    check("Gh neutral", gls::limit_Gh(0,Gh0,Ghcri,Ghmin), 0);
    const Real singular = Real(2)*Ghcri-Gh0;
    check("inactive zero denominator", gls::limit_Gh(singular,Gh0,Ghcri,Ghmin), singular);
    check("Gh transition", gls::limit_Gh(Ghcri,Gh0,Ghcri,Ghmin), Ghcri);
    check("Gh upper end", gls::limit_Gh(Gh0,Gh0,Ghcri,Ghmin), Real(0.03145));
    Real previous = Ghmin;
    for (int n = 0; n <= 1000; ++n) {
        const Real Gh = Ghmin + (Gh0-Ghmin)*Real(n)/Real(1000);
        const Real result = gls::limit_Gh(Gh,Gh0,Ghcri,Ghmin);
        if (!std::isfinite(result) || result < previous || result > Gh) { ++failures; }
        previous = result;
    }
    const Real step = Real(32)*std::numeric_limits<Real>::epsilon();
    check("Gh continuous below", gls::limit_Gh(Ghcri-step,Gh0,Ghcri,Ghmin), Ghcri-step);
    check("Gh continuous above", gls::limit_Gh(Ghcri+step,Gh0,Ghcri,Ghmin), Ghcri+step);
    std::printf("GLS flux and stability arithmetic: %d failures\n", failures);
    return failures != 0;
}
