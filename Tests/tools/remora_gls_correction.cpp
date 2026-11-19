#include <REMORA.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>
#include <limits>

using namespace amrex;
std::string inputs_name;

namespace {
void test_defaults ()
{
    ParmParse amr("amr"), remora("remora"), geometry("geometry");
    Vector<int> cells, periodic;
    Vector<Real> lo, hi;
    remora.getarr("n_cell", cells);
    remora.getarr("is_periodic", periodic);
    remora.getarr("prob_lo", lo);
    remora.getarr("prob_hi", hi);
    amr.addarr("n_cell", cells);
    int max_level = 0;
    amr.queryAdd("max_level",max_level);
    amr.add("max_grid_size_z", cells[2]);
    amr.add("blocking_factor", 1);
    amr.add("refine_grid_layout_x", 1);
    amr.add("refine_grid_layout_y", 1);
    amr.add("refine_grid_layout_z", 0);
    geometry.addarr("prob_lo", lo);
    geometry.addarr("prob_hi", hi);
    geometry.addarr("is_periodic", periodic);
    Vector<int> tiles = {16,16,1024000};
    remora.queryarr("omp_tile_size",tiles);
    ParmParse("fabarray").addarr("mfiter_tile_size",tiles);
}

void evolve (REMORA& model, bool require_regrid)
{
    const BoxArray initial = model.finestLevel() > 0 ? model.boxArray(1) : BoxArray();
    model.Evolve();
    if (require_regrid) {
        AMREX_ALWAYS_ASSERT(model.finestLevel() == 1 && model.boxArray(1) != initial);
    }
    Real Kmin = Real(7.6e-6), Pmin = Real(1.e-12);
    ParmParse pp("remora");
    pp.query("gls_Kmin",Kmin); pp.query("gls_Pmin",Pmin);
    for (int lev = 0; lev <= model.finestLevel(); ++lev) {
        for (auto* field : {model.vec_tke[lev].get(),model.vec_gls[lev].get(),
                           model.vec_Lscale[lev].get(),model.vec_Akv[lev].get(),
                           model.vec_Akt[lev].get(),model.vec_Akk[lev].get(),
                           model.vec_Akp[lev].get()}) {
            AMREX_ALWAYS_ASSERT(!field->contains_nan(0,field->nComp(),0));
            AMREX_ALWAYS_ASSERT(!field->contains_inf(0,field->nComp(),0));
        }
        for (int n = 0; n < 2; ++n) {
            AMREX_ALWAYS_ASSERT(model.vec_tke[lev]->min(n) >= Kmin);
            AMREX_ALWAYS_ASSERT(model.vec_gls[lev]->min(n) >= Pmin);
        }
    }
}

void check_field (const MultiFab& field, int component, Real expected, const char* message)
{
    AMREX_ALWAYS_ASSERT(!field.contains_nan(component,1,0));
    AMREX_ALWAYS_ASSERT(!field.contains_inf(component,1,0));
    const Real tolerance = Real(1024)*std::numeric_limits<Real>::epsilon()*std::abs(expected);
    if (std::abs(field.min(component)-expected) > tolerance ||
        std::abs(field.max(component)-expected) > tolerance) {
        amrex::Print().SetPrecision(17) << message << ": min " << field.min(component)
            << " max " << field.max(component) << " expected " << expected << "\n";
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(field.min(component)-expected) <= tolerance,
                                    message);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(field.max(component)-expected) <= tolerance,
                                    message);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real profile (int i, int j, int nx, int ny)
{
    i = (i % nx + nx) % nx;
    j = (j % ny + ny) % ny;
    return Real(0.01) + Real(0.000001)*(i*i*i + 2*j*j*j);
}

// Expanded third-order face coefficients from ROMS, independent of the production helper.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real reference_flux (int i, int j, int nx, int ny, int dir, Real transport)
{
    const int di = dir == 0, dj = dir == 1;
    if (transport > Real(0)) {
        return transport * (-profile(i-2*di,j-2*dj,nx,ny) +
            Real(5)*profile(i-di,j-dj,nx,ny) + Real(2)*profile(i,j,nx,ny))/Real(6);
    }
    return transport * (Real(2)*profile(i-di,j-dj,nx,ny) +
        Real(5)*profile(i,j,nx,ny)-profile(i+di,j+dj,nx,ny))/Real(6);
}

void predictor (REMORA& model)
{
    const int nx = model.Geom(0).Domain().length(0);
    const int ny = model.Geom(0).Domain().length(1);
    const Real H = Real(4), dt = Real(10);
    model.vec_Hz[0]->setVal(H);
    model.vec_pm[0]->setVal(Real(1));
    model.vec_pn[0]->setVal(Real(1));
    model.vec_msku[0]->setVal(Real(1));
    model.vec_mskv[0]->setVal(Real(1));
    MultiFab W(model.vec_tke[0]->boxArray(), model.DistributionMap(0), 1,
               model.vec_tke[0]->nGrowVect());
    W.setVal(Real(0));
    for (Real sign : {Real(-1),Real(1)}) {
        const Real u = sign*Real(0.01), v = -sign*Real(0.02);
        model.vec_Huon[0]->setVal(u);
        model.vec_Hvom[0]->setVal(v);
        for (int first : {0,1}) {
            for (MFIter mfi(*model.vec_tke[0]); mfi.isValid(); ++mfi) {
                auto tke = model.vec_tke[0]->array(mfi);
                auto gls = model.vec_gls[0]->array(mfi);
                ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const Real q = profile(i,j,nx,ny);
                    tke(i,j,k,0) = Real(3)*q;
                    tke(i,j,k,1) = q;
                    tke(i,j,k,2) = Real(-99);
                    gls(i,j,k,0) = Real(2.1)*q;
                    gls(i,j,k,1) = Real(0.7)*q;
                    gls(i,j,k,2) = Real(-99);
                });
            }
            model.gls_prestep(0,model.vec_gls[0].get(),model.vec_tke[0].get(),W,
                model.vec_msku[0].get(),model.vec_mskv[0].get(),1,0,first,0,2,dt);
            MultiFab error(model.vec_tke[0]->boxArray(),model.DistributionMap(0),2,0);
            error.setVal(Real(0));
            for (MFIter mfi(error); mfi.isValid(); ++mfi) {
                auto err = error.array(mfi);
                auto tke = model.vec_tke[0]->const_array(mfi);
                auto gls = model.vec_gls[0]->const_array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (k != 1 && k != 2) { return; }
                    const Real divergence = reference_flux(i+1,j,nx,ny,0,u) -
                        reference_flux(i,j,nx,ny,0,u) + reference_flux(i,j+1,nx,ny,1,v) -
                        reference_flux(i,j,nx,ny,1,v);
                    const Real q = profile(i,j,nx,ny);
                    const Real expected = first == 0 ? q-dt/(Real(2)*H)*divergence :
                        Real(5)/Real(3)*q-Real(5)*dt/(Real(6)*H)*divergence;
                    err(i,j,k,0) = tke(i,j,k,2)-expected;
                    err(i,j,k,1) = gls(i,j,k,2)-Real(0.7)*expected;
                });
            }
            AMREX_ALWAYS_ASSERT(!error.contains_nan());
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norm0(0) < Real(128)*
                std::numeric_limits<Real>::epsilon()*Real(0.02), "Predictor differs from upstream reference");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norm0(1) < Real(128)*
                std::numeric_limits<Real>::epsilon()*Real(0.02), "GLS predictor uses the wrong history or flux");
        }
    }
}

void corrector (REMORA& model, bool solve, bool cap)
{
    Real Kmin = Real(7.6e-6), Pmin = Real(1.e-5), cmu = Real(0.5477), c2 = Real(1.92);
    Real Zos = Real(0.02);
    ParmParse pp("remora");
    pp.query("gls_Kmin",Kmin); pp.query("gls_Pmin",Pmin);
    pp.query("gls_cmu0",cmu); pp.query("gls_c2",c2); pp.query("Zos",Zos);
    Zos = amrex::max(Zos,Real(0.0001));
    const int N = model.Geom(0).Domain().bigEnd(2);
    AMREX_ALWAYS_ASSERT(N == (solve ? 3 : 2));
    const Real dt = cap ? Real(0) : Real(10), old_k = Real(0.01);
    const Real old_p = solve ? Real(0.01) : Real(1.e-12);
    const Real decay = old_p/old_k; // k-epsilon: epsilon/k, cmu**(3 + p/n) = 1.
    const Real nu = solve ? Real(0.02) : Real(0);
    MultiFab W(model.vec_tke[0]->boxArray(),model.DistributionMap(0),1,
               model.vec_tke[0]->nGrowVect());
    for (Real H : {Real(0.25),Real(4)}) {
        for (int vertical = 0; vertical < 2; ++vertical) {
            model.vec_Hz[0]->setVal(H);
            model.vec_pm[0]->setVal(Real(1)); model.vec_pn[0]->setVal(Real(1));
            model.vec_Huon[0]->setVal(Real(0)); model.vec_Hvom[0]->setVal(Real(0));
            model.xvel_old[0]->setVal(Real(0)); model.yvel_old[0]->setVal(Real(0));
            model.vec_bvf[0]->setVal(Real(0));
            model.vec_Akv[0]->setVal(Real(0)); model.vec_Akt[0]->setVal(Real(0));
            model.vec_Akk[0]->setVal(Real(0)); model.vec_Akp[0]->setVal(nu);
            model.vec_Lscale[0]->setVal(Real(0)); model.vec_ZoBot[0]->setVal(Real(1.e6));
            for (auto* stress : {model.vec_sustr[0].get(),model.vec_svstr[0].get(),
                                model.vec_bustr[0].get(),model.vec_bvstr[0].get()}) {
                stress->setVal(Real(0));
            }
            const Real speed = solve ? Real(0) : (vertical ? Real(1) : Real(-0.1));
            for (MFIter mfi(*model.vec_tke[0]); mfi.isValid(); ++mfi) {
                auto tke = model.vec_tke[0]->array(mfi);
                auto gls = model.vec_gls[0]->array(mfi);
                auto zw = model.vec_z_w[0]->array(mfi);
                auto w = W.array(mfi);
                ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    tke(i,j,k,0) = old_k; gls(i,j,k,0) = old_p;
                    tke(i,j,k,1) = cap ? H*old_k : H*Kmin*(vertical ? Real(2) : Real(0.25));
                    gls(i,j,k,1) = cap ? H*Pmin : solve ? H*Pmin*(k == N ? Real(100) : (k == 2 && vertical ? Real(50) : Real(1))) :
                                          H*Pmin*(vertical ? Real(2) : Real(0.25));
                    tke(i,j,k,2) = Kmin; gls(i,j,k,2) = Pmin;
                    zw(i,j,k) = H*k; w(i,j,k) = speed*k;
                });
            }
            model.gls_corrector(0,model.vec_gls[0].get(),model.vec_tke[0].get(),W,
                model.vec_Akv[0].get(),model.vec_Akt[0].get(),model.vec_Akk[0].get(),
                model.vec_Akp[0].get(),model.vec_mskr[0].get(),model.vec_msku[0].get(),
                model.vec_mskv[0].get(),0,1,N,dt);
            Vector<Real> expected_k(N,Kmin), expected_p(N,Pmin);
            if (cap) {
                Real expected_length = Real(0.4)*H*(N+1);
                bool enabled = true;
                pp.query("gls_length_cap",enabled);
                if (!enabled) {
                    expected_length = cmu*cmu*cmu*std::pow(old_k,Real(1.5))/Pmin;
                }
                for (int n = 0; n < N; ++n) {
                    expected_k[n] = old_k;
                    expected_p[n] = cmu*cmu*cmu*std::pow(old_k,Real(1.5))/expected_length;
                }

            } else if (!solve) {
                const Real weighted = amrex::max(H, H-dt*speed);
                for (int n = 0; n < N; ++n) {
                    expected_k[n] = amrex::max(Kmin, weighted*Kmin/(H*(Real(1)+dt*decay)));
                    expected_p[n] = amrex::max(Pmin, weighted*Pmin/(H*(Real(1)+dt*c2*decay)));
                }
            } else {
                // One profile limits both bottom and middle; the other leaves the middle
                // above its floor so the bottom limit's effect is not overwritten.
                const Real a = dt*nu/H;
                const Real diagonal = H*(Real(1)+dt*c2*decay)+a;
                const Real factor = -dt*cmu*cmu*cmu;
                const Real flux_top = factor*std::pow(Kmin,Real(1.5))/Real(0.41)*nu/
                    ((Zos+H/Real(2))*(Zos+H/Real(2)));
                const Real flux_bottom = factor*std::pow(Kmin,Real(1.5))/Real(0.41)*nu/
                    ((Real(1.e6)+H/Real(2))*(Real(1.e6)+H/Real(2)));
                const Real upper = (Real(100)*H*Pmin-flux_top)/diagonal;
                const Real middle_diagonal = diagonal+a-a*a/diagonal;
                const Real middle = ((vertical ? Real(50) : Real(1))*H*Pmin+a*upper)/
                                    middle_diagonal;
                const Real lower = (H*Pmin+a*middle-flux_bottom)/
                                   (diagonal-a*a/middle_diagonal);
                AMREX_ALWAYS_ASSERT(lower < Pmin && upper > Real(2)*Pmin);
                AMREX_ALWAYS_ASSERT(vertical ? middle+a/middle_diagonal*Pmin > Pmin :
                                               middle+a/middle_diagonal*Pmin < Pmin);
                expected_p[0] = Pmin;
                expected_p[1] = amrex::max(Pmin,middle+a/middle_diagonal*Pmin);
                expected_p[2] = upper+a/diagonal*expected_p[1];
            }
            for (int n = 0; n < N; ++n) {
                // Only the interior interfaces enter the implicit solve; boundary values
                // are Dirichlet. Copy each horizontal slice for global extrema assertions.
                const Box slice = makeSlab(amrex::convert(model.Geom(0).Domain(),
                                                        IntVect(0,0,1)),2,n+1);
                if (cap) {
                    MultiFab selected(BoxArray(slice),DistributionMapping(BoxArray(slice)),1,0);
                    selected.ParallelCopy(*model.vec_Lscale[0],0,0,1,0,0);
                    bool enabled = true;
                    pp.query("gls_length_cap",enabled);
                    const Real expected = enabled ? Real(0.4)*H*(N+1) :
                        cmu*cmu*cmu*std::pow(old_k,Real(1.5))/Pmin;
                    check_field(selected,0,expected,"Length cap differs from ROMS 0.4*depth");
                }
                for (auto field : {model.vec_tke[0].get(),model.vec_gls[0].get()}) {
                    MultiFab selected(BoxArray(slice),DistributionMapping(BoxArray(slice)),1,0);
                    selected.ParallelCopy(*field,1,0,1,0,0);
                    check_field(selected,0,field == model.vec_tke[0].get() ? expected_k[n] :
                        expected_p[n], solve ? "GLS substitution floor differs from reference" :
                                             "Thickness-weighted advection floor differs from reference");
                }
            }
        }
    }
}
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv,true,MPI_COMM_WORLD,test_defaults);
    inputs_name = argv[1];
    {
        REMORA model;
        model.InitData();
        std::string name;
        ParmParse("test").get("case",name);
        if (name == "predictor") { predictor(model); }
        else if (name == "evolve" || name == "regrid") { evolve(model,name == "regrid"); }
        else { corrector(model,name == "solve",name == "cap"); }
        amrex::Print() << "GLS " << name << " reference checks passed\n";
    }
    amrex::Finalize();
}
