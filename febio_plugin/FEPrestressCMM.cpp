#include "FEPrestressCMM.h"
#include <FECore/FEModel.h>
#include <FECore/log.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>

//=============================================================================
BEGIN_FECORE_CLASS(FEPrestressCMM, FEUncoupledMaterial)
    ADD_PARAMETER(m_ce   , FE_RANGE_GREATER(0.0)         , "c_e"    );
    ADD_PARAMETER(m_phie , FE_RANGE_CLOSED(0.0, 1.0)     , "phi_e"  );
    ADD_PARAMETER(m_c1   , FE_RANGE_GREATER_OR_EQUAL(0.0), "c1"     );
    ADD_PARAMETER(m_c2   , FE_RANGE_GREATER(0.0)         , "c2"     );
    ADD_PARAMETER(m_G    , FE_RANGE_GREATER(0.0)         , "lam_dep");
    ADD_PARAMETER(m_phic0, FE_RANGE_CLOSED(0.0, 1.0)     , "phi_c"  );
    ADD_PARAMETER(m_kd   , FE_RANGE_GREATER_OR_EQUAL(0.0), "k_d"    );
    ADD_PARAMETER(m_Ksig , FE_RANGE_GREATER_OR_EQUAL(0.0), "K_sigma");
    ADD_PARAMETER(m_tgr  ,                                 "t_gr"   );
    ADD_PARAMETER(m_avg  , FE_RANGE_CLOSED(0, 1)         , "average_stimulus");
    ADD_PARAMETER(m_lamref,                                "lam_ref");
    ADD_PARAMETER(m_Jref ,                                 "J_ref"  );
    ADD_PARAMETER(m_fiber,                                 "fiber"  );
END_FECORE_CLASS();

//=============================================================================
// material point
//=============================================================================
FEMaterialPointData* FECMMPoint::Copy()
{
    FECMMPoint* pt = new FECMMPoint(*this);
    if (m_pNext) pt->m_pNext = m_pNext->Copy();
    return pt;
}

void FECMMPoint::Init()
{
    FEMaterialPointData::Init();
    m_phic = m_phic_p = (m_mat ? m_mat->m_phic0 : 0.0);
    m_lamn = m_lamn_p = (m_mat ? 1.0 / m_mat->m_G : 1.0);
    m_Jg = 1.0;
    m_twork = m_tp = 0.0;
    m_rphi = m_rlam = 0.0;
    m_cached = false;
    m_lame = (m_mat ? m_mat->m_G : 1.0);
    m_sigr = 1.0;
}

void FECMMPoint::Serialize(DumpStream& ar)
{
    FEMaterialPointData::Serialize(ar);
    ar & m_phic & m_lamn & m_Jg & m_twork & m_phic_p & m_lamn_p & m_tp
       & m_rphi & m_rlam & m_lamref & m_Jref & m_M & m_cached & m_lame & m_sigr;
}

void FECMMPoint::Update(const FETimeInfo& timeInfo)
{
    FEMaterialPointData::Update(timeInfo);
    if (m_mat) m_mat->UpdateState(*this, timeInfo);
}

//=============================================================================
// material
//=============================================================================
FEPrestressCMM::FEPrestressCMM(FEModel* pfem) : FEUncoupledMaterial(pfem)
{
    m_lamref = 1.0;
    m_Jref = 1.0;
    m_fiber = vec3d(0, 1, 0);
}

FEMaterialPointData* FEPrestressCMM::CreateMaterialPointData()
{
    return new FECMMPoint(new FEElasticMaterialPoint, this);
}

bool FEPrestressCMM::Init()
{
    if (FEUncoupledMaterial::Init() == false) return false;
    m_sigh = FiberStress(m_G);
    if (m_phic0 > 0.0 && m_sigh <= 0.0)
    {
        feLogError("prestress_cmm: the homeostatic fibre stress must be positive "
                   "(lam_dep has to exceed 1).");
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
double FEPrestressCMM::FiberStress2PK(double lam_e) const
{
    const double i4 = lam_e * lam_e;
    return m_c1 * (i4 - 1.0) * exp(m_c2 * (i4 - 1.0) * (i4 - 1.0));
}

double FEPrestressCMM::FiberStress(double lam_e) const
{
    return lam_e * lam_e * FiberStress2PK(lam_e);
}

//-----------------------------------------------------------------------------
//! Deviatoric Cauchy stress of the mixture.
//!
//! The elastin term is identical to FEBio's uncoupled Mooney-Rivlin with
//! c1 = phi_e c_e and c2 = 0, i.e. Eq. 11 of the manuscript.  The fibre term is
//! the deviatoric projection of the constrained-mixture fibre stress.  Growth
//! enters through an additive pressure, because
//!
//!     d/dJ [K/2 ln(J/J_g)^2] = K ln(J)/J - K ln(J_g)/J,
//!
//! i.e. exactly FEBio's U'(J) plus the correction returned here.
mat3ds FEPrestressCMM::DevStress(FEMaterialPoint& mp)
{
    mat3ds s = DevStressCore(mp, true);
    // report the state through the (unused) velocity field; only done here, so
    // that the perturbed evaluations of DevTangent do not overwrite it
    FEElasticMaterialPoint& et = *mp.ExtractData<FEElasticMaterialPoint>();
    FECMMPoint& pt = *mp.ExtractData<FECMMPoint>();
    et.m_v = vec3d(pt.m_phic, pt.m_lame, pt.m_sigr);
    return s;
}

mat3ds FEPrestressCMM::DevStressCore(FEMaterialPoint& mp, bool with_growth)
{
    FEElasticMaterialPoint& et = *mp.ExtractData<FEElasticMaterialPoint>();
    FECMMPoint& pt = *mp.ExtractData<FECMMPoint>();

    const mat3d& F = et.m_F;
    const double J = et.m_J;

    if (!pt.m_cached)
    {
        pt.m_lamref = m_lamref(mp);
        pt.m_Jref = m_Jref(mp);
        vec3d M = m_fiber(mp);
        M.unit();
        pt.m_M = M;
        pt.m_elem = mp.m_elem;
        pt.m_index = mp.m_index;
        pt.m_cached = true;
    }

    // --- growth tensor: transversely isotropic, perpendicular to the fibre
    //     F_g = M (x) M + sqrt(J_g) (I - M (x) M),  det F_g = J_g
    //     (Braeu et al. 2017).  Mass is added across the wall rather than
    //     along the fibre, so the fibre stretch is unaffected by growth,
    //     F_g^-1 M = M, and the wall thickens as it grows.
    const double Jg = pt.m_Jg;
    const double Je = J / Jg;
    mat3d Fe = F;
    if (Jg != 1.0)
    {
        const double a = 1.0 / sqrt(Jg);
        mat3ds MM = dyad(pt.m_M);
        const mat3d Fgi = mat3dd(a) + mat3d(MM) * (1.0 - a);  // a (I - M(x)M) + M(x)M
        Fe = F * Fgi;
    }

    // --- elastin
    const double c1e = m_phie * m_ce;
    mat3ds b = (Fe * Fe.transpose()).sym();
    mat3ds bbar = b * pow(Je, -2.0 / 3.0);
    mat3ds s = (bbar - mat3dd(bbar.tr() / 3.0)) * (2.0 * c1e / J);

    // --- embedded fibre family
    if (pt.m_phic > 0.0)
    {
        const vec3d f = F * pt.m_M;
        const double ln = pt.m_lamref * pt.m_lamn;
        const double lam_e = f.norm() / ln;
        const double rho = pt.m_phic * pt.m_Jref;
        mat3ds sf = dyad(f) * (rho * FiberStress2PK(lam_e) / (J * ln * ln));
        s += sf.dev();
        pt.m_lame = lam_e;
        pt.m_sigr = (m_sigh > 0.0 ? FiberStress(lam_e) / m_sigh : 1.0);
    }

    // --- growth: shift of the stress-free volume.  This is a pure pressure;
    //     it is excluded from the finite-difference tangent and added back
    //     analytically in DevTangent, because K log(J_g) is large and would
    //     otherwise swamp the difference quotient.
    if (with_growth && Jg != 1.0) s += mat3dd(-m_K * log(Jg) / J);

    return s;
}

//-----------------------------------------------------------------------------
//! 2nd Piola-Kirchhoff stress associated with DevStress, evaluated for a
//! prescribed Lagrange strain (mirrors FESolidMaterial::PK2Stress).
mat3ds FEPrestressCMM::DevPK2Stress(FEMaterialPoint& mp, const mat3ds E)
{
    mat3ds C = mat3dd(1) + E * 2;

    vec3d v[3];
    double lam[3];
    C.eigen2(lam, v);
    lam[0] = sqrt(lam[0]); lam[1] = sqrt(lam[1]); lam[2] = sqrt(lam[2]);
    mat3ds U = dyad(v[0]) * lam[0] + dyad(v[1]) * lam[1] + dyad(v[2]) * lam[2];
    const double J = lam[0] * lam[1] * lam[2];

    FEElasticMaterialPoint& pt = *mp.ExtractData<FEElasticMaterialPoint>();
    mat3d Fsafe = pt.m_F;
    double Jsafe = pt.m_J;
    pt.m_F = U;
    pt.m_J = J;

    mat3ds s = DevStressCore(mp, false);

    pt.m_F = Fsafe;
    pt.m_J = Jsafe;

    mat3ds Ui = dyad(v[0]) / lam[0] + dyad(v[1]) / lam[1] + dyad(v[2]) / lam[2];
    return (Ui * s * Ui).sym() * J;
}

//-----------------------------------------------------------------------------
//! Numerical deviatoric spatial tangent (forward differences in the material frame).
tens4ds FEPrestressCMM::DevTangent(FEMaterialPoint& mp)
{
    FEElasticMaterialPoint& pt = *mp.ExtractData<FEElasticMaterialPoint>();
    const mat3d F = pt.m_F;
    const double J = pt.m_J;
    const mat3ds E = pt.Strain();

    const mat3ds S = DevPK2Stress(mp, E);

    const double eps = 1e-9;
    vec3d e[3] = { vec3d(1, 0, 0), vec3d(0, 1, 0), vec3d(0, 0, 1) };

    tens4dmm C;
    for (int k = 0; k < 3; ++k)
        for (int l = k; l < 3; ++l)
        {
            mat3ds dE = dyads(e[k], e[l]) * (eps / 2);
            mat3ds dS = (DevPK2Stress(mp, E + dE) - S) / eps;

            C(0, 0, k, l) = C(0, 0, l, k) = dS.xx();
            C(1, 1, k, l) = C(1, 1, l, k) = dS.yy();
            C(2, 2, k, l) = C(2, 2, l, k) = dS.zz();
            C(0, 1, k, l) = C(1, 0, k, l) = C(0, 1, l, k) = C(1, 0, l, k) = dS.xy();
            C(1, 2, k, l) = C(2, 1, k, l) = C(1, 2, l, k) = C(2, 1, l, k) = dS.yz();
            C(2, 0, k, l) = C(0, 2, k, l) = C(2, 0, l, k) = C(0, 2, l, k) = dS.xz();
        }

    tens4dmm c = C.pp(F) / J;
    tens4ds cs = c.supersymm();

    // analytic contribution of the growth pressure p(J) = -K ln(J_g)/J:
    // for sigma += p(J) I the spatial tangent is (p + J dp/dJ) IxI - 2p IoI,
    // and p + J dp/dJ vanishes identically for this p.
    const FECMMPoint& gt = *mp.ExtractData<FECMMPoint>();
    if (gt.m_Jg != 1.0)
    {
        const double p = -m_K * log(gt.m_Jg) / J;
        cs += dyad4s(mat3dd(1.0)) * (-2.0 * p);
    }
    return cs;
}

//-----------------------------------------------------------------------------
double FEPrestressCMM::DevStrainEnergyDensity(FEMaterialPoint& mp)
{
    FEElasticMaterialPoint& et = *mp.ExtractData<FEElasticMaterialPoint>();
    FECMMPoint& pt = *mp.ExtractData<FECMMPoint>();
    const double J = et.m_J;
    mat3ds C = et.RightCauchyGreen();
    const double I1bar = C.tr() * pow(J, -2.0 / 3.0);

    double W = m_phie * m_ce * (I1bar - 3.0);
    if (pt.m_phic > 0.0)
    {
        const double i4 = pt.m_lame * pt.m_lame;
        W += pt.m_phic * pt.m_Jref * m_c1 / (4.0 * m_c2)
             * (exp(m_c2 * (i4 - 1.0) * (i4 - 1.0)) - 1.0);
    }
    return W;
}

//-----------------------------------------------------------------------------
//! Rates of the two state variables, evaluated with the last converged
//! deformation.
//!
//! The mechanobiological stimulus is averaged over the element by default.
//! Evaluating it at each integration point separately couples the pointwise
//! state to the element-internal deformation modes of a HEX8, which a
//! displacement-based element barely restrains: a checkerboard perturbation of
//! the fibre stretch then grows exponentially even from an exactly prestressed
//! configuration.  Averaging removes that feedback and is consistent with the
//! stimulus being a tissue-level, not a point-level, quantity.
void FEPrestressCMM::Rates(FECMMPoint& pt)
{
    FEElasticMaterialPoint* et = pt.ExtractData<FEElasticMaterialPoint>();
    if (et == nullptr) return;

    double lam_fib = 0.0;
    int n = 0;
    if (m_avg && pt.m_elem)
    {
        for (int j = 0; j < pt.m_elem->GaussPoints(); ++j)
        {
            FEMaterialPoint* mj = pt.m_elem->GetMaterialPoint(j);
            FECMMPoint* pj = (mj ? mj->ExtractData<FECMMPoint>() : nullptr);
            FEElasticMaterialPoint* ej = (mj ? mj->ExtractData<FEElasticMaterialPoint>() : nullptr);
            if (pj == nullptr || ej == nullptr || !pj->m_cached) continue;
            lam_fib += (ej->m_F * pj->m_M).norm() / pj->m_lamref;
            ++n;
        }
    }
    if (n == 0)
    {
        lam_fib = (et->m_F * pt.m_M).norm() / pt.m_lamref;
        n = 1;
    }
    lam_fib /= n;

    const double lam_e = lam_fib / pt.m_lamn_p;
    const double dev = FiberStress(lam_e) / m_sigh - 1.0;
    const double ups = std::max(0.0, 1.0 + m_Ksig * dev);

    pt.m_rphi = pt.m_phic_p * m_kd * m_Ksig * dev;
    pt.m_rlam = m_kd * ups * (lam_fib / m_G - pt.m_lamn_p);
}

//-----------------------------------------------------------------------------
//! Explicit update of the growth & remodelling state.  Called once per time
//! step from PreSolveUpdate, i.e. with the converged deformation of the
//! previous step (homogenized constrained mixture, Cyron et al. 2016).
void FEPrestressCMM::UpdateState(FECMMPoint& pt, const FETimeInfo& tp)
{
    const double t = tp.currentTime;
    const double eps = 1e-12;

    if (m_kd <= 0.0 || m_phic0 <= 0.0 || !pt.m_cached)
    {
        pt.m_twork = pt.m_tp = t;
        return;
    }

    // The working state belongs to m_twork.  If the solver has moved past that
    // time, that step converged and the working state becomes the committed
    // one; the rates are then re-evaluated with its converged deformation.
    // On a retry the solver comes back with t <= m_twork and nothing is
    // committed, so the step is simply recomputed from the same state.
    if (t > pt.m_twork + eps)
    {
        pt.m_phic_p = pt.m_phic;
        pt.m_lamn_p = pt.m_lamn;
        pt.m_tp = pt.m_twork;

        pt.m_rphi = pt.m_rlam = 0.0;
        if (pt.m_tp >= m_tgr - eps) Rates(pt);
    }

    // optional diagnostics (run with PRESTRESS_CMM_DEBUG=1 and OMP_NUM_THREADS=1)
    static bool dbg = (getenv("PRESTRESS_CMM_DEBUG") != nullptr);
    if (dbg)
    {
        static double tlog = -1.0;
        static double lo[4], hi[4];
        FEElasticMaterialPoint* et2 = pt.ExtractData<FEElasticMaterialPoint>();
        if (et2)
        {
            const double lf = (et2->m_F * pt.m_M).norm() / pt.m_lamref;
            const double le = lf / pt.m_lamn;
            const double dv = FiberStress(le) / m_sigh - 1.0;
            const double v[4] = { pt.m_lamn, lf, le, dv };
            if (t != tlog) { for (int i=0;i<4;++i) { lo[i]=hi[i]=v[i]; } tlog = t; }
            else for (int i=0;i<4;++i) { if (v[i]<lo[i]) lo[i]=v[i]; if (v[i]>hi[i]) hi[i]=v[i]; }
            feLog("GR t=%.5f dt=%.5f lamn=[%.12f,%.12f] lamfib=[%.12f,%.12f] "
                  "lam_e=[%.12f,%.12f] dev=[%.4e,%.4e] phic=%.10f\n",
                  t, t - pt.m_tp, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                  lo[3], hi[3], pt.m_phic);
        }
    }

    // forward Euler from the committed state to the requested time
    const double dt = std::max(0.0, t - pt.m_tp);
    pt.m_phic = std::max(1e-12, pt.m_phic_p + dt * pt.m_rphi);
    pt.m_lamn = std::max(1e-12, pt.m_lamn_p + dt * pt.m_rlam);
    pt.m_Jg = (m_phie + pt.m_phic) / (m_phie + m_phic0);
    pt.m_twork = t;
}
