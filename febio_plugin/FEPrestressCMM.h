//=============================================================================
// FEBio plugin: constrained mixture of an isotropic ground substance (elastin)
// and one family of embedded fibres that may turn over (growth & remodelling).
//
// The material is formulated with respect to the *stress-free elastin*
// configuration X_e produced by the augmented iterative method (AIM).  Two
// per-integration-point scalars link that configuration back to the in vivo
// (homeostatic) configuration x:
//
//     lam_ref = |G_e M|      stretch of the fibre direction from X_e to x
//     J_ref   = det(G_e)     volume ratio from X_e to x
//
// With these, a model built on X_e reproduces exactly the elastin stress of the
// AIM solve and, at the in vivo state, the prescribed homeostatic fibre stress.
//
// Constitutive model (manuscript Eqs. 10-11):
//     psi_e  = phi_e c_e (I1bar - 3) + K/2 ln(J/J_g)^2
//     psi_ef = phi_ef c1/(4 c2) [exp(c2 (I4e - 1)^2) - 1]
//
// Growth & remodelling: homogenized constrained mixture (Cyron, Aydin &
// Humphrey 2016; Braeu et al. 2017)
//     d(phi_c)/dt   = phi_c k_d K_sigma (sigma_c/sigma_h - 1)
//     d(lam_n)/dt   = k_d Upsilon (lam_fib/G - lam_n),  Upsilon = max(0, 1 + K_sigma (...))
//     J_g           = (phi_e + phi_c) / (phi_e + phi_c0)
//     F_g           = M (x) M + sqrt(J_g) (I - M (x) M)
//
// Growth is transversely isotropic and perpendicular to the fibre, so mass is
// added across the wall: the fibre stretch is unaffected by growth and the wall
// thickens instead of the lumen widening.  At lam_fib = 1 and lam_n = 1/G all
// rates vanish, so a perfectly prestressed configuration is a fixed point.
//
// The stimulus is averaged over the element by default (average_stimulus = 1);
// evaluating it per integration point excites an element-internal checkerboard
// mode of the HEX8 -- see docs/growth.md.
//=============================================================================
#pragma once

#include "FEBioMech/FEUncoupledMaterial.h"
#include <FECore/FEModelParam.h>

class FEPrestressCMM;

//-----------------------------------------------------------------------------
class FECMMPoint : public FEMaterialPointData
{
public:
    FECMMPoint(FEMaterialPointData* pt, FEPrestressCMM* mat)
        : FEMaterialPointData(pt), m_mat(mat) {}

    FEMaterialPointData* Copy() override;
    void Init() override;
    void Update(const FETimeInfo& timeInfo) override;
    void Serialize(DumpStream& ar) override;

public:
    FEPrestressCMM* m_mat = nullptr;

    // working state (the trial state of the step currently being solved)
    double m_phic  = 0.0;   //!< current fibre mass fraction (referred to the in vivo state)
    double m_lamn  = 1.0;   //!< normalised mean natural stretch of the fibre family
    double m_Jg    = 1.0;   //!< growth volume ratio
    double m_twork = 0.0;   //!< time the working state belongs to

    // last converged state and the rates evaluated there; keeping these makes
    // the explicit update safe when FEBio retries a time step
    double m_phic_p = 0.0;
    double m_lamn_p = 1.0;
    double m_tp     = 0.0;
    double m_rphi   = 0.0;  //!< d(phi_c)/dt at the last converged state
    double m_rlam   = 0.0;  //!< d(lam_n)/dt at the last converged state

    // cached per-point quantities, filled by DevStress()
    double m_lamref = 1.0;
    double m_Jref   = 1.0;
    vec3d  m_M      = vec3d(0, 1, 0);
    bool   m_cached = false;
    FEElement* m_elem = nullptr;   //!< element this point belongs to
    int    m_index  = 0;           //!< local integration point index

    // reporting
    double m_lame = 1.0;    //!< elastic fibre stretch
    double m_sigr = 1.0;    //!< sigma_c / sigma_h
};

//-----------------------------------------------------------------------------
class FEPrestressCMM : public FEUncoupledMaterial
{
public:
    FEPrestressCMM(FEModel* pfem);

    FEMaterialPointData* CreateMaterialPointData() override;
    bool Init() override;

    //! deviatoric Cauchy stress (plus the hydrostatic correction due to growth)
    mat3ds DevStress(FEMaterialPoint& mp) override;

    //! the same, with the growth pressure optionally left out
    mat3ds DevStressCore(FEMaterialPoint& mp, bool with_growth);

    //! numerical deviatoric spatial tangent
    tens4ds DevTangent(FEMaterialPoint& mp) override;

    double DevStrainEnergyDensity(FEMaterialPoint& mp) override;

    //! 2nd PK stress associated with DevStress, for the numerical tangent
    mat3ds DevPK2Stress(FEMaterialPoint& mp, const mat3ds E);

    //! advance the growth & remodelling state of one material point
    void UpdateState(FECMMPoint& pt, const FETimeInfo& tp);

    //! rates of the state variables at the last converged deformation
    void Rates(FECMMPoint& pt);

    //! constituent-level fibre Cauchy stress for an elastic fibre stretch
    double FiberStress(double lam_e) const;
    double FiberStress2PK(double lam_e) const;

public:
    // elastin
    double m_ce = 1000.0;     //!< stiffness-like parameter c_e [kPa]
    double m_phie = 0.67;     //!< elastin mass fraction
    // embedded fibre family
    double m_c1 = 235.0;      //!< fibre stiffness-like parameter [kPa]
    double m_c2 = 4.08;       //!< fibre nonlinearity parameter [-]
    double m_G = 1.1;         //!< deposition (pre-)stretch
    double m_phic0 = 0.0;     //!< fibre mass fraction (0 -> elastin only)
    // growth & remodelling
    double m_kd = 0.0;        //!< mass turnover rate [1/time] (0 -> no G&R)
    double m_Ksig = 1.0;      //!< gain of the stress-mediated production
    double m_tgr = 1.0;       //!< time at which G&R starts
    int    m_avg = 1;         //!< 1: average the G&R stimulus over the element

    // per-integration-point maps
    FEParamDouble m_lamref;   //!< |G_e M|
    FEParamDouble m_Jref;     //!< det(G_e)
    FEParamVec3   m_fiber;    //!< fibre direction in the reference configuration

    double m_sigh = 0.0;      //!< homeostatic constituent-level fibre stress

    DECLARE_FECORE_CLASS();
};
