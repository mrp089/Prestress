#include <FECore/FECoreKernel.h>
#include "FEPrestressCMM.h"

FECORE_EXPORT unsigned int GetSDKVersion()
{
    return FE_SDK_VERSION;
}

FECORE_EXPORT void PluginInitialize(FECoreKernel& febio)
{
    FECoreKernel::SetInstance(&febio);
    REGISTER_FECORE_CLASS(FEPrestressCMM, "prestress_cmm");
}

FECORE_EXPORT void PluginCleanup()
{
}
