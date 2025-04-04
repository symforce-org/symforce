// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

namespace sym {

/**
 * Given input symbols `x` and `y`, return a list of expressions which provide good test coverage
 * over symbolic functions supported by symforce.
 *
 * The intention is that generating this function for a given backend should provide good test
 * coverage indicating that the printer for that backend is implemented correctly.
 *
 * This does not attempt to test the rest of the backend (any geo, cam, matrix, or DataBuffer use),
 * just the printer itself.
 */
__host__ __device__ void BackendTestFunctionFloat64(
    const double x, const double y, double* const __restrict__ res0 = nullptr,
    double* const __restrict__ res1 = nullptr, double* const __restrict__ res2 = nullptr,
    double* const __restrict__ res3 = nullptr, double* const __restrict__ res4 = nullptr,
    double* const __restrict__ res5 = nullptr, double* const __restrict__ res6 = nullptr,
    double* const __restrict__ res7 = nullptr, double* const __restrict__ res8 = nullptr,
    double* const __restrict__ res9 = nullptr, double* const __restrict__ res10 = nullptr,
    double* const __restrict__ res11 = nullptr, double* const __restrict__ res12 = nullptr,
    double* const __restrict__ res13 = nullptr, double* const __restrict__ res14 = nullptr,
    double* const __restrict__ res15 = nullptr, double* const __restrict__ res16 = nullptr,
    double* const __restrict__ res17 = nullptr, double* const __restrict__ res18 = nullptr,
    double* const __restrict__ res19 = nullptr, double* const __restrict__ res20 = nullptr,
    double* const __restrict__ res21 = nullptr, double* const __restrict__ res22 = nullptr,
    double* const __restrict__ res23 = nullptr, double* const __restrict__ res24 = nullptr,
    double* const __restrict__ res25 = nullptr, double* const __restrict__ res26 = nullptr,
    double* const __restrict__ res27 = nullptr, double* const __restrict__ res28 = nullptr,
    double* const __restrict__ res29 = nullptr, double* const __restrict__ res30 = nullptr,
    double* const __restrict__ res31 = nullptr, double* const __restrict__ res32 = nullptr,
    double* const __restrict__ res33 = nullptr, double* const __restrict__ res34 = nullptr,
    double* const __restrict__ res35 = nullptr, double* const __restrict__ res36 = nullptr,
    double* const __restrict__ res37 = nullptr, double* const __restrict__ res38 = nullptr,
    double* const __restrict__ res39 = nullptr, double* const __restrict__ res40 = nullptr,
    double* const __restrict__ res41 = nullptr, double* const __restrict__ res42 = nullptr,
    double* const __restrict__ res43 = nullptr, double* const __restrict__ res44 = nullptr,
    double* const __restrict__ res45 = nullptr, double* const __restrict__ res46 = nullptr,
    double* const __restrict__ res47 = nullptr, double* const __restrict__ res48 = nullptr,
    double* const __restrict__ res49 = nullptr, double* const __restrict__ res50 = nullptr,
    double* const __restrict__ res51 = nullptr, double* const __restrict__ res52 = nullptr,
    double* const __restrict__ res53 = nullptr, double* const __restrict__ res54 = nullptr,
    double* const __restrict__ res55 = nullptr, double* const __restrict__ res56 = nullptr,
    double* const __restrict__ res57 = nullptr, double* const __restrict__ res58 = nullptr,
    double* const __restrict__ res59 = nullptr);

}  // namespace sym
