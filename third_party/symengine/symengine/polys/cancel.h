#ifndef SYMENGINE_CANCEL_H
#define SYMENGINE_CANCEL_H

#include <symengine/basic.h>
#include <symengine/polys/basic_conversions.h>

namespace SymEngine
{
// Declaration of cancel function
template <typename Poly>
inline void cancel(const RCP<const Basic> &numer, const RCP<const Basic> &denom,
                   const Ptr<RCP<const Poly>> &result_numer,
                   const Ptr<RCP<const Poly>> &result_denom,
                   const Ptr<RCP<const Poly>> &common)
{
    // Converting basic to Poly
    umap_basic_num numer_gens = _find_gens_poly(numer);
    umap_basic_num denom_gens = _find_gens_poly(denom);

    if (numer_gens.size() != 1 && denom_gens.size() != 1) {
        // only considering univariate here
        return;
    }
    RCP<const Basic> numer_var = numer_gens.begin()->first;
    RCP<const Basic> denom_var = denom_gens.begin()->first;

    RCP<const Poly> numer_poly = from_basic<Poly>(numer, numer_var);
    RCP<const Poly> denom_poly = from_basic<Poly>(denom, denom_var);

    // Finding common factors of numer_poly and denom_poly
    RCP<const Poly> gcd_poly = gcd_upoly(*numer_poly, *denom_poly);

    // Dividing both by common factors
    divides_upoly(*gcd_poly, *numer_poly, outArg(*result_numer));
    divides_upoly(*gcd_poly, *denom_poly, outArg(*result_denom));
    *common = gcd_poly;
}
} // namespace SymEngine
#endif // SYMENGINE_CANCEL_H
