#include <symengine/polys/uintpoly.h>

namespace SymEngine
{

UIntPoly::UIntPoly(const RCP<const Basic> &var, UIntDict &&dict)
    : USymEnginePoly(
        var, std::move(dict)){SYMENGINE_ASSIGN_TYPEID()
                                  SYMENGINE_ASSERT(is_canonical(get_poly()))}

      hash_t UIntPoly::__hash__() const
{
    hash_t seed = SYMENGINE_UINTPOLY;

    seed += get_var()->hash();
    for (const auto &it : get_poly().dict_) {
        hash_t temp = SYMENGINE_UINTPOLY;
        hash_combine<unsigned int>(temp, it.first);
        hash_combine<long long int>(temp, mp_get_si(it.second));
        seed += temp;
    }
    return seed;
}

bool divides_upoly(const UIntPoly &a, const UIntPoly &b,
                   const Ptr<RCP<const UIntPoly>> &out)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    auto a_poly = a.get_poly();
    auto b_poly = b.get_poly();
    if (a_poly.size() == 0)
        return false;

    map_uint_mpz res;
    UIntDict tmp;
    integer_class q, r;
    unsigned int a_deg, b_deg;

    while (b_poly.size() >= a_poly.size()) {
        a_deg = a_poly.degree();
        b_deg = b_poly.degree();

        mp_tdiv_qr(q, r, b_poly.get_lc(), a_poly.get_lc());
        if (r != 0)
            return false;

        res[b_deg - a_deg] = q;
        UIntDict tmp = UIntDict({{b_deg - a_deg, q}});
        b_poly -= (a_poly * tmp);
    }

    if (b_poly.empty()) {
        *out = UIntPoly::from_dict(a.get_var(), std::move(res));
        return true;
    } else {
        return false;
    }
}

} // namespace SymEngine
