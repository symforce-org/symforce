#include <symengine/symbol.h>
#include <symengine/polys/uintpoly_flint.h>

namespace SymEngine
{

UIntPolyFlint::UIntPolyFlint(const RCP<const Basic> &var, fzp_t &&dict)
    : UFlintPoly(var, std::move(dict)){SYMENGINE_ASSIGN_TYPEID()}

      hash_t UIntPolyFlint::__hash__() const
{
    std::hash<std::string> str_hash;
    hash_t seed = SYMENGINE_UINTPOLYFLINT;

    seed += get_var()->hash();
    hash_combine(seed, str_hash(get_poly().to_string()));
    return seed;
}

URatPolyFlint::URatPolyFlint(const RCP<const Basic> &var, fqp_t &&dict)
    : UFlintPoly(var, std::move(dict)){SYMENGINE_ASSIGN_TYPEID()}

      hash_t URatPolyFlint::__hash__() const
{
    std::hash<std::string> str_hash;
    hash_t seed = SYMENGINE_URATPOLYFLINT;

    seed += get_var()->hash();
    hash_combine(seed, str_hash(get_poly().to_string()));
    return seed;
}
} // namespace SymEngine
