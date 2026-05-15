#include <symengine/tuple.h>
#include <symengine/dict.h>

namespace SymEngine
{

Tuple::Tuple(const vec_basic &container) : container_(container)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(Tuple::is_canonical(container_));
}

bool Tuple::is_canonical(const vec_basic &container)
{
    return true;
}

hash_t Tuple::__hash__() const
{
    hash_t seed = SYMENGINE_TUPLE;
    for (const auto &a : container_)
        hash_combine<Basic>(seed, *a);
    return seed;
}

bool Tuple::__eq__(const Basic &o) const
{
    if (is_a<Tuple>(o)) {
        const Tuple &other = down_cast<const Tuple &>(o);
        return unified_eq(container_, other.container_);
    }
    return false;
}

int Tuple::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Tuple>(o))
    const Tuple &other = down_cast<const Tuple &>(o);
    return unified_compare(container_, other.container_);
}

RCP<const Basic> tuple(const vec_basic &arg)
{
    return make_rcp<const Tuple>(arg);
}

} // namespace SymEngine
