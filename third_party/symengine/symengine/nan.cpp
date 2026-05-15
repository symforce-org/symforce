#include <symengine/nan.h>
#include <symengine/constants.h>

namespace SymEngine
{

NaN::NaN(){SYMENGINE_ASSIGN_TYPEID()}

hash_t NaN::__hash__() const
{
    hash_t seed = SYMENGINE_NOT_A_NUMBER;
    return seed;
}

bool NaN::__eq__(const Basic &o) const
{
    if (is_a<NaN>(o))
        return true;
    else
        return false;
}

int NaN::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<NaN>(o))
    return 0;
}

RCP<const Basic> NaN::conjugate() const
{
    return Nan;
}

RCP<const Number> NaN::add(const Number &other) const
{
    return rcp_from_this_cast<Number>();
}

RCP<const Number> NaN::mul(const Number &other) const
{
    return rcp_from_this_cast<Number>();
}

RCP<const Number> NaN::div(const Number &other) const
{
    return rcp_from_this_cast<Number>();
}

RCP<const Number> NaN::pow(const Number &other) const
{
    return rcp_from_this_cast<Number>();
}

RCP<const Number> NaN::rpow(const Number &other) const
{
    return rcp_from_this_cast<Number>();
}

class EvaluateNaN : public Evaluate
{
    RCP<const Basic> sin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> cos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> tan(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> cot(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> sec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> csc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> asin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> acos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> acsc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> asec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> atan(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> acot(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> sinh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> csch(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> cosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> sech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> tanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> coth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> asinh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> acosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> acsch(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> asech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> atanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> acoth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> abs(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> log(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> gamma(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> exp(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> floor(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> ceiling(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> truncate(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> erf(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
    RCP<const Basic> erfc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<NaN>(x))
        return Nan;
    }
};

Evaluate &NaN::get_eval() const
{
    static EvaluateNaN evaluate_NaN;
    return evaluate_NaN;
}

} // namespace SymEngine
