#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/rational.h>

namespace SymEngine
{

RCP<const Basic> Number::conjugate() const
{
    if (not is_complex()) {
        return this->rcp_from_this();
    }
    throw NotImplementedError("Not Implemented.");
}

RCP<const Number> Number::sub(const Number &other) const
{
    return add(*other.mul(*integer(-1)));
}

RCP<const Number> Number::rsub(const Number &other) const
{
    return mul(*integer(-1))->add(other);
}

RCP<const Number> Number::div(const Number &other) const
{
    return mul(*other.pow(*integer(-1)));
}

RCP<const Number> Number::rdiv(const Number &other) const
{
    return other.mul(*pow(*integer(-1)));
}

} // namespace SymEngine
