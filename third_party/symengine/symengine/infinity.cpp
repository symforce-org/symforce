#include <symengine/complex.h>
#include <symengine/complex_double.h>
#include <symengine/constants.h>
#include <symengine/infinity.h>
#include <symengine/functions.h>
#include <symengine/symengine_exception.h>
#include <symengine/complex_mpc.h>

using SymEngine::ComplexMPC;

namespace SymEngine
{

Infty::Infty(const RCP<const Number> &direction)
{
    SYMENGINE_ASSIGN_TYPEID()
    _direction = direction;
    SYMENGINE_ASSERT(is_canonical(_direction));
}

Infty::Infty(const Infty &inf)
{
    SYMENGINE_ASSIGN_TYPEID()
    _direction = inf.get_direction();
    SYMENGINE_ASSERT(is_canonical(_direction))
}

RCP<const Infty> Infty::from_direction(const RCP<const Number> &direction)
{
    return make_rcp<Infty>(direction);
}

RCP<const Infty> Infty::from_int(const int val)
{
    SYMENGINE_ASSERT(val >= -1 && val <= 1)
    return make_rcp<Infty>(integer(val));
}

//! Canonical when the direction is -1, 0 or 1.
bool Infty::is_canonical(const RCP<const Number> &num) const
{
    if (is_a<Complex>(*num) || is_a<ComplexDouble>(*num))
        throw NotImplementedError("Not implemented for all directions");

    if (num->is_one() || num->is_zero() || num->is_minus_one())
        return true;

    return false;
}

hash_t Infty::__hash__() const
{
    hash_t seed = SYMENGINE_INFTY;
    hash_combine<Basic>(seed, *_direction);
    return seed;
}

bool Infty::__eq__(const Basic &o) const
{
    if (is_a<Infty>(o)) {
        const Infty &s = down_cast<const Infty &>(o);
        return eq(*_direction, *(s.get_direction()));
    }

    return false;
}

int Infty::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Infty>(o))
    const Infty &s = down_cast<const Infty &>(o);
    return _direction->compare(*(s.get_direction()));
}

bool Infty::is_unsigned_infinity() const
{
    return _direction->is_zero();
}

bool Infty::is_positive_infinity() const
{
    return _direction->is_positive();
}

bool Infty::is_negative_infinity() const
{
    return _direction->is_negative();
}

RCP<const Basic> Infty::conjugate() const
{
    if (is_positive_infinity() or is_negative_infinity()) {
        return infty(_direction);
    }
    return make_rcp<const Conjugate>(ComplexInf);
}

RCP<const Number> Infty::add(const Number &other) const
{
    if (not is_a<Infty>(other))
        return rcp_from_this_cast<Number>();

    const Infty &s = down_cast<const Infty &>(other);

    if (not eq(*s.get_direction(), *_direction))
        return Nan;
    else if (is_unsigned_infinity())
        return Nan;
    else
        return rcp_from_this_cast<Number>();
}

RCP<const Number> Infty::mul(const Number &other) const
{
    if (is_a<Complex>(other))
        throw NotImplementedError(
            "Multiplication with Complex not implemented");

    if (is_a<Infty>(other)) {
        const Infty &s = down_cast<const Infty &>(other);
        return make_rcp<const Infty>(this->_direction->mul(*(s._direction)));
    } else {
        if (other.is_positive())
            return rcp_from_this_cast<Number>();
        else if (other.is_negative())
            return make_rcp<const Infty>(this->_direction->mul(*minus_one));
        else
            return Nan;
    }
}

RCP<const Number> Infty::div(const Number &other) const
{
    if (is_a<Infty>(other)) {
        return Nan;
    } else {
        if (other.is_positive())
            return rcp_from_this_cast<Number>();
        else if (other.is_zero())
            return infty(0);
        else
            return infty(this->_direction->mul(*minus_one));
    }
}

RCP<const Number> Infty::pow(const Number &other) const
{
    if (is_a<Infty>(other)) {
        if (is_positive_infinity()) {
            if (other.is_negative()) {
                return zero;
            } else if (other.is_positive()) {
                return rcp_from_this_cast<Number>();
            } else {
                return Nan;
            }
        } else if (is_negative_infinity()) {
            return Nan;
        } else {
            if (other.is_positive()) {
                return infty(0);
            } else if (other.is_negative()) {
                return zero;
            } else {
                return Nan;
            }
        }
    } else if (is_a<Complex>(other)) {
        throw NotImplementedError(
            "Raising to the Complex powers not yet implemented");
    } else {
        if (other.is_negative()) {
            return zero;
        } else if (other.is_zero()) {
            return one;
        } else {
            if (is_positive_infinity()) {
                return rcp_from_this_cast<Number>();
            } else if (is_negative_infinity()) {
                throw NotImplementedError("Raising Negative Infty to the "
                                          "Positive Real powers not yet "
                                          "implemented");
            } else {
                return infty(0);
            }
        }
    }
}

RCP<const Number> Infty::rpow(const Number &other) const
{
    if (is_a_Complex(other)) {
        throw NotImplementedError(
            "Raising Complex powers to Infty not yet implemented");
    } else {
        if (other.is_negative()) {
            throw NotImplementedError("Raising Negative numbers to infinite "
                                      "powers not yet implemented");
        } else if (other.is_zero()) {
            throw SymEngineException("Indeterminate Expression: `0 ** +- "
                                     "unsigned Infty` encountered");
        } else {
            const Number &s = down_cast<const Number &>(other);
            if (s.is_one()) {
                return Nan;
            } else if (is_positive_infinity()) {
                if (s.sub(*one)->is_negative()) {
                    return zero;
                } else {
                    return rcp_from_this_cast<Number>();
                }
            } else if (is_negative_infinity()) {
                if (s.sub(*one)->is_negative()) {
                    return infty(0);
                } else {
                    return zero;
                }
            } else {
                throw SymEngineException("Indeterminate Expression: `Positive "
                                         "Real Number ** unsigned Infty` "
                                         "encountered");
            }
        }
    }
}

inline RCP<const Infty> infty(const RCP<const Number> &direction)
{
    return make_rcp<Infty>(direction);
}

class EvaluateInfty : public Evaluate
{
    RCP<const Basic> sin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("sin is not defined for infinite values");
    }
    RCP<const Basic> cos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("cos is not defined for infinite values");
    }
    RCP<const Basic> tan(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("tan is not defined for infinite values");
    }
    RCP<const Basic> cot(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("cot is not defined for infinite values");
    }
    RCP<const Basic> sec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("sec is not defined for infinite values");
    }
    RCP<const Basic> csc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("csc is not defined for infinite values");
    }
    RCP<const Basic> asin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("asin is not defined for infinite values");
    }
    RCP<const Basic> acos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("acos is not defined for infinite values");
    }
    RCP<const Basic> acsc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("acsc is not defined for infinite values");
    }
    RCP<const Basic> asec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        throw DomainError("asec is not defined for infinite values");
    }
    RCP<const Basic> atan(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return div(pi, integer(2));
        } else if (s.is_negative()) {
            return mul(minus_one, (div(pi, integer(2))));
        } else {
            throw DomainError("atan is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> acot(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return zero;
        } else {
            throw DomainError("acot is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> sinh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return infty(s.get_direction());
        } else {
            throw DomainError("sinh is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> csch(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return zero;
        } else {
            throw DomainError("csch is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> cosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return Inf;
        } else {
            throw DomainError("cosh is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> sech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return zero;
        } else {
            throw DomainError("sech is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> tanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return one;
        } else if (s.is_negative()) {
            return minus_one;
        } else {
            throw DomainError("tanh is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> coth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return one;
        } else if (s.is_negative()) {
            return minus_one;
        } else {
            throw DomainError("coth is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> asinh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return infty(s.get_direction());
        } else {
            throw DomainError("asinh is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> acosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return Inf;
        } else {
            throw DomainError("acosh is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> acsch(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return zero;
        } else {
            throw DomainError("acsch is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> asech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return mul(mul(I, pi), div(one, integer(2)));
        } else {
            throw DomainError("asech is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> atanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return mul(minus_one, div(mul(pi, I), integer(2)));
        } else if (s.is_negative()) {
            return div(mul(pi, I), integer(2));
        } else {
            throw DomainError("atanh is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> acoth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return zero;
        } else {
            throw DomainError("acoth is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> abs(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        return Inf;
    }
    RCP<const Basic> log(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive() or s.is_negative()) {
            return Inf;
        } else {
            return ComplexInf;
        }
    }
    RCP<const Basic> gamma(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return Inf;
        } else {
            return ComplexInf;
        }
    }
    RCP<const Basic> exp(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return Inf;
        } else if (s.is_negative()) {
            return zero;
        } else {
            throw DomainError("exp is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> floor(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return Inf;
        } else if (s.is_negative()) {
            return NegInf;
        } else {
            throw DomainError("floor is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> ceiling(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return Inf;
        } else if (s.is_negative()) {
            return NegInf;
        } else {
            throw DomainError("ceiling is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> truncate(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return Inf;
        } else if (s.is_negative()) {
            return NegInf;
        } else {
            throw DomainError("truncate is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> erf(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return one;
        } else if (s.is_negative()) {
            return minus_one;
        } else {
            throw DomainError("erf is not defined for Complex Infinity");
        }
    }
    RCP<const Basic> erfc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<Infty>(x))
        const Infty &s = down_cast<const Infty &>(x);
        if (s.is_positive()) {
            return zero;
        } else if (s.is_negative()) {
            return integer(2);
        } else {
            throw DomainError("erfc is not defined for Complex Infinity");
        }
    }
};

Evaluate &Infty::get_eval() const
{
    static EvaluateInfty evaluate_infty;
    return evaluate_infty;
}

} // namespace SymEngine
