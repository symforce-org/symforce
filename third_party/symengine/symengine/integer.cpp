#include <symengine/rational.h>
#include <symengine/pow.h>
#include <symengine/symengine_exception.h>
#include <symengine/symengine_casts.h>

namespace SymEngine
{

hash_t Integer::__hash__() const
{
    // only the least significant bits that fit into "long long int" are
    // hashed:
    return ((hash_t)mp_get_ui(this->i)) * (hash_t)(mp_sign(this->i));
}

bool Integer::__eq__(const Basic &o) const
{
    if (is_a<Integer>(o)) {
        const Integer &s = down_cast<const Integer &>(o);
        return this->i == s.i;
    }
    return false;
}

int Integer::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Integer>(o))
    const Integer &s = down_cast<const Integer &>(o);
    if (i == s.i)
        return 0;
    return i < s.i ? -1 : 1;
}

signed long int Integer::as_int() const
{
    // mp_get_si() returns "signed long int", so that's what we return from
    // "as_int()" and we leave it to the user to do any possible further integer
    // conversions.
    if (not(mp_fits_slong_p(this->i))) {
        throw SymEngineException("as_int: Integer larger than int");
    }
    return mp_get_si(this->i);
}

unsigned long int Integer::as_uint() const
{
    // mp_get_ui() returns "unsigned long int", so that's what we return from
    // "as_uint()" and we leave it to the user to do any possible further
    // integer
    // conversions.
    if (this->i < 0u) {
        throw SymEngineException("as_uint: negative Integer");
    }
    if (not(mp_fits_ulong_p(this->i))) {
        throw SymEngineException("as_uint: Integer larger than uint");
    }
    return mp_get_ui(this->i);
}

RCP<const Number> Integer::divint(const Integer &other) const
{
    if (other.i == 0) {
        if (this->i == 0) {
            return Nan;
        } else {
            return ComplexInf;
        }
    }
    rational_class q(this->i, other.i);

    // This is potentially slow, but has to be done, since q might not
    // be in canonical form.
    canonicalize(q);

    return Rational::from_mpq(std::move(q));
}

RCP<const Number> Integer::rdiv(const Number &other) const
{
    if (is_a<Integer>(other)) {
        if (this->i == 0) {
            if (other.is_zero()) {
                return Nan;
            } else {
                return ComplexInf;
            }
        }
        rational_class q((down_cast<const Integer &>(other)).i, this->i);

        // This is potentially slow, but has to be done, since q might not
        // be in canonical form.
        canonicalize(q);

        return Rational::from_mpq(std::move(q));
    } else {
        throw NotImplementedError("Not Implemented");
    }
}

RCP<const Number> Integer::pow_negint(const Integer &other) const
{
    RCP<const Number> tmp = powint(*other.neg());
    if (is_a<Integer>(*tmp)) {
        const integer_class &j = down_cast<const Integer &>(*tmp).i;
#if SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP
        // boost::multiprecision::cpp_rational lacks an (int, cpp_int)
        // constructor. must use cpp_rational(cpp_int,cpp_int)
        rational_class q(integer_class(mp_sign(j)), mp_abs(j));
#else
        rational_class q(mp_sign(j), mp_abs(j));
#endif
        return Rational::from_mpq(std::move(q));
    } else {
        throw SymEngineException("powint returned non-integer");
    }
}

RCP<const Integer> isqrt(const Integer &n)
{
    return integer(mp_sqrt(n.as_integer_class()));
}

RCP<const Integer> iabs(const Integer &n)
{
    return integer(mp_abs(n.as_integer_class()));
}

int i_nth_root(const Ptr<RCP<const Integer>> &r, const Integer &a,
               unsigned long int n)
{
    if (n == 0)
        throw SymEngineException("i_nth_root: Can not find Zeroth root");

    int ret_val;
    integer_class t;

    ret_val = mp_root(t, a.as_integer_class(), n);
    *r = integer(std::move(t));

    return ret_val;
}

bool perfect_square(const Integer &n)
{
    return mp_perfect_square_p(n.as_integer_class());
}

bool perfect_power(const Integer &n)
{
    return mp_perfect_power_p(n.as_integer_class());
}

} // namespace SymEngine
