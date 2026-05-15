#include <symengine/ntheory.h>
#include <symengine/ntheory_funcs.h>
#include <symengine/prime_sieve.h>

namespace SymEngine
{

PrimePi::PrimePi(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool PrimePi::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg) or is_a<Constant>(*arg)) {
        return false;
    } else {
        return true;
    }
}

RCP<const Basic> PrimePi::create(const RCP<const Basic> &arg) const
{
    return primepi(arg);
}

RCP<const Basic> primepi(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        if (is_a<NaN>(*arg)) {
            return arg;
        } else if (is_a<Infty>(*arg)) {
            if (down_cast<const Infty &>(*arg).is_negative_infinity()) {
                return make_rcp<const Integer>(integer_class(0));
            } else {
                return arg;
            }
        } else if (down_cast<const Number &>(*arg).is_complex()) {
            throw SymEngineException("Complex can't be passed to primepi!");
        } else if (down_cast<const Number &>(*arg).is_negative()) {
            return make_rcp<const Integer>(integer_class(0));
        }
    }
    if (is_a_Number(*arg) or is_a<Constant>(*arg)) {
        unsigned int num
            = (unsigned int)down_cast<const Integer &>(*SymEngine::floor(arg))
                  .as_uint();
        Sieve::iterator pi(num);
        unsigned long int p = 0;
        while ((pi.next_prime()) <= num) {
            p++;
        }
        return make_rcp<const Integer>(integer_class(p));
    }
    return make_rcp<const PrimePi>(arg);
}

Primorial::Primorial(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Primorial::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg) or is_a<Constant>(*arg)) {
        return false;
    } else {
        return true;
    }
}

RCP<const Basic> Primorial::create(const RCP<const Basic> &arg) const
{
    return primorial(arg);
}

RCP<const Basic> primorial(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        if (is_a<NaN>(*arg)) {
            return arg;
        }
        if (down_cast<const Number &>(*arg).is_positive()) {
            if (is_a<Infty>(*arg)) {
                return arg;
            }
        } else {
            throw SymEngineException(
                "Only positive numbers are allowed for primorial!");
        }
    }
    if (is_a_Number(*arg) or is_a<Constant>(*arg)) {
        unsigned long n
            = down_cast<const Integer &>(*SymEngine::floor(arg)).as_uint();
        return make_rcp<const Integer>(mp_primorial(n));
    }
    return make_rcp<const Primorial>(arg);
}

/**
 * @brief The n:th s-gonal number
 * @param s Number of sides of the polygon. Must be greater than 2
 * @param n Must be greater than 0
 * @returns The n:th s-gonal number
 *
 * Symbolic calculation of the n:th s-gonal number.
 */
RCP<const Basic> polygonal_number(const RCP<const Basic> &s,
                                  const RCP<const Basic> &n)
{
    if (is_a_Number(*s)) {
        if (not is_a<Integer>(*s)
            or not down_cast<const Integer &>(*sub(s, integer(2)))
                       .is_positive()) {
            throw DomainError("The number of sides of the polygon must be an "
                              "integer greater than 2");
        }
    }
    if (is_a_Number(*n)) {
        if (not is_a<Integer>(*n)
            or not down_cast<const Integer &>(*n).is_positive()) {
            throw DomainError("n must be an integer greater than 0");
        }
    }

    if (is_a_Number(*s) and is_a_Number(*n)) {
        // Optimized numeric calculation
        auto s_int = down_cast<const Integer &>(*s).as_integer_class();
        auto n_int = down_cast<const Integer &>(*n).as_integer_class();
        auto res = mp_polygonal_number(s_int, n_int);
        return make_rcp<const Integer>(res);
    }

    RCP<const Integer> m1 = integer(-1);
    RCP<const Integer> m2 = integer(-2);
    RCP<const Integer> p2 = integer(2);
    RCP<const Integer> p4 = integer(4);
    RCP<const Basic> x = div(
        add(mul(add(s, m2), pow(n, p2)), mul(add(p4, mul(m1, s)), n)), p2);
    return x;
}

/**
 * @brief The principal s-gonal root of x
 * @param s Number of sides of the polygon. Must be greater than 2
 * @param n Must be greater than 0
 * @returns The n:th s-gonal number
 *
 * Symbolic calculation of the principal (i.e. positive) s-gonal root of x.
 */
RCP<const Basic> principal_polygonal_root(const RCP<const Basic> &s,
                                          const RCP<const Basic> &x)
{
    if (is_a_Number(*s)) {
        if (not is_a<Integer>(*s)
            or not down_cast<const Integer &>(*sub(s, integer(2)))
                       .is_positive()) {
            throw DomainError("The number of sides of the polygon must be an "
                              "integer greater than 2");
        }
    }
    if (is_a_Number(*x)) {
        if (not is_a<Integer>(*x)
            or not down_cast<const Integer &>(*x).is_positive()) {
            throw DomainError("x must be an integer greater than 0");
        }
    }

    if (is_a_Number(*s) and is_a_Number(*x)) {
        // Optimized numeric calculation
        auto s_int = down_cast<const Integer &>(*s).as_integer_class();
        auto x_int = down_cast<const Integer &>(*x).as_integer_class();
        auto res = mp_principal_polygonal_root(s_int, x_int);
        return make_rcp<const Integer>(res);
    }

    RCP<const Integer> m2 = integer(-2);
    RCP<const Integer> m4 = integer(-4);
    RCP<const Integer> p2 = integer(2);
    RCP<const Integer> p8 = integer(8);
    RCP<const Basic> root
        = sqrt(add(mul(mul(p8, add(s, m2)), x), pow(add(s, m4), p2)));
    RCP<const Basic> n = div(add(root, add(s, m4)), mul(p2, add(s, m2)));
    return n;
}

} // namespace SymEngine
