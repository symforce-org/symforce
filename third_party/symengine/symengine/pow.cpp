#include <symengine/pow.h>
#include <symengine/add.h>
#include <symengine/complex.h>
#include <symengine/symengine_exception.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

Pow::Pow(const RCP<const Basic> &base, const RCP<const Basic> &exp)
    : base_{base}, exp_{exp}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(*base, *exp))
}

bool Pow::is_canonical(const Basic &base, const Basic &exp) const
{
    // e.g. 0**x
    if (is_a<Integer>(base) and down_cast<const Integer &>(base).is_zero()) {
        if (is_a_Number(exp)) {
            return false;
        } else {
            return true;
        }
    }
    // e.g. 1**x
    if (is_a<Integer>(base) and down_cast<const Integer &>(base).is_one())
        return false;
    // e.g. x**0.0
    if (is_number_and_zero(exp))
        return false;
    // e.g. x**1
    if (is_a<Integer>(exp) and down_cast<const Integer &>(exp).is_one())
        return false;
    // e.g. 2**3, (2/3)**4
    if ((is_a<Integer>(base) or is_a<Rational>(base)) and is_a<Integer>(exp))
        return false;
    // e.g. (x*y)**2, should rather be x**2*y**2
    if (is_a<Mul>(base) and is_a<Integer>(exp))
        return false;
    // e.g. (x**y)**2, should rather be x**(2*y)
    if (is_a<Pow>(base) and is_a<Integer>(exp))
        return false;
    // If exp is a rational, it should be between 0  and 1, i.e. we don't
    // allow things like 2**(-1/2) or 2**(3/2)
    if ((is_a<Rational>(base) or is_a<Integer>(base)) and is_a<Rational>(exp)
        and (down_cast<const Rational &>(exp).as_rational_class() < 0
             or down_cast<const Rational &>(exp).as_rational_class() > 1))
        return false;
    // Purely Imaginary complex numbers with integral powers are expanded
    // e.g (2I)**3
    if (is_a<Complex>(base) and down_cast<const Complex &>(base).is_re_zero()
        and is_a<Integer>(exp))
        return false;
    // e.g. 0.5^2.0 should be represented as 0.25
    if (is_a_Number(base) and not down_cast<const Number &>(base).is_exact()
        and is_a_Number(exp) and not down_cast<const Number &>(exp).is_exact())
        return false;
    return true;
}

hash_t Pow::__hash__() const
{
    hash_t seed = SYMENGINE_POW;
    hash_combine<Basic>(seed, *base_);
    hash_combine<Basic>(seed, *exp_);
    return seed;
}

bool Pow::__eq__(const Basic &o) const
{
    if (is_a<Pow>(o) and eq(*base_, *(down_cast<const Pow &>(o).base_))
        and eq(*exp_, *(down_cast<const Pow &>(o).exp_)))
        return true;

    return false;
}

int Pow::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Pow>(o))
    const Pow &s = down_cast<const Pow &>(o);
    int base_cmp = base_->__cmp__(*s.base_);
    if (base_cmp == 0)
        return exp_->__cmp__(*s.exp_);
    else
        return base_cmp;
}

RCP<const Basic> pow(const RCP<const Basic> &a, const RCP<const Basic> &b)
{
    if (is_number_and_zero(*b)) {
        // addnum is used for converting to the type of `b`.
        return addnum(one, rcp_static_cast<const Number>(b));
    }
    if (eq(*b, *one))
        return a;

    if (eq(*a, *zero)) {
        if (is_a_Number(*b)
            and rcp_static_cast<const Number>(b)->is_positive()) {
            return zero;
        } else if (is_a_Number(*b)
                   and rcp_static_cast<const Number>(b)->is_negative()) {
            return ComplexInf;
        } else {
            return make_rcp<const Pow>(a, b);
        }
    }

    if (eq(*a, *one) and not is_a_Number(*b))
        return one;
    if (eq(*a, *minus_one)) {
        if (is_a<Integer>(*b)) {
            return is_a<Integer>(*div(b, integer(2))) ? one : minus_one;
        } else if (is_a<Rational>(*b) and eq(*b, *rational(1, 2))) {
            return I;
        }
    }

    if (is_a_Number(*b)) {
        if (is_a_Number(*a)) {
            if (is_a<Integer>(*b)) {
                return down_cast<const Number &>(*a).pow(
                    *rcp_static_cast<const Number>(b));
            } else if (is_a<Rational>(*b)) {
                if (is_a<Rational>(*a)) {
                    return down_cast<const Rational &>(*a).powrat(
                        down_cast<const Rational &>(*b));
                } else if (is_a<Integer>(*a)) {
                    return down_cast<const Rational &>(*b).rpowrat(
                        down_cast<const Integer &>(*a));
                } else if (is_a<Complex>(*a)) {
                    return make_rcp<const Pow>(a, b);
                } else {
                    return down_cast<const Number &>(*a).pow(
                        *rcp_static_cast<const Number>(b));
                }
            } else if (is_a<Complex>(*b)
                       and down_cast<const Number &>(*a).is_exact()) {
                return make_rcp<const Pow>(a, b);
            } else {
                return down_cast<const Number &>(*a).pow(
                    *rcp_static_cast<const Number>(b));
            }
        } else if (eq(*a, *E)) {
            RCP<const Number> p = rcp_static_cast<const Number>(b);
            if (not p->is_exact()) {
                // Evaluate E**0.2, but not E**2
                return p->get_eval().exp(*p);
            }
        } else if (is_a<Mul>(*a)) {
            // Expand (x*y)**b = x**b*y**b
            map_basic_basic d;
            RCP<const Number> coef = one;
            down_cast<const Mul &>(*a).power_num(
                outArg(coef), d, rcp_static_cast<const Number>(b));
            return Mul::from_dict(coef, std::move(d));
        }
    }
    if (is_a<Pow>(*a) and is_a<Integer>(*b)) {
        // Convert (x**y)**b = x**(b*y), where 'b' is an integer. This holds for
        // any complex 'x', 'y' and integer 'b'.
        RCP<const Pow> A = rcp_static_cast<const Pow>(a);
        return pow(A->get_base(), mul(A->get_exp(), b));
    }
    if (is_a<Pow>(*a)
        and eq(*down_cast<const Pow &>(*a).get_exp(), *minus_one)) {
        // Convert (x**-1)**b = x**(-b)
        RCP<const Pow> A = rcp_static_cast<const Pow>(a);
        return pow(A->get_base(), neg(b));
    }
    return make_rcp<const Pow>(a, b);
}

// This function can overflow, but it is fast.
// TODO: figure out condition for (m, n) when it overflows and raise an
// exception.
void multinomial_coefficients(unsigned m, unsigned n, map_vec_uint &r)
{
    vec_uint t;
    unsigned j, tj, start, k;
    unsigned long long int v;
    if (m < 2)
        throw SymEngineException("multinomial_coefficients: m >= 2 must hold.");
    t.assign(m, 0);
    t[0] = n;
    r[t] = 1;
    if (n == 0)
        return;
    j = 0;
    while (j < m - 1) {
        tj = t[j];
        if (j) {
            t[j] = 0;
            t[0] = tj;
        }
        if (tj > 1) {
            t[j + 1] += 1;
            j = 0;
            start = 1;
            v = 0;
        } else {
            j += 1;
            start = j + 1;
            v = r[t];
            t[j] += 1;
        }
        for (k = start; k < m; k++) {
            if (t[k]) {
                t[k] -= 1;
                v += r[t];
                t[k] += 1;
            }
        }
        t[0] -= 1;
        r[t] = (v * tj) / (n - t[0]);
    }
}

// Slower, but returns exact (possibly large) integers (as mpz)
void multinomial_coefficients_mpz(unsigned m, unsigned n, map_vec_mpz &r)
{
    vec_uint t;
    unsigned j, tj, start, k;
    integer_class v;
    if (m < 2)
        throw SymEngineException("multinomial_coefficients: m >= 2 must hold.");
    t.assign(m, 0);
    t[0] = n;
    r[t] = 1;
    if (n == 0)
        return;
    j = 0;
    while (j < m - 1) {
        tj = t[j];
        if (j) {
            t[j] = 0;
            t[0] = tj;
        }
        if (tj > 1) {
            t[j + 1] += 1;
            j = 0;
            start = 1;
            v = 0;
        } else {
            j += 1;
            start = j + 1;
            v = r[t];
            t[j] += 1;
        }
        for (k = start; k < m; k++) {
            if (t[k]) {
                t[k] -= 1;
                v += r[t];
                t[k] += 1;
            }
        }
        t[0] -= 1;
        r[t] = (v * tj) / (n - t[0]);
    }
}

vec_basic Pow::get_args() const
{
    return {base_, exp_};
}

RCP<const Basic> exp(const RCP<const Basic> &x)
{
    return pow(E, x);
}

} // SymEngine
