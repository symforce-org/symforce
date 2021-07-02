#include <symengine/rational.h>
#include <symengine/pow.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{

bool Rational::is_canonical(const rational_class &i) const
{
    rational_class x = i;
    canonicalize(x);
    // If 'x' is an integer, it should not be Rational:
    if (SymEngine::get_den(x) == 1)
        return false;
    // if 'i' is not in canonical form:
    if (SymEngine::get_num(x) != SymEngine::get_num(i))
        return false;
    if (SymEngine::get_den(x) != SymEngine::get_den(i))
        return false;
    return true;
}

RCP<const Number> Rational::from_mpq(const rational_class &i)
{
    // If the result is an Integer, return an Integer:
    if (SymEngine::get_den(i) == 1) {
        return integer(SymEngine::get_num(i));
    } else {
        rational_class j(i);
        return make_rcp<const Rational>(std::move(j));
    }
}

RCP<const Number> Rational::from_mpq(rational_class &&i)
{
    // If the result is an Integer, return an Integer:
    if (SymEngine::get_den(i) == 1) {
        return integer(SymEngine::get_num(i));
    } else {
        return make_rcp<const Rational>(std::move(i));
    }
}

RCP<const Number> Rational::from_two_ints(const Integer &n, const Integer &d)
{
    if (d.as_integer_class() == 0) {
        if (n.as_integer_class() == 0) {
            return Nan;
        } else {
            return ComplexInf;
        }
    }
    rational_class q(n.as_integer_class(), d.as_integer_class());

    // This is potentially slow, but has to be done, since 'n/d' might not be
    // in canonical form.
    canonicalize(q);

    return Rational::from_mpq(std::move(q));
}

RCP<const Number> Rational::from_two_ints(long n, long d)
{
    if (d == 0) {
        if (n == 0) {
            return Nan;
        } else {
            return ComplexInf;
        }
    }
    rational_class q(n, d);

    // This is potentially slow, but has to be done, since 'n/d' might not be
    // in canonical form.
    canonicalize(q);

    return Rational::from_mpq(q);
}

hash_t Rational::__hash__() const
{
    // only the least significant bits that fit into "signed long int" are
    // hashed:
    hash_t seed = SYMENGINE_RATIONAL;
    hash_combine<long long int>(seed, mp_get_si(SymEngine::get_num(this->i)));
    hash_combine<long long int>(seed, mp_get_si(SymEngine::get_den(this->i)));
    return seed;
}

bool Rational::__eq__(const Basic &o) const
{
    if (is_a<Rational>(o)) {
        const Rational &s = down_cast<const Rational &>(o);
        return this->i == s.i;
    }
    return false;
}

int Rational::compare(const Basic &o) const
{
    if (is_a<Rational>(o)) {
        const Rational &s = down_cast<const Rational &>(o);
        if (i == s.i)
            return 0;
        return i < s.i ? -1 : 1;
    }
    if (is_a<Integer>(o)) {
        const Integer &s = down_cast<const Integer &>(o);
        return i < s.as_integer_class() ? -1 : 1;
    }
    throw NotImplementedError("unhandled comparison of Rational");
}

void get_num_den(const Rational &rat, const Ptr<RCP<const Integer>> &num,
                 const Ptr<RCP<const Integer>> &den)
{
    *num = integer(SymEngine::get_num(rat.as_rational_class()));
    *den = integer(SymEngine::get_den(rat.as_rational_class()));
}

bool Rational::is_perfect_power(bool is_expected) const
{
    const integer_class &num = SymEngine::get_num(i);
    if (num == 1)
        return mp_perfect_power_p(SymEngine::get_den(i));

    const integer_class &den = SymEngine::get_den(i);
    // TODO: fix this
    if (not is_expected) {
        if (mp_cmpabs(num, den) > 0) {
            if (!mp_perfect_power_p(den))
                return false;
        } else {
            if (!mp_perfect_power_p(num))
                return false;
        }
    }
    integer_class prod = num * den;
    return mp_perfect_power_p(prod);
}

bool Rational::nth_root(const Ptr<RCP<const Number>> &the_rat,
                        unsigned long n) const
{
    if (n == 0)
        throw SymEngineException("i_nth_root: Can not find Zeroth root");

#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
    rational_class r;
    int ret = mp_root(SymEngine::get_num(r), SymEngine::get_num(i), n);
    if (ret == 0)
        return false;
    ret = mp_root(SymEngine::get_den(r), SymEngine::get_den(i), n);
    if (ret == 0)
        return false;
#else
    // boost::multiprecision::cpp_rational doesn't provide
    // non-const get_num and get_den
    integer_class num, den;
    int ret = mp_root(num, SymEngine::get_num(i), n);
    if (ret == 0)
        return false;
    ret = mp_root(den, SymEngine::get_den(i), n);
    if (ret == 0)
        return false;
    rational_class r(num, den);
#endif
    // No need to canonicalize since `this` is in canonical form
    *the_rat = make_rcp<const Rational>(std::move(r));
    return true;
}

RCP<const Basic> Rational::powrat(const Rational &other) const
{
    return SymEngine::mul(other.rpowrat(*this->get_num()),
                          other.neg()->rpowrat(*this->get_den()));
}

RCP<const Basic> Rational::rpowrat(const Integer &other) const
{
    if (not(mp_fits_ulong_p(SymEngine::get_den(i))))
        throw SymEngineException("powrat: den of 'exp' does not fit ulong.");
    unsigned long exp = mp_get_ui(SymEngine::get_den(i));
    RCP<const Integer> res;
    if (other.is_negative()) {
        if (i_nth_root(outArg(res), *other.neg(), exp)) {
            if (exp % 2 == 0) {
                return I->pow(*get_num())->mul(*res->powint(*get_num()));
            } else {
                return SymEngine::neg(res->powint(*get_num()));
            }
        }
    } else {
        if (i_nth_root(outArg(res), other, exp)) {
            return res->powint(*get_num());
        }
    }
    integer_class q, r;
    auto num = SymEngine::get_num(i);
    auto den = SymEngine::get_den(i);

    mp_fdiv_qr(q, r, num, den);
    // Here we make the exponent postive and a fraction between
    // 0 and 1. We multiply numerator and denominator appropriately
    // to achieve this
    RCP<const Number> coef = other.powint(*integer(q));
    map_basic_basic surd;

    if ((other.is_negative()) and den == 2) {
        imulnum(outArg(coef), I);
        // if other.neg() is one, no need to add it to dict
        if (other.as_integer_class() != -1)
            insert(surd, other.neg(),
                   Rational::from_mpq(rational_class(r, den)));
    } else {
        insert(surd, other.rcp_from_this(),
               Rational::from_mpq(rational_class(r, den)));
    }
    return Mul::from_dict(coef, std::move(surd));
}

} // SymEngine
