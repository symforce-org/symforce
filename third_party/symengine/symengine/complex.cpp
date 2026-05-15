#include <symengine/complex.h>
#include <symengine/ntheory.h>

namespace SymEngine
{

bool ComplexBase::is_re_zero() const
{
    return this->real_part()->is_zero();
}

Complex::Complex(rational_class real, rational_class imaginary)
    : real_{real}, imaginary_{imaginary}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(this->real_, this->imaginary_))
}

bool Complex::is_canonical(const rational_class &real,
                           const rational_class &imaginary) const
{
    rational_class re = real;
    rational_class im = imaginary;
    canonicalize(re);
    canonicalize(im);
    // If 'im' is 0, it should not be Complex:
    if (get_num(im) == 0)
        return false;
    // if 'real' or `imaginary` are not in canonical form:
    if (get_num(re) != get_num(real))
        return false;
    if (get_den(re) != get_den(real))
        return false;
    if (get_num(im) != get_num(imaginary))
        return false;
    if (get_den(im) != get_den(imaginary))
        return false;
    return true;
}

hash_t Complex::__hash__() const
{
    // only the least significant bits that fit into "signed long int" are
    // hashed:
    hash_t seed = SYMENGINE_COMPLEX;
    hash_combine<long long int>(seed, mp_get_si(get_num(this->real_)));
    hash_combine<long long int>(seed, mp_get_si(get_den(this->real_)));
    hash_combine<long long int>(seed, mp_get_si(get_num(this->imaginary_)));
    hash_combine<long long int>(seed, mp_get_si(get_den(this->imaginary_)));
    return seed;
}

bool Complex::__eq__(const Basic &o) const
{
    if (is_a<Complex>(o)) {
        const Complex &s = down_cast<const Complex &>(o);
        return ((this->real_ == s.real_)
                and (this->imaginary_ == s.imaginary_));
    }
    return false;
}

int Complex::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Complex>(o))
    const Complex &s = down_cast<const Complex &>(o);
    if (real_ == s.real_) {
        if (imaginary_ == s.imaginary_) {
            return 0;
        } else {
            return imaginary_ < s.imaginary_ ? -1 : 1;
        }
    } else {
        return real_ < s.real_ ? -1 : 1;
    }
}

RCP<const Number> Complex::real_part() const
{
    return Rational::from_mpq(real_);
}

RCP<const Number> Complex::imaginary_part() const
{
    return Rational::from_mpq(imaginary_);
}

RCP<const Basic> Complex::conjugate() const
{
    return Complex::from_mpq(real_, -imaginary_);
}

RCP<const Number> Complex::from_mpq(const rational_class re,
                                    const rational_class im)
{
    // It is assumed that `re` and `im` are already in canonical form.
    if (get_num(im) == 0) {
        return Rational::from_mpq(re);
    } else {
        return make_rcp<const Complex>(re, im);
    }
}

RCP<const Number> Complex::from_two_rats(const Rational &re, const Rational &im)
{
    return Complex::from_mpq(re.as_rational_class(), im.as_rational_class());
}

RCP<const Number> Complex::from_two_nums(const Number &re, const Number &im)
{
    if (is_a<Integer>(re) and is_a<Integer>(im)) {
        rational_class re_mpq(
            down_cast<const Integer &>(re).as_integer_class(),
            down_cast<const Integer &>(*one).as_integer_class());
        rational_class im_mpq(
            down_cast<const Integer &>(im).as_integer_class(),
            down_cast<const Integer &>(*one).as_integer_class());
        return Complex::from_mpq(re_mpq, im_mpq);
    } else if (is_a<Rational>(re) and is_a<Integer>(im)) {
        rational_class re_mpq
            = down_cast<const Rational &>(re).as_rational_class();
        rational_class im_mpq(
            down_cast<const Integer &>(im).as_integer_class(),
            down_cast<const Integer &>(*one).as_integer_class());
        return Complex::from_mpq(re_mpq, im_mpq);
    } else if (is_a<Integer>(re) and is_a<Rational>(im)) {
        rational_class re_mpq(
            down_cast<const Integer &>(re).as_integer_class(),
            down_cast<const Integer &>(*one).as_integer_class());
        rational_class im_mpq
            = down_cast<const Rational &>(im).as_rational_class();
        return Complex::from_mpq(re_mpq, im_mpq);
    } else if (is_a<Rational>(re) and is_a<Rational>(im)) {
        rational_class re_mpq
            = down_cast<const Rational &>(re).as_rational_class();
        rational_class im_mpq
            = down_cast<const Rational &>(im).as_rational_class();
        return Complex::from_mpq(re_mpq, im_mpq);
    } else {
        throw SymEngineException(
            "Invalid Format: Expected Integer or Rational");
    }
}

RCP<const Number> pow_number(const Complex &x, unsigned long n)
{
    unsigned long mask = 1;
    rational_class r_re(1);
    rational_class r_im(0);

    rational_class p_re = x.real_;
    rational_class p_im = x.imaginary_;

    rational_class tmp;

    while (true) {
        if (n & mask) {
            // Multiply r by p
            tmp = r_re * p_re - r_im * p_im;
            r_im = r_re * p_im + r_im * p_re;
            r_re = tmp;
        }
        mask = mask << 1;
        if (not(mask > 0 and n >= mask)) {
            break;
        }
        // Multiply p by p
        tmp = p_re * p_re - p_im * p_im;
        p_im = 2 * p_re * p_im;
        p_re = tmp;
    }
    return Complex::from_mpq(r_re, r_im);
}

RCP<const Number> Complex::powcomp(const Integer &other) const
{
    if (this->is_re_zero()) {
        // Imaginary Number raised to an integer power.
        RCP<const Number> im = Rational::from_mpq(this->imaginary_);
        long rem = mod_f(other, *integer(4))->as_int();
        RCP<const Number> res;
        if (rem == 0) {
            res = one;
        } else if (rem == 1) {
            res = I;
        } else if (rem == 2) {
            res = minus_one;
        } else {
            res = mulnum(I, minus_one);
        }
        return mulnum(im->pow(other), res);
    } else if (other.is_positive()) {
        return pow_number(*this, other.as_int());
    } else {
        return one->div(*pow_number(*this, -1 * other.as_int()));
    }
}
} // namespace SymEngine
