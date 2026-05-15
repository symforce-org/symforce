/**
 *  \file RealDouble.h
 *  Class for RealDouble built on top of Number class
 *
 **/
#include <symengine/complex_double.h>
#include <symengine/eval_double.h>

namespace SymEngine
{

RealDouble::RealDouble(double i)
{
    SYMENGINE_ASSIGN_TYPEID()
    this->i = i;
}

hash_t RealDouble::__hash__() const
{
    hash_t seed = SYMENGINE_REAL_DOUBLE;
    hash_combine<double>(seed, i);
    return seed;
}

bool RealDouble::__eq__(const Basic &o) const
{
    if (is_a<RealDouble>(o)) {
        const RealDouble &s = down_cast<const RealDouble &>(o);
        return this->i == s.i;
    }
    return false;
}

int RealDouble::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<RealDouble>(o))
    const RealDouble &s = down_cast<const RealDouble &>(o);
    if (i == s.i)
        return 0;
    return i < s.i ? -1 : 1;
}

RCP<const RealDouble> real_double(double x)
{
    return make_rcp<const RealDouble>(x);
}

RCP<const Number> number(std::complex<double> x)
{
    return complex_double(x);
}

RCP<const Number> number(double x)
{
    return real_double(x);
}

//! Evaluate functions with double precision
template <class T>
class EvaluateDouble : public Evaluate
{
    RCP<const Basic> sin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::sin(down_cast<const T &>(x).i));
    }
    RCP<const Basic> cos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::cos(down_cast<const T &>(x).i));
    }
    RCP<const Basic> tan(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::tan(down_cast<const T &>(x).i));
    }
    RCP<const Basic> cot(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(1.0 / std::tan(down_cast<const T &>(x).i));
    }
    RCP<const Basic> sec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(1.0 / std::cos(down_cast<const T &>(x).i));
    }
    RCP<const Basic> csc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(1.0 / std::sin(down_cast<const T &>(x).i));
    }
    RCP<const Basic> atan(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::atan(down_cast<const T &>(x).i));
    }
    RCP<const Basic> acot(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::atan(1.0 / down_cast<const T &>(x).i));
    }
    RCP<const Basic> sinh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::sinh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> csch(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(1.0 / std::sinh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> cosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::cosh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> sech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(1.0 / std::cosh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> tanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::tanh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> coth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(1.0 / std::tanh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> asinh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::asinh(down_cast<const T &>(x).i));
    }
    RCP<const Basic> acsch(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::asinh(1.0 / down_cast<const T &>(x).i));
    }
    RCP<const Basic> abs(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::abs(down_cast<const T &>(x).i));
    }
    RCP<const Basic> exp(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<T>(x))
        return number(std::exp(down_cast<const T &>(x).i));
    }
};

class EvaluateRealDouble : public EvaluateDouble<RealDouble>
{
    RCP<const Basic> gamma(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        return number(std::tgamma(down_cast<const RealDouble &>(x).i));
    }
    RCP<const Basic> asin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d <= 1.0 and d >= -1.0) {
            return number(std::asin(d));
        } else {
            return number(std::asin(std::complex<double>(d)));
        }
    }
    RCP<const Basic> acos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d <= 1.0 and d >= -1.0) {
            return number(std::acos(d));
        } else {
            return number(std::acos(std::complex<double>(d)));
        }
    }
    RCP<const Basic> acsc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d >= 1.0 or d <= -1.0) {
            return number(std::asin(1.0 / d));
        } else {
            return number(std::asin(1.0 / std::complex<double>(d)));
        }
    }
    RCP<const Basic> asec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d >= 1.0 or d <= -1.0) {
            return number(std::acos(1.0 / d));
        } else {
            return number(std::acos(1.0 / std::complex<double>(d)));
        }
    }
    RCP<const Basic> acosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d >= 1.0) {
            return number(std::acosh(d));
        } else {
            return number(std::acosh(std::complex<double>(d)));
        }
    }
    RCP<const Basic> atanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d <= 1.0 and d >= -1.0) {
            return number(std::atanh(d));
        } else {
            return number(std::atanh(std::complex<double>(d)));
        }
    }
    RCP<const Basic> acoth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d >= 1.0 or d <= -1.0) {
            return number(std::atanh(1.0 / d));
        } else {
            return number(std::atanh(1.0 / std::complex<double>(d)));
        }
    }
    RCP<const Basic> asech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d <= 1.0 and d >= 0.0) {
            return number(std::acosh(1.0 / d));
        } else {
            return number(std::acosh(1.0 / std::complex<double>(d)));
        }
    }
    RCP<const Basic> log(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        double d = down_cast<const RealDouble &>(x).i;
        if (d >= 0.0) {
            return number(std::log(d));
        } else {
            return number(std::log(std::complex<double>(d)));
        }
    }
    RCP<const Basic> floor(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        integer_class i;
        mp_set_d(i, std::floor(down_cast<const RealDouble &>(x).i));
        return integer(std::move(i));
    }
    RCP<const Basic> ceiling(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        integer_class i;
        mp_set_d(i, std::ceil(down_cast<const RealDouble &>(x).i));
        return integer(std::move(i));
    }
    RCP<const Basic> truncate(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        integer_class i;
        mp_set_d(i, std::trunc(down_cast<const RealDouble &>(x).i));
        return integer(std::move(i));
    }
    RCP<const Basic> erf(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        return number(std::erf(down_cast<const RealDouble &>(x).i));
    }
    RCP<const Basic> erfc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<RealDouble>(x))
        return number(std::erfc(down_cast<const RealDouble &>(x).i));
    }
};

class EvaluateComplexDouble : public EvaluateDouble<ComplexDouble>
{
    RCP<const Basic> gamma(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        throw NotImplementedError("Not Implemented.");
    }
    RCP<const Basic> asin(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::asin(down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> acos(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::acos(down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> acsc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::asin(1.0 / down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> asec(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::acos(1.0 / down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> acosh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::acosh(down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> atanh(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::atanh(down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> acoth(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::atanh(1.0 / down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> asech(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::acosh(1.0 / down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> log(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        return number(std::log(down_cast<const ComplexDouble &>(x).i));
    }
    RCP<const Basic> floor(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        integer_class re, im;
        mp_set_d(re, std::floor(down_cast<const ComplexDouble &>(x).i.real()));
        mp_set_d(im, std::floor(down_cast<const ComplexDouble &>(x).i.imag()));
        return Complex::from_two_nums(*integer(std::move(re)),
                                      *integer(std::move(im)));
    }
    RCP<const Basic> ceiling(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        integer_class re, im;
        mp_set_d(re, std::ceil(down_cast<const ComplexDouble &>(x).i.real()));
        mp_set_d(im, std::ceil(down_cast<const ComplexDouble &>(x).i.imag()));
        return Complex::from_two_nums(*integer(std::move(re)),
                                      *integer(std::move(im)));
    }
    RCP<const Basic> truncate(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        integer_class re, im;
        mp_set_d(re, std::trunc(down_cast<const ComplexDouble &>(x).i.real()));
        mp_set_d(im, std::trunc(down_cast<const ComplexDouble &>(x).i.imag()));
        return Complex::from_two_nums(*integer(std::move(re)),
                                      *integer(std::move(im)));
    }
    RCP<const Basic> erf(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        throw NotImplementedError("erf is not implemented for Complex numbers");
    }
    RCP<const Basic> erfc(const Basic &x) const override
    {
        SYMENGINE_ASSERT(is_a<ComplexDouble>(x))
        throw NotImplementedError(
            "erfc is not implemented for Complex numbers");
    }
};

Evaluate &RealDouble::get_eval() const
{
    static EvaluateRealDouble evaluate_real_double;
    return evaluate_real_double;
}

Evaluate &ComplexDouble::get_eval() const
{
    static EvaluateComplexDouble evaluate_complex_double;
    return evaluate_complex_double;
}

} // namespace SymEngine
