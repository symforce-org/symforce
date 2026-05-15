#include "catch.hpp"

#include <symengine/as_real_imag.cpp>
#include <symengine/symengine_casts.h>

using SymEngine::Abs;
using SymEngine::add;
using SymEngine::asin;
using SymEngine::asinh;
using SymEngine::Basic;
using SymEngine::Complex;
using SymEngine::ComplexInf;
using SymEngine::cos;
using SymEngine::cosh;
using SymEngine::cot;
using SymEngine::coth;
using SymEngine::csc;
using SymEngine::csch;
using SymEngine::I;
using SymEngine::Inf;
using SymEngine::integer;
using SymEngine::minus_one;
using SymEngine::mul;
using SymEngine::Nan;
using SymEngine::neg;
using SymEngine::one;
using SymEngine::rational;
using SymEngine::Rational;
using SymEngine::RCP;
using SymEngine::sec;
using SymEngine::sech;
using SymEngine::sin;
using SymEngine::sinh;
using SymEngine::sqrt;
using SymEngine::sub;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::tan;
using SymEngine::tanh;
using SymEngine::zero;

TEST_CASE("RealImag: Number and Symbol", "[as_real_imag]")
{
    RCP<const Basic> re, im;
    auto i2 = integer(2), i3 = integer(3);
    auto r1 = Rational::from_two_ints(*i2, *i3);
    auto r2 = Rational::from_two_ints(*i3, *i2);

    as_real_imag(Complex::from_two_nums(*r1, *r2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *r1));
    REQUIRE(eq(*im, *r2));

    as_real_imag(r1, outArg(re), outArg(im));
    REQUIRE(eq(*re, *r1));
    REQUIRE(eq(*im, *zero));

    as_real_imag(neg(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *neg(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(Inf, outArg(re), outArg(im));
    REQUIRE(eq(*re, *Inf));
    REQUIRE(eq(*im, *zero));

    as_real_imag(ComplexInf, outArg(re), outArg(im));
    REQUIRE(eq(*re, *Nan));
    REQUIRE(eq(*im, *Nan));

    // Symbol
    CHECK_THROWS_AS(
        as_real_imag(mul(add(i2, I), symbol("x")), outArg(re), outArg(im)),
        SymEngineException);
}

TEST_CASE("RealImag: Mul", "[as_real_imag]")
{
    RCP<const Basic> re, im;
    auto i2 = integer(2), i3 = integer(3);

    as_real_imag(neg(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *neg(i2)));
    REQUIRE(eq(*im, *neg(one)));

    as_real_imag(mul(i2, pow(I, rational(2, 3))), outArg(re), outArg(im));
    REQUIRE(eq(*re, *one));
    REQUIRE(eq(*im, *sqrt(i3)));

    as_real_imag(mul(add(i2, I), pow(I, rational(2, 3))), outArg(re),
                 outArg(im));
    REQUIRE(eq(*re, *sub(one, div(sqrt(i3), i2))));
    REQUIRE(eq(*im, *add(sqrt(i3), div(one, i2))));

    CHECK_THROWS_AS(as_real_imag(mul(add(i2, I), add(i2, mul(symbol("x"), I))),
                                 outArg(re), outArg(im)),
                    SymEngineException);
}

TEST_CASE("RealImag: Add", "[as_real_imag]")
{
    RCP<const Basic> re, im;
    auto i2 = integer(2), i3 = integer(3);

    as_real_imag(add(add(i2, I), pow(I, rational(2, 3))), outArg(re),
                 outArg(im));
    REQUIRE(eq(*re, *rational(5, 2)));
    REQUIRE(eq(*im, *add(one, div(sqrt(i3), i2))));

    as_real_imag(add(add(i2, I), sqrt(I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *add(i2, div(sqrt(i2), i2))));
    REQUIRE(eq(*im, *add(one, div(sqrt(i2), i2))));
}

TEST_CASE("RealImag: Pow", "[as_real_imag]")
{
    auto sq = sqrt(neg(I));
    RCP<const Basic> re, im;
    auto i2 = integer(2);

    as_real_imag(pow(I, rational(2, 3)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(one, i2)));
    REQUIRE(eq(*im, *div(sqrt(integer(3)), i2)));

    as_real_imag(pow(sub(add(i2, sqrt(i2)), sub(sqrt(i2), I)), 2), outArg(re),
                 outArg(im));
    REQUIRE(eq(*re, *integer(3)));
    REQUIRE(eq(*im, *integer(4)));

    as_real_imag(sqrt(neg(I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(sqrt(i2), i2)));
    REQUIRE(eq(*im, *neg(div(sqrt(i2), i2))));

    as_real_imag(neg(sqrt(neg(I))), outArg(re), outArg(im));
    REQUIRE(eq(*re, *neg(div(sqrt(i2), i2))));
    REQUIRE(eq(*im, *div(sqrt(i2), i2)));

    CHECK_THROWS_AS(as_real_imag(pow(I, symbol("x")), outArg(re), outArg(im)),
                    SymEngineException);
}

TEST_CASE("RealImag: Trigonometric functions", "[as_real_imag]")
{
    auto sq = sqrt(neg(I));
    RCP<const Basic> re, im;
    auto i2 = integer(2);

    as_real_imag(sin(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *sin(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(sin(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *mul(sin(i2), cosh(one))));
    REQUIRE(eq(*im, *mul(cos(i2), sinh(one))));

    as_real_imag(cos(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *cos(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(cos(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *mul(cos(i2), cosh(one))));
    REQUIRE(eq(*im, *mul({minus_one, sin(i2), sinh(one)})));

    as_real_imag(tan(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *tan(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(tan(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(sin(integer(4)), add(cosh(i2), cos(integer(4))))));
    REQUIRE(eq(*im, *div(sinh(i2), add(cosh(i2), cos(integer(4))))));

    as_real_imag(csc(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(one, sin(i2))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(csc(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(mul(sin(i2), cosh(one)),
                         add(mul(pow(cos(i2), i2), pow(sinh(one), i2)),
                             mul(pow(sin(i2), i2), pow(cosh(one), i2))))));
    REQUIRE(eq(*im, *div(mul({minus_one, cos(i2), sinh(one)}),
                         add(mul(pow(cos(i2), i2), pow(sinh(one), i2)),
                             mul(pow(sin(i2), i2), pow(cosh(one), i2))))));

    as_real_imag(sec(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(one, cos(i2))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(sec(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(mul(cos(i2), cosh(one)),
                         add(mul(pow(cos(i2), i2), pow(cosh(one), i2)),
                             mul(pow(sin(i2), i2), pow(sinh(one), i2))))));
    REQUIRE(eq(*im, *div(mul(sin(i2), sinh(one)),
                         add(mul(pow(cos(i2), i2), pow(cosh(one), i2)),
                             mul(pow(sin(i2), i2), pow(sinh(one), i2))))));

    as_real_imag(cot(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *cot(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(cot(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(
        eq(*re, *div(neg(sin(integer(4))), sub(cos(integer(4)), cosh(i2)))));
    REQUIRE(eq(*im, *div(neg(sinh(i2)), sub(cos(integer(4)), cosh(i2)))));

    as_real_imag(sinh(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *sinh(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(sinh(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *mul(sinh(i2), cos(one))));
    REQUIRE(eq(*im, *mul(cosh(i2), sin(one))));

    as_real_imag(cosh(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *cosh(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(cosh(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *mul(cosh(i2), cos(one))));
    REQUIRE(eq(*im, *mul(sinh(i2), sin(one))));

    as_real_imag(tanh(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *tanh(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(tanh(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(mul(sinh(i2), cosh(i2)),
                         add(pow(cos(one), i2), pow(sinh(i2), i2)))));
    REQUIRE(eq(*im, *div(mul(sin(one), cos(one)),
                         add(pow(cos(one), i2), pow(sinh(i2), i2)))));

    as_real_imag(csch(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(one, sinh(i2))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(csch(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(mul(sinh(i2), cos(one)),
                         add(mul(pow(cosh(i2), i2), pow(sin(one), i2)),
                             mul(pow(sinh(i2), i2), pow(cos(one), i2))))));
    REQUIRE(eq(*im, *div(mul({minus_one, cosh(i2), sin(one)}),
                         add(mul(pow(cosh(i2), i2), pow(sin(one), i2)),
                             mul(pow(sinh(i2), i2), pow(cos(one), i2))))));

    as_real_imag(sech(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(one, cosh(i2))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(sech(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(mul(cosh(i2), cos(one)),
                         add(mul(pow(cosh(i2), i2), pow(cos(one), i2)),
                             mul(pow(sinh(i2), i2), pow(sin(one), i2))))));
    REQUIRE(eq(*im, *div(mul({minus_one, sin(one), sinh(i2)}),
                         add(mul(pow(cosh(i2), i2), pow(cos(one), i2)),
                             mul(pow(sinh(i2), i2), pow(sin(one), i2))))));

    as_real_imag(coth(i2), outArg(re), outArg(im));
    REQUIRE(eq(*re, *coth(i2)));
    REQUIRE(eq(*im, *zero));

    as_real_imag(coth(add(i2, I)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *div(mul(sinh(i2), cosh(i2)),
                         add(pow(sin(one), i2), pow(sinh(i2), i2)))));
    REQUIRE(eq(*im, *div(mul({minus_one, sin(one), cos(one)}),
                         add(pow(sin(one), i2), pow(sinh(i2), i2)))));

    CHECK_THROWS_AS(as_real_imag(asin(i2), outArg(re), outArg(im)),
                    SymEngineException);
    CHECK_THROWS_AS(as_real_imag(asinh(i2), outArg(re), outArg(im)),
                    SymEngineException);
}

TEST_CASE("RealImag: Absolute Value Function", "[as_real_imag]")
{
    RCP<const Basic> re, im;
    auto x = symbol("x");
    auto i2 = integer(2), i3 = integer(3);

    as_real_imag(abs(add(add(i2, I), add(i3, I))), outArg(re), outArg(im));
    REQUIRE(eq(*re, *sqrt(add(pow(add(i2, i3), i2), pow(i2, i2)))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(abs(neg(i2)), outArg(re), outArg(im));
    REQUIRE(eq(*re, *abs(neg(i2))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(abs(add(x, neg(i2))), outArg(re), outArg(im));
    REQUIRE(eq(*re, *abs(add(x, neg(i2)))));
    REQUIRE(eq(*im, *zero));

    as_real_imag(abs(add(x, pow(I, i2))), outArg(re), outArg(im));
    REQUIRE(eq(*re, *abs(add(x, neg(one)))));
    REQUIRE(eq(*im, *zero));
}
