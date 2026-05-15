#include "catch.hpp"

#include <symengine/eval_double.h>
#include <symengine/numer_denom.cpp>
#include <symengine/symengine_exception.h>

using SymEngine::Add;
using SymEngine::as_numer_denom;
using SymEngine::Basic;
using SymEngine::Complex;
using SymEngine::has_symbol;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::is_a;
using SymEngine::make_rcp;
using SymEngine::Mul;
using SymEngine::neg;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pow;
using SymEngine::Rational;
using SymEngine::RCP;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::zero;

TEST_CASE("NumerDenom: Basic", "[as_numer_denom]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r1, num, den;

    r1 = add(x, y);
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *r1));
    REQUIRE(eq(*den, *one));

    r1 = add(x, mul(y, pow(x, y)));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *r1));
    REQUIRE(eq(*den, *one));
}

TEST_CASE("NumerDenom: Rational", "[as_numer_denom]")
{
    RCP<const Basic> num, den, r2_5, r6_m2;

    r2_5 = Rational::from_two_ints(*integer(2), *integer(5));
    r6_m2 = Rational::from_two_ints(*integer(6), *integer(-2));

    as_numer_denom(r2_5, outArg(num), outArg(den));
    REQUIRE(eq(*num, *integer(2)));
    REQUIRE(eq(*den, *integer(5)));
    REQUIRE(is_a<Integer>(*num));

    as_numer_denom(r6_m2, outArg(num), outArg(den));
    REQUIRE(eq(*num, *integer(-3)));
    REQUIRE(eq(*den, *one));
    REQUIRE(is_a<Integer>(*num));
}

TEST_CASE("NumerDenom: Mul", "[as_numer_denom]")
{
    RCP<const Basic> num, den, r2_5, rm6_2, r1;
    RCP<const Symbol> x = symbol("x");

    r2_5 = Rational::from_two_ints(*integer(2), *integer(5));
    rm6_2 = Rational::from_two_ints(*integer(-6), *integer(2));

    r1 = mul(x, r2_5);
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *mul(x, integer(2))));
    REQUIRE(eq(*den, *integer(5)));

    r1 = mul(r1, rm6_2);
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *mul(x, integer(-6))));
    REQUIRE(eq(*den, *integer(5)));

    r1 = div(exp(neg(x)), pow(add(one, exp(neg(x))), integer(2)));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *exp(x)));
    REQUIRE(eq(*den, *pow(add(one, exp(x)), integer(2))));

    r1 = neg(sqrt(div(integer(1), integer(2))));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *neg(sqrt(integer(2)))));
    REQUIRE(eq(*den, *integer(2)));
}

TEST_CASE("NumerDenom: Pow", "[as_numer_denom]")
{
    RCP<const Basic> num, den, r2_5, i3, im3, r1;
    RCP<const Symbol> x = symbol("x");

    i3 = integer(3);
    im3 = integer(-3);
    r2_5 = Rational::from_two_ints(*integer(2), *integer(5));

    r1 = pow(i3, r2_5);
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *r1));
    REQUIRE(eq(*den, *one));

    r1 = pow(r2_5, i3);
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *pow(integer(2), i3)));
    REQUIRE(eq(*den, *pow(integer(5), i3)));

    r1 = pow(r2_5, im3);
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *pow(integer(5), i3)));
    REQUIRE(eq(*den, *pow(integer(2), i3)));

    r1 = pow(r2_5, mul(im3, x));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *pow(integer(5), mul(i3, x))));
    REQUIRE(eq(*den, *pow(integer(2), mul(i3, x))));
}

TEST_CASE("NumerDenom: Add", "[as_numer_denom]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> r1, num, den;

    // (1/x^2) + x^2
    r1 = add(pow(x, integer(2)), pow(x, integer(-2)));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *add(pow(x, integer(4)), one))); // x^4 + 1
    REQUIRE(eq(*den, *pow(x, integer(2))));           // x^2

    // (1/x^3) + (1/x^6)
    r1 = add(pow(x, integer(-3)), pow(x, integer(-6)));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *add(pow(x, integer(3)), one))); // x^3 + 1
    REQUIRE(eq(*den, *pow(x, integer(6))));           // x^6

    // (x/4) + (y/6)
    r1 = add(div(x, integer(4)), div(y, integer(6)));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(
        eq(*num, *add(mul(integer(3), x), mul(integer(2), y)))); // 3*x + 2*y
    REQUIRE(eq(*den, *integer(12)));                             // 12

    // (1/xy) + (1/yz)
    r1 = add(div(one, mul(x, y)), div(one, mul(y, z)));
    as_numer_denom(r1, outArg(num), outArg(den));
    REQUIRE(eq(*num, *add(x, z)));         // x + z
    REQUIRE(eq(*den, *mul(x, mul(y, z)))); // x*y*z
}

TEST_CASE("Complex: Basic", "[basic]")
{
    RCP<const Number> r1, r2, r3, c, cnum;
    RCP<const Basic> num, den;

    r1 = Rational::from_two_ints(*integer(2), *integer(4));
    r2 = Rational::from_two_ints(*integer(7), *integer(6));
    r3 = Rational::from_two_ints(*integer(-5), *integer(8));

    c = Complex::from_two_nums(*r1, *r2);
    cnum = Complex::from_two_nums(*integer(3), *integer(7));
    as_numer_denom(c, outArg(num), outArg(den));
    REQUIRE(eq(*num, *cnum));
    REQUIRE(eq(*den, *integer(6)));

    c = Complex::from_two_nums(*r1, *r3);
    cnum = Complex::from_two_nums(*integer(4), *integer(-5));
    as_numer_denom(c, outArg(num), outArg(den));
    REQUIRE(eq(*num, *cnum));
    REQUIRE(eq(*den, *integer(8)));

    c = Complex::from_two_nums(*r2, *r3);
    cnum = Complex::from_two_nums(*integer(28), *integer(-15));
    as_numer_denom(c, outArg(num), outArg(den));
    REQUIRE(eq(*num, *cnum));
    REQUIRE(eq(*den, *integer(24)));
}
