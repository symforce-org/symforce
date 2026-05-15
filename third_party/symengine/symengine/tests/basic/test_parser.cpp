#include "catch.hpp"

#include <symengine/visitor.h>
#include <symengine/eval_double.h>
#include <symengine/parser.h>
#include <symengine/polys/basic_conversions.h>
#include <symengine/symengine_exception.h>
#include <symengine/parser/parser.h>

using SymEngine::Add;
using SymEngine::Basic;
using SymEngine::boolFalse;
using SymEngine::boolTrue;
using SymEngine::Complex;
using SymEngine::ComplexInf;
using SymEngine::down_cast;
using SymEngine::E;
using SymEngine::Eq;
using SymEngine::erf;
using SymEngine::erfc;
using SymEngine::from_basic;
using SymEngine::function_symbol;
using SymEngine::gamma;
using SymEngine::Ge;
using SymEngine::Gt;
using SymEngine::has_symbol;
using SymEngine::I;
using SymEngine::Inf;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::is_a;
using SymEngine::Le;
using SymEngine::loggamma;
using SymEngine::logical_and;
using SymEngine::logical_nand;
using SymEngine::logical_nor;
using SymEngine::logical_not;
using SymEngine::logical_or;
using SymEngine::logical_xnor;
using SymEngine::logical_xor;
using SymEngine::Lt;
using SymEngine::make_rcp;
using SymEngine::max;
using SymEngine::min;
using SymEngine::minus_one;
using SymEngine::Mul;
using SymEngine::Ne;
using SymEngine::NegInf;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::parse;
using SymEngine::ParseError;
using SymEngine::pi;
using SymEngine::piecewise;
using SymEngine::pow;
using SymEngine::Rational;
using SymEngine::rational;
using SymEngine::RCP;
using SymEngine::real_double;
using SymEngine::RealDouble;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::UIntPoly;
using SymEngine::zero;

using namespace SymEngine::literals;

TEST_CASE("Parsing: integers, basic operations", "[parser]")
{
    std::string s;
    RCP<const Basic> res;

    s = "-1^2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(-1)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "+1^2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(1)));

    s = "-2^2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(-4)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "-a^2";
    res = parse(s);
    REQUIRE(eq(*res, *neg(parse("a^2"))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "-2a^2";
    res = parse(s);
    REQUIRE(eq(*res, *mul(integer(-2), parse("a^2"))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "-3-5";
    res = parse(s);
    REQUIRE(eq(*res, *integer(-8)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "((3)+(1*0))";
    res = parse(s);
    REQUIRE(eq(*res, *integer(3)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "((2))*(1+(2*3))";
    res = parse(s);
    REQUIRE(eq(*res, *integer(14)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(1+1)*((1+1)+(1+1))";
    res = parse(s);
    REQUIRE(eq(*res, *integer(8)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(1*3)*(2+4)*(2)";
    res = parse(s);
    REQUIRE(eq(*res, *integer(36)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(1+3)/(2+4)";
    res = parse(s);
    REQUIRE(eq(*res, *div(integer(2), integer(3))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2*3 + 50*2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(106)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2**(3+2)+ 10";
    res = parse(s);
    REQUIRE(eq(*res, *integer(42)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2^(3+2)+ 10";
    res = parse(s);
    REQUIRE(eq(*res, *integer(42)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(5**3)/8 + 12";
    res = parse(s);
    REQUIRE(eq(*res, *add(div(integer(125), integer(8)), integer(12))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(5^3)/8 + 12";
    res = parse(s);
    REQUIRE(eq(*res, *add(div(integer(125), integer(8)), integer(12))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "3*2+3-5+2/2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(5)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "4**2/2+2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(10)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "4^2/2+2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(10)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(1+2*(3+1)-5/(2+2))";
    res = parse(s);
    REQUIRE(eq(*res, *add(integer(9), div(integer(-5), integer(4)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2 + -3";
    res = parse(s);
    REQUIRE(eq(*res, *integer(-1)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2**-3*2";
    res = parse(s);
    REQUIRE(eq(*res, *div(one, integer(4))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2^-2n*y";
    res = parse(s);
    REQUIRE(eq(*res, *parse("(2^(-2*n))*y")));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2^2n*y";
    res = parse(s);
    REQUIRE(eq(*res, *parse("(2^(2*n))*y")));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "10000000000000000000000000";
    res = parse(s);
    REQUIRE(eq(*res, *pow(integer(10), integer(25))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    // Make sure that parsing and printing works correctly
    s = "0.123123123e-10";
    res = parse(s);
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "123123123123123.0";
    res = parse(s);
    REQUIRE(eq(*res, *parse(res->__str__())));

#ifdef HAVE_SYMENGINE_MPFR
    s = "1.231231232123123123123123123123e8";
    res = parse(s);
    REQUIRE(eq(*res, *parse(res->__str__())));
#endif
}

TEST_CASE("Parsing: symbols", "[parser]")
{
    std::string s;
    RCP<const Basic> res;
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> w = symbol("w1");
    RCP<const Basic> l = symbol("l0ngn4me");

    s = "x + 2*y";
    res = parse(s);
    REQUIRE(eq(*res, *add(x, mul(integer(2), y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x + 2y";
    res = parse(s);
    REQUIRE(eq(*res, *add(x, mul(integer(2), y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "w1*y";
    res = parse(s);
    REQUIRE(eq(*res, *mul(w, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x**(3+w1)-2/y";
    res = parse(s);
    REQUIRE(eq(*res, *add(pow(x, add(integer(3), w)), div(integer(-2), y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "l0ngn4me - w1*y + 2**(x)";
    res = parse(s);
    REQUIRE(eq(*res, *add(add(l, neg(mul(w, y))), pow(integer(2), x))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "4*x/8 - (w1*y)";
    res = parse(s);
    REQUIRE(eq(*res, *add(neg(mul(w, y)), div(x, integer(2)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "3*y + (2*y)";
    res = parse(s);
    REQUIRE(eq(*res, *mul(y, integer(5))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "3*y/(1+x)";
    res = parse(s);
    REQUIRE(eq(*res, *div(mul(y, integer(3)), add(x, integer(1)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "y/x*x";
    res = parse(s);
    REQUIRE(eq(*res, *y));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x * -y";
    res = parse(s);
    REQUIRE(eq(*res, *mul(x, mul(y, integer(-1)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x ** --y";
    res = parse(s);
    REQUIRE(eq(*res, *pow(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

#if !defined(_MSC_VER) || !defined(_DEBUG)
    // test unicode
    s = "μ + 1";
    res = parse(s);
    REQUIRE(eq(*res, *add(symbol("μ"), one)));
    REQUIRE(eq(*res, *parse(res->__str__())));
#endif

    s = "x**2e-1+3e+2-2e-2";
    res = parse(s);
    REQUIRE(eq(*res, *add(real_double(299.98), pow(x, real_double(0.2)))));
    REQUIRE(eq(*res, *parse(res->__str__())));
}

TEST_CASE("Parsing: functions", "[parser]")
{
    std::string s;
    RCP<const Basic> res;
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> z = symbol("z");

    s = "sin(x)";
    res = parse(s);
    REQUIRE(eq(*res, *sin(x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "asin(-1)";
    res = parse(s);
    REQUIRE(eq(*res, *neg(div(pi, integer(2)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "asin(sin(x))";
    res = parse(s);
    REQUIRE(eq(*res, *asin(sin(x))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "floor(5.2)";
    res = parse(s);
    REQUIRE(eq(*res, *integer(5)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "ceiling(5.2)";
    res = parse(s);
    REQUIRE(eq(*res, *integer(6)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "floor(x) + ceiling(y)";
    res = parse(s);
    REQUIRE(eq(*res, *add(floor(x), ceiling(y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "beta(x, y)";
    res = parse(s);
    REQUIRE(eq(*res, *beta(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "erf(erf(x*y)) + y";
    res = parse(s);
    REQUIRE(eq(*res, *add(erf(erf(mul(x, y))), y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "erfc(sin(x))+erfc(x*y)";
    res = parse(s);
    REQUIRE(eq(*res, *add(erfc(mul(x, y)), erfc(sin(x)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "beta(sin(x+3), gamma(2**y+sin(y)))";
    res = parse(s);
    REQUIRE(eq(*res, *beta(sin(add(x, integer(3))),
                           gamma(add(sin(y), pow(integer(2), y))))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "y**(abs(sin(3) + x)) + sinh(2)";
    res = parse(s);
    REQUIRE(
        eq(*res, *add(pow(y, abs(add(sin(integer(3)), x))), sinh(integer(2)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "y^(abs(sin(3) + x)) + sinh(2)";
    res = parse(s);
    REQUIRE(
        eq(*res, *add(pow(y, abs(add(sin(integer(3)), x))), sinh(integer(2)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2 + zeta(2, x) + zeta(ln(3))";
    res = parse(s);
    REQUIRE(eq(*res, *add(integer(2),
                          add(zeta(integer(2), x), zeta(log(integer(3)))))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "sin(asin(x)) + y";
    res = parse(s);
    REQUIRE(eq(*res, *add(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "log(x, gamma(y))*sin(3)";
    res = parse(s);
    REQUIRE(eq(*res, *mul(log(x, gamma(y)), sin(integer(3)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "loggamma(x)*gamma(y)";
    res = parse(s);
    REQUIRE(eq(*res, *mul(loggamma(x), gamma(y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "loggamma(x)+loggamma(x)";
    res = parse(s);
    REQUIRE(eq(*res, *mul(integer(2), loggamma(x))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "max(x, x, y)";
    res = parse(s);
    REQUIRE(eq(*res, *max({x, y})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "max(x, y, max(x))";
    res = parse(s);
    REQUIRE(eq(*res, *max({x, y})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "sin(max(log(x, y), min(x, y)))";
    res = parse(s);
    REQUIRE(eq(*res, *sin(max({log(x, y), min({x, y})}))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "atan2(x, y)";
    res = parse(s);
    REQUIRE(eq(*res, *atan2(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Eq(x)";
    res = parse(s);
    CHECK(eq(*res, *Eq(x, integer(0))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "And(Equality(x), Unequality(y, 1))";
    res = parse(s);
    CHECK(eq(*res, *logical_and({Eq(x, integer(0)), Ne(y, integer(1))})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Eq(x, y)";
    res = parse(s);
    CHECK(eq(*res, *Eq(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Ne(x, y)";
    res = parse(s);
    CHECK(eq(*res, *Ne(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Ge(x, y)";
    res = parse(s);
    CHECK(eq(*res, *Le(y, x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Gt(x, y)";
    res = parse(s);
    CHECK(eq(*res, *Lt(y, x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Le(x, y)";
    res = parse(s);
    CHECK(eq(*res, *Le(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Lt(x, y)";
    res = parse(s);
    CHECK(eq(*res, *Lt(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x == y";
    res = parse(s);
    CHECK(eq(*res, *Eq(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x >= y";
    res = parse(s);
    CHECK(eq(*res, *Le(y, x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x > y";
    res = parse(s);
    CHECK(eq(*res, *Lt(y, x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x <= y";
    res = parse(s);
    CHECK(eq(*res, *Le(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x < y";
    res = parse(s);
    CHECK(eq(*res, *Lt(x, y)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x + y < x*y";
    res = parse(s);
    CHECK(eq(*res, *Lt(add(x, y), mul(x, y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x + y >= x*y";
    res = parse(s);
    CHECK(eq(*res, *Le(mul(x, y), add(x, y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x - y == x/y";
    res = parse(s);
    CHECK(eq(*res, *Eq(sub(x, y), div(x, y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "x - y <= x/y";
    res = parse(s);
    CHECK(eq(*res, *Le(sub(x, y), div(x, y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(2pi) > x";
    res = parse(s);
    REQUIRE(eq(*res, *Lt(x, mul(integer(2), pi))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "sin(pi/2) == 1";
    res = parse(s);
    REQUIRE(eq(*res, *boolTrue));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "log(e) > 2";
    res = parse(s);
    REQUIRE(eq(*res, *boolFalse));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "ln(e/e) + sin(pi*2/2) + 3*x == -1";
    res = parse(s);
    REQUIRE(eq(*res, *Eq(mul(integer(3), x), minus_one)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "3*y/(1+x) > y/x*x";
    res = parse(s);
    REQUIRE(eq(*res, *Lt(y, div(mul(y, integer(3)), add(x, integer(1))))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) & (w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_and({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) | (w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_or({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "~(x < y)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_not(Lt(x, y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) ^ (w >= z)";
    res = parse(s, false);
    REQUIRE(eq(*res, *logical_xor({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) & (w >= z) | (y == z)";
    res = parse(s);
    REQUIRE(
        eq(*res, *logical_or({logical_and({Lt(x, y), Le(z, w)}), Eq(y, z)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) & ((w >= z) | (y == z))";
    res = parse(s);
    REQUIRE(
        eq(*res, *logical_and({logical_or({Eq(y, z), Le(z, w)}), Lt(x, y)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "~ (x < y) & (w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_and({logical_not(Lt(x, y)), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "~ (x < y) | (w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_or({logical_not(Lt(x, y)), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "~ (x < y) ^ (w >= z)";
    res = parse(s, false);
    REQUIRE(eq(*res, *logical_xor({logical_not(Lt(x, y)), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) | (w >= z) ^ (y == z)";
    res = parse(s, false);
    REQUIRE(
        eq(*res, *logical_or({Lt(x, y), logical_xor({Le(z, w), Eq(y, z)})})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(x < y) & (w >= z) ^ (y == z)";
    res = parse(s, false);
    REQUIRE(
        eq(*res, *logical_xor({logical_and({Lt(x, y), Le(z, w)}), Eq(y, z)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "And(x < y, w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_and({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Or(x < y, w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_or({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Nor(x < y, w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_nor({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Nand(x < y, w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_nand({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Xor(x < y, w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_xor({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Xnor(x < y, w >= z)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_xnor({Lt(x, y), Le(z, w)})));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "Not(x < y)";
    res = parse(s);
    REQUIRE(eq(*res, *logical_not(Lt(x, y))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    {
        s = "Piecewise((x, x <= 2), (y, And(x > 2, x <= 5)), (x+y, True))";
        res = parse(s);

        auto cond1 = Le(x, integer(2));
        auto cond2 = logical_and({Gt(x, integer(2)), Le(x, integer(5))});
        auto p = piecewise({{x, cond1}, {y, cond2}, {add(x, y), boolTrue}});

        REQUIRE(eq(*res, *p));
    }
    s = "Piecewise((2x, True))";
    res = parse(s);
    REQUIRE(eq(*res, *piecewise({{mul(integer(2), x), boolTrue}})));
    REQUIRE(eq(*res, *parse(res->__str__())));
}

TEST_CASE("Parsing: constants", "[parser]")
{
    std::string s;
    RCP<const Basic> res;
    RCP<const Basic> x = symbol("x");

    s = "E*pi";
    res = parse(s);
    REQUIRE(eq(*res, *mul(E, pi)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "2pi";
    res = parse(s);
    REQUIRE(eq(*res, *mul(integer(2), pi)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "sin(pi/2)";
    res = parse(s);
    REQUIRE(eq(*res, *one));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "log(e)";
    res = parse(s);
    REQUIRE(eq(*res, *one));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "ln(e/e) + sin(pi*2/2) + 3*x";
    res = parse(s);
    REQUIRE(eq(*res, *mul(integer(3), x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(3+4*I)/(5+cos(pi/2)*I)";
    res = parse(s);
    REQUIRE(eq(*res, *div(Complex::from_two_nums(*integer(3), *integer(4)),
                          integer(5))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "(2*I +6*I)*3*I + 4*I";
    res = parse(s);
    REQUIRE(eq(*res, *Complex::from_two_nums(*integer(-24), *integer(4))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "oo";
    res = parse(s);
    REQUIRE(eq(*res, *Inf));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "-oo";
    res = parse(s);
    REQUIRE(eq(*res, *NegInf));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "1/oo + 2";
    res = parse(s);
    REQUIRE(eq(*res, *integer(2)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "zoo";
    res = parse(s);
    REQUIRE(eq(*res, *ComplexInf));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "True";
    res = parse(s);
    REQUIRE(eq(*res, *boolTrue));
    REQUIRE(eq(*res, *parse(res->__str__())));
}

TEST_CASE("Parsing: local_constants", "[parser]")
{
    // local constants take precedence over parser built-ins
    SymEngine::Parser parser({{"pi", integer(3)}, {"pie", pi}});
    std::string s;
    RCP<const Basic> res;

    s = "E*pi";
    res = parser.parse(s);
    REQUIRE(eq(*res, *mul(E, integer(3))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "E*pie";
    res = parser.parse(s);
    REQUIRE(eq(*res, *mul(E, pi)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "pi*pie";
    res = parser.parse(s);
    REQUIRE(eq(*res, *mul(integer(3), pi)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    // julia uses "im" for I
    std::map<const std::string, const RCP<const Basic>> constants(
        {{"I", symbol("I")}, {"im", I}});

    s = "2*I";
    res = parse(s, true, constants);
    REQUIRE(eq(*res, *mul(integer(2), symbol("I"))));
    REQUIRE(eq(*res, *parse(julia_str(*res), true, constants)));

    s = "2*im";
    res = parse(s, true, constants);
    REQUIRE(eq(*res, *mul(integer(2), I)));
    REQUIRE(eq(*res, *parse(julia_str(*res), true, constants)));
}

TEST_CASE("Parsing: function_symbols", "[parser]")
{
    std::string s;
    RCP<const Basic> res;
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("wt");

    s = "f(x)";
    res = parse(s);
    REQUIRE(eq(*res, *function_symbol("f", x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "my_func(x, wt) + sin(f(y))";
    res = parse(s);
    REQUIRE(eq(*res, *add(function_symbol("my_func", {x, z}),
                          sin(function_symbol("f", y)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "func(x, y, wt) + f(sin(x))";
    res = parse(s);
    REQUIRE(eq(*res, *add(function_symbol("func", {x, y, z}),
                          function_symbol("f", sin(x)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "f(g(2**x))";
    res = parse(s);
    REQUIRE(eq(
        *res, *function_symbol("f", function_symbol("g", pow(integer(2), x)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "f(g(2^x))";
    res = parse(s);
    REQUIRE(eq(
        *res, *function_symbol("f", function_symbol("g", pow(integer(2), x)))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "primepi(23)";
    res = parse(s);
    REQUIRE(eq(*res, *integer(9)));

    s = "primorial(15.9)";
    res = parse(s);
    REQUIRE(eq(*res, *integer(30030)));
}

TEST_CASE("Parsing: multi-arg functions", "[parser]")
{
    std::string s;
    RCP<const Basic> res;
    RCP<const Basic> x1 = symbol("x1");
    RCP<const Basic> x2 = symbol("x2");

    s = "x1*pow(x2,-1)";
    res = parse(s);
    REQUIRE(eq(*res, *div(x1, x2)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "z + f(x + y, g(x), h(g(x)))";
    res = parse(s);
    REQUIRE(res->__str__() == s);
    REQUIRE(eq(*res, *parse(res->__str__())));
}

TEST_CASE("Parsing: doubles", "[parser]")
{
    std::string s;
    double d;
    RCP<const Basic> res;
    RCP<const Basic> x = symbol("x");

    s = "1.324";
    res = parse(s);
    REQUIRE(eq(*res, *real_double(1.324)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "0.0324*x + 2*3";
    res = parse(s);
    REQUIRE(eq(*res, *add(mul(real_double(0.0324), x), integer(6))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "0.324e-1x + 2*3";
    res = parse(s);
    REQUIRE(eq(*res, *add(mul(real_double(0.0324), x), integer(6))));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "1.324/(2+3)";
    res = parse(s);
    REQUIRE(is_a<RealDouble>(*res));
    d = down_cast<const RealDouble &>(*res).as_double();
    REQUIRE(std::abs(d - 0.2648) < 1e-12);
    // note: printing an expression containing a RealDouble & reparsing
    // does not always compare equal, as the stored double has ~17 significant
    // digits, but only 15 digits are printed.
    // first print & parse res again to get only 15 significant figures:
    res = parse(res->__str__());
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "sqrt(2.0)+5";
    res = parse(s);
    REQUIRE(is_a<RealDouble>(*res));
    d = down_cast<const RealDouble &>(*res).as_double();
    REQUIRE(std::abs(d - (std::sqrt(2) + 5)) < 1e-12);
    // as above: first print to get doubles with 15 significant figures
    res = parse(res->__str__());
    REQUIRE(eq(*res, *parse(res->__str__())));

    // Test that https://github.com/symengine/symengine/issues/1413 is fixed

    s = "inflation";
    res = parse(s);
    REQUIRE(eq(*res, *symbol("inflation")));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "nano";
    res = parse(s);
    REQUIRE(eq(*res, *symbol("nano")));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "inf";
    res = parse(s);
    REQUIRE(eq(*res, *Inf));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "nan";
    res = parse(s);
    REQUIRE(eq(*res, *SymEngine::Nan));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "-0.12x";
    res = parse(s);
    REQUIRE(eq(*res, *mul(real_double(-0.12), x)));
    REQUIRE(eq(*res, *parse(res->__str__())));

    s = "-.12x";
    res = parse(s);
    REQUIRE(eq(*res, *mul(real_double(-0.12), x)));
    REQUIRE(eq(*res, *parse(res->__str__())));
}

TEST_CASE("Parsing: polys", "[parser]")
{
    std::string s;
    RCP<const UIntPoly> poly1, poly2, poly3, poly4;
    RCP<const Basic> x = symbol("x");

    s = "x + 2*x**2 + 1";
    poly1 = from_basic<UIntPoly>(parse(s));
    poly2 = UIntPoly::from_vec(x, {{1_z, 1_z, 2_z}});
    REQUIRE(eq(*poly1, *poly2));

    s = "2*(x+1)**10 + 3*(x+2)**5";
    poly1 = from_basic<UIntPoly>(parse(s));
    // double-braced initialization of 2-element vector
    // causes compiler error with boost.multiprecision
    // so use single brace.
    poly2 = pow_upoly(*UIntPoly::from_vec(x, {1_z, 1_z}), 10);
    poly3 = UIntPoly::from_vec(x, {{2_z}});
    poly2 = mul_upoly(*poly2, *poly3);
    poly3 = pow_upoly(*UIntPoly::from_vec(x, {2_z, 1_z}), 5);
    poly4 = UIntPoly::from_vec(x, {{3_z}});
    poly3 = mul_upoly(*poly4, *poly3);
    poly2 = add_upoly(*poly2, *poly3);
    REQUIRE(eq(*poly1, *poly2));

    s = "((x+1)**5)*(x+2)*(2*x + 1)**3";
    poly1 = from_basic<UIntPoly>(parse(s));

    poly2 = pow_upoly(*UIntPoly::from_vec(x, {1_z, 1_z}), 5);
    poly3 = UIntPoly::from_vec(x, {2_z, 1_z});
    poly2 = mul_upoly(*poly2, *poly3);
    poly3 = pow_upoly(*UIntPoly::from_vec(x, {1_z, 2_z}), 3);
    poly2 = mul_upoly(*poly2, *poly3);
    REQUIRE(eq(*poly1, *poly2));
}

TEST_CASE("Parsing: errors", "[parser]")
{
    std::string s;

    s = "x+y+";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "x + (y))";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "x + max((3, 2+1)";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "2..33 + 2";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "(2)(3)";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "sin(x y)";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "max(,3,2)";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "x+%y+z";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "Piecewise((x, y))";
    CHECK_THROWS_AS(parse(s), ParseError);

    s = "And(x, y)";
    CHECK_THROWS_AS(parse(s), ParseError);
}

TEST_CASE("Parsing: bison stack reallocation", "[parser]")
{
    std::size_t n{5000};
    std::string s{};
    for (std::size_t i = 0; i < n; ++i) {
        s.append("sin(");
    }
    s.append("0");
    for (std::size_t i = 0; i < n; ++i) {
        s.append(")");
    }
    REQUIRE(eq(*parse(s), *integer(0)));
}
