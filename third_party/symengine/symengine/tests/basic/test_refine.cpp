#include "catch.hpp"
#include <symengine/refine.h>

using SymEngine::Assumptions;
using SymEngine::infty;
using SymEngine::integer;
using SymEngine::integers;
using SymEngine::max;
using SymEngine::min;
using SymEngine::pi;
using SymEngine::Rational;
using SymEngine::reals;
using SymEngine::symbol;

TEST_CASE("Test refine", "[refine]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto z = symbol("z");

    auto expr = abs(x);
    Assumptions a = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = abs(x);
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = abs(x);
    a = Assumptions({Le(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *neg(x)));

    expr = abs(conjugate(x));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *abs(x)));

    expr = sign(x);
    a = Assumptions({Lt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(-1)));

    expr = sign(x);
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = sign(x);
    a = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(1)));

    expr = sign(x);
    a = Assumptions({Eq(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(0)));

    expr = sign(abs(x));
    a = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(1)));

    expr = floor(x);
    a = Assumptions({integers()->contains(x)});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = floor(x);
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = ceiling(x);
    a = Assumptions({integers()->contains(x)});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = ceiling(x);
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = ceiling(neg(x));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *neg(floor(x))));

    expr = floor(neg(x));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *neg(ceiling(x))));

    expr = conjugate(x);
    a = Assumptions({reals()->contains(x)});
    REQUIRE(eq(*refine(expr, &a), *x));

    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = max({integer(-1), x});
    a = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = max({integer(-1), x});
    a = Assumptions({Ge(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = max({integer(-1), x});
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = max({x, y, z});
    a = Assumptions({Ge(x, integer(0)), Ge(y, integer(0)), Ge(z, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = max({x, y, z});
    a = Assumptions({Ge(z, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = max({integer(1), x});
    a = Assumptions({Le(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(1)));

    expr = max({integer(-1), x});
    a = Assumptions({Le(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = min({integer(-1), x});
    a = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(-1)));

    expr = min({integer(-1), x});
    a = Assumptions({Ge(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *integer(-1)));

    expr = min({integer(-1), x});
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = min({x, y, z});
    a = Assumptions({Ge(x, integer(0)), Ge(y, integer(0)), Ge(z, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = min({x, y, z});
    a = Assumptions({Ge(z, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = min({integer(1), x});
    a = Assumptions({Le(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = min({integer(-1), x});
    a = Assumptions({Le(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = conjugate(x);
    auto a16 = Assumptions({});
    REQUIRE(eq(*refine(expr, &a16), *expr));

    expr = pow(x, integer(2));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = pow(pow(x, integer(2)),
               Rational::from_two_ints(*integer(1), *integer(2)));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));

    expr = pow(pow(x, integer(2)),
               Rational::from_two_ints(*integer(1), *integer(2)));
    a = Assumptions({reals()->contains(x)});
    REQUIRE(eq(*refine(expr, &a), *abs(x)));

    expr = pow(pow(x, integer(2)),
               Rational::from_two_ints(*integer(1), *integer(2)));
    a = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *x));

    expr = pow(pow(x, integer(6)),
               Rational::from_two_ints(*integer(1), *integer(2)));
    a = Assumptions({reals()->contains(x)});
    REQUIRE(eq(*refine(expr, &a), *pow(abs(x), integer(3))));

    expr = log(pow(x, y));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *expr));
    a = Assumptions({reals()->contains(y), Gt(x, integer(0))});
    REQUIRE(eq(*refine(expr, &a), *mul(y, log(x))));

    expr = log(pow(integer(2), y));
    a = Assumptions({reals()->contains(y)});
    REQUIRE(eq(*refine(expr, &a), *mul(y, log(integer(2)))));

    expr = log(integer(8));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *mul(integer(3), log(integer(2)))));

    expr = log(integer(23));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *log(integer(23))));

    expr = log(integer(49));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *mul(integer(2), log(integer(7)))));

    expr = div(log(integer(4)), integer(2));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *log(integer(2))));

    expr = interval(infty(-1), infty(1));
    a = Assumptions({});
    REQUIRE(eq(*refine(expr, &a), *reals()));
    expr = interval(integer(1), integer(2));
    REQUIRE(eq(*refine(expr, &a), *expr));
}
