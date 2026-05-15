#include "catch.hpp"
#include <symengine/simplify.h>

using SymEngine::Assumptions;
using SymEngine::integer;
using SymEngine::integers;
using SymEngine::pi;
using SymEngine::reals;
using SymEngine::symbol;
using SymEngine::unevaluated_expr;

TEST_CASE("Test simplify", "[simplify]")
{
    auto x = symbol("x");

    // The following tests visits Mul
    auto expr = div(integer(2), csc(x));
    REQUIRE(eq(*simplify(expr), *mul(integer(2), sin(x))));
    expr = div(integer(2), sec(x));
    REQUIRE(eq(*simplify(expr), *mul(integer(2), cos(x))));
    expr = div(integer(2), cot(x));
    REQUIRE(eq(*simplify(expr), *mul(integer(2), tan(x))));
    expr = div(integer(2), csc(div(integer(2), csc(x))));
    REQUIRE(
        eq(*simplify(expr), *mul(integer(2), sin(mul(integer(2), sin(x))))));

    expr = div(integer(2), csc(abs(x)));
    Assumptions a1 = Assumptions({Gt(x, integer(0))});
    REQUIRE(eq(*simplify(expr, &a1), *mul(integer(2), sin(x))));

    // The following tests visits Pow
    expr = div(integer(1), csc(x));
    REQUIRE(eq(*simplify(expr), *sin(x)));
    expr = div(integer(1), sec(x));
    REQUIRE(eq(*simplify(expr), *cos(x)));
    expr = div(integer(1), cot(x));
    REQUIRE(eq(*simplify(expr), *tan(x)));

    // No simplifications
    expr = x;
    REQUIRE(eq(*simplify(expr), *expr));
    expr = sin(x);
    REQUIRE(eq(*simplify(expr), *expr));
    expr = div(integer(1), sin(x));
    REQUIRE(eq(*simplify(expr), *expr));
    expr = div(integer(2), sin(x));
    REQUIRE(eq(*simplify(expr), *expr));
}
