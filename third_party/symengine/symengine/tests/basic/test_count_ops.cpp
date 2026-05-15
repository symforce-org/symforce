#include "catch.hpp"

#include <symengine/visitor.h>

using SymEngine::Basic;
using SymEngine::count_ops;
using SymEngine::I;
using SymEngine::integer;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::RCP;
using SymEngine::symbol;

TEST_CASE("CountOps", "[count_ops]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1;

    r1 = add(add(one, x), y);
    REQUIRE(count_ops({r1}) == 2);

    r1 = add(add(x, x), y);
    REQUIRE(count_ops({r1}) == 2);

    r1 = mul(mul(x, x), y);
    REQUIRE(count_ops({r1}) == 2);

    r1 = mul(mul(i2, x), y);
    REQUIRE(count_ops({r1}) == 2);

    r1 = add(add(I, one), sin(x));
    REQUIRE(count_ops({r1}) == 3);

    r1 = add(add(mul(i2, I), one), sin(x));
    REQUIRE(count_ops({r1}) == 4);

    r1 = add(I, pi);
    REQUIRE(count_ops({r1}) == 1);

    r1 = pow(pi, pi);
    REQUIRE(count_ops({r1}) == 1);

    REQUIRE(count_ops({x, y}) == 0);
}
