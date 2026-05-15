#include "catch.hpp"
#include <chrono>

#include <symengine/polys/cancel.h>
#include <symengine/polys/uintpoly_flint.h>

using SymEngine::Basic;
using SymEngine::cancel;
using SymEngine::exp;
using SymEngine::integer;
using SymEngine::mul;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::sub;
using SymEngine::symbol;
using SymEngine::UIntPolyFlint;

using namespace SymEngine::literals;

TEST_CASE("cancel", "[Basic]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const UIntPolyFlint> numer, denom, common;

    // 2*x / x = 2 / 1
    cancel(mul(x, integer(2)), x, outArg(numer), outArg(denom), outArg(common));
    REQUIRE(numer->__str__() == "2");
    REQUIRE(denom->__str__() == "1");
    REQUIRE(common->__str__() == "x");

    // (x**2 - 4) / (x + 2) = x - 2
    cancel(sub(mul(x, x), integer(4)), add(x, integer(2)), outArg(numer),
           outArg(denom), outArg(common));
    REQUIRE(numer->__str__() == "x - 2");
    REQUIRE(denom->__str__() == "1");
    REQUIRE(common->__str__() == "x + 2");

    // (x**2 - 4) / (x - 2) = x + 2
    cancel(sub(mul(x, x), integer(4)), sub(x, integer(2)), outArg(numer),
           outArg(denom), outArg(common));
    REQUIRE(numer->__str__() == "x + 2");
    REQUIRE(denom->__str__() == "1");
    REQUIRE(common->__str__() == "x - 2");

    // (x**2 + x + 1) / (x + 1) = x - 1
    cancel(sub(pow(x, 3), integer(1)), sub(pow(x, 2), integer(1)),
           outArg(numer), outArg(denom), outArg(common));
    REQUIRE(numer->__str__() == "x**2 + x + 1");
    REQUIRE(denom->__str__() == "x + 1");
    REQUIRE(common->__str__() == "x - 1");

    // (exp(2*x) + 2*exp(x) + 1) / (exp(x) + 1) = exp(x) + 1
    cancel(
        add(add(exp(mul(integer(2), x)), mul(integer(2), exp(x))), integer(1)),
        add(exp(x), integer(1)), outArg(numer), outArg(denom), outArg(common));
    REQUIRE(numer->__str__() == "exp(x) + 1");
    REQUIRE(denom->__str__() == "1");
    REQUIRE(common->__str__() == "exp(x) + 1");
}
