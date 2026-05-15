#include "catch.hpp"
#include <iostream>
#include <symengine/basic.h>
#include <symengine/nan.h>
#include <symengine/symengine_rcp.h>
#include <symengine/constants.h>
#include <symengine/functions.h>
#include <symengine/pow.h>

using SymEngine::Basic;
using SymEngine::gamma;
using SymEngine::Inf;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::is_a;
using SymEngine::minus_one;
using SymEngine::NaN;
using SymEngine::Nan;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::zero;

TEST_CASE("Hash Size for NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;
    RCP<const NaN> b = Nan;

    REQUIRE(a->__hash__() == b->__hash__());
}

TEST_CASE("NaN Constants", "[NaN]")
{
    REQUIRE(Nan->__str__() == "nan");
}

TEST_CASE("Boolean tests for NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;

    REQUIRE((not a->is_zero() && not a->is_one() && not a->is_minus_one()
             && not a->is_positive() && not a->is_negative()
             && not a->is_complex() && not a->is_exact() && is_a<NaN>(*a)));
}

TEST_CASE("Comparing NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;
    RCP<const NaN> b = Nan;
    RCP<const Basic> i1 = integer(1);

    REQUIRE(a->compare(*b) == 0);
    REQUIRE(eq(*a, *b));
    REQUIRE(neq(*a, *i1));
}

TEST_CASE("Check Derivative", "[NaN]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const NaN> b = Nan;
    REQUIRE(eq(*b->diff(x), *zero));
}

TEST_CASE("Adding to NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;

    RCP<const Basic> n1 = a->add(*one);
    REQUIRE(eq(*n1, *Nan));
}

TEST_CASE("Subtracting from NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;
    RCP<const Basic> r1 = a->sub(*a);
    REQUIRE(eq(*r1, *Nan));
}

TEST_CASE("Multiplication with NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;

    RCP<const Basic> n1;
    n1 = a->mul(*integer(-10));
    REQUIRE(eq(*n1, *Nan));
}

TEST_CASE("Powers of NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;

    RCP<const Basic> n1;
    n1 = a->pow(*integer(-10));
    REQUIRE(eq(*n1, *Nan));
}

TEST_CASE("Powers to NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;

    RCP<const Basic> n1;
    n1 = integer(-10)->pow(*a);
    REQUIRE(eq(*n1, *Nan));
}

TEST_CASE("Evaluate Class of NaN", "[NaN]")
{
    RCP<const NaN> a = Nan;
    RCP<const Basic> n1;

    n1 = sin(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = cos(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = tan(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = csc(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = sec(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = cot(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = asin(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = acos(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = atan(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = acsc(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = asec(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = acot(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = sinh(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = cosh(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = tanh(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = csch(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = sech(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = coth(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = asinh(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = acosh(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = atanh(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = acsch(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = asech(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = acoth(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = log(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = gamma(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = abs(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = exp(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = erf(a);
    REQUIRE(eq(*n1, *Nan));
    n1 = erfc(a);
    REQUIRE(eq(*n1, *Nan));
}