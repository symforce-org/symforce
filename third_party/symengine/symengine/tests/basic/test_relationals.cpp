#include "catch.hpp"
#include <iostream>
#include <symengine/logic.h>
#include <symengine/add.h>
#include <symengine/real_double.h>
#include <symengine/complex_double.h>

using SymEngine::Basic;
using SymEngine::Boolean;
using SymEngine::boolFalse;
using SymEngine::boolTrue;
using SymEngine::complex_double;
using SymEngine::ComplexInf;
using SymEngine::Eq;
using SymEngine::Equality;
using SymEngine::gamma;
using SymEngine::Ge;
using SymEngine::Gt;
using SymEngine::I;
using SymEngine::Inf;
using SymEngine::integer;
using SymEngine::Le;
using SymEngine::logical_not;
using SymEngine::Lt;
using SymEngine::make_rcp;
using SymEngine::Nan;
using SymEngine::Ne;
using SymEngine::NegInf;
using SymEngine::one;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::real_double;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::zero;

TEST_CASE("Hash Size for Relationals", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> a = Eq(x, y);
    RCP<const Basic> b = Eq(x, y);
    CHECK(a->__hash__() == b->__hash__());

    a = Eq(one, zero);
    b = Eq(one);
    CHECK(a->__hash__() == b->__hash__());
}

TEST_CASE("String Printing", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> a = Eq(x, y);
    CHECK(a->__str__() == "x == y");

    a = Ne(x, y);
    CHECK(a->__str__() == "x != y");

    a = Ge(x, y);
    CHECK(a->__str__() == "y <= x");

    a = Gt(x, y);
    CHECK(a->__str__() == "y < x");

    a = Le(x, y);
    CHECK(a->__str__() == "x <= y");

    a = Lt(x, y);
    CHECK(a->__str__() == "x < y");
}

TEST_CASE("Comparing Relationals", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> a = Eq(x, y);
    RCP<const Basic> b = Eq(x, y);
    CHECK(a->compare(*b) == 0);
    CHECK(eq(*a, *b));

    b = Eq(y, x);
    CHECK(a->compare(*b) == 0);
    CHECK(eq(*a, *b));

    a = Ne(x, y);
    b = Ne(y, x);
    CHECK(eq(*a, *b));

    a = Eq(one, zero);
    b = Ne(one, one);
    CHECK(eq(*a, *b));

    a = Le(x, y);
    b = Le(x, y);
    CHECK(eq(*a, *b));

    a = Lt(x, y);
    b = Lt(x, y);
    CHECK(eq(*a, *b));

    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i0 = integer(0);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> r1 = complex_double(std::complex<double>(0.1, 0.2));
    RCP<const Basic> r2 = complex_double(std::complex<double>(1, 0.2));
    RCP<const Basic> comp = integer(r1->compare(*r1));
    CHECK(eq(*comp, *i0));
    comp = integer(r1->compare(*r2));
    CHECK(eq(*comp, *im1));
    comp = integer(r2->compare(*r1));
    CHECK(eq(*comp, *i1));
    r2 = complex_double(std::complex<double>(0.1, 0.3));
    comp = integer(r2->compare(*r1));
    CHECK(eq(*comp, *i1));
}

TEST_CASE("Canonicalization", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Equality> r = make_rcp<Equality>(x, y);
    CHECK(not(r->is_canonical(zero, one)));
    CHECK(not(r->is_canonical(gamma(integer(2)), one)));
    CHECK(not(r->is_canonical(boolTrue, boolTrue)));
}

TEST_CASE("Eq", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> a = Eq(x);
    RCP<const Basic> b = Eq(x, zero);
    CHECK(eq(*a, *b));

    a = add(x, y);
    b = add(a, zero);
    CHECK(eq(*Eq(a, b), *boolTrue));

    b = add(a, real_double(0.0));
    CHECK(eq(*Eq(a, b), *boolTrue));
    CHECK(eq(*Eq(sub(b, y), x), *boolTrue));
    CHECK(eq(*Eq(add(x, real_double(0.0)), x), *boolTrue));
}

TEST_CASE("Infinity", "[Relationals]")
{
    RCP<const Basic> a = Eq(Inf, Inf);
    CHECK(eq(*a, *boolTrue));

    a = Ne(Inf, Inf);
    CHECK(eq(*a, *boolFalse));

    a = Lt(Inf, Inf);
    CHECK(eq(*a, *boolFalse));

    a = Lt(Inf, NegInf);
    CHECK(eq(*a, *boolFalse));

    a = Lt(Inf, one);
    CHECK(eq(*a, *boolFalse));

    a = Le(Inf, Inf);
    CHECK(eq(*a, *boolTrue));

    a = Le(Inf, NegInf);
    CHECK(eq(*a, *boolFalse));

    a = Le(Inf, one);
    CHECK(eq(*a, *boolFalse));

    a = Lt(NegInf, Inf);
    CHECK(eq(*a, *boolTrue));

    a = Lt(NegInf, NegInf);
    CHECK(eq(*a, *boolFalse));

    a = Lt(NegInf, one);
    CHECK(eq(*a, *boolTrue));

    a = Le(NegInf, Inf);
    CHECK(eq(*a, *boolTrue));

    a = Le(NegInf, NegInf);
    CHECK(eq(*a, *boolTrue));

    a = Le(NegInf, one);
    CHECK(eq(*a, *boolTrue));

    CHECK_THROWS_AS(Lt(ComplexInf, zero), SymEngineException);
    CHECK_THROWS_AS(Le(ComplexInf, zero), SymEngineException);
}

TEST_CASE("Boolean Values", "[Relationals]")
{
    RCP<const Basic> a = Eq(zero, zero);
    CHECK(eq(*a, *boolTrue));

    a = Eq(boolTrue, boolTrue);
    CHECK(eq(*a, *boolTrue));

    a = Eq(boolFalse, boolTrue);
    CHECK(eq(*a, *boolFalse));

    a = Eq(boolTrue, boolFalse);
    CHECK(eq(*a, *boolFalse));

    a = Eq(boolFalse, boolFalse);
    CHECK(eq(*a, *boolTrue));

    a = Eq(one, zero);
    CHECK(eq(*a, *boolFalse));

    a = Ne(zero, zero);
    CHECK(eq(*a, *boolFalse));

    a = Eq(I, one);
    CHECK(eq(*a, *boolFalse));

    a = Ne(I, one);
    CHECK(eq(*a, *boolTrue));

    a = Ne(one, zero);
    CHECK(eq(*a, *boolTrue));

    a = Lt(zero, one);
    CHECK(eq(*a, *boolTrue));

    a = Lt(one, zero);
    CHECK(eq(*a, *boolFalse));

    a = Ge(zero, one);
    CHECK(eq(*a, *boolFalse));

    a = Ge(one, zero);
    CHECK(eq(*a, *boolTrue));

    a = Le(zero, zero);
    CHECK(eq(*a, *boolTrue));

    a = Le(zero, one);
    CHECK(eq(*a, *boolTrue));

    a = Le(one, zero);
    CHECK(eq(*a, *boolFalse));

    a = Gt(zero, one);
    CHECK(eq(*a, *boolFalse));

    a = Gt(one, zero);
    CHECK(eq(*a, *boolTrue));

    CHECK_THROWS_AS(Ge(I, one), SymEngineException);
    CHECK_THROWS_AS(Gt(I, one), SymEngineException);
    CHECK_THROWS_AS(Lt(I, one), SymEngineException);
    CHECK_THROWS_AS(Le(I, one), SymEngineException);
}

TEST_CASE("Logical Not", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Boolean> a = Eq(x, zero);
    RCP<const Boolean> b = Ne(x, zero);
    CHECK(eq(*logical_not(a), *b));

    a = Ne(x, y);
    b = Eq(y, x);
    CHECK(eq(*logical_not(a), *b));

    a = Le(x, y);
    b = Lt(y, x);
    CHECK(eq(*logical_not(a), *b));

    a = Lt(x, y);
    b = Le(y, x);
    CHECK(eq(*logical_not(a), *b));
}

TEST_CASE("Subs", "[Relationals]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");

    RCP<const Basic> a = Eq(x, y)->subs({{x, y}, {y, z}});
    RCP<const Basic> b = Eq(y, z);
    CHECK(eq(*a, *b));

    a = Gt(x, y)->subs({{x, integer(2)}, {y, integer(3)}});
    CHECK(eq(*a, *boolFalse));

    a = Gt(x, y)->subs({{x, integer(3)}, {y, integer(2)}});
    CHECK(eq(*a, *boolTrue));
}

TEST_CASE("Nan Exceptions", "[Relationals]")
{
    RCP<const Basic> a = Eq(Nan, Nan);
    CHECK(eq(*a, *boolFalse));

    a = Ne(Nan, Nan);
    CHECK(eq(*a, *boolTrue));

    CHECK_THROWS_AS(Gt(Nan, one), SymEngineException);
    CHECK_THROWS_AS(Ge(Nan, one), SymEngineException);
    CHECK_THROWS_AS(Lt(Nan, one), SymEngineException);
    CHECK_THROWS_AS(Le(Nan, one), SymEngineException);
}

TEST_CASE("Boolean Exceptions", "[Relationals]")
{
    CHECK_THROWS_AS(Gt(boolFalse, boolTrue), SymEngineException);
    CHECK_THROWS_AS(Ge(boolTrue, boolTrue), SymEngineException);
    CHECK_THROWS_AS(Lt(boolFalse, boolTrue), SymEngineException);
    CHECK_THROWS_AS(Le(boolTrue, boolTrue), SymEngineException);
}