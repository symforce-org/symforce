#include "catch.hpp"
#include <iostream>

#include <symengine/complex.h>
#include <symengine/basic.h>
#include <symengine/infinity.h>
#include <symengine/symengine_rcp.h>
#include <symengine/constants.h>
#include <symengine/symengine_exception.h>
#include <symengine/functions.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/complex_double.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Complex;
using SymEngine::complex_double;
using SymEngine::ComplexInf;
using SymEngine::DomainError;
using SymEngine::erf;
using SymEngine::erfc;
using SymEngine::gamma;
using SymEngine::I;
using SymEngine::Inf;
using SymEngine::Infty;
using SymEngine::infty;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::is_a;
using SymEngine::make_rcp;
using SymEngine::minus_one;
using SymEngine::Nan;
using SymEngine::NegInf;
using SymEngine::NotImplementedError;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::Rational;
using SymEngine::rational;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::zero;

TEST_CASE("Constructors for Infinity", "[Infinity]")
{
    RCP<const Number> r1 = rational(1, 1);
    RCP<const Number> rm1 = rational(-1, 1);
    RCP<const Number> r0 = rational(0, 1);

    RCP<const Integer> im1 = integer(-1);
    RCP<const Integer> i0 = integer(0);

    RCP<const Infty> a = Infty::from_direction(r1);
    RCP<const Infty> b = Infty::from_direction(rm1);
    RCP<const Infty> c = Infty::from_direction(r0);

    REQUIRE(eq(*a, *Inf));
    REQUIRE(eq(*b, *NegInf));
    REQUIRE(eq(*c, *ComplexInf));

    CHECK_THROWS_AS(a->is_canonical(complex_double(std::complex<double>(2, 3))),
                    NotImplementedError);
    CHECK_THROWS_AS(
        a->is_canonical(Complex::from_two_nums(*integer(1), *integer(2))),
        NotImplementedError);

    REQUIRE(not(a->is_canonical(integer(2))));

    a = infty();
    b = infty(-1);
    c = infty(0);

    REQUIRE(eq(*a, *Inf));
    REQUIRE(eq(*b, *NegInf));
    REQUIRE(eq(*c, *ComplexInf));

    a = Infty::from_int(1);
    b = Infty::from_direction(im1);
    REQUIRE(eq(*a, *Inf));
    REQUIRE(eq(*b, *NegInf));

    //! Checking copy constructor
    Infty inf2 = Infty(*NegInf);
    REQUIRE(eq(inf2, *NegInf));

    // RCP<const Number> cx = Complex::from_two_nums(*integer(1), *integer(1));
    // CHECK_THROWS_AS(Infty::from_direction(cx), SymEngineException);
}

TEST_CASE("Hash Size for Infinity", "[Infinity]")
{
    RCP<const Infty> a = infty(1);
    RCP<const Infty> b = infty(0);

    REQUIRE(not eq(*a, *b));
    REQUIRE(not(a->__hash__() == b->__hash__()));
    REQUIRE(eq(*a, *infty()));
    REQUIRE(a->__hash__() == infty(1)->__hash__());
}

TEST_CASE("Infinity Constants", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    REQUIRE(eq(*a, *Inf));
    REQUIRE(eq(*b, *NegInf));
    REQUIRE(eq(*c, *ComplexInf));
}

TEST_CASE("Boolean tests for Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    REQUIRE((not a->is_zero() && not a->is_one() && not a->is_minus_one()
             && not a->is_negative_infinity() && a->is_positive_infinity()
             && not a->is_unsigned_infinity() && a->is_positive()
             && not a->is_negative() && is_a<Infty>(*a)));
    REQUIRE((not b->is_zero() && not b->is_one() && not b->is_minus_one()
             && b->is_negative_infinity() && not b->is_positive_infinity()
             && not b->is_unsigned_infinity() && not b->is_positive()
             && b->is_negative() && is_a<Infty>(*b)));
    REQUIRE((not c->is_zero() && not c->is_one() && not c->is_minus_one()
             && not c->is_negative_infinity() && not c->is_positive_infinity()
             && c->is_unsigned_infinity() && not c->is_positive()
             && not c->is_negative() && c->is_complex() && is_a<Infty>(*c)));
}

TEST_CASE("Comparing Infinitys", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    REQUIRE(a->compare(*b) == 1);
    REQUIRE(c->compare(*c) == 0);
    REQUIRE(c->compare(*a) == -1);
    REQUIRE(a->compare(*c) == 1);
    REQUIRE(not c->__eq__(*a));
    REQUIRE(b->__eq__(*b));
    REQUIRE(not c->__eq__(*zero));
}

TEST_CASE("Checking arguments returned", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    REQUIRE(eq(*a->get_direction(), *one));
    REQUIRE(eq(*b->get_direction(), *minus_one));
    REQUIRE(eq(*c->get_direction(), *zero));
    REQUIRE(eq(*a->get_args()[0], *one));
    REQUIRE(eq(*b->get_args()[0], *minus_one));
    REQUIRE(eq(*c->get_args()[0], *zero));
}

TEST_CASE("Check Derivative", "[Infinity]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Infty> b = NegInf;
    REQUIRE(eq(*b->diff(x), *zero));
}

TEST_CASE("Adding to Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    RCP<const Number> n1 = a->add(*one);
    REQUIRE(eq(*n1, *Inf));
    n1 = b->add(*b);
    REQUIRE(eq(*n1, *NegInf));
    n1 = c->add(*minus_one);
    REQUIRE(eq(*n1, *ComplexInf));

    n1 = c->add(*a);
    REQUIRE(eq(*n1, *Nan));
    n1 = c->add(*c);
    REQUIRE(eq(*n1, *Nan));
    n1 = b->add(*a);
    REQUIRE(eq(*n1, *Nan));
}

TEST_CASE("Subtracting from Infinity", "[Infinity]")
{
    REQUIRE(eq(*Inf->sub(*NegInf), *Inf));
}

TEST_CASE("Multiplication with Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    RCP<const Number> n1 = b->mul(*integer(-10));
    REQUIRE(eq(*n1, *Inf));
    n1 = c->mul(*integer(5));
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = c->mul(*integer(-5));
    REQUIRE(eq(*n1, *ComplexInf));

    RCP<const Number> n2 = a->mul(*a);
    REQUIRE(eq(*n2, *Inf));
    n2 = b->mul(*a);
    REQUIRE(eq(*n2, *NegInf));
    n2 = b->mul(*c);
    REQUIRE(eq(*n2, *ComplexInf));
    n2 = b->mul(*b);
    REQUIRE(eq(*n2, *Inf));
    n2 = c->mul(*c);
    REQUIRE(eq(*n2, *ComplexInf));

    n2 = a->mul(*zero);
    REQUIRE(eq(*n2, *Nan));
    n2 = b->mul(*zero);
    REQUIRE(eq(*n2, *Nan));
    n2 = c->mul(*zero);
    REQUIRE(eq(*n2, *Nan));

    RCP<const Number> cx = Complex::from_two_nums(*integer(1), *integer(1));
    CHECK_THROWS_AS(c->mul(*cx), NotImplementedError);
}

TEST_CASE("Division of Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    RCP<const Number> n1 = b->div(*integer(-10));
    REQUIRE(eq(*n1, *Inf));
    n1 = b->div(*integer(10));
    REQUIRE(eq(*n1, *NegInf));
    n1 = c->div(*minus_one);
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = a->div(*zero);
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = b->div(*zero);
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = c->div(*zero);
    REQUIRE(eq(*n1, *ComplexInf));

    n1 = a->div(*b);
    REQUIRE(eq(*n1, *Nan));
    n1 = b->div(*c);
    REQUIRE(eq(*n1, *Nan));
    n1 = c->div(*c);
    REQUIRE(eq(*n1, *Nan));
}

TEST_CASE("Powers of Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    RCP<const Number> n1 = a->pow(*integer(-10));
    REQUIRE(eq(*n1, *zero));
    n1 = a->pow(*integer(10));
    REQUIRE(eq(*n1, *Inf));
    n1 = a->pow(*zero);
    REQUIRE(eq(*n1, *one));
    n1 = a->pow(*b);
    REQUIRE(eq(*n1, *zero));
    n1 = a->pow(*a);
    REQUIRE(eq(*n1, *Inf));
    n1 = b->pow(*integer(-10));
    REQUIRE(eq(*n1, *zero));
    n1 = b->pow(*zero);
    REQUIRE(eq(*n1, *one));
    n1 = c->pow(*a);
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = c->pow(*b);
    REQUIRE(eq(*n1, *zero));
    n1 = c->pow(*integer(-10));
    REQUIRE(eq(*n1, *zero));
    n1 = c->pow(*zero);
    REQUIRE(eq(*n1, *one));
    n1 = c->pow(*integer(10));
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = a->pow(*c);
    REQUIRE(eq(*n1, *Nan));
    n1 = b->pow(*a);
    REQUIRE(eq(*n1, *Nan));
    n1 = b->pow(*c);
    REQUIRE(eq(*n1, *Nan));
    n1 = c->pow(*c);
    REQUIRE(eq(*n1, *Nan));

    RCP<const Number> cx = Complex::from_two_nums(*integer(1), *integer(1));
    CHECK_THROWS_AS(b->pow(*integer(2)), NotImplementedError);
    CHECK_THROWS_AS(b->pow(*cx), NotImplementedError);
}

TEST_CASE("Powers to Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;

    RCP<const Number> n1;
    n1 = integer(10)->pow(*a);
    REQUIRE(eq(*n1, *Inf));
    n1 = rational(2, 5)->pow(*a);
    REQUIRE(eq(*n1, *zero));
    n1 = rational(5, 2)->pow(*a);
    REQUIRE(eq(*n1, *Inf));
    n1 = integer(10)->pow(*b);
    REQUIRE(eq(*n1, *zero));
    n1 = rational(2, 5)->pow(*b);
    REQUIRE(eq(*n1, *ComplexInf));
    n1 = rational(5, 2)->pow(*b);
    REQUIRE(eq(*n1, *zero));

    n1 = integer(1)->pow(*c);
    REQUIRE(eq(*n1, *Nan));
    n1 = rational(3, 3)->pow(*c);

    RCP<const Number> cx = Complex::from_two_nums(*integer(1), *integer(1));
    CHECK_THROWS_AS(integer(-10)->pow(*a), NotImplementedError);
    CHECK_THROWS_AS(integer(0)->pow(*b), SymEngineException);
    CHECK_THROWS_AS(integer(10)->pow(*c), SymEngineException);
    CHECK_THROWS_AS(integer(-3)->pow(*c), SymEngineException);
    CHECK_THROWS_AS(cx->pow(*c), NotImplementedError);

    RCP<const Basic> x = symbol("x");
    RCP<const Basic> r = exp(add(c, x));
    CHECK_THROWS_AS(div(r, exp(x)), DomainError);
}

TEST_CASE("Evaluate Class of Infinity", "[Infinity]")
{
    RCP<const Infty> a = Inf;
    RCP<const Infty> b = NegInf;
    RCP<const Infty> c = ComplexInf;
    RCP<const Basic> r1, r2;

    CHECK_THROWS_AS(sin(Inf), DomainError);
    CHECK_THROWS_AS(cos(Inf), DomainError);
    CHECK_THROWS_AS(tan(Inf), DomainError);
    CHECK_THROWS_AS(csc(Inf), DomainError);
    CHECK_THROWS_AS(sec(Inf), DomainError);
    CHECK_THROWS_AS(cot(Inf), DomainError);
    CHECK_THROWS_AS(asin(Inf), DomainError);
    CHECK_THROWS_AS(acos(Inf), DomainError);
    CHECK_THROWS_AS(acsc(Inf), DomainError);
    CHECK_THROWS_AS(asec(Inf), DomainError);
    CHECK_THROWS_AS(sin(ComplexInf), DomainError);
    CHECK_THROWS_AS(asech(ComplexInf), DomainError);
    CHECK_THROWS_AS(erfc(ComplexInf), DomainError);
    CHECK_THROWS_AS(atan(ComplexInf), DomainError);
    CHECK_THROWS_AS(acot(ComplexInf), DomainError);
    CHECK_THROWS_AS(sinh(ComplexInf), DomainError);
    CHECK_THROWS_AS(csch(ComplexInf), DomainError);
    CHECK_THROWS_AS(cosh(ComplexInf), DomainError);
    CHECK_THROWS_AS(sech(ComplexInf), DomainError);
    CHECK_THROWS_AS(tanh(ComplexInf), DomainError);
    CHECK_THROWS_AS(coth(ComplexInf), DomainError);
    CHECK_THROWS_AS(asinh(ComplexInf), DomainError);
    CHECK_THROWS_AS(acosh(ComplexInf), DomainError);
    CHECK_THROWS_AS(atanh(ComplexInf), DomainError);
    CHECK_THROWS_AS(acoth(ComplexInf), DomainError);
    CHECK_THROWS_AS(acsch(ComplexInf), DomainError);
    CHECK_THROWS_AS(exp(ComplexInf), DomainError);
    CHECK_THROWS_AS(erf(ComplexInf), DomainError);

    r1 = atan(Inf);
    REQUIRE(eq(*r1, *div(pi, integer(2))));

    r1 = acot(Inf);
    REQUIRE(eq(*r1, *zero));

    r1 = abs(ComplexInf);
    REQUIRE(eq(*r1, *a));

    r1 = atan(NegInf);
    REQUIRE(eq(*r1, *mul(minus_one, (div(pi, integer(2))))));

    r1 = acot(Inf);
    REQUIRE(eq(*r1, *zero));

    r1 = csch(Inf);
    REQUIRE(eq(*r1, *zero));

    r1 = cosh(Inf);
    REQUIRE(eq(*r1, *Inf));

    r1 = sech(Inf);
    REQUIRE(eq(*r1, *zero));

    r1 = tanh(Inf);
    REQUIRE(eq(*r1, *one));

    r1 = tanh(NegInf);
    REQUIRE(eq(*r1, *minus_one));

    r1 = coth(Inf);
    REQUIRE(eq(*r1, *one));

    r1 = coth(NegInf);
    REQUIRE(eq(*r1, *minus_one));

    r1 = asinh(Inf);
    REQUIRE(eq(*r1, *Inf));

    r1 = acsch(Inf);
    REQUIRE(eq(*r1, *zero));

    r1 = acosh(NegInf);
    REQUIRE(eq(*r1, *Inf));

    r1 = atanh(Inf);
    REQUIRE(eq(*r1, *mul(minus_one, div(mul(pi, I), integer(2)))));

    r1 = atanh(NegInf);
    REQUIRE(eq(*r1, *div(mul(pi, I), integer(2))));

    r1 = acoth(NegInf);
    REQUIRE(eq(*r1, *zero));

    r1 = abs(ComplexInf);
    REQUIRE(eq(*r1, *a));

    r1 = gamma(Inf);
    REQUIRE(eq(*r1, *a));

    r1 = sinh(NegInf);
    REQUIRE(eq(*r1, *b));

    r1 = exp(NegInf);
    REQUIRE(eq(*r1, *zero));

    r1 = asech(Inf);
    r2 = mul(mul(I, pi), div(one, integer(2)));
    REQUIRE(eq(*r1, *r2));

    r1 = exp(NegInf);
    REQUIRE(eq(*r1, *zero));

    r1 = erf(Inf);
    REQUIRE(eq(*r1, *one));

    r1 = erfc(Inf);
    REQUIRE(eq(*r1, *zero));

    r1 = erf(NegInf);
    REQUIRE(eq(*r1, *minus_one));

    r1 = erfc(NegInf);
    REQUIRE(eq(*r1, *integer(2)));

    r1 = exp(Inf);
    REQUIRE(eq(*r1, *Inf));

    r1 = gamma(NegInf);
    REQUIRE(eq(*r1, *ComplexInf));
}
