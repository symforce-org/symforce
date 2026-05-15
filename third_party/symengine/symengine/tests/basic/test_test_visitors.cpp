#include "catch.hpp"
#include <symengine/test_visitors.h>
#include <symengine/sets.h>

using SymEngine::abs;
using SymEngine::Assumptions;
using SymEngine::Basic;
using SymEngine::boolTrue;
using SymEngine::Catalan;
using SymEngine::Complex;
using SymEngine::complexes;
using SymEngine::constant;
using SymEngine::cos;
using SymEngine::E;
using SymEngine::EulerGamma;
using SymEngine::GoldenRatio;
using SymEngine::Inf;
using SymEngine::integer;
using SymEngine::integers;
using SymEngine::interval;
using SymEngine::Nan;
using SymEngine::Number;
using SymEngine::pi;
using SymEngine::Rational;
using SymEngine::rationals;
using SymEngine::RCP;
using SymEngine::real_double;
using SymEngine::reals;
using SymEngine::Set;
using SymEngine::sin;
using SymEngine::symbol;
using SymEngine::Symbol;
using SymEngine::SymEngineException;
using SymEngine::tribool;

TEST_CASE("Test is zero", "[is_zero]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rat2 = Rational::from_two_ints(*integer(0), *integer(1));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);

    REQUIRE(is_zero(*x) == tribool::indeterminate);
    REQUIRE(is_true(is_zero(*i1)));
    REQUIRE(is_zero(*i2) == tribool::trifalse);
    REQUIRE(is_zero(*rat1) == tribool::trifalse);
    REQUIRE(is_zero(*rat2) == tribool::tritrue);
    REQUIRE_THROWS_AS(is_zero(*s1), SymEngineException);
    REQUIRE(is_zero(*c1) == tribool::trifalse);
    REQUIRE_THROWS_AS(is_zero(*rel1), SymEngineException);
    REQUIRE(is_zero(*pi) == tribool::trifalse);
    REQUIRE(is_zero(*d1) == tribool::indeterminate);
    REQUIRE_THROWS_AS(is_zero(*boolTrue), SymEngineException);
    REQUIRE(is_zero(*pi) == tribool::trifalse);
    REQUIRE(is_indeterminate(is_zero(*abs(x))));
    REQUIRE(is_indeterminate(is_zero(*conjugate(x))));
    REQUIRE(is_indeterminate(is_zero(*sign(x))));
    REQUIRE(is_indeterminate(is_zero(*primepi(x))));

    const auto a1 = Assumptions({Eq(x, integer(0))});
    REQUIRE(is_true(is_zero(*x, &a1)));

    const auto a2 = Assumptions({Gt(x, integer(0))});
    REQUIRE(is_false(is_zero(*x, &a2)));

    const auto a3 = Assumptions({Ne(x, integer(0))});
    REQUIRE(is_false(is_zero(*x, &a3)));
}

TEST_CASE("Test is nonzero", "[is_nonzero]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rat2 = Rational::from_two_ints(*integer(0), *integer(1));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);

    REQUIRE(is_indeterminate(is_nonzero(*x)));
    REQUIRE(is_false(is_nonzero(*i1)));
    REQUIRE(is_true(is_nonzero(*i2)));
    REQUIRE(is_true(is_nonzero(*rat1)));
    REQUIRE(is_false(is_nonzero(*rat2)));
    REQUIRE_THROWS_AS(is_nonzero(*s1), SymEngineException);
    REQUIRE(is_true(is_nonzero(*c1)));
    REQUIRE_THROWS_AS(is_nonzero(*rel1), SymEngineException);
    REQUIRE(is_true(is_nonzero(*pi)));
    REQUIRE(is_indeterminate(is_nonzero(*d1)));
    REQUIRE_THROWS_AS(is_nonzero(*boolTrue), SymEngineException);
    REQUIRE(is_true(is_nonzero(*pi)));
}

TEST_CASE("Test is positive", "[is_positive]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Number> i3 = integer(-2);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rat2 = Rational::from_two_ints(*integer(0), *integer(1));
    RCP<const Basic> rat3 = Rational::from_two_ints(*integer(-1), *integer(3));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Number> c2 = Complex::from_two_nums(*i3, *i3);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);

    REQUIRE(is_indeterminate(is_positive(*x)));
    REQUIRE(is_false(is_positive(*i1)));
    REQUIRE(is_true(is_positive(*i2)));
    REQUIRE(is_false(is_positive(*i3)));
    REQUIRE(is_true(is_positive(*rat1)));
    REQUIRE(is_false(is_positive(*rat2)));
    REQUIRE(is_false(is_positive(*rat3)));
    REQUIRE_THROWS_AS(is_positive(*s1), SymEngineException);
    REQUIRE(is_false(is_positive(*c1)));
    REQUIRE(is_false(is_positive(*c2)));
    REQUIRE_THROWS_AS(is_positive(*rel1), SymEngineException);
    REQUIRE(is_true(is_positive(*pi)));
    REQUIRE(is_indeterminate(is_positive(*d1)));
    REQUIRE_THROWS_AS(is_positive(*boolTrue), SymEngineException);

    const auto a1 = Assumptions({Lt(x, integer(0))});
    REQUIRE(is_false(is_positive(*x, &a1)));
    REQUIRE(is_false(is_positive(*add(x, integer(-1)), &a1)));
    REQUIRE(is_indeterminate(is_positive(*add(x, integer(2)), &a1)));

    const auto a2 = Assumptions({Lt(x, integer(1))});
    REQUIRE(is_indeterminate(is_positive(*x, &a2)));

    const auto a3 = Assumptions({Gt(x, integer(1))});
    REQUIRE(is_true(is_positive(*x, &a3)));
    REQUIRE(is_true(is_positive(*add(x, integer(2)), &a3)));
    REQUIRE(is_indeterminate(is_positive(*add(x, integer(-1)), &a3)));

    const auto a4 = Assumptions({reals()->contains(x), reals()->contains(y)});
    REQUIRE(is_indeterminate(is_positive(*add(x, y), &a4)));

    const auto a5 = Assumptions({Lt(x, integer(0)), Lt(y, integer(0))});
    REQUIRE(is_false(is_positive(*add(x, mul(y, integer(2))), &a5)));
    REQUIRE(is_true(
        is_positive(*add(mul(x, integer(-2)), mul(y, integer(-2))), &a5)));

    const auto a6 = Assumptions({Gt(x, integer(0)), Gt(y, integer(0))});
    REQUIRE(is_false(
        is_positive(*add(mul(x, integer(-2)), mul(y, integer(-2))), &a6)));
    REQUIRE(is_true(
        is_positive(*add(mul(x, integer(2)), mul(y, integer(2))), &a6)));
}

TEST_CASE("Test is non positive", "[is_nonpositive]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Number> i3 = integer(-2);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rat2 = Rational::from_two_ints(*integer(0), *integer(1));
    RCP<const Basic> rat3 = Rational::from_two_ints(*integer(-1), *integer(3));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Number> c2 = Complex::from_two_nums(*i3, *i3);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);

    REQUIRE(is_indeterminate(is_nonpositive(*x)));
    REQUIRE(is_true(is_nonpositive(*i1)));
    REQUIRE(is_false(is_nonpositive(*i2)));
    REQUIRE(is_true(is_nonpositive(*i3)));
    REQUIRE(is_false(is_nonpositive(*rat1)));
    REQUIRE(is_true(is_nonpositive(*rat2)));
    REQUIRE(is_true(is_nonpositive(*rat3)));
    REQUIRE_THROWS_AS(is_nonpositive(*s1), SymEngineException);
    REQUIRE(is_false(is_nonpositive(*c1)));
    REQUIRE(is_false(is_nonpositive(*c2)));
    REQUIRE_THROWS_AS(is_nonpositive(*rel1), SymEngineException);
    REQUIRE(is_false(is_nonpositive(*pi)));
    REQUIRE(is_indeterminate(is_nonpositive(*d1)));
    REQUIRE_THROWS_AS(is_nonpositive(*boolTrue), SymEngineException);

    const auto a1 = Assumptions({Lt(x, integer(0))});
    REQUIRE(is_true(is_nonpositive(*x, &a1)));

    const auto a2 = Assumptions({Lt(x, integer(1))});
    REQUIRE(is_indeterminate(is_nonpositive(*x, &a2)));

    const auto a3 = Assumptions({Gt(x, integer(1))});
    REQUIRE(is_false(is_nonpositive(*x, &a3)));
}

TEST_CASE("Test is negative", "[is_negative]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Number> i3 = integer(-2);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rat2 = Rational::from_two_ints(*integer(0), *integer(1));
    RCP<const Basic> rat3 = Rational::from_two_ints(*integer(-1), *integer(3));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Number> c2 = Complex::from_two_nums(*i3, *i3);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);

    REQUIRE(is_indeterminate(is_negative(*x)));
    REQUIRE(is_false(is_negative(*i1)));
    REQUIRE(is_false(is_negative(*i2)));
    REQUIRE(is_true(is_negative(*i3)));
    REQUIRE(is_false(is_negative(*rat1)));
    REQUIRE(is_false(is_negative(*rat2)));
    REQUIRE(is_true(is_negative(*rat3)));
    REQUIRE_THROWS_AS(is_negative(*s1), SymEngineException);
    REQUIRE(is_false(is_negative(*c1)));
    REQUIRE(is_false(is_negative(*c2)));
    REQUIRE_THROWS_AS(is_negative(*rel1), SymEngineException);
    REQUIRE(is_false(is_negative(*pi)));
    REQUIRE(is_indeterminate(is_negative(*d1)));
    REQUIRE_THROWS_AS(is_negative(*boolTrue), SymEngineException);

    const auto a1 = Assumptions({Lt(x, integer(0))});
    REQUIRE(is_true(is_negative(*x, &a1)));

    const auto a2 = Assumptions({Lt(x, integer(1))});
    REQUIRE(is_indeterminate(is_negative(*x, &a2)));

    const auto a3 = Assumptions({Gt(x, integer(1))});
    REQUIRE(is_false(is_negative(*x, &a3)));
}

TEST_CASE("Test is nonnegative", "[is_nonnegative]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Number> i3 = integer(-2);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rat2 = Rational::from_two_ints(*integer(0), *integer(1));
    RCP<const Basic> rat3 = Rational::from_two_ints(*integer(-1), *integer(3));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Number> c2 = Complex::from_two_nums(*i3, *i3);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);

    REQUIRE(is_indeterminate(is_nonnegative(*x)));
    REQUIRE(is_true(is_nonnegative(*i1)));
    REQUIRE(is_true(is_nonnegative(*i2)));
    REQUIRE(is_false(is_nonnegative(*i3)));
    REQUIRE(is_true(is_nonnegative(*rat1)));
    REQUIRE(is_true(is_nonnegative(*rat2)));
    REQUIRE(is_false(is_nonnegative(*rat3)));
    REQUIRE_THROWS_AS(is_nonnegative(*s1), SymEngineException);
    REQUIRE(is_false(is_nonnegative(*c1)));
    REQUIRE(is_false(is_nonnegative(*c2)));
    REQUIRE_THROWS_AS(is_nonnegative(*rel1), SymEngineException);
    REQUIRE(is_true(is_nonnegative(*pi)));
    REQUIRE(is_indeterminate(is_nonnegative(*d1)));
    REQUIRE_THROWS_AS(is_nonnegative(*boolTrue), SymEngineException);

    const auto a1 = Assumptions({Lt(x, integer(0))});
    REQUIRE(is_false(is_nonnegative(*x, &a1)));

    const auto a2 = Assumptions({Lt(x, integer(1))});
    REQUIRE(is_indeterminate(is_nonnegative(*x, &a2)));

    const auto a3 = Assumptions({Gt(x, integer(1))});
    REQUIRE(is_true(is_nonnegative(*x, &a3)));
}

TEST_CASE("Test is_integer", "[is_integer]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);
    RCP<const Basic> e1 = add(x, x);
    RCP<const Basic> e2 = add(x, Inf);
    RCP<const Basic> e3 = add(x, c1);
    RCP<const Basic> e4 = add(x, y);
    RCP<const Basic> e5 = mul(mul(i2, x), y);

    REQUIRE(is_indeterminate(is_integer(*x)));
    REQUIRE(is_true(is_integer(*i1)));
    REQUIRE(is_true(is_integer(*i2)));
    REQUIRE(is_false(is_integer(*rat1)));
    REQUIRE(is_false(is_integer(*s1)));
    REQUIRE(is_false(is_integer(*c1)));
    REQUIRE(is_false(is_integer(*rel1)));
    REQUIRE(is_false(is_integer(*pi)));
    REQUIRE(is_indeterminate(is_integer(*d1)));
    REQUIRE(is_false(is_integer(*boolTrue)));
    REQUIRE(is_indeterminate(is_integer(*e1)));
    REQUIRE(is_indeterminate(is_integer(*e2)));
    REQUIRE(is_indeterminate(is_integer(*e3)));
    REQUIRE(is_false(is_integer(*Inf)));
    REQUIRE(is_false(is_integer(*Nan)));
    REQUIRE(is_true(is_integer(*kronecker_delta(i2, x))));

    const auto a1 = Assumptions({reals()->contains(x)});
    REQUIRE(is_indeterminate(is_integer(*x, &a1)));

    const auto a2 = Assumptions({integers()->contains(x)});
    REQUIRE(is_true(is_integer(*x, &a2)));
    REQUIRE(is_true(is_integer(*conjugate(x), &a2)));

    const auto a3
        = Assumptions({rationals()->contains(x), rationals()->contains(y)});
    REQUIRE(is_indeterminate(is_integer(*e4, &a3)));
    REQUIRE(is_indeterminate(is_integer(*e5, &a3)));

    const auto a4
        = Assumptions({integers()->contains(x), integers()->contains(y)});
    REQUIRE(is_true(is_integer(*e4, &a4)));
    REQUIRE(is_true(is_integer(*e5, &a4)));
}

TEST_CASE("Test is_real", "[is_real]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);
    RCP<const Basic> e1 = add(x, x);
    RCP<const Basic> e2 = add(x, Inf);
    RCP<const Basic> e3 = add(x, c1);
    RCP<const Basic> e4 = add(x, y);
    RCP<const Basic> e5 = pow(integer(2), x);
    RCP<const Basic> e6 = pow(integer(-1), x);
    RCP<const Basic> e7 = mul(x, y);
    RCP<const Basic> e8 = mul(c1, x);
    RCP<const Basic> e9 = pow(i2, c1);

    REQUIRE(is_indeterminate(is_real(*x)));
    REQUIRE(is_true(is_real(*i1)));
    REQUIRE(is_true(is_real(*i2)));
    REQUIRE(is_true(is_real(*rat1)));
    REQUIRE(is_false(is_real(*s1)));
    REQUIRE(is_false(is_real(*c1)));
    REQUIRE(is_false(is_real(*rel1)));
    REQUIRE(is_true(is_real(*pi)));
    REQUIRE(is_indeterminate(is_real(*d1)));
    REQUIRE(is_false(is_real(*boolTrue)));
    REQUIRE(is_indeterminate(is_real(*e1)));
    REQUIRE(is_indeterminate(is_real(*e2)));
    REQUIRE(is_indeterminate(is_real(*e3)));
    REQUIRE(is_indeterminate(is_real(*e5)));
    REQUIRE(is_indeterminate(is_real(*e6)));
    REQUIRE(is_indeterminate(is_real(*e7)));
    REQUIRE(is_indeterminate(is_real(*e8)));
    REQUIRE(is_indeterminate(is_real(*e9)));
    REQUIRE(is_false(is_real(*Inf)));
    REQUIRE(is_false(is_real(*Nan)));

    const auto a1 = Assumptions({reals()->contains(x)});
    REQUIRE(is_true(is_real(*x, &a1)));

    const auto a2 = Assumptions({integers()->contains(x)});
    REQUIRE(is_true(is_real(*x, &a2)));

    const auto a3
        = Assumptions({rationals()->contains(x), rationals()->contains(y)});
    REQUIRE(is_true(is_real(*e4, &a3)));

    const auto a4 = Assumptions({reals()->contains(x)});
    REQUIRE(is_true(is_real(*e5, &a4)));

    const auto a5 = Assumptions({complexes()->contains(x)});
    REQUIRE(is_indeterminate(is_real(*e5, &a5)));

    const auto a6 = Assumptions({reals()->contains(x), reals()->contains(y)});
    REQUIRE(is_true(is_real(*e7, &a6)));

    const auto a7 = Assumptions({reals()->contains(x)});
    REQUIRE(is_false(is_real(*e8, &a7)));
}

TEST_CASE("Test is_complex", "[is_complex]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Number> i3 = integer(1);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Number> c2 = Complex::from_two_nums(*integer(0), *integer(-1));
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);
    RCP<const Basic> e1 = add(x, x);
    RCP<const Basic> e2 = add(x, Inf);
    RCP<const Basic> e3 = add(x, c1);
    RCP<const Basic> e4 = add(x, y);
    RCP<const Basic> e5 = pow(x, y);
    RCP<const Basic> e6 = pow(x, i2);
    RCP<const Basic> e7 = mul(x, y);
    RCP<const Basic> e8 = mul(i2, mul(x, y));
    RCP<const Basic> e9 = mul(i2, mul(x, add(x, y)));

    REQUIRE(is_indeterminate(is_complex(*x)));
    REQUIRE(is_true(is_complex(*i1)));
    REQUIRE(is_true(is_complex(*i2)));
    REQUIRE(is_true(is_complex(*rat1)));
    REQUIRE(is_false(is_complex(*s1)));
    REQUIRE(is_true(is_complex(*c1)));
    REQUIRE(is_false(is_complex(*rel1)));
    REQUIRE(is_true(is_complex(*pi)));
    REQUIRE(is_indeterminate(is_complex(*d1)));
    REQUIRE(is_false(is_complex(*boolTrue)));
    REQUIRE(is_indeterminate(is_complex(*e1)));
    REQUIRE(is_false(is_complex(*e2)));
    REQUIRE(is_indeterminate(is_complex(*e3)));
    REQUIRE(is_false(is_complex(*Inf)));
    REQUIRE(is_false(is_complex(*Nan)));
    REQUIRE(is_indeterminate(is_complex(*e5)));
    REQUIRE(is_indeterminate(is_complex(*e6)));
    REQUIRE(is_indeterminate(is_complex(*e7)));
    REQUIRE(is_indeterminate(is_complex(*e8)));
    REQUIRE(is_indeterminate(is_complex(*e9)));
    REQUIRE(is_indeterminate(is_complex(*cos(x))));
    REQUIRE(is_indeterminate(is_complex(*sin(e9))));
    REQUIRE(is_indeterminate(is_complex(*sinh(e4))));
    REQUIRE(is_indeterminate(is_complex(*cosh(e4))));
    REQUIRE(is_indeterminate(is_complex(*cos(sin(cos(x))))));
    REQUIRE(is_indeterminate(is_complex(*tan(x))));
    REQUIRE(is_indeterminate(is_complex(*atan(x))));
    REQUIRE(is_indeterminate(is_complex(*atanh(x))));
    REQUIRE(is_indeterminate(is_complex(*acoth(x))));
    REQUIRE(is_false(is_complex(*acoth(i3))));
    REQUIRE(is_true(is_complex(*atan(c1))));
    REQUIRE(is_false(is_complex(*atan(c2))));
    REQUIRE(is_indeterminate(is_complex(*cot(x))));
    REQUIRE(is_indeterminate(is_complex(*acot(x))));
    REQUIRE(is_false(is_complex(*acot(c2))));
    REQUIRE(is_indeterminate(is_complex(*csc(x))));
    REQUIRE(is_indeterminate(is_complex(*csc(x))));
    REQUIRE(is_indeterminate(is_complex(*asec(x))));
    REQUIRE(is_true(is_complex(*asec(i2))));
    REQUIRE(is_false(is_complex(*asec(i1))));
    REQUIRE(is_indeterminate(is_complex(*acsc(x))));
    REQUIRE(is_indeterminate(is_complex(*acsch(x))));
    REQUIRE(is_indeterminate(is_complex(*sec(x))));
    REQUIRE(is_false(is_complex(*asech(i1))));
    REQUIRE(is_indeterminate(is_complex(*sign(x))));
    REQUIRE(is_indeterminate(is_complex(*floor(x))));
    REQUIRE(is_indeterminate(is_complex(*ceiling(x))));
    REQUIRE(is_indeterminate(is_complex(*abs(x))));
    REQUIRE(is_indeterminate(is_complex(*log(x))));
    REQUIRE(is_true(is_complex(*log(i2))));
    REQUIRE(is_true(is_complex(*kronecker_delta(i2, x))));

    const auto a1 = Assumptions({reals()->contains(x)});
    REQUIRE(is_true(is_complex(*x, &a1)));
    REQUIRE(is_indeterminate(is_complex(*e5, &a1)));
    REQUIRE(is_true(is_complex(*e6, &a1)));
    REQUIRE(is_indeterminate(is_complex(*tan(x), &a1)));
    // For atan and acot we could now that these are true
    // but is_zero cannot (yet) see that (x - i) can never be zero
    // if x is real.
    REQUIRE(is_indeterminate(is_complex(*atan(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*acot(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*cot(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*csc(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*acsch(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*sec(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*asech(x), &a1)));
    REQUIRE(is_true(is_complex(*sign(x), &a1)));
    REQUIRE(is_true(is_complex(*floor(x), &a1)));
    REQUIRE(is_true(is_complex(*ceiling(x), &a1)));
    REQUIRE(is_true(is_complex(*abs(x), &a1)));
    REQUIRE(is_true(is_complex(*conjugate(x), &a1)));
    REQUIRE(is_indeterminate(is_complex(*log(x), &a1)));

    const auto a2 = Assumptions({integers()->contains(x)});
    REQUIRE(is_true(is_complex(*x, &a2)));
    REQUIRE(is_indeterminate(is_complex(*e7, &a2)));

    const auto a3
        = Assumptions({rationals()->contains(x), rationals()->contains(y)});
    REQUIRE(is_true(is_complex(*e4, &a3)));
    REQUIRE(is_true(is_complex(*e5, &a3)));
    REQUIRE(is_true(is_complex(*e7, &a3)));
    REQUIRE(is_true(is_complex(*e8, &a3)));
    REQUIRE(is_true(is_complex(*e9, &a3)));
    REQUIRE(is_true(is_complex(*cos(x), &a3)));
    REQUIRE(is_true(is_complex(*acos(x), &a3)));
    REQUIRE(is_true(is_complex(*asin(x), &a3)));
    REQUIRE(is_true(is_complex(*cos(sin(cos(x))), &a3)));
    REQUIRE(is_true(is_complex(*sin(e9), &a3)));
    REQUIRE(is_true(is_complex(*sinh(e4), &a3)));
    REQUIRE(is_true(is_complex(*cosh(e4), &a3)));
}

TEST_CASE("Test is_polynomial", "[is_polynomial]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Number> i1 = integer(1);
    RCP<const Number> i2 = integer(3);
    RCP<const Number> i3 = integer(-2);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> e1 = add(x, i1);
    RCP<const Basic> e2 = add(x, y);
    RCP<const Basic> e3 = add(add(x, pi), y);
    RCP<const Basic> e4 = mul(x, y);
    RCP<const Basic> e5 = mul(i2, mul(x, y));
    RCP<const Basic> e6 = mul(i2, mul(x, add(x, y)));
    RCP<const Basic> e7 = pow(x, y);
    RCP<const Basic> e8 = pow(x, i2);
    RCP<const Basic> e9 = pow(x, rat1);
    RCP<const Basic> e10 = pow(integer(2), x);
    RCP<const Basic> e11 = div(i1, x);
    RCP<const Basic> e12 = pow(x, i3);
    RCP<const Basic> e13 = pow(x, x);
    RCP<const Basic> e14 = pow(x, mul(i3, y));
    RCP<const Basic> e15 = pow(mul(i2, x), y);
    RCP<const Basic> e16 = add(add(mul(x, x), mul(i2, x)), i3);
    RCP<const Basic> e17 = sqrt(x);
    RCP<const Basic> e18 = pow(sqrt(x), i2);
    RCP<const Basic> e19 = sqrt(y);
    RCP<const Basic> e20 = add(add(mul(x, x), mul(i2, mul(x, e19))), i3);
    RCP<const Basic> e21
        = add(add(mul(mul(x, x), mul(y, y)), mul(x, pow(y, i2))), exp(i3));
    RCP<const Basic> e22
        = add(add(mul(mul(x, x), mul(y, y)), mul(x, pow(y, i2))), exp(x));
    RCP<const Basic> e23 = sin(x);
    RCP<const Basic> b1 = interval(i1, i2);
    RCP<const Basic> rel1 = Eq(x, i1);

    REQUIRE(is_polynomial(*x));
    REQUIRE(is_polynomial(*i1));
    REQUIRE(is_polynomial(*rat1));
    REQUIRE(is_polynomial(*pi));
    REQUIRE(is_polynomial(*e1));
    REQUIRE(is_polynomial(*e2));
    REQUIRE(is_polynomial(*e3));
    REQUIRE(is_polynomial(*e4));
    REQUIRE(is_polynomial(*e5));
    REQUIRE(is_polynomial(*e6));
    REQUIRE(!is_polynomial(*e7));
    REQUIRE(is_polynomial(*e8));
    REQUIRE(!is_polynomial(*e9));
    REQUIRE(!is_polynomial(*e10));
    REQUIRE(!is_polynomial(*e11));
    REQUIRE(!is_polynomial(*b1));
    REQUIRE(is_polynomial(*rat1, {x, y, z}));
    REQUIRE(is_polynomial(*pi, {x, y, z}));
    REQUIRE(is_polynomial(*x, {x}));
    REQUIRE(is_polynomial(*x, {y}));
    REQUIRE(is_polynomial(*e8, {x}));
    REQUIRE(!is_polynomial(*e12, {x}));
    REQUIRE(is_polynomial(*e12, {y}));
    REQUIRE(!is_polynomial(*e10, {x}));
    REQUIRE(is_polynomial(*e10, {y}));
    REQUIRE(!is_polynomial(*e7, {x}));
    REQUIRE(!is_polynomial(*e7, {y}));
    REQUIRE(!is_polynomial(*e13, {x}));
    REQUIRE(!is_polynomial(*e14, {x}));
    REQUIRE(!is_polynomial(*e15, {x}));
    REQUIRE(is_polynomial(*e16, {x}));
    REQUIRE(is_polynomial(*e16, {y}));
    REQUIRE(is_polynomial(*e16));
    REQUIRE(!is_polynomial(*e17, {x}));
    REQUIRE(is_polynomial(*e17, {y}));
    REQUIRE(!is_polynomial(*e17));
    REQUIRE(!is_polynomial(*e18, {x}));
    REQUIRE(is_polynomial(*e20, {x}));
    REQUIRE(!is_polynomial(*e20, {y}));
    REQUIRE(is_polynomial(*e21));
    REQUIRE(!is_polynomial(*e22));
    REQUIRE(is_polynomial(*e21, {x, y}));
    REQUIRE(!is_polynomial(*e22, {x, y}));
    REQUIRE(!is_polynomial(*e23));
    REQUIRE(is_polynomial(*e23, {y}));
    REQUIRE(!is_polynomial(*rel1));
}

TEST_CASE("Test is_rational", "[is_rational]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> irr1 = real_double(2.0);
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);
    RCP<const Basic> e1 = add(x, x);
    RCP<const Basic> e2 = add(x, Inf);
    RCP<const Basic> e3 = add(x, c1);

    REQUIRE(is_indeterminate(is_rational(*x)));
    REQUIRE(is_true(is_rational(*i1)));
    REQUIRE(is_true(is_rational(*i2)));
    REQUIRE(is_true(is_rational(*rat1)));
    REQUIRE(is_false(is_rational(*s1)));
    REQUIRE(is_false(is_rational(*c1)));
    REQUIRE(is_false(is_rational(*rel1)));
    REQUIRE(is_false(is_rational(*pi)));
    REQUIRE(is_indeterminate(is_rational(*d1)));
    REQUIRE(is_false(is_rational(*boolTrue)));
    REQUIRE(is_indeterminate(is_rational(*e1)));
    REQUIRE(is_indeterminate(is_rational(*e2)));
    REQUIRE(is_indeterminate(is_rational(*e3)));
    REQUIRE(is_false(is_rational(*Inf)));
    REQUIRE(is_false(is_rational(*Nan)));
    REQUIRE(is_false(is_rational(*irr1)));
    REQUIRE(is_indeterminate(is_rational(*constant("catalan"))));
}

TEST_CASE("Test is_irrational", "[is_irrational]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Number> i1 = integer(0);
    RCP<const Number> i2 = integer(3);
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> irr1 = real_double(2.0);
    RCP<const Basic> s1 = interval(i1, i2);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Basic> rel1 = Eq(x, i1);
    RCP<const Symbol> t = symbol("t");
    RCP<const Basic> f = function_symbol("f", t);
    RCP<const Basic> d1 = f->diff(t);
    RCP<const Basic> e1 = add(x, x);
    RCP<const Basic> e2 = add(x, Inf);
    RCP<const Basic> e3 = add(x, c1);

    REQUIRE(is_indeterminate(is_irrational(*x)));
    REQUIRE(is_false(is_irrational(*i1)));
    REQUIRE(is_false(is_irrational(*i2)));
    REQUIRE(is_false(is_irrational(*rat1)));
    REQUIRE(is_false(is_irrational(*s1)));
    REQUIRE(is_false(is_irrational(*c1)));
    REQUIRE(is_false(is_irrational(*rel1)));
    REQUIRE(is_true(is_irrational(*pi)));
    REQUIRE(is_indeterminate(is_irrational(*d1)));
    REQUIRE(is_false(is_irrational(*boolTrue)));
    REQUIRE(is_indeterminate(is_irrational(*e1)));
    REQUIRE(is_indeterminate(is_irrational(*e2)));
    REQUIRE(is_indeterminate(is_irrational(*e3)));
    REQUIRE(is_false(is_irrational(*Inf)));
    REQUIRE(is_false(is_irrational(*Nan)));
    REQUIRE(is_true(is_irrational(*irr1)));
    REQUIRE(is_indeterminate(is_irrational(*constant("catalan"))));
}

TEST_CASE("Test is_finite", "[is_finite]")
{
    auto x = symbol("x");

    REQUIRE(is_false(is_finite(*Inf)));
    REQUIRE(is_true(is_infinite(*Inf)));

    REQUIRE_THROWS_AS(is_finite(*boolTrue), SymEngineException);
    REQUIRE_THROWS_AS(is_infinite(*boolTrue), SymEngineException);
    REQUIRE_THROWS_AS(is_finite(*Eq(symbol("x"), integer(1))),
                      SymEngineException);
    REQUIRE_THROWS_AS(is_infinite(*Eq(symbol("x"), integer(1))),
                      SymEngineException);
    REQUIRE_THROWS_AS(is_finite(*integers()), SymEngineException);
    REQUIRE_THROWS_AS(is_infinite(*integers()), SymEngineException);
    REQUIRE_THROWS_AS(is_finite(*Nan), SymEngineException);
    REQUIRE_THROWS_AS(is_infinite(*Nan), SymEngineException);

    REQUIRE(is_true(is_finite(*pi)));
    REQUIRE(is_false(is_infinite(*pi)));

    REQUIRE(is_true(is_finite(*integer(23))));
    REQUIRE(is_false(is_infinite(*integer(23))));

    REQUIRE(is_indeterminate(is_finite(*x)));
    REQUIRE(is_indeterminate(is_infinite(*x)));

    const auto a1 = Assumptions({reals()->contains(x)});
    REQUIRE(is_true(is_finite(*x, &a1)));
    REQUIRE(is_false(is_infinite(*x, &a1)));

    // Not yet supported
    REQUIRE(is_indeterminate(is_finite(*add(x, x))));
}

TEST_CASE("Test is_even / is_odd", "[is_even]")
{
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Number> c1 = Complex::from_two_nums(*integer(2), *integer(3));

    REQUIRE(is_true(is_odd(*integer(3))));
    REQUIRE(is_false(is_odd(*integer(2))));
    REQUIRE(is_false(is_even(*integer(3))));
    REQUIRE(is_true(is_even(*integer(2))));
    REQUIRE(is_false(is_odd(*rat1)));
    REQUIRE(is_false(is_even(*rat1)));
    REQUIRE(is_false(is_odd(*c1)));
    REQUIRE(is_false(is_even(*c1)));
}

TEST_CASE("Test is_algebraic/is_transcendental", "[is_algebraic]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> rat1 = Rational::from_two_ints(*integer(5), *integer(6));

    REQUIRE(is_true(is_algebraic(*integer(23))));
    REQUIRE(is_false(is_transcendental(*integer(23))));
    REQUIRE(is_true(is_algebraic(*rat1)));
    REQUIRE(is_false(is_transcendental(*rat1)));
    REQUIRE_THROWS_AS(is_algebraic(*boolTrue), SymEngineException);
    REQUIRE_THROWS_AS(is_transcendental(*boolTrue), SymEngineException);
    REQUIRE_THROWS_AS(is_algebraic(*integers()), SymEngineException);
    REQUIRE_THROWS_AS(is_transcendental(*integers()), SymEngineException);
    REQUIRE_THROWS_AS(is_algebraic(*Eq(x, integer(1))), SymEngineException);
    REQUIRE_THROWS_AS(is_transcendental(*Eq(x, integer(1))),
                      SymEngineException);
    REQUIRE(is_false(is_algebraic(*pi)));
    REQUIRE(is_true(is_transcendental(*pi)));
    REQUIRE(is_indeterminate(is_algebraic(*EulerGamma)));
    REQUIRE(is_indeterminate(is_transcendental(*EulerGamma)));
    REQUIRE(is_indeterminate(is_algebraic(*Catalan)));
    REQUIRE(is_indeterminate(is_transcendental(*Catalan)));
    REQUIRE(is_false(is_algebraic(*E)));
    REQUIRE(is_true(is_transcendental(*E)));
    REQUIRE(is_true(is_algebraic(*GoldenRatio)));
    REQUIRE(is_false(is_transcendental(*GoldenRatio)));

    REQUIRE(is_indeterminate(is_algebraic(*add(x, integer(1)))));

    Assumptions a = Assumptions({integers()->contains(x)});
    REQUIRE(is_true(is_algebraic(*x, &a)));

    a = Assumptions({rationals()->contains(x), rationals()->contains(y)});
    REQUIRE(is_true(is_algebraic(*x, &a)));
    REQUIRE(is_true(is_algebraic(*add(x, integer(23)), &a)));
    REQUIRE(is_true(is_algebraic(*add(x, add(y, integer(23))), &a)));
    REQUIRE(is_true(is_transcendental(*add(x, add(y, pi)), &a)));
    REQUIRE(is_indeterminate(is_transcendental(*add(x, add(E, pi)), &a)));

    a = Assumptions({reals()->contains(x)});
    REQUIRE(is_indeterminate(is_algebraic(*x, &a)));
    REQUIRE(is_indeterminate(is_algebraic(*add(x, integer(23)), &a)));

    REQUIRE(is_true(is_transcendental(*sin(integer(1)))));
    REQUIRE(is_true(is_transcendental(*sinh(integer(2)))));
    REQUIRE(is_true(is_transcendental(*lambertw(integer(2)))));
}
