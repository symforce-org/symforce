#include "catch.hpp"
#include <cmath>
#include <iostream>

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/integer.h>
#include <symengine/rational.h>
#include <symengine/complex.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/visitor.h>
#include <symengine/eval_double.h>
#include <symengine/eval_mpfr.h>
#include <symengine/eval_mpc.h>
#include <symengine/eval.h>
#include <symengine/symengine_rcp.h>
#include <symengine/real_double.h>
#include <symengine/complex_double.h>
#include <symengine/symengine_casts.h>
#ifdef HAVE_SYMENGINE_MPFR
#include <symengine/real_mpfr.h>
#endif // HAVE_SYMENGINE_MPFR
#ifdef HAVE_SYMENGINE_MPC
#include <symengine/complex_mpc.h>
#endif // HAVE_SYMENGINE_MPC
#include <symengine/symengine_exception.h>

using SymEngine::Basic;
using SymEngine::Complex;
using SymEngine::complex_double;
using SymEngine::ComplexDouble;
using SymEngine::constant;
using SymEngine::real_double;
using SymEngine::RealDouble;
using SymEngine::SymEngineException;
#ifdef HAVE_SYMENGINE_MPFR
using SymEngine::real_mpfr;
using SymEngine::RealMPFR;
#endif // HAVE_SYMENGINE_MPFR
#ifdef HAVE_SYMENGINE_MPC
using SymEngine::complex_mpc;
using SymEngine::ComplexMPC;
#endif // HAVE_SYMENGINE_MPC
using SymEngine::acos;
using SymEngine::acosh;
using SymEngine::acot;
using SymEngine::acoth;
using SymEngine::acsc;
using SymEngine::asec;
using SymEngine::asin;
using SymEngine::asinh;
using SymEngine::atan;
using SymEngine::atanh;
using SymEngine::cos;
using SymEngine::cosh;
using SymEngine::cot;
using SymEngine::coth;
using SymEngine::csc;
using SymEngine::down_cast;
using SymEngine::E;
using SymEngine::erf;
using SymEngine::EulerGamma;
using SymEngine::EvalfDomain;
using SymEngine::I;
using SymEngine::integer;
using SymEngine::integer_class;
using SymEngine::levi_civita;
using SymEngine::log;
using SymEngine::loggamma;
using SymEngine::max;
using SymEngine::min;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::Rational;
using SymEngine::rational_class;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::sec;
using SymEngine::sin;
using SymEngine::sinh;
using SymEngine::sub;
using SymEngine::symbol;
using SymEngine::tan;
using SymEngine::tanh;
using SymEngine::vec_basic;
using SymEngine::zero;
using SymEngine::zeta;

TEST_CASE("evalf: real_double", "[evalf]")
{
    RCP<const Basic> r1, r2;
    r1 = sin(integer(2));
    r2 = evalf(*r1, 53, EvalfDomain::Real);
    REQUIRE(r2->get_type_code() == SymEngine::SYMENGINE_REAL_DOUBLE);
    double d1 = 0.909297;
    double d2 = (down_cast<const RealDouble &>(*r2)).as_double();
    d1 = fabs(d1 - d2);
    d2 = 0.000001;
    REQUIRE(d1 < d2);
}

TEST_CASE("evalf: symbols", "[evalf]")
{
    RCP<const Basic> r1, r2, r3, x;
    x = symbol("x");
    r1 = add(add(sin(integer(2)), pi), x);
    r2 = evalf(*r1, 53, EvalfDomain::Symbolic);
    r3 = sub(r2, x);
    REQUIRE(r3->get_type_code() == SymEngine::SYMENGINE_REAL_DOUBLE);
    double d1 = 4.05089;
    double d2 = (down_cast<const RealDouble &>(*r3)).as_double();
    d1 = fabs(d1 - d2);
    d2 = 0.000001;
    REQUIRE(d1 < d2);
}

#ifdef HAVE_SYMENGINE_MPFR
TEST_CASE("evalf: real_mpfr", "[evalf]")
{
    RCP<const Basic> r1, r2;
    r1 = mul(pi, integer(integer_class("1963319607")));
    r2 = integer(integer_class("6167950454"));
    r1 = sub(r1, r2);

    r2 = evalf(*r1, 100, EvalfDomain::Real);
    REQUIRE(r2->get_type_code() == SymEngine::SYMENGINE_REAL_MPFR);
    REQUIRE(!(down_cast<const RealMPFR &>(*r2)).is_zero());

    r2 = evalf(*r1, 60, EvalfDomain::Real);
    REQUIRE(r2->get_type_code() == SymEngine::SYMENGINE_REAL_MPFR);
    REQUIRE((down_cast<const RealMPFR &>(*r2)).is_zero());
}
#endif // HAVE_SYMENGINE_MPFR

TEST_CASE("evalf: complex_double", "[evalf]")
{
    RCP<const Basic> r1, r2;
    r1 = add(sin(integer(4)), mul(sin(integer(3)), I));
    // r1 = sin(4) + sin(3)i
    r2 = add(sin(integer(2)), mul(sin(integer(7)), I));
    // r2 = sin(2) + sin(7)i

    r1 = mul(r1, r2);
    // r1 = (sin(4) + sin(3)i) * (sin(2) + sin(7)i)

    r2 = evalf(*r1, 53, EvalfDomain::Complex);
    REQUIRE(r2->get_type_code() == SymEngine::SYMENGINE_COMPLEX_DOUBLE);

    r1 = (down_cast<const ComplexDouble &>(*r2)).real_part();
    REQUIRE(r1->get_type_code() == SymEngine::SYMENGINE_REAL_DOUBLE);

    double d1 = (rcp_static_cast<const RealDouble>(
                     (down_cast<const ComplexDouble &>(*r2)).real_part()))
                    ->as_double();
    double d2 = -0.780872515;
    d2 = fabs(d1 - d2);
    d1 = 0.000001;
    REQUIRE(d2 < d1);

    r1 = (down_cast<const ComplexDouble &>(*r2)).imaginary_part();
    REQUIRE(r1->get_type_code() == SymEngine::SYMENGINE_REAL_DOUBLE);
    d1 = (rcp_static_cast<const RealDouble>(
              (down_cast<const ComplexDouble &>(*r2)).imaginary_part()))
             ->as_double();
    d2 = -0.3688890370;
    d2 = fabs(d1 - d2);
    d1 = 0.000001;
    REQUIRE(d2 < d1);
}

#ifdef HAVE_SYMENGINE_MPC
TEST_CASE("evalf: complex_mpc", "[evalf]")
{
    RCP<const Basic> r1, r2, c1, c2;
    r1 = mul(pi, integer(integer_class("1963319607")));
    r2 = integer(integer_class("6167950454"));
    c1 = add(r1, mul(r1, I));
    c2 = add(r2, mul(r2, I));

    r1 = evalf(*c1, 100, EvalfDomain::Complex);

    REQUIRE(static_cast<SymEngine::TypeID>(r1->get_type_code())
            == SymEngine::SYMENGINE_COMPLEX_MPC);
    REQUIRE(!(down_cast<const ComplexMPC &>(*r1)).is_zero());
}
#endif // HAVE_SYMENGINE_MPC
