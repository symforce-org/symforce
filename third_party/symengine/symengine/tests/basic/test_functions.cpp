#include "catch.hpp"
#include <chrono>

#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/complex_mpc.h>
#include <symengine/eval_double.h>
#include <symengine/eval_mpc.h>
#include <symengine/eval_mpfr.h>
#include <symengine/logic.h>
#include <symengine/symengine_exception.h>
#include <symengine/parser.h>

using SymEngine::abs;
using SymEngine::ACos;
using SymEngine::acos;
using SymEngine::ACosh;
using SymEngine::acosh;
using SymEngine::ACot;
using SymEngine::acot;
using SymEngine::ACoth;
using SymEngine::acoth;
using SymEngine::ACsc;
using SymEngine::acsc;
using SymEngine::ACsch;
using SymEngine::acsch;
using SymEngine::Add;
using SymEngine::ASec;
using SymEngine::asec;
using SymEngine::ASech;
using SymEngine::asech;
using SymEngine::ASin;
using SymEngine::asin;
using SymEngine::ASinh;
using SymEngine::asinh;
using SymEngine::ATan;
using SymEngine::atan;
using SymEngine::ATan2;
using SymEngine::atan2;
using SymEngine::ATanh;
using SymEngine::atanh;
using SymEngine::Basic;
using SymEngine::Beta;
using SymEngine::beta;
using SymEngine::Complex;
using SymEngine::complex_double;
using SymEngine::ComplexDouble;
using SymEngine::ComplexInf;
using SymEngine::conjugate;
using SymEngine::cos;
using SymEngine::Cos;
using SymEngine::Cosh;
using SymEngine::cosh;
using SymEngine::cot;
using SymEngine::Cot;
using SymEngine::Coth;
using SymEngine::coth;
using SymEngine::csc;
using SymEngine::Csc;
using SymEngine::Csch;
using SymEngine::csch;
using SymEngine::Derivative;
using SymEngine::dirichlet_eta;
using SymEngine::Dirichlet_eta;
using SymEngine::down_cast;
using SymEngine::dummy;
using SymEngine::E;
using SymEngine::erf;
using SymEngine::Erf;
using SymEngine::erfc;
using SymEngine::Erfc;
using SymEngine::EulerGamma;
using SymEngine::eval_double;
using SymEngine::exp;
using SymEngine::function_symbol;
using SymEngine::FunctionWrapper;
using SymEngine::gamma;
using SymEngine::Gamma;
using SymEngine::I;
using SymEngine::Inf;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::integer_class;
using SymEngine::is_a;
using SymEngine::kronecker_delta;
using SymEngine::KroneckerDelta;
using SymEngine::LambertW;
using SymEngine::lambertw;
using SymEngine::levi_civita;
using SymEngine::LeviCivita;
using SymEngine::log;
using SymEngine::loggamma;
using SymEngine::LogGamma;
using SymEngine::lowergamma;
using SymEngine::LowerGamma;
using SymEngine::make_rcp;
using SymEngine::max;
using SymEngine::Max;
using SymEngine::min;
using SymEngine::Min;
using SymEngine::minus_one;
using SymEngine::Mul;
using SymEngine::multinomial_coefficients;
using SymEngine::Nan;
using SymEngine::neg;
using SymEngine::NegInf;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::polygamma;
using SymEngine::PolyGamma;
using SymEngine::Pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::rational;
using SymEngine::Rational;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::real_double;
using SymEngine::RealDouble;
using SymEngine::sec;
using SymEngine::Sec;
using SymEngine::Sech;
using SymEngine::sech;
using SymEngine::sign;
using SymEngine::sin;
using SymEngine::Sin;
using SymEngine::Sinh;
using SymEngine::sinh;
using SymEngine::sqrt;
using SymEngine::Subs;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::tan;
using SymEngine::Tan;
using SymEngine::Tanh;
using SymEngine::tanh;
using SymEngine::umap_basic_num;
using SymEngine::uppergamma;
using SymEngine::UpperGamma;
using SymEngine::vec_basic;
using SymEngine::zero;
using SymEngine::zeta;
using SymEngine::Zeta;
#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
using SymEngine::get_mpz_t;
#endif
using SymEngine::ceiling;
using SymEngine::Conjugate;
using SymEngine::digamma;
using SymEngine::Eq;
using SymEngine::floor;
using SymEngine::mul;
using SymEngine::NotImplementedError;
using SymEngine::parse;
using SymEngine::rewrite_as_cos;
using SymEngine::rewrite_as_exp;
using SymEngine::rewrite_as_sin;
using SymEngine::SymEngineException;
using SymEngine::trigamma;
using SymEngine::truncate;
using SymEngine::unevaluated_expr;

using namespace SymEngine::literals;

#ifdef HAVE_SYMENGINE_MPFR
using SymEngine::eval_mpfr;
using SymEngine::mpfr_class;
using SymEngine::real_mpfr;
using SymEngine::RealMPFR;
#endif

#ifdef HAVE_SYMENGINE_MPC
using SymEngine::complex_mpc;
using SymEngine::ComplexMPC;
using SymEngine::eval_mpc;
using SymEngine::mpc_class;
#endif

TEST_CASE("Sin: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");

    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i12 = integer(12);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = sin(x);
    r2 = sin(x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = sin(zero);
    r2 = zero;
    std::cout << *r1 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = sin(x)->diff(x);
    r2 = cos(x);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(i2, x)->diff(x);
    r2 = i2;
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = sin(mul(i2, x))->diff(x);
    r2 = mul(i2, cos(mul(i2, x)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, sin(x))->diff(x);
    r2 = add(sin(x), mul(x, cos(x)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, sin(x))->diff(x)->diff(x);
    r2 = add(mul(i2, cos(x)), neg(mul(x, sin(x))));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(sin(x), cos(x))->diff(x);
    r2 = sub(pow(cos(x), i2), pow(sin(x), i2));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    // sin(-y) = -sin(y)
    r1 = sin(mul(im1, y));
    r2 = mul(im1, sin(y));
    REQUIRE(eq(*r1, *r2));

    // sin(pi - y) = sin(y)
    r1 = sin(sub(pi, y));
    r2 = sin(y);
    REQUIRE(eq(*r1, *r2));

    // sin(asin(x)) = x
    r1 = sin(asin(x));
    REQUIRE(eq(*r1, *x));

    // sin(acsc(x)) = 1/x
    r1 = sin(acsc(x));
    REQUIRE(eq(*r1, *div(one, x)));

    // sin(pi + y) = -sin(y)
    r1 = sin(add(pi, y));
    r2 = mul(im1, sin(y));
    REQUIRE(eq(*r1, *r2));

    // sin(2*pi - y) = -sin(y)
    r1 = sin(sub(mul(i2, pi), y));
    r2 = mul(im1, sin(y));
    REQUIRE(eq(*r1, *r2));

    // sin(12*pi + y) = sin(y)
    r1 = sin(add(mul(i12, pi), y));
    r2 = sin(y);
    REQUIRE(eq(*r1, *r2));

    // sin(3*pi - y) = sin(y)
    r1 = sin(sub(mul(i3, pi), y));
    r2 = sin(y);
    REQUIRE(eq(*r1, *r2));

    // sin(-3*pi - y) = sin(y)
    r1 = sin(sub(mul(neg(i3), pi), y));
    r2 = sin(y);
    REQUIRE(eq(*r1, *r2));

    // sin(pi/2 + y) = cos(y)
    r1 = sin(add(div(pi, i2), y));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // sin(pi/2 - y) = cos(y)
    r1 = sin(sub(div(pi, i2), y));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // sin(-pi/2 - y) = -cos(y)
    r1 = sin(sub(neg(div(pi, i2)), y));
    r2 = neg(cos(y));
    REQUIRE(eq(*r1, *r2));

    // sin(12*pi + y + pi/2) = cos(y)
    r1 = sin(add(add(mul(i12, pi), y), div(pi, i2)));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // sin(12*pi - y + pi/2) = cos(y)
    r1 = sin(add(sub(mul(i12, pi), y), div(pi, i2)));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // sin(2*pi/3 + y) = cos(pi/6 + y)
    r1 = sin(add(div(mul(i2, pi), i3), y));
    r2 = cos(add(div(pi, i6), y));
    REQUIRE(eq(*r1, *r2));

    r1 = sin(real_double(1.0));
    r2 = sin(sub(div(pi, i2), real_double(2.0)));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.841470984807897)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i + 0.416146836547142)
            < 1e-12);

    // Test is_canonical()
    RCP<const Sin> r4 = make_rcp<Sin>(i2); // dummy Sin
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(mul(pi, i2))));
    REQUIRE(not(r4->is_canonical(add(mul(pi, i2), div(pi, i2)))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));

    r1 = rewrite_as_exp(sin(x));
    r2 = div(sub(exp(mul(I, x)), exp(mul(neg(I), x))), mul(integer(2), I));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_sin(sin(x));
    r2 = sin(x);
    REQUIRE(eq(*r1, *r2));
    // Parsing to evaluate the unevaluated_expr
    REQUIRE(eq(*parse(r1->__str__()), *sin(x)));

    r1 = rewrite_as_cos(sin(x));
    r2 = cos(unevaluated_expr(sub(x, div(pi, integer(2)))));
    REQUIRE(eq(*r1, *r2));
    REQUIRE(eq(*parse(r1->__str__()), *sin(x)));
}

TEST_CASE("Cos: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i12 = integer(12);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = cos(x);
    r2 = cos(x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = cos(zero);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = cos(x)->diff(x);
    r2 = mul(im1, sin(x));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    // cos(-pi) = -1
    r1 = cos(neg(pi));
    r2 = im1;
    REQUIRE(eq(*r1, *r2));

    // cos(-y) = cos(y)
    r1 = cos(mul(im1, y));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // cos(x - 12) = cos(12 - x)
    r1 = cos(sub(x, i12));
    r2 = cos(sub(i12, x));
    REQUIRE(eq(*r1, *r2));

    // cos(acos(x)) = x
    r1 = cos(acos(x));
    REQUIRE(eq(*r1, *x));

    // cos(asec(x)) = 1/x
    r1 = cos(asec(x));
    REQUIRE(eq(*r1, *div(one, x)));

    // cos(pi - y) = -cos(y)
    r1 = cos(sub(pi, y));
    r2 = mul(im1, cos(y));
    REQUIRE(eq(*r1, *r2));

    // cos(pi + y) = -cos(y)
    r1 = cos(add(pi, y));
    r2 = mul(im1, cos(y));
    REQUIRE(eq(*r1, *r2));

    // cos(2*pi - y) = cos(y)
    r1 = cos(sub(mul(i2, pi), y));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // cos(12*pi + y) = cos(y)
    r1 = cos(add(mul(i12, pi), y));
    r2 = cos(y);
    REQUIRE(eq(*r1, *r2));

    // cos(3*pi - y) = -cos(y)
    r1 = cos(sub(mul(i3, pi), y));
    r2 = mul(im1, cos(y));
    REQUIRE(eq(*r1, *r2));

    // cos(pi/2 + y) = -sin(y)
    r1 = cos(add(div(pi, i2), y));
    r2 = mul(im1, sin(y));
    REQUIRE(eq(*r1, *r2));

    // cos(pi/2 - y) = sin(y)
    r1 = cos(sub(div(pi, i2), y));
    r2 = sin(y);
    REQUIRE(eq(*r1, *r2));

    // cos(12*pi + y + pi/2) = -sin(y)
    r1 = cos(add(add(mul(i12, pi), y), div(pi, i2)));
    r2 = mul(im1, sin(y));
    REQUIRE(eq(*r1, *r2));

    // cos(12*pi - y + pi/2) = sin(y)
    r1 = cos(add(sub(mul(i12, pi), y), div(pi, i2)));
    r2 = sin(y);
    REQUIRE(eq(*r1, *r2));

    // cos(2*pi/3 + y) = -sin(pi/6 + y)
    r1 = cos(add(div(mul(i2, pi), i3), y));
    r2 = neg(sin(add(div(pi, i6), y)));
    REQUIRE(eq(*r1, *r2));

    r1 = cos(real_double(1.0));
    r2 = cos(sub(div(pi, i2), real_double(2.0)));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.540302305868140)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 0.909297426825682)
            < 1e-12);

    RCP<const Cos> r4 = make_rcp<Cos>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(mul(pi, i2))));
    REQUIRE(not(r4->is_canonical(add(mul(pi, i2), div(pi, i2)))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));

    r1 = rewrite_as_exp(cos(x));
    r2 = div(add(exp(mul(I, x)), exp(mul(neg(I), x))), integer(2));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_sin(cos(x));
    r2 = sin(unevaluated_expr(add(x, div(pi, integer(2)))));
    REQUIRE(eq(*r1, *r2));
    REQUIRE(eq(*parse(r1->__str__()), *cos(x)));

    r1 = rewrite_as_cos(cos(x));
    r2 = cos(x);
    REQUIRE(eq(*r1, *r2));
    REQUIRE(eq(*parse(r1->__str__()), *cos(x)));
}

TEST_CASE("Tan: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> i23 = integer(23);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = tan(x);
    r2 = tan(x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = tan(zero);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = tan(x)->diff(x);
    r2 = add(pow(tan(x), i2), i1);
    REQUIRE(eq(*r1, *r2));

    r1 = tan(mul(i2, x))->diff(x);
    r2 = mul(i2, add(pow(tan(mul(i2, x)), i2), i1));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, tan(x))->diff(x);
    r2 = add(tan(x), mul(x, add(pow(tan(x), i2), i1)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    // tan(-y) = -tan(y)
    r1 = tan(mul(im1, y));
    r2 = mul(im1, tan(y));
    REQUIRE(eq(*r1, *r2));

    // tan(pi - y) = -tan(y)
    r1 = tan(sub(pi, y));
    r2 = mul(im1, tan(y));
    REQUIRE(eq(*r1, *r2));

    // tan(atan(x)) = x
    r1 = tan(atan(x));
    REQUIRE(eq(*r1, *x));

    // tan(acot(x)) = 1/x
    r1 = tan(acot(x));
    REQUIRE(eq(*r1, *div(one, x)));

    // tan(pi + y) = -tan(y)
    r1 = tan(add(pi, y));
    r2 = tan(y);
    REQUIRE(eq(*r1, *r2));

    // tan(2*pi - y) = -tan(y)
    r1 = tan(sub(mul(i2, pi), y));
    r2 = mul(im1, tan(y));
    REQUIRE(eq(*r1, *r2));

    // tan(12*pi + y) = tan(y)
    r1 = tan(add(mul(i12, pi), y));
    r2 = tan(y);
    REQUIRE(eq(*r1, *r2));

    // tan(3*pi - y) = -tan(y)
    r1 = tan(sub(mul(i3, pi), y));
    r2 = mul(im1, tan(y));
    REQUIRE(eq(*r1, *r2));

    // tan(pi/2 + y) = -cot(y)
    r1 = tan(add(div(pi, i2), y));
    r2 = mul(im1, cot(y));
    REQUIRE(eq(*r1, *r2));

    // tan(pi/2 - y) = cot(y)
    r1 = tan(sub(div(pi, i2), y));
    r2 = cot(y);
    REQUIRE(eq(*r1, *r2));

    // tan(12*pi + y + pi/2) = -cot(y)
    r1 = tan(add(add(mul(i12, pi), y), div(pi, i2)));
    r2 = mul(im1, cot(y));
    REQUIRE(eq(*r1, *r2));

    // tan(12*pi - y + pi/2) = cot(y)
    r1 = tan(add(sub(mul(i12, pi), y), div(pi, i2)));
    r2 = cot(y);
    REQUIRE(eq(*r1, *r2));

    // tan(23*pi/3 + y) = -cot(pi/6 + y)
    r1 = tan(add(div(mul(i23, pi), i3), y));
    r2 = neg(cot(add(div(pi, i6), y)));
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*tan(mul(integer(5), div(pi, i2))), *ComplexInf));

    r1 = tan(real_double(3.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i + 0.142546543074278)
            < 1e-12);

    RCP<const Tan> r4 = make_rcp<Tan>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(mul(pi, i2))));
    REQUIRE(not(r4->is_canonical(add(mul(pi, i2), div(pi, i2)))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));

    r1 = rewrite_as_exp(tan(x));
    r2 = div(sub(exp(mul(I, x)), exp(mul(neg(I), x))),
             mul(add(exp(mul(I, x)), exp(mul(neg(I), x))), I));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_sin(tan(x));
    r2 = div(mul(integer(2), pow(sin(x), integer(2))), sin(mul(integer(2), x)));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_cos(tan(x));
    r2 = div(cos(unevaluated_expr(sub(x, div(pi, integer(2))))), cos(x));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Cot: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = cot(x);
    r2 = cot(x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = cot(x)->diff(x);
    r2 = mul(im1, add(pow(cot(x), i2), i1));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = cot(mul(i2, x))->diff(x);
    r2 = mul(integer(-2), add(pow(cot(mul(i2, x)), i2), i1));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, cot(x))->diff(x);
    r2 = add(cot(x), mul(x, mul(add(pow(cot(x), i2), i1), im1)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    // cot(-y) = -cot(y)
    r1 = cot(mul(im1, y));
    r2 = mul(im1, cot(y));
    REQUIRE(eq(*r1, *r2));

    // cot(pi - y) = -cot(y)
    r1 = cot(sub(pi, y));
    r2 = mul(im1, cot(y));
    REQUIRE(eq(*r1, *r2));

    // cot(acot(x)) = x
    r1 = cot(acot(x));
    REQUIRE(eq(*r1, *x));

    // cot(atan(x)) = 1/x
    r1 = cot(atan(x));
    REQUIRE(eq(*r1, *div(one, x)));

    // cot(pi + y) = -cot(y)
    r1 = cot(add(pi, y));
    r2 = cot(y);
    REQUIRE(eq(*r1, *r2));

    // cot(2*pi - y) = -cot(y)
    r1 = cot(sub(mul(i2, pi), y));
    r2 = mul(im1, cot(y));
    REQUIRE(eq(*r1, *r2));

    // cot(12*pi + y) = cot(y)
    r1 = cot(add(mul(i12, pi), y));
    r2 = cot(y);
    REQUIRE(eq(*r1, *r2));

    // cot(3*pi - y) = -cot(y)
    r1 = cot(sub(mul(i3, pi), y));
    r2 = mul(im1, cot(y));
    REQUIRE(eq(*r1, *r2));

    // cot(pi/2 + y) = -tan(y)
    r1 = cot(add(div(pi, i2), y));
    r2 = mul(im1, tan(y));
    REQUIRE(eq(*r1, *r2));

    // cot(pi/2 - y) = cot(y)
    r1 = cot(sub(div(pi, i2), y));
    r2 = tan(y);
    REQUIRE(eq(*r1, *r2));

    // cot(12*pi + y + pi/2) = -tan(y)
    r1 = cot(add(add(mul(i12, pi), y), div(pi, i2)));
    r2 = mul(im1, tan(y));
    REQUIRE(eq(*r1, *r2));

    // cot(12*pi - y + pi/2) = tan(y)
    r1 = cot(add(sub(mul(i12, pi), y), div(pi, i2)));
    r2 = tan(y);
    REQUIRE(eq(*r1, *r2));

    // cot(100*pi/7 + y) = cot(2*pi/7 + y)
    r1 = cot(add(div(mul(integer(100), pi), integer(7)), y));
    r2 = cot(add(div(mul(i2, pi), integer(7)), y));
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*cot(mul(integer(7), pi)), *ComplexInf));

    r1 = cot(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i + 0.457657554360286)
            < 1e-12);

    RCP<const Cot> r4 = make_rcp<Cot>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(mul(pi, i2))));
    REQUIRE(not(r4->is_canonical(add(mul(pi, i2), div(pi, i2)))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));

    r1 = rewrite_as_exp(cot(x));
    r2 = div(mul(add(exp(mul(I, x)), exp(mul(neg(I), x))), I),
             sub(exp(mul(I, x)), exp(mul(neg(I), x))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_sin(cot(x));
    r2 = div(sin(mul(integer(2), x)), mul(integer(2), pow(sin(x), integer(2))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_cos(cot(x));
    r2 = div(cos(x), cos(unevaluated_expr(sub(x, div(pi, integer(2))))));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Csc: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> i12 = integer(12);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = csc(x);
    r2 = csc(x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = csc(x)->diff(x);
    r2 = mul(im1, mul(cot(x), csc(x)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = csc(mul(i2, x))->diff(x);
    r2 = mul(integer(-2), mul(cot(mul(i2, x)), csc(mul(i2, x))));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, csc(x))->diff(x);
    r2 = add(csc(x), mul(x, mul(im1, mul(cot(x), csc(x)))));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    // csc(-y) = -csc(y)
    r1 = csc(mul(im1, y));
    r2 = mul(im1, csc(y));
    REQUIRE(eq(*r1, *r2));

    // csc(pi - y) = csc(y)
    r1 = csc(sub(pi, y));
    r2 = csc(y);
    REQUIRE(eq(*r1, *r2));

    // csc(acsc(x)) = x
    r1 = csc(acsc(x));
    REQUIRE(eq(*r1, *x));

    // csc(asin(x)) = 1/x
    r1 = csc(asin(x));
    REQUIRE(eq(*r1, *div(one, x)));

    // csc(pi + y) = -csc(y)
    r1 = csc(add(pi, y));
    r2 = mul(im1, csc(y));
    REQUIRE(eq(*r1, *r2));

    // csc(2*pi - y) = -csc(y)
    r1 = csc(sub(mul(i2, pi), y));
    r2 = mul(im1, csc(y));
    REQUIRE(eq(*r1, *r2));

    // csc(12*pi + y) = csc(y)
    r1 = csc(add(mul(i12, pi), y));
    r2 = csc(y);
    REQUIRE(eq(*r1, *r2));

    // csc(3*pi - y) = csc(y)
    r1 = csc(sub(mul(i3, pi), y));
    r2 = csc(y);
    REQUIRE(eq(*r1, *r2));

    // csc(pi/2 + y) = sec(y)
    r1 = csc(add(div(pi, i2), y));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // csc(pi/2 - y) = sec(y)
    r1 = csc(sub(div(pi, i2), y));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // csc(12*pi + y + pi/2) = sec(y)
    r1 = csc(add(add(mul(i12, pi), y), div(pi, i2)));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // csc(12*pi - y + pi/2) = sec(y)
    r1 = csc(add(sub(mul(i12, pi), y), div(pi, i2)));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // csc(pi/5 + y) unchanged
    r1 = rcp_dynamic_cast<const Csc>(csc(add(div(pi, i5), y)))->get_arg();
    r2 = add(div(pi, i5), y);
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*csc(mul(integer(7), pi)), *ComplexInf));
    REQUIRE(eq(*csc(integer(0)), *ComplexInf));

    r1 = csc(real_double(3.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 7.08616739573719)
            < 1e-12);

    RCP<const Csc> r4 = make_rcp<Csc>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(mul(pi, i2))));
    REQUIRE(not(r4->is_canonical(add(mul(pi, i2), div(pi, i2)))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));

    r1 = rewrite_as_exp(csc(x));
    r2 = div(mul(integer(2), I), sub(exp(mul(I, x)), exp(mul(neg(I), x))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_sin(csc(x));
    r2 = div(integer(1), sin(x));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_cos(csc(x));
    r2 = div(integer(1), cos(unevaluated_expr(sub(x, div(pi, integer(2))))));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Sec: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = sec(x);
    r2 = sec(x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = sec(x)->diff(x);
    r2 = mul(tan(x), sec(x));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = sec(mul(i2, x))->diff(x);
    r2 = mul(i2, mul(tan(mul(i2, x)), sec(mul(i2, x))));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, sec(x))->diff(x);
    r2 = add(sec(x), mul(x, mul(tan(x), sec(x))));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    // sec(0) = zero
    r1 = sec(zero);
    REQUIRE(eq(*r1, *i1));

    // sec(-y) = sec(y)
    r1 = sec(mul(im1, y));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // sec(pi - y) = -sec(y)
    r1 = sec(sub(pi, y));
    r2 = mul(im1, sec(y));
    REQUIRE(eq(*r1, *r2));

    // sec(asec(x)) = x
    r1 = sec(asec(x));
    REQUIRE(eq(*r1, *x));

    // sec(acos(x)) = 1/x
    r1 = sec(acos(x));
    REQUIRE(eq(*r1, *div(one, x)));

    // sec(pi + y) = -sec(y)
    r1 = sec(add(pi, y));
    r2 = mul(im1, sec(y));
    REQUIRE(eq(*r1, *r2));

    // sec(2*pi - y) = sec(y)
    r1 = sec(sub(mul(i2, pi), y));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // sec(12*pi + y) = sec(y)
    r1 = sec(add(mul(i12, pi), y));
    r2 = sec(y);
    REQUIRE(eq(*r1, *r2));

    // sec(3*pi - y) = -sec(y)
    r1 = sec(sub(mul(i3, pi), y));
    r2 = mul(im1, sec(y));
    REQUIRE(eq(*r1, *r2));

    // sec(pi/2 + y) = -csc(y)
    r1 = sec(add(div(pi, i2), y));
    r2 = mul(im1, csc(y));
    REQUIRE(eq(*r1, *r2));

    // sec(pi/2 - y) = csc(y)
    r1 = sec(sub(div(pi, i2), y));
    r2 = csc(y);
    REQUIRE(eq(*r1, *r2));

    // sec(12*pi + y + pi/2) = -csc(y)
    r1 = sec(add(add(mul(i12, pi), y), div(pi, i2)));
    r2 = mul(im1, csc(y));
    REQUIRE(eq(*r1, *r2));

    // sec(12*pi - y + pi/2) = csc(y)
    r1 = sec(add(sub(mul(i12, pi), y), div(pi, i2)));
    r2 = csc(y);
    REQUIRE(eq(*r1, *r2));

    // sec(pi/3 + y) unchanged
    r1 = rcp_dynamic_cast<const Sec>(sec(add(div(pi, i3), y)))->get_arg();
    r2 = add(div(pi, i3), y);
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*sec(mul(integer(7), div(pi, i2))), *ComplexInf));

    r1 = sec(real_double(3.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i + 1.01010866590799)
            < 1e-12);

    RCP<const Sec> r4 = make_rcp<Sec>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(mul(pi, i2))));
    REQUIRE(not(r4->is_canonical(add(mul(pi, i2), div(pi, i2)))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));

    r1 = rewrite_as_exp(sec(x));
    r2 = div(integer(2), add(exp(mul(I, x)), exp(mul(neg(I), x))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_sin(sec(x));
    r2 = div(integer(1), sin(unevaluated_expr(add(x, div(pi, integer(2))))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_cos(sec(x));
    r2 = div(integer(1), cos(x));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("TrigFunction: trig_to_sqrt", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> one_m_x2 = sub(one, pow(x, i2));
    RCP<const Basic> one_m_xm2 = sub(one, pow(x, im2));
    RCP<const Basic> one_p_x2 = add(one, pow(x, i2));
    RCP<const Basic> one_p_xm2 = add(one, pow(x, im2));

    r1 = sin(acos(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *sqrt(one_m_x2)));

    r1 = sin(atan(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(x, sqrt(one_p_x2))));

    r1 = sin(asec(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *sqrt(one_m_xm2)));

    r1 = sin(acot(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, mul(x, sqrt(one_p_xm2)))));

    r1 = cos(asin(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *sqrt(one_m_x2)));

    r1 = cos(atan(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, sqrt(one_p_x2))));

    r1 = cos(acsc(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *sqrt(one_m_xm2)));

    r1 = cos(acot(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, sqrt(one_p_xm2))));

    r1 = tan(asin(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(x, sqrt(one_m_x2))));

    r1 = tan(acos(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(sqrt(one_m_x2), x)));

    r1 = tan(acsc(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, mul(x, sqrt(one_m_xm2)))));

    r1 = tan(asec(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *mul(x, sqrt(one_m_xm2))));

    r1 = csc(acos(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, sqrt(one_m_x2))));

    r1 = csc(atan(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(sqrt(one_p_x2), x)));

    r1 = csc(asec(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, sqrt(one_m_xm2))));

    r1 = csc(acot(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *mul(x, sqrt(one_p_xm2))));

    r1 = sec(asin(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, sqrt(one_m_x2))));

    r1 = sec(acos(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, x)));

    r1 = sec(atan(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *sqrt(one_p_x2)));

    r1 = sec(acsc(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, sqrt(one_m_xm2))));

    r1 = sec(acot(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *sqrt(one_p_xm2)));

    r1 = cot(asin(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(sqrt(one_m_x2), x)));

    r1 = cot(acos(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(x, sqrt(one_m_x2))));

    r1 = cot(acsc(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *mul(x, sqrt(one_m_xm2))));

    r1 = cot(asec(x));
    REQUIRE(eq(*trig_to_sqrt(r1), *div(one, mul(x, sqrt(one_m_xm2)))));
}

TEST_CASE("function_symbol: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = function_symbol("f", x);
    r2 = function_symbol("f", x);
    std::cout << *r1 << std::endl;

    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *zero));

    r1 = function_symbol("f", x);
    r2 = function_symbol("g", x);
    REQUIRE(neq(*r1, *r2));

    r1 = function_symbol("f", x);
    r2 = function_symbol("f", y);
    REQUIRE(neq(*r1, *r2));

    r1 = function_symbol("f", {x, y});
    r2 = function_symbol("f", {x, y});
    REQUIRE(eq(*r1, *r2));

    r1 = function_symbol("f", {x, y});
    r2 = function_symbol("f", {y, x});
    REQUIRE(neq(*r1, *r2));

    r1 = function_symbol("f", {x, y});
    r2 = function_symbol("f", x);
    REQUIRE(neq(*r1, *r2));

    r1 = function_symbol("f", zero);
    r2 = one;
    REQUIRE(neq(*r1, *r2));

    r1 = function_symbol("f", x)->diff(y);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = function_symbol("f", {x, y})->diff(z);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(i2, pow(function_symbol("f", add(add(x, y), z)), i2));
    r2 = mul(i2, pow(function_symbol("f", add(add(y, z), x)), i2));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Derivative: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> _x = symbol("_x");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Symbol> __xi_1 = symbol("__xi_1");
    RCP<const Symbol> _xi_2 = symbol("_xi_2");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> f = function_symbol("f", x);
    RCP<const Basic> g = function_symbol("g", x);

    RCP<const Basic> r1, r2, r3;

    r1 = f->diff(x);
    r2 = Derivative::create(f, {x});
    r3 = Derivative::create(g, {x});
    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *r3));
    REQUIRE(neq(*r2, *r3));
    REQUIRE(unified_eq(r1->get_args(), {f, x}));

    r1 = f->diff(x)->diff(x);
    r2 = Derivative::create(f, {x, x});
    REQUIRE(eq(*r1, *r2));
    REQUIRE(unified_eq(r1->get_args(), {f, x, x}));

    f = function_symbol("f", {x, y});
    r1 = f->diff(x)->diff(y);
    r2 = f->diff(y)->diff(x);
    r3 = f->diff(x)->diff(z);
    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *r3));

    r1 = Derivative::create(f, {x, y, x});
    r2 = Derivative::create(f, {x, x, y});
    REQUIRE(eq(*r1, *r2));

    f = function_symbol("f", pow(x, integer(2)));
    r1 = f->diff(x);
    std::cout << *f << " " << *r1 << std::endl;
    r2 = Derivative::create(function_symbol("f", _xi_1), {_xi_1});
    r2 = Subs::create(r2, {{_xi_1, pow(x, integer(2))}});
    REQUIRE(eq(*r1, *mul(mul(integer(2), x), r2)));

    f = function_symbol("f", {x, x});
    r1 = f->diff(x);
    std::cout << *f << " " << *r1 << std::endl;
    r2 = Derivative::create(function_symbol("f", {_xi_1, x}), {_xi_1});
    r2 = Subs::create(r2, {{_xi_1, x}});
    r3 = Derivative::create(function_symbol("f", {x, _xi_2}), {_xi_2});
    r3 = Subs::create(r3, {{_xi_2, x}});
    REQUIRE(eq(*r1, *add(r2, r3)));

    f = function_symbol("f", {y, add(x, y)});
    r1 = f->diff(x);
    std::cout << *f << " " << *r1 << std::endl;
    r2 = Derivative::create(function_symbol("f", {y, _xi_2}), {_xi_2});
    r2 = Subs::create(r2, {{_xi_2, add(y, x)}});
    REQUIRE(eq(*r1, *r2));

    r1 = function_symbol("f", add(_xi_1, x))->diff(_xi_1);
    std::cout << *f << " " << *r1 << std::endl;
    r2 = Subs::create(
        Derivative::create(function_symbol("f", __xi_1), {__xi_1}),
        {{__xi_1, add(_xi_1, x)}});
    REQUIRE(eq(*r1, *r2));

    f = function_symbol("f", x);
    RCP<const Derivative> r4 = Derivative::create(f, {x});
    REQUIRE(r4->is_canonical(function_symbol("f", {y, x}), {x}));
    REQUIRE(not r4->is_canonical(function_symbol("f", y), {x}));
    REQUIRE(not r4->is_canonical(function_symbol("f", x), {x, y, x, x}));
    REQUIRE(
        not(r4->is_canonical(function_symbol("f", x), {pow(x, integer(2))})));

    // Test get_args()
    r1 = Derivative::create(function_symbol("f", {x, y, pow(z, integer(2))}),
                            {x, x, y});
    REQUIRE(vec_basic_eq_perm(
        r1->get_args(),
        {function_symbol("f", {x, y, pow(z, integer(2))}), x, x, y}));

    // Test Derivative::subs
    r1 = Derivative::create(function_symbol("f", {x, add(y, y)}), {x});
    r2 = r1->subs({{x, y}});
    r3 = Subs::create(
        Derivative::create(function_symbol("f", {x, add(y, y)}), {x}),
        {{x, y}});
    REQUIRE(eq(*r2, *r3));

    r2 = r1->subs({{x, z}});
    r3 = Derivative::create(function_symbol("f", {z, add(y, y)}), {z});
    REQUIRE(eq(*r2, *r3));

    r2 = r1->subs({{y, z}});
    r3 = Derivative::create(function_symbol("f", {x, add(z, z)}), {x});
    REQUIRE(eq(*r2, *r3));

    // r1 = Derivative::create(kronecker_delta(x, y), {y});

    r1 = EulerGamma->diff(x);
    REQUIRE(eq(*r1, *zero));
}

TEST_CASE("Subs: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Symbol> _x = symbol("_x");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Symbol> _xi_2 = symbol("_xi_2");
    RCP<const Basic> r1, r2, r3, r4;

    // Test Subs::subs
    r1 = Subs::create(Derivative::create(function_symbol("f", {y, x}), {x}),
                      {{x, add(x, y)}});
    r2 = Subs::create(Derivative::create(function_symbol("f", {y, x}), {x}),
                      {{x, z}, {y, z}});
    r3 = Subs::create(Derivative::create(function_symbol("f", {y, x}), {x}),
                      {{y, z}, {x, z}});
    REQUIRE(eq(*r2, *r3));

    r2 = r1->subs({{y, z}});
    r3 = Subs::create(Derivative::create(function_symbol("f", {z, x}), {x}),
                      {{x, add(x, z)}});
    REQUIRE(eq(*r2, *r3));

    r2 = r1->subs({{x, z}});
    r3 = Subs::create(Derivative::create(function_symbol("f", {y, x}), {x}),
                      {{x, add(z, y)}});
    REQUIRE(eq(*r2, *r3));

    r2 = r1->subs({{r1, r3}});
    REQUIRE(eq(*r2, *r3));

    // Test Subs::diff
    r1 = function_symbol("f", {add(y, y), add(x, y)})->diff(x);

    r2 = r1->diff(_x);
    r3 = zero;
    REQUIRE(eq(*r2, *r3));

    r2 = r1->diff(x);
    r3 = Subs::create(
        Derivative::create(function_symbol("f", {add(y, y), _xi_2}),
                           {_xi_2, _xi_2}),
        {{_xi_2, add(x, y)}});
    REQUIRE(eq(*r2, *r3));

    r2 = r1->diff(y);
    r3 = Subs::create(
        Derivative::create(function_symbol("f", {add(y, y), _xi_2}),
                           {_xi_2, _xi_2}),
        {{_xi_2, add(x, y)}});
    r4 = Subs::create(Derivative::create(function_symbol("f", {_xi_1, _xi_2}),
                                         {_xi_1, _xi_2}),
                      {{_xi_2, add(x, y)}, {_xi_1, add(y, y)}});
    r3 = add(r3, add(r4, r4));
    REQUIRE(eq(*r2, *r3));
}

TEST_CASE("Get pi shift: functions", "[functions]")
{
    RCP<const Basic> r;
    RCP<const Basic> r1;
    RCP<const Number> n;
    bool b;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> i8 = integer(8);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    RCP<const Symbol> x = symbol("x");

    // arg = k + n*pi
    r = add(i3, mul(i2, pi));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == true);
    REQUIRE(eq(*n, *integer(2)));
    REQUIRE(eq(*r1, *i3));

    // arg = n*pi/12
    r = mul(pi, div(one, integer(12)));
    get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(eq(*n, *div(one, integer(12))));
    REQUIRE(eq(*r1, *zero));

    // arg = 2*pi/3
    r = mul(pi, div(i2, integer(3)));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE((eq(*n, *div(i2, integer(3))) and (b == true) and eq(*r1, *zero)));

    // arg = 2 * pi / 5
    r = mul(pi, div(i2, integer(5)));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(((b == true) and eq(*n, *div(i2, integer(5)))));

    // arg neq theta + n*pi (no pi symbol, pi as pow)
    r = mul(pow(pi, i2), div(i2, integer(3)));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == false);

    // arg neq theta + n*pi (no pi symbol, pi as mul form)
    r = mul(mul(pi, x), div(i2, integer(3)));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == false);

    // arg = theta + n*pi (theta is just another symbol)
    r = add(mul(i2, x), mul(pi, div(i2, integer(3))));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == true);
    REQUIRE(eq(*n, *div(i2, integer(3))));
    REQUIRE(eq(*r1, *mul(i2, x)));

    // arg = theta + n*pi (theta is constant plus a symbol)
    r = add(i2, add(x, mul(pi, div(i2, integer(3)))));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == true);
    REQUIRE(eq(*n, *div(i2, integer(3))));
    REQUIRE(eq(*r1, *add(i2, x)));

    // arg = theta + n*pi (theta is an expression)
    r = add(i2, add(mul(x, i2), mul(pi, div(i2, integer(3)))));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == true);
    REQUIRE(eq(*n, *div(i2, integer(3))));
    REQUIRE(eq(*r1, *add(i2, mul(x, i2))));

    // arg neq n*pi (n is not rational)
    r = mul(pi, real_double(0.1));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == false);

    // arg neq n*pi (pi is not in form of symbol)
    r = mul(pow(pi, i2), div(i2, integer(3)));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == false);

    // arg = pi (it is neither of form add nor mul, just a symbol)
    b = get_pi_shift(pi, outArg(n), outArg(r1));
    REQUIRE(((b == true) and eq(*n, *one) and eq(*r1, *zero)));

    // arg = theta + n*pi (theta is an expression of >1 symbols)
    r = add(add(mul(i2, x), mul(i2, symbol("y"))),
            mul(pi, div(i2, integer(3))));
    b = get_pi_shift(r, outArg(n), outArg(r1));
    REQUIRE(b == true);
    REQUIRE(eq(*n, *div(i2, integer(3))));
    REQUIRE(eq(*r1, *add(mul(i2, x), mul(i2, symbol("y")))));
}

TEST_CASE("Sin table: functions", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    // sin(2pi + pi/6) = 1/2
    r1 = sin(add(mul(pi, i2), mul(div(pi, i12), i2)));
    r2 = div(one, i2);
    REQUIRE(eq(*r1, *r2));

    // sin(n*pi + pi/6) = 1/2
    r1 = sin(add(mul(pi, integer(10)), mul(div(pi, i12), i2)));
    r2 = div(one, i2);
    REQUIRE(eq(*r1, *r2));

    // sin(n*pi) = 0
    r1 = sin(mul(pi, i12));
    REQUIRE(eq(*r1, *zero));

    // sin(2pi + pi/2) = 1
    r1 = sin(add(mul(pi, i2), div(pi, i2)));
    REQUIRE(eq(*r1, *one));

    // sin(pi/3) = sqrt(3)/2
    r1 = sin(div(pi, integer(3)));
    r2 = div(sq3, i2);
    REQUIRE(eq(*r1, *r2));

    // sin(pi/4) = 1/sqrt(2)
    r1 = sin(div(pi, integer(4)));
    r2 = div(sq2, i2);
    REQUIRE(eq(*r1, *r2));

    // sin(pi/12) = (sqrt(3) - 1)/(2*sqrt(2))
    r1 = sin(div(pi, i12));
    r2 = div(sub(sq3, one), mul(i2, sq2));
    REQUIRE(eq(*r1, *r2));

    // sin(5*pi/12) = (sqrt(3) + 1)/(2*sqrt(2))
    r1 = sin(mul(div(pi, i12), integer(5)));
    r2 = div(add(sq3, one), mul(i2, sq2));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Cos table: functions", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> i13 = integer(13);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    // cos(2pi + pi/6) = sqrt(3)/2
    r1 = cos(add(mul(pi, i2), mul(div(pi, i12), i2)));
    r2 = div(sq3, i2);
    REQUIRE(eq(*r1, *r2));

    // cos(n*pi + pi/6) = sqrt(3)/2
    r1 = cos(add(mul(pi, integer(10)), mul(div(pi, i12), i2)));
    r2 = div(sq3, i2);
    REQUIRE(eq(*r1, *r2));

    // cos((2n - 1)*pi) = -1
    r1 = cos(mul(pi, i13));
    REQUIRE(eq(*r1, *im1));

    // cos(2pi + pi/2) = 0
    r1 = cos(add(mul(pi, i2), div(pi, i2)));
    REQUIRE(eq(*r1, *zero));

    // cos(pi/3) = 1/2
    r1 = cos(div(pi, integer(3)));
    r2 = div(one, i2);
    REQUIRE(eq(*r1, *r2));

    // cos(pi/4) = 1/sqrt(2)
    r1 = cos(div(pi, integer(4)));
    r2 = div(sq2, i2);
    REQUIRE(eq(*r1, *r2));

    // cos(5*pi/12) = (sqrt(3) - 1)/(2*sqrt(2))
    r1 = cos(mul(div(pi, i12), integer(5)));
    r2 = div(sub(sq3, one), mul(i2, sq2));
    REQUIRE(eq(*r1, *r2));

    // cos(pi/12) = (sqrt(3) + 1)/(2*sqrt(2))
    r1 = cos(div(pi, i12));
    r2 = div(add(sq3, one), mul(i2, sq2));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Sec table: functions", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> i13 = integer(13);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    // sec(2pi + pi/6) = 2/sqrt(3)
    r1 = sec(add(mul(pi, i2), mul(div(pi, i12), i2)));
    r2 = div(i2, sq3);
    REQUIRE(eq(*r1, *r2));

    // sec(n*pi + pi/6) = 2/sqrt(3)
    r1 = sec(add(mul(pi, integer(10)), mul(div(pi, i12), i2)));
    r2 = div(i2, sq3);
    REQUIRE(eq(*r1, *r2));

    // sec((2n - 1)*pi) = -1
    r1 = sec(mul(pi, i13));
    REQUIRE(eq(*r1, *im1));

    // sec(pi/3) = 2
    r1 = sec(div(pi, integer(3)));
    REQUIRE(eq(*r1, *i2));

    // sec(pi/4) = sqrt(2)
    r1 = sec(div(pi, integer(4)));
    REQUIRE(eq(*r1, *sq2));

    // sec(5*pi/12) = (2*sqrt(2))/(sqrt(3) - 1)
    r1 = sec(mul(div(pi, i12), integer(5)));
    r2 = div(mul(i2, sq2), sub(sq3, one));
    REQUIRE(eq(*r1, *r2));

    // sec(pi/12) = (2*sqrt(2))/(sqrt(3) + 1)
    r1 = sec(div(pi, i12));
    r2 = div(mul(i2, sq2), add(sq3, one));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Csc table: functions", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    // csc(2pi + pi/6) = 2
    r1 = csc(add(mul(pi, i2), mul(div(pi, i12), i2)));
    REQUIRE(eq(*r1, *i2));

    // csc(n*pi + pi/6) = 2
    r1 = csc(add(mul(pi, integer(10)), mul(div(pi, i12), i2)));
    REQUIRE(eq(*r1, *i2));

    // csc(2pi + pi/2) = 1
    r1 = csc(add(mul(pi, i2), div(pi, i2)));
    REQUIRE(eq(*r1, *one));

    // csc(pi/3) = 2/sqrt(3)
    r1 = csc(div(pi, integer(3)));
    r2 = div(i2, sq3);
    REQUIRE(eq(*r1, *r2));

    // csc(pi/4) = sqrt(2)
    r1 = csc(div(pi, integer(4)));
    REQUIRE(eq(*r1, *sq2));

    // csc(pi/12) = (2*sqrt(2))/(sqrt(3) - 1)
    r1 = csc(div(pi, i12));
    r2 = div(mul(i2, sq2), sub(sq3, one));
    REQUIRE(eq(*r1, *r2));

    // csc(5*pi/12) = (2*sqrt(2))/(sqrt(3) + 1)
    r1 = csc(mul(div(pi, i12), integer(5)));
    r2 = div(mul(i2, sq2), add(sq3, one));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Tan table: functions", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> i13 = integer(13);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    // tan(2pi + pi/6) = 1/sqrt(3)
    r1 = tan(add(mul(pi, i2), mul(div(pi, i12), i2)));
    r2 = div(sq3, i3);
    REQUIRE(eq(*r1, *r2));

    // tan(n*pi + pi/6) = 1/sqrt(3)
    r1 = tan(add(mul(pi, integer(10)), mul(div(pi, i12), i2)));
    r2 = div(sq3, i3);
    REQUIRE(eq(*r1, *r2));

    // tan(n*pi) = 0
    r1 = tan(mul(pi, i13));
    REQUIRE(eq(*r1, *zero));

    // tan(pi/3) = sq3
    r1 = tan(div(pi, integer(3)));
    REQUIRE(eq(*r1, *sq3));

    // tan(pi/4) = 1
    r1 = tan(div(pi, integer(4)));
    REQUIRE(eq(*r1, *one));

    // tan(5*pi/12) = (1 + 3**(1/2))/(-1 + 3**(1/2))
    r1 = tan(mul(div(integer(5), i12), pi));
    r2 = div(add(one, sq3), add(im1, sq3));
    REQUIRE(eq(*r1, *r2));

    // tan(pi/12) = (-1 + 3**(1/2))/(1 + 3**(1/2))
    r1 = tan(div(pi, i12));
    r2 = div(sub(sq3, one), add(one, sq3));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Cot table: functions", "[functions]")
{
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> sq3 = sqrt(i3);
    RCP<const Basic> sq2 = sqrt(i2);

    // cot(2pi + pi/6) = sqrt(3)
    r1 = cot(add(mul(pi, i2), mul(div(pi, i12), i2)));
    REQUIRE(eq(*r1, *sq3));

    // cot(n*pi + pi/6) = sqrt(3)
    r1 = cot(add(mul(pi, integer(10)), mul(div(pi, i12), i2)));
    REQUIRE(eq(*r1, *sq3));

    // cot(pi/2) = 0
    r1 = cot(div(pi, i2));
    REQUIRE(eq(*r1, *zero));

    // cot(pi/3) = 1/sq3
    r1 = cot(div(pi, integer(3)));
    r2 = div(one, sq3);
    REQUIRE(eq(*r1, *r2));

    // cot(pi/4) = 1
    r1 = cot(div(pi, integer(4)));
    REQUIRE(eq(*r1, *one));

    // cot(pi/12) = (1 + 3**(1/2))/(-1 + 3**(1/2))
    r1 = cot(div(pi, i12));
    r2 = div(add(one, sq3), sub(sq3, one));
    REQUIRE(eq(*r1, *r2));

    // cot(5*pi/12) = (-1 + 3**(1/2))/(1 + 3**(1/2))
    r1 = cot(div(mul(integer(5), pi), i12));
    r2 = div(sub(sq3, one), add(one, sq3));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Could extract minus: functions", "[functions]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");

    RCP<const Number> i2 = integer(2);
    RCP<const Number> im1 = integer(-1);
    RCP<const Basic> r, s;
    bool b, c;

    r = add(mul(im1, x), mul(im1, mul(i2, y)));
    b = could_extract_minus(*r);
    REQUIRE(b == true);

    r = add(mul(im1, x), mul(i2, y));
    s = add(x, mul(mul(i2, y), im1));
    b = could_extract_minus(*r);
    c = could_extract_minus(*s);
    REQUIRE(b != c);

    r = mul(mul(x, integer(-10)), y);
    b = could_extract_minus(*r);
    REQUIRE(b == true);

    r = mul(mul(x, i2), y);
    b = could_extract_minus(*r);
    REQUIRE(b == false);

    r = add(mul(im1, x), mul(im1, div(mul(i2, y), integer(3))));
    b = could_extract_minus(*r);
    REQUIRE(b == true);

    r = mul(div(x, i2), y);
    b = could_extract_minus(*r);
    REQUIRE(b == false);

    r = Complex::from_two_nums(*i2, *im1);
    b = could_extract_minus(*r);
    REQUIRE(b == false);

    r = Complex::from_two_nums(*im1, *i2);
    b = could_extract_minus(*r);
    REQUIRE(b == true);

    r = Complex::from_two_nums(*zero, *i2);
    b = could_extract_minus(*r);
    REQUIRE(b == false);

    r = Complex::from_two_nums(*zero, *im1);
    b = could_extract_minus(*r);
    REQUIRE(b == true);

    r = im1;
    b = could_extract_minus(*r);
    REQUIRE(b == true);
}

TEST_CASE("Asin: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = asin(x)->diff(x);
    r2 = div(one, sqrt(sub(one, pow(x, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = asin(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = asin(zero);
    REQUIRE(eq(*r1, *zero));

    r1 = asin(im1);
    r2 = mul(im1, div(pi, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = asin(div(im1, i2));
    r2 = div(pi, mul(im2, i3));
    REQUIRE(eq(*r1, *r2));

    r1 = asin(div(sqrt(i2), i2));
    r2 = div(pi, mul(i2, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = asin(div(add(sqrt(i3), i1), mul(i2, sqrt(i2))));
    r2 = div(pi, mul(i3, pow(i2, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = asin(div(sqrt(sub(i5, sqrt(i5))), integer(8)));
    r2 = div(pi, i5);
    REQUIRE(eq(*r1, *r2));

    r1 = asin(mul(div(sub(sqrt(i5), i1), integer(4)), im1));
    r2 = div(pi, mul(im2, i5));
    REQUIRE(eq(*r1, *r2));

    r1 = asin(real_double(0.5));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.523598775598299)
            < 1e-12);

    r1 = asin(complex_double(std::complex<double>(1, 1)));
    r2 = asin(real_double(2.0));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.2530681300031)
            < 1e-10);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 2.0498241882037)
            < 1e-10);

    RCP<const ASin> r4 = make_rcp<ASin>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Acos: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = acos(x)->diff(x);
    r2 = div(minus_one, sqrt(sub(one, pow(x, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = acos(im1);
    r2 = pi;
    REQUIRE(eq(*r1, *r2));

    r1 = acos(zero);
    r2 = div(pi, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = acos(div(im1, i2));
    r2 = mul(i2, div(pi, i3));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(div(sqrt(i2), i2));
    r2 = div(pi, mul(i2, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(div(add(sqrt(i3), i1), mul(i2, sqrt(i2))));
    r2 = mul(i5, div(pi, mul(i3, pow(i2, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(div(sqrt(sub(i5, sqrt(i5))), integer(8)));
    r2 = mul(i3, div(pi, mul(i2, i5)));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(mul(div(sub(sqrt(i5), i1), integer(4)), im1));
    r2 = mul(i3, div(pi, i5));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(div(sub(sqrt(i5), i1), integer(4)));
    r2 = mul(i2, div(pi, i5));
    REQUIRE(eq(*r1, *r2));

    r1 = acos(real_double(0.5));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.04719755119660)
            < 1e-12);

    r1 = acos(complex_double(std::complex<double>(1, 2)));
    r2 = acos(real_double(4.0));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.90908861124732)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 2.06343706889556)
            < 1e-12);

    RCP<const ACos> r4 = make_rcp<ACos>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Asec: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = asec(x)->diff(x);
    r2 = div(one, mul(pow(x, i2), sqrt(sub(one, div(one, pow(x, i2))))));
    REQUIRE(eq(*r1, *r2));

    r1 = asec(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = asec(im1);
    r2 = pi;
    REQUIRE(eq(*r1, *r2));

    r1 = asec(div(i2, im1));
    r2 = mul(i2, div(pi, i3));
    std::cout << r1->__str__() << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = asec(sqrt(i2));
    r2 = div(pi, mul(i2, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = asec(div(mul(i2, sqrt(i2)), add(sqrt(i3), i1)));
    r2 = mul(i5, div(pi, mul(i3, pow(i2, i2))));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = asec(div(integer(8), sqrt(sub(i5, sqrt(i5)))));
    r2 = mul(i3, div(pi, mul(i2, i5)));
    REQUIRE(eq(*r1, *r2));

    r1 = asec(mul(div(integer(4), sub(sqrt(i5), i1)), im1));
    r2 = mul(i3, div(pi, i5));
    REQUIRE(eq(*r1, *r2));

    r1 = asec(div(integer(4), sub(sqrt(i5), i1)));
    r2 = mul(i2, div(pi, i5));
    REQUIRE(eq(*r1, *r2));

    r1 = asec(real_double(-2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 2.09439510239320)
            < 1e-12);

    r1 = asec(complex_double(std::complex<double>(1, 2)));
    r2 = asec(real_double(0.5));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.44015500855881)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 1.31695789692482)
            < 1e-12);

    RCP<const ASec> r4 = make_rcp<ASec>(i5);
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(r4->is_canonical(i5));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Acsc: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = acsc(x)->diff(x);
    r2 = div(minus_one, mul(pow(x, i2), sqrt(sub(one, div(one, pow(x, i2))))));
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = acsc(im1);
    r2 = mul(im1, div(pi, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(im2);
    r2 = div(pi, mul(im2, i3));
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(sqrt(i2));
    r2 = div(pi, mul(i2, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(div(mul(i2, sqrt(i2)), add(sqrt(i3), i1)));
    r2 = div(pi, mul(i3, pow(i2, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(div(integer(8), sqrt(sub(i5, sqrt(i5)))));
    r2 = div(pi, i5);
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(mul(div(integer(4), sub(sqrt(i5), i1)), im1));
    r2 = div(pi, mul(im2, i5));
    REQUIRE(eq(*r1, *r2));

    r1 = acsc(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.523598775598299)
            < 1e-12);

    r1 = acsc(complex_double(std::complex<double>(1, 2)));
    r2 = acsc(real_double(0.4));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 0.438156111929239)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 2.21861690006402)
            < 1e-12);

    RCP<const ACsc> r4 = make_rcp<ACsc>(i5);
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(r4->is_canonical(i5));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("atan: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = atan(x)->diff(x);
    r2 = div(one, add(one, pow(x, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = atan(i1);
    r2 = div(pi, integer(4));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(zero);
    REQUIRE(eq(*r1, *zero));

    r1 = atan(im1);
    r2 = div(pi, integer(-4));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(div(one, sqrt(i3)));
    r2 = div(pi, integer(6));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(mul(im1, add(one, sqrt(i2))));
    r2 = div(mul(pi, i3), integer(-8));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(sub(sqrt(i2), one));
    r2 = div(pi, integer(8));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(sub(i2, sqrt(i3)));
    r2 = div(pi, integer(12));
    REQUIRE(eq(*r1, *r2));

    r1 = atan(mul(im1, sqrt(add(i5, mul(i2, sqrt(i5))))));
    r2 = div(mul(pi, im2), i5);
    REQUIRE(eq(*r1, *r2));

    r1 = atan(real_double(3.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.24904577239825)
            < 1e-12);

    RCP<const ATan> r4 = make_rcp<ATan>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Acot: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = acot(x)->diff(x);
    r2 = div(minus_one, add(one, pow(x, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = acot(i1);
    r2 = div(pi, integer(4));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(zero);
    r2 = div(pi, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = acot(im1);
    r2 = mul(i3, div(pi, integer(4)));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(div(one, sqrt(i3)));
    r2 = div(pi, i3);
    REQUIRE(eq(*r1, *r2));

    r1 = acot(mul(im1, add(one, sqrt(i2))));
    r2 = div(mul(pi, integer(7)), integer(8));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(sub(sqrt(i2), one));
    r2 = mul(i3, div(pi, integer(8)));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(sub(i2, sqrt(i3)));
    r2 = mul(i5, div(pi, integer(12)));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(mul(im1, sqrt(add(i5, mul(i2, sqrt(i5))))));
    r2 = div(mul(pi, integer(9)), mul(i5, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = acot(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.463647609000806)
            < 1e-12);

    RCP<const ACot> r4 = make_rcp<ACot>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Atan2: functions", "[functions]")
{
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = atan2(i1, i1);
    r2 = div(pi, integer(4));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(im1, i1);
    r2 = div(pi, integer(-4));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(i1, im1);
    r2 = div(mul(i3, pi), integer(4));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(im1, im1);
    r2 = div(mul(i3, pi), integer(-4));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(one, sqrt(i3));
    r2 = div(pi, integer(6));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(add(one, sqrt(i2)), im1);
    r2 = div(mul(pi, i3), integer(-8));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(sub(sqrt(i2), one), i1);
    r2 = div(pi, integer(8));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(sub(i2, sqrt(i3)), i1);
    r2 = div(pi, integer(12));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(sqrt(add(i5, mul(i2, sqrt(i5)))), im1);
    r2 = div(mul(pi, im2), i5);
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(y, x)->diff(x);
    r2 = div(mul(im1, y), add(pow(x, i2), pow(y, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(y, x)->diff(y);
    r2 = div(x, add(pow(x, i2), pow(y, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(y, x);
    REQUIRE(unified_eq(r1->get_args(), {y, x}));

    r1 = atan2(zero, zero);
    REQUIRE(eq(*r1, *Nan));

    r1 = atan2(zero, i2);
    REQUIRE(eq(*r1, *zero));

    r1 = atan2(zero, im1);
    REQUIRE(eq(*r1, *pi));

    r1 = atan2(i2, zero);
    r2 = div(pi, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = atan2(im2, zero);
    r2 = mul(div(pi, i2), minus_one);
    REQUIRE(eq(*r1, *r2));

    RCP<const ATan2> r4 = make_rcp<ATan2>(i2, i3);
    REQUIRE(not(r4->is_canonical(zero, i2)));
    REQUIRE(not(r4->is_canonical(zero, zero)));
    REQUIRE(not(r4->is_canonical(i2, i2)));
    REQUIRE(not(r4->is_canonical(i2, neg(i2))));
    REQUIRE(r4->is_canonical(i2, i3));
    REQUIRE(not(r4->is_canonical(one, sqrt(i3))));
}

TEST_CASE("Lambertw: functions", "[functions]")
{
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = lambertw(zero);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = lambertw(E);
    REQUIRE(eq(*r1, *one));

    r1 = lambertw(neg(exp(im1)));
    r2 = im1;
    REQUIRE(eq(*r1, *r2));

    r1 = lambertw(div(log(i2), im2));
    r2 = log(div(one, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = lambertw(x)->diff(y);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = lambertw(mul(i2, x))->diff(x);
    r2 = div(lambertw(mul(i2, x)), mul(x, add(lambertw(mul(i2, x)), one)));
    REQUIRE(eq(*r1, *r2));

    RCP<const LambertW> r4 = make_rcp<LambertW>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(E)));
    REQUIRE(not(r4->is_canonical(neg(div(one, E)))));
    REQUIRE(not(r4->is_canonical(div(log(i2), im2))));
    REQUIRE(r4->is_canonical(i2));
}

TEST_CASE("Sinh: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = sinh(zero);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = sinh(im1);
    r2 = mul(im1, sinh(one));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(sinh(x));
    r2 = div(add(exp(x), mul(im1, exp(mul(im1, x)))), i2);
    REQUIRE(eq(*r1, *r2));
    // tests cosh(-x) = cosh(x) and sinh(x)->diff(x) = cosh(x)
    r1 = sinh(mul(im1, x))->diff(x);
    r2 = mul(im1, cosh(x));
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*sinh(sub(x, y)), *neg(sinh(sub(y, x)))));

    r1 = sinh(real_double(1.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.1752011936438)
            < 1e-12);

    RCP<const Sinh> r4 = make_rcp<Sinh>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Csch: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = csch(im1);
    r2 = mul(im1, csch(one));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(csch(x));
    r2 = div(i2, add(exp(x), mul(im1, exp(mul(im1, x)))));
    REQUIRE(eq(*r1, *r2));
    r1 = csch(mul(im1, x))->diff(x);
    r2 = mul(csch(x), coth(x));
    REQUIRE(eq(*r1, *r2));

    r1 = csch(zero);
    REQUIRE(eq(*r1, *ComplexInf));
    REQUIRE(eq(*csch(sub(x, y)), *neg(csch(sub(y, x)))));

    r1 = csch(real_double(1.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.850918128239322)
            < 1e-12);

    RCP<const Csch> r4 = make_rcp<Csch>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Cosh: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = cosh(zero);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = cosh(im1);
    r2 = cosh(one);
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(cosh(x));
    r2 = div(add(exp(x), exp(mul(im1, x))), i2);
    REQUIRE(eq(*r1, *r2));
    // tests sinh(-x) = -sinh(x) and cosh(x)->diff(x) = sinh(x)
    r1 = cosh(mul(im1, x))->diff(x);
    r2 = mul(im1, sinh(mul(im1, x)));
    REQUIRE(eq(*r1, *r2));
    REQUIRE(eq(*cosh(sub(x, y)), *cosh(sub(y, x))));

    r1 = cosh(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 3.76219569108363)
            < 1e-12);

    RCP<const Cosh> r4 = make_rcp<Cosh>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Sech: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = sech(zero);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = sech(im1);
    r2 = sech(one);
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(sech(x));
    r2 = div(i2, add(exp(x), exp(mul(im1, x))));
    REQUIRE(eq(*r1, *r2));

    r1 = sech(mul(im1, x))->diff(x);
    r2 = mul(im1, mul(sech(x), tanh(x)));
    REQUIRE(eq(*r1, *r2));
    REQUIRE(eq(*sech(sub(x, y)), *sech(sub(y, x))));

    r1 = sech(real_double(4.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.0366189934736865)
            < 1e-12);

    RCP<const Sech> r4 = make_rcp<Sech>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Tanh: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = tanh(zero);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = tanh(im1);
    r2 = mul(im1, tanh(one));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(tanh(x));
    r2 = div(sub(exp(x), exp(mul(im1, x))), add(exp(x), exp(mul(im1, x))));
    REQUIRE(eq(*r1, *r2));

    r1 = tanh(mul(im1, x))->diff(x);
    r2 = add(pow(tanh(x), i2), im1);
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    // REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*tanh(sub(x, y)), *neg(tanh(sub(y, x)))));

    r1 = tanh(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.964027580075817)
            < 1e-12);

    RCP<const Tanh> r4 = make_rcp<Tanh>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Coth: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> im2 = integer(-2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = coth(im1);
    r2 = mul(im1, coth(one));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(coth(x));
    r2 = div(add(exp(x), exp(mul(im1, x))), sub(exp(x), exp(mul(im1, x))));
    REQUIRE(eq(*r1, *r2));

    r1 = coth(mul(im1, x))->diff(x);
    r2 = pow(sinh(x), im2);
    REQUIRE(eq(*r1, *r2));

    r1 = coth(zero);
    REQUIRE(eq(*r1, *ComplexInf));

    REQUIRE(eq(*coth(sub(x, y)), *neg(coth(sub(y, x)))));

    r1 = coth(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.03731472072755)
            < 1e-12);

    RCP<const Coth> r4 = make_rcp<Coth>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Asinh: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = asinh(zero);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = asinh(one);
    r2 = log(add(sqrt(i2), one));
    REQUIRE(eq(*r1, *r2));

    r1 = asinh(im1);
    r2 = log(add(sqrt(i2), im1));
    REQUIRE(eq(*r1, *r2));

    r1 = asinh(neg(i2));
    r2 = asinh(i2);
    REQUIRE(eq(*r1, *neg(r2)));

    r1 = asinh(mul(im1, x))->diff(x);
    r2 = div(im1, sqrt(add(pow(x, i2), one)));
    REQUIRE(eq(*r1, *r2));

    r1 = asinh(mul(i2, y))->diff(y);
    r2 = div(i2, sqrt(add(mul(i4, pow(y, i2)), one)));
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*asinh(sub(x, y)), *neg(asinh(sub(y, x)))));

    r1 = asinh(real_double(3.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.81844645923207)
            < 1e-12);

    RCP<const ASinh> r4 = make_rcp<ASinh>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Acsch: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> one = integer(1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = acsch(one);
    r2 = log(add(sqrt(i2), one));
    REQUIRE(eq(*r1, *r2));

    r1 = acsch(im1);
    r2 = log(add(sqrt(i2), im1));
    REQUIRE(eq(*r1, *r2));

    r1 = acsch(x)->diff(x);
    r2 = div(im1, mul(sqrt(add(one, div(one, pow(x, i2)))), pow(x, i2)));
    REQUIRE(eq(*r1, *r2));

    REQUIRE(eq(*acsch(sub(x, y)), *neg(acsch(sub(y, x)))));

    r1 = acsch(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.481211825059603)
            < 1e-12);

    RCP<const ACsch> r4 = make_rcp<ACsch>(i2);
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Acosh: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = acosh(one);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = acosh(mul(im1, x))->diff(x);
    r2 = div(im1, sqrt(add(pow(x, i2), im1)));
    REQUIRE(eq(*r1, *r2));

    r1 = acosh(mul(i2, y))->diff(y);
    r2 = div(i2, sqrt(add(mul(i4, pow(y, i2)), im1)));
    REQUIRE(eq(*r1, *r2));

    r1 = acosh(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.31695789692482)
            < 1e-12);

    r1 = acosh(complex_double(std::complex<double>(1, 2)));
    r2 = acosh(real_double(-1.0));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.90908861124732)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 3.14159265358979)
            < 1e-7);

    RCP<const ACosh> r4 = make_rcp<ACosh>(i2);
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Atanh: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = atanh(zero);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = atanh(mul(im1, x))->diff(x);
    r2 = div(im1, sub(one, pow(x, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = atanh(mul(i2, y))->diff(y);
    r2 = div(i2, sub(one, mul(i4, pow(y, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = atanh(mul(im1, x));
    r2 = mul(im1, atanh(x));
    REQUIRE(eq(*r1, *r2));

    r1 = atanh(neg(i2));
    r2 = atanh(i2);
    REQUIRE(eq(*r1, *neg(r2)));

    r1 = atanh(real_double(0.5));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.549306144334055)
            < 1e-12);

    r1 = atanh(complex_double(std::complex<double>(1, 1)));
    r2 = atanh(real_double(2.0));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.09390752881482)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 1.66407281705924)
            < 1e-12);

    REQUIRE(eq(*atanh(sub(x, y)), *neg(atanh(sub(y, x)))));

    RCP<const ATanh> r4 = make_rcp<ATanh>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Acoth: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = acoth(mul(im1, x))->diff(x);
    r2 = div(im1, sub(one, pow(x, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = acoth(mul(im1, x));
    r2 = mul(im1, acoth(x));
    REQUIRE(eq(*r1, *r2));

    r1 = acoth(neg(i2));
    r2 = acoth(i2);
    REQUIRE(eq(*r1, *neg(r2)));

    REQUIRE(eq(*acoth(sub(x, y)), *neg(acoth(sub(y, x)))));

    r1 = acoth(real_double(3.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.346573590279973)
            < 1e-12);

    r1 = acoth(complex_double(std::complex<double>(1, 2)));
    r2 = acoth(real_double(0.5));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 0.429232899644131)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 1.66407281705924)
            < 1e-12);

    RCP<const ACoth> r4 = make_rcp<ACoth>(i2);
    REQUIRE(not(r4->is_canonical(neg(i2))));
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Asech: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = asech(one);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = asech(zero);
    REQUIRE(eq(*r1, *Inf));

    r1 = asech(real_double(0.5));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 1.31695789692482)
            < 1e-12);

    r1 = asech(real_double(-0.5));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 3.40646187463796)
            < 1e-12);

    r1 = asech(complex_double(std::complex<double>(1, 1)));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.2380058304943624757)
            < 1e-12);

    r1 = asech(x)->diff(x);
    r2 = div(im1, mul(sqrt(sub(one, pow(x, i2))), x));
    REQUIRE(eq(*r1, *r2));

    RCP<const ASech> r4 = make_rcp<ASech>(i2);
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Kronecker Delta: functions", "[functions]")
{
    RCP<const Symbol> i = symbol("i");
    RCP<const Symbol> j = symbol("j");
    RCP<const Symbol> _x1 = symbol("_xi_1");
    RCP<const Symbol> _x2 = symbol("_xi_2");
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = kronecker_delta(i, i);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(i, j)->diff(i);
    r2 = Derivative::create(kronecker_delta(i, j), {i});
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(i, j)->diff(j);
    r2 = Derivative::create(kronecker_delta(i, j), {j});
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(i, mul(j, j))->diff(j);
    r2 = mul(i2, mul(j, Subs::create(
                            Derivative::create(kronecker_delta(i, _x2), {_x2}),
                            {{_x2, mul(j, j)}})));
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(i, mul(j, j))->diff(i);
    r2 = Derivative::create(kronecker_delta(i, mul(j, j)), {i});
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(mul(i, i), j)->diff(i);
    r2 = mul(i2, mul(i, Subs::create(
                            Derivative::create(kronecker_delta(_x1, j), {_x1}),
                            {{_x1, mul(i, i)}})));
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(mul(i, i), j)->diff(j);
    r2 = Derivative::create(kronecker_delta(mul(i, i), j), {j});
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(mul(i, i), i)->diff(i);
    r2 = mul(i2, mul(i, Subs::create(
                            Derivative::create(kronecker_delta(_x1, i), {_x1}),
                            {{_x1, mul(i, i)}})));
    r2 = add(r2, Subs::create(
                     Derivative::create(kronecker_delta(mul(i, i), _x2), {_x2}),
                     {{_x2, i}}));
    REQUIRE(eq(*r1, *r2));

    r1 = kronecker_delta(i, add(i, one));
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    RCP<const KroneckerDelta> r4 = make_rcp<KroneckerDelta>(i2, i);
    REQUIRE(not(r4->is_canonical(i2, i2)));
    REQUIRE(not(r4->is_canonical(i, i)));
    REQUIRE(not(r4->is_canonical(i2, add(i2, i2))));
    REQUIRE(r4->is_canonical(i, j));
}

TEST_CASE("Zeta: functions", "[functions]")
{
    RCP<const Symbol> s = symbol("s");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> im3 = integer(-3);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i3 = integer(3);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = zeta(zero, x);
    r2 = sub(div(one, i2), x);
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(one);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = zeta(zero, im1);
    r2 = div(integer(3), i2);
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(zero, i2);
    r2 = div(integer(-3), i2);
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(i2, i2);
    r2 = add(div(pow(pi, i2), integer(6)), im1);
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(im3, i2);
    r2 = rational(-119, 120);
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(integer(-5), integer(3));
    r2 = rational(-8317, 252);
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(integer(3), i2);
    REQUIRE(r1->__str__() == "zeta(3, 2)");

    r1 = zeta(x, i2);
    REQUIRE(r1->__str__() == "zeta(x, 2)");

    r2 = zeta(s, x);
    REQUIRE(r1->compare(*r2) == 1);

    r1 = zeta(i1, i2);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = zeta(one, x)->diff(x);
    REQUIRE(eq(*r1, *zero));

    r1 = zeta(zero, x)->diff(x);
    REQUIRE(eq(*r1, *im1));

    r1 = zeta(s, x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = zeta(i2, x)->diff(x);
    r2 = mul(im2, zeta(i3, x));
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(s, x)->diff(s);
    r2 = Derivative::create(zeta(s, x), {s});
    REQUIRE(eq(*r1, *r2));
    REQUIRE(r1->compare(*r2) == 0);

    r1 = zeta(one, i2)->diff(x);
    REQUIRE(eq(*r1, *zero));

    r1 = zeta(pow(x, i2), pow(i2, x))->diff(x);
    r2 = add(mul(mul(mul(mul(pow(i2, x), pow(x, i2)), log(i2)),
                     zeta(add(pow(x, i2), i1), pow(i2, x))),
                 im1),
             mul(mul(i2, x), Subs::create(Derivative::create(
                                              zeta(_xi_1, pow(i2, x)), {_xi_1}),
                                          {{_xi_1, pow(x, i2)}})));
    REQUIRE(eq(*r1, *r2));

    r1 = zeta(s, pow(s, i2))->diff(x);
    REQUIRE(eq(*r1, *zero));

    r1 = zeta(pow(s, i3), pow(i2, x))->diff(s);
    r2 = mul(i3,
             mul(pow(s, i2), Subs::create(Derivative::create(
                                              zeta(_xi_1, pow(i2, x)), {_xi_1}),
                                          {{_xi_1, pow(s, i3)}})));
    REQUIRE(eq(*r1, *r2));

    RCP<const Zeta> r4 = make_rcp<Zeta>(i2, x);
    REQUIRE(not(r4->is_canonical(zero, i2)));
    REQUIRE(not(r4->is_canonical(one, i2)));
    REQUIRE(not(r4->is_canonical(i2, add(i2, i2))));
    REQUIRE(r4->is_canonical(x, y));
}

TEST_CASE("Levi Civita: functions", "[functions]")
{
    RCP<const Symbol> i = symbol("i");
    RCP<const Symbol> j = symbol("j");
    RCP<const Symbol> k = symbol("k");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Symbol> _xi_2 = symbol("_xi_2");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = levi_civita({one, i2, i3});
    REQUIRE(eq(*r1, *one));

    r1 = levi_civita({one, i3, i2});
    REQUIRE(eq(*r1, *im1));

    r1 = levi_civita({one, one, i2});
    REQUIRE(eq(*r1, *zero));

    r1 = levi_civita({i, j, i});
    REQUIRE(eq(*r1, *zero));

    r1 = levi_civita({i2, i4});
    REQUIRE(eq(*r1, *i2));

    r1 = levi_civita({one, i2, i3})->diff(i);
    REQUIRE(eq(*r1, *zero));

    r1 = levi_civita({i, j, k})->diff(k);
    r2 = Derivative::create(levi_civita({i, j, k}), {k});
    REQUIRE(eq(*r1, *r2));

    r1 = levi_civita({one, i2, k})->diff(k);
    r2 = Derivative::create(levi_civita({one, i2, k}), {k});
    REQUIRE(eq(*r1, *r2));

    r1 = levi_civita({i, j, k})->diff(x);
    REQUIRE(eq(*r1, *zero));

    r1 = levi_civita({j, j, k})->diff(j);
    REQUIRE(eq(*r1, *zero));

    r1 = levi_civita({i, j, k});
    r2 = levi_civita({j, k, x});
    REQUIRE(r1->compare(*r2) == -1);

    r1 = levi_civita({pow(x, i2), y})->diff(x);
    r2 = mul(mul(i2, x),
             Subs::create(Derivative::create(levi_civita({_xi_1, y}), {_xi_1}),
                          {{_xi_1, pow(x, i2)}}));
    REQUIRE(eq(*r1, *r2));

    r1 = levi_civita({pow(x, i2), mul(i2, x)})->diff(x);
    r2 = add(
        mul(mul(i2, x),
            Subs::create(
                Derivative::create(levi_civita({_xi_1, mul(i2, x)}), {_xi_1}),
                {{_xi_1, pow(x, i2)}})),
        mul(i2, Subs::create(Derivative::create(
                                 levi_civita({pow(x, i2), _xi_2}), {_xi_2}),
                             {{_xi_2, mul(i2, x)}})));
    REQUIRE(eq(*r1, *r2));

    vec_basic temp = {i2, i};
    RCP<const LeviCivita> r4 = make_rcp<LeviCivita>(std::move(temp));
    REQUIRE(not r4->is_canonical({i2, i2}));
    REQUIRE(not(r4->is_canonical({i2, i, i})));
    REQUIRE(r4->is_canonical({i2, i}));
}

TEST_CASE("Dirichlet Eta: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = dirichlet_eta(one);
    r2 = log(i2);
    REQUIRE(eq(*r1, *r2));

    r1 = dirichlet_eta(zero);
    r2 = div(one, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = dirichlet_eta(zeta(one));
    r2 = dirichlet_eta(ComplexInf);
    REQUIRE(eq(*r1, *r2));

    r1 = dirichlet_eta(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = mul(x, dirichlet_eta(x))->diff(x);
    r2 = add(dirichlet_eta(x), mul(x, dirichlet_eta(x)->diff(x)));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, dirichlet_eta(y))->diff(x);
    r2 = dirichlet_eta(y);
    REQUIRE(eq(*r1, *r2));

    r1 = dirichlet_eta(x)->diff(x);
    r2 = Derivative::create(dirichlet_eta(x), {x});
    REQUIRE(eq(*r1, *r2));

    r1 = dirichlet_eta(pow(x, i2))->diff(x);
    r2 = mul(mul(i2, x),
             Subs::create(Derivative::create(dirichlet_eta(_xi_1), {_xi_1}),
                          {{_xi_1, pow(x, i2)}}));
    REQUIRE(eq(*r1, *r2));

    r1 = dirichlet_eta(pow(x, x))->diff(x);
    r2 = mul(mul(pow(x, x), add(log(x), one)),
             Subs::create(Derivative::create(dirichlet_eta(_xi_1), {_xi_1}),
                          {{_xi_1, pow(x, x)}}));
    REQUIRE(eq(*r1, *r2));

    r1 = make_rcp<Dirichlet_eta>(i3)->rewrite_as_zeta();
    r2 = mul(sub(one, pow(i2, sub(one, i3))), zeta(i3));
    REQUIRE(eq(*r1, *r2));

    RCP<const Dirichlet_eta> r4 = make_rcp<Dirichlet_eta>(i3);
    REQUIRE(not r4->is_canonical(one));
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(i2)));
}

TEST_CASE("Erf: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i2 = integer(2);

    r1 = erf(zero);
    REQUIRE(eq(*r1, *zero));

    r1 = erf(real_double(1.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.84270079294971)
            < 1e-12);

    CHECK_THROWS_AS(erf(complex_double(std::complex<double>(1, 1))),
                    NotImplementedError);

    r1 = erf(mul(i2, x));
    r2 = exp(mul(integer(-4), (mul(x, x))));
    r2 = div(mul(integer(4), r2), sqrt(pi));
    REQUIRE(eq(*r1->diff(x), *r2));

    r2 = add(x, y);
    r1 = erf(r2);
    r2 = exp(neg(mul(r2, r2)));
    r2 = mul(div(i2, sqrt(pi)), r2);
    REQUIRE(eq(*r1->diff(x), *r2));

    REQUIRE(eq(*erf(neg(x)), *neg(erf(x))));
    REQUIRE(eq(*erf(sub(x, y)), *neg(erf(sub(y, x)))));

    RCP<const Erf> r4 = make_rcp<Erf>(i2);
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Erfc: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i2 = integer(2);

    r1 = erfc(zero);
    REQUIRE(eq(*r1, *one));

    r1 = erfc(real_double(1.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.15729920705028)
            < 1e-12);

    CHECK_THROWS_AS(erfc(complex_double(std::complex<double>(1, 1))),
                    NotImplementedError);

    r1 = erfc(mul(i3, x));
    r2 = exp(mul(integer(-9), (mul(x, x))));
    r2 = div(mul(integer(6), r2), sqrt(pi));
    REQUIRE(eq(*r1->diff(x), *neg(r2)));

    REQUIRE(eq(*erfc(neg(x)), *sub(i2, erfc(x))));
    REQUIRE(eq(*erfc(sub(y, x)), *sub(i2, erfc(sub(x, y)))));

    RCP<const Erfc> r4 = make_rcp<Erfc>(i2);
    REQUIRE(not(r4->is_canonical(neg(x))));
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("Gamma: functions", "[functions]")
{
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> sqrt_pi = sqrt(pi);
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> r1;
    RCP<const Basic> r2;
    RCP<const Basic> r3;

    r1 = gamma(one);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(minus_one);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = gamma(mul(i2, i2));
    r2 = mul(i2, i3);
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(div(i3, i2));
    r2 = div(sqrt(pi), i2);
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(div(one, i2));
    r2 = sqrt(pi);
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(div(im1, i2));
    r2 = mul(mul(im1, i2), sqrt(pi));
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(real_double(3.7));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 4.17065178379660)
            < 1e-12);

    CHECK_THROWS_AS(gamma(complex_double(std::complex<double>(1, 1))),
                    NotImplementedError);

    r1 = gamma(div(integer(-15), i2));
    r2 = mul(div(integer(256), integer(2027025)), sqrt(pi));
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = gamma(x)->diff(x);
    r2 = mul(gamma(x), polygamma(zero, x));
    REQUIRE(eq(*r1, *r2));

    r3 = add(mul(x, x), y);
    r1 = gamma(r3)->diff(x);
    r2 = mul(mul(gamma(r3), polygamma(zero, r3)), mul(i2, x));
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(r3)->diff(y);
    r2 = mul(gamma(r3), polygamma(zero, r3));
    REQUIRE(eq(*r1, *r2));

    r3 = sub(im1, x);
    r1 = gamma(r3)->diff(x);
    r2 = neg((mul(gamma(r3), polygamma(zero, r3))));
    REQUIRE(eq(*r1, *r2));

    r1 = gamma(add(x, y))->subs({{x, y}});
    r2 = gamma(add(y, y));
    REQUIRE(eq(*r1, *r2));

    RCP<const Gamma> r4 = make_rcp<Gamma>(x);
    REQUIRE(not(r4->is_canonical(i2)));
    REQUIRE(r4->is_canonical(y));
    REQUIRE(not(r4->is_canonical(div(one, i2))));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
}

TEST_CASE("LogGamma: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = loggamma(integer(1));
    REQUIRE(eq(*r1, *zero));

    r2 = loggamma(integer(2));
    REQUIRE(eq(*r2, *zero));

    r1 = loggamma(integer(3));
    REQUIRE(eq(*r1, *log(integer(2))));

    r1 = loggamma(integer(0));
    REQUIRE(eq(*r1, *Inf));

    r1 = loggamma(x);
    r1 = SymEngine::rcp_dynamic_cast<const LogGamma>(r1)->rewrite_as_gamma();
    REQUIRE(eq(*r1, *log(gamma(x))));

    r1 = loggamma(x)->diff(x);
    r2 = polygamma(zero, x);
    REQUIRE(eq(*r1, *r2));

    r1 = loggamma(x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r2 = mul(x, y);
    r1 = loggamma(r2)->diff(x);
    r2 = mul(polygamma(zero, r2), y);
    REQUIRE(eq(*r1, *r2));

    r1 = loggamma(x)->subs({{x, y}});
    r2 = loggamma(y);
    REQUIRE(eq(*r1, *r2));

    r1 = loggamma(add(y, mul(x, y)))->subs({{y, x}});
    r2 = loggamma(add(x, mul(x, x)));
    REQUIRE(eq(*r1, *r2));

    RCP<const LogGamma> r4 = make_rcp<LogGamma>(x);
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(integer(2))));
    REQUIRE(not(r4->is_canonical(integer(3))));
}

TEST_CASE("Lowergamma: functions", "[functions]")
{
    RCP<const Symbol> s = symbol("s");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = lowergamma(one, i2);
    r2 = sub(one, exp(mul(im1, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(i2, i2);
    r2 = sub(one, mul(i3, exp(mul(im1, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(div(i3, i2), i2);
    r2 = add(mul(mul(minus_one, sqrt(i2)), exp(mul(i2, minus_one))),
             mul(mul(div(one, integer(2)), sqrt(pi)), erf(sqrt(i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(mul(i2, i3), i2);
    r2 = sub(integer(120), mul(integer(872), exp(mul(im1, i2))));
    REQUIRE(eq(*expand(r1), *r2));

    r1 = lowergamma(s, x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = lowergamma(s, x)->diff(s);
    r2 = Derivative::create(lowergamma(s, x), {s});
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(pow(s, i2), x)->diff(s);
    r2 = mul(mul(i2, s),
             Subs::create(Derivative::create(lowergamma(_xi_1, x), {_xi_1}),
                          {{_xi_1, pow(s, i2)}}));
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(pow(s, i2), x)->diff(x);
    r2 = mul(pow(x, add(pow(s, i2), minus_one)), exp(mul(minus_one, x)));
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(pow(s, i2), pow(x, i2))->diff(x);
    r2 = mul(mul(pow(pow(x, i2), add(pow(s, i2), minus_one)),
                 exp(mul(minus_one, pow(x, i2)))),
             mul(x, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = lowergamma(pow(x, i2), pow(i2, x))->diff(x);
    r2 = add(mul(mul(pow(pow(i2, x), add(pow(x, i2), minus_one)),
                     mul(exp(mul(minus_one, pow(i2, x))), log(i2))),
                 pow(i2, x)),
             mul(mul(i2, x),
                 Subs::create(
                     Derivative::create(lowergamma(_xi_1, pow(i2, x)), {_xi_1}),
                     {{_xi_1, pow(x, i2)}})));
    REQUIRE(eq(*r1, *r2));

    RCP<const LowerGamma> r4 = make_rcp<LowerGamma>(x, y);
    REQUIRE(not(r4->is_canonical(one, x)));
    REQUIRE(not(r4->is_canonical(i2, y)));
    REQUIRE(not(r4->is_canonical(div(one, i2), i2)));
}

TEST_CASE("Uppergamma: functions", "[functions]")
{
    RCP<const Symbol> s = symbol("s");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> _xi_1 = symbol("_xi_1");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im1 = integer(-1);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = uppergamma(one, i2);
    r2 = exp(mul(im1, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(i2, i2);
    r2 = mul(i3, exp(mul(im1, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(div(i3, i2), i2);
    r2 = add(mul(sqrt(i2), exp(mul(i2, minus_one))),
             mul(mul(div(one, integer(2)), sqrt(pi)), erfc(sqrt(i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(mul(i2, i3), i2);
    r2 = mul(integer(872), exp(mul(im1, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(s, x)->diff(y);
    REQUIRE(eq(*r1, *zero));

    r1 = uppergamma(s, x)->diff(s);
    r2 = Derivative::create(uppergamma(s, x), {s});
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(pow(s, i2), x)->diff(s);
    r2 = mul(mul(i2, s),
             Subs::create(Derivative::create(uppergamma(_xi_1, x), {_xi_1}),
                          {{_xi_1, pow(s, i2)}}));
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(pow(s, i2), x)->diff(x);
    r2 = mul(mul(pow(x, add(pow(s, i2), minus_one)), exp(mul(minus_one, x))),
             minus_one);
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(pow(s, i2), pow(x, i2))->diff(x);
    r2 = mul(mul(mul(pow(pow(x, i2), add(pow(s, i2), minus_one)),
                     exp(mul(minus_one, pow(x, i2)))),
                 mul(x, i2)),
             minus_one);
    REQUIRE(eq(*r1, *r2));

    r1 = uppergamma(pow(x, i2), pow(i2, x))->diff(x);
    r2 = add(mul(mul(mul(pow(pow(i2, x), add(pow(x, i2), minus_one)),
                         mul(exp(mul(minus_one, pow(i2, x))), log(i2))),
                     pow(i2, x)),
                 minus_one),
             mul(mul(i2, x),
                 Subs::create(
                     Derivative::create(uppergamma(_xi_1, pow(i2, x)), {_xi_1}),
                     {{_xi_1, pow(x, i2)}})));
    REQUIRE(eq(*r1, *r2));

    RCP<const UpperGamma> r4 = make_rcp<UpperGamma>(x, y);
    REQUIRE(not(r4->is_canonical(one, x)));
    REQUIRE(not(r4->is_canonical(i2, y)));
    REQUIRE(not(r4->is_canonical(div(one, i2), i2)));
}

TEST_CASE("Beta: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> r2_5 = Rational::from_two_ints(*integer(2), *integer(5));
    RCP<const Basic> r5_2 = Rational::from_two_ints(*integer(5), *integer(2));

    RCP<const Basic> r1;
    RCP<const Basic> r2;
    RCP<const Basic> r3;

    r1 = beta(i3, i2);
    r2 = beta(i2, i3);
    REQUIRE(eq(*r1, *r2));
    r3 = div(mul(gamma(i3), gamma(i2)), gamma(add(i2, i3)));
    REQUIRE(eq(*r1, *r3));
    r2 = div(one, integer(12));
    REQUIRE(eq(*r1, *r2));

    r1 = beta(div(one, i2), i2);
    r2 = beta(i2, div(one, i2));
    REQUIRE(eq(*r1, *r2));
    r3 = div(i4, i3);
    REQUIRE(eq(*r3, *r1));

    r1 = beta(div(integer(7), i2), div(integer(9), i2));
    r2 = beta(div(integer(9), i2), div(integer(7), i2));
    REQUIRE(eq(*r1, *r2));
    r3 = div(mul(integer(5), pi), integer(2048));
    REQUIRE(eq(*r3, *r1));

    r1 = beta(div(one, i2), div(i3, i2));
    r2 = div(pi, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = beta(x, y);
    r2 = beta(y, x);
    REQUIRE(eq(*r1, *r2));
    REQUIRE(r1->__hash__() == r2->__hash__());

    r1 = beta(x, y)->diff(x);
    r2 = mul(beta(x, y), sub(polygamma(zero, x), polygamma(zero, add(x, y))));
    REQUIRE(eq(*r1, *r2));

    r1 = beta(x, y)->diff(x);
    r2 = beta(y, x)->diff(x);
    REQUIRE(eq(*r1, *r2));

    r1 = beta(x, mul(x, x))->diff(x);
    r2 = mul(
        beta(x, mul(x, x)),
        add(mul(mul(i2, x), polygamma(zero, mul(x, x))),
            sub(polygamma(zero, x), mul(add(mul(i2, x), one),
                                        polygamma(zero, add(x, mul(x, x)))))));
    REQUIRE(eq(*r1, *r2));

    r1 = beta(i2, im1);
    REQUIRE(eq(*r1, *ComplexInf));
    r1 = beta(i3, im1);
    REQUIRE(eq(*r1, *ComplexInf));
    r1 = beta(im1, im1);
    REQUIRE(eq(*r1, *ComplexInf));
    r1 = beta(r2_5, im1);
    REQUIRE(eq(*r1, *ComplexInf));
    r1 = beta(one, r5_2);
    r2 = beta(r5_2, one);
    REQUIRE(eq(*r1, *r2));

    r1 = make_rcp<Beta>(y, x)->rewrite_as_gamma();
    r2 = div(mul(gamma(y), gamma(x)), gamma(add(x, y)));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Digamma: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im3 = integer(-3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = digamma(x);
    r2 = polygamma(zero, x);
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(zero);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = digamma(one);
    r2 = neg(EulerGamma);
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(im2);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = digamma(div(integer(5), i2));
    r2 = add(sub(mul(im2, log(i2)), EulerGamma), div(integer(8), integer(3)));
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(div(one, i3));
    r2 = add(neg(div(div(pi, i2), sqrt(i3))),
             sub(div(mul(im3, log(i3)), i2), EulerGamma));
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(div(one, i2));
    r2 = sub(mul(im2, log(i2)), EulerGamma);
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(div(i2, i3));
    r2 = add(div(div(pi, i2), sqrt(i3)),
             sub(div(mul(im3, log(i3)), i2), EulerGamma));
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(div(one, i4));
    r2 = add(neg(div(pi, i2)), sub(mul(im3, log(i2)), EulerGamma));
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(div(i3, i4));
    r2 = add(div(pi, i2), sub(mul(im3, log(i2)), EulerGamma));
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(x)->diff(x);
    r2 = polygamma(one, x);
    REQUIRE(eq(*r1, *r2));

    r1 = digamma(x)->diff(y);
    REQUIRE(eq(*r1, *zero));
}

TEST_CASE("Trigamma: functions", "[functions]")
{
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = trigamma(i3);
    r2 = add(div(integer(-5), i4), div(pow(pi, i2), integer(6)));
    REQUIRE(eq(*r1, *r2));

    r1 = trigamma(im2);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = digamma(x)->diff(x);
    r2 = trigamma(x);
    REQUIRE(eq(*r1, *r2));

    r1 = trigamma(x);
    r2 = polygamma(one, x);
    REQUIRE(eq(*r1, *r2));

    r1 = trigamma(x)->diff(x);
    r2 = polygamma(integer(2), x);
    REQUIRE(eq(*r1, *r2));

    r1 = trigamma(x)->diff(y);
    REQUIRE(eq(*r1, *zero));
}

TEST_CASE("Polygamma: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> _x = symbol("_xi_1");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = polygamma(i2, im2);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = SymEngine::rcp_dynamic_cast<const PolyGamma>(polygamma(i2, x))
             ->rewrite_as_zeta();
    r2 = neg(mul(i2, zeta(i3, x)));
    REQUIRE(eq(*r1, *r2));

    r1 = SymEngine::rcp_dynamic_cast<const PolyGamma>(polygamma(i3, x))
             ->rewrite_as_zeta();
    r2 = mul(integer(6), zeta(i4, x));
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(x, y)->subs({{x, zero}, {y, one}});
    r2 = neg(EulerGamma);
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(x, y)->subs({{y, x}});
    r2 = polygamma(x, x);
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(y, mul(x, i2))->diff(x);
    r2 = mul(i2, polygamma(add(y, one), mul(x, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(y, mul(x, i2))->diff(x);
    r2 = mul(i2, polygamma(add(y, one), mul(x, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(x, y)->diff(x);
    r2 = Derivative::create(polygamma(x, y), {x});
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(mul(i2, x), y)->diff(x);
    r2 = mul(i2, Subs::create(Derivative::create(polygamma(_x, y), {_x}),
                              {{_x, mul(i2, x)}}));
    REQUIRE(eq(*r1, *r2));

    r1 = polygamma(mul(i2, x), mul(i3, x))->diff(x);
    r2 = mul(i2,
             Subs::create(Derivative::create(polygamma(_x, mul(i3, x)), {_x}),
                          {{_x, mul(i2, x)}}));
    r2 = add(r2, mul(i3, polygamma(add(mul(i2, x), one), mul(i3, x))));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Abs: functions", "[functions]")
{
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> sqrt_pi = sqrt(pi);
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    REQUIRE(eq(*abs(add(i2, mul(I, im1))), *sqrt(integer(5))));
    REQUIRE(eq(*abs(add(i2, mul(I, i3))), *sqrt(integer(13))));
    REQUIRE(eq(*abs(x), *abs(neg(x))));
    REQUIRE(eq(*abs(one), *one));
    REQUIRE(eq(*abs(i2), *i2));
    REQUIRE(eq(*abs(im1), *one));
    REQUIRE(eq(*abs(integer(-5)), *integer(5)));
    REQUIRE(neq(*abs(sqrt_pi), *sqrt_pi));
    REQUIRE(eq(*abs(sqrt_pi), *abs(sqrt_pi)));
    REQUIRE(eq(*abs(div(i2, i3)), *div(i2, i3)));
    REQUIRE(eq(*abs(neg(div(i2, i3))), *div(i2, i3)));
    REQUIRE(neq(*abs(x)->diff(x), *integer(0)));
    REQUIRE(eq(*abs(x)->diff(y), *integer(0)));
    REQUIRE(eq(*abs(sub(x, y)), *abs(sub(y, x))));
    REQUIRE(eq(*abs(real_double(-1.0)), *real_double(1.0)));
    REQUIRE(eq(*abs(abs(x)), *abs(x)));
}

class MySin : public FunctionWrapper
{
public:
    MySin(RCP<const Basic> arg) : FunctionWrapper("MySin", arg) {}
    RCP<const Number> eval(long bits) const
    {
        return real_double(::sin(eval_double(*get_vec()[0])));
    }
    RCP<const Basic> create(const vec_basic &v) const
    {
        if (eq(*zero, *v[0])) {
            return zero;
        } else {
            return make_rcp<MySin>(v[0]);
        }
    }
    RCP<const Basic> diff_impl(const RCP<const Symbol> &x) const
    {
        return mul(cos(get_vec()[0]), get_vec()[0]->diff(x));
    }
};

TEST_CASE("FunctionWrapper: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> e = add(one, make_rcp<MySin>(x));
    RCP<const Basic> f;

    f = e->subs({{x, integer(0)}});
    REQUIRE(eq(*f, *one));

    f = e->diff(x);
    REQUIRE(eq(*f, *cos(x)));

    f = e->subs({{x, integer(1)}});
    double d = eval_double(*f);
    REQUIRE(std::fabs(d - 1.84147098480789) < 1e-12);

    REQUIRE(e->__str__() == "1 + MySin(x)");

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class aa(100);
    eval_mpfr(aa.get_mpfr_t(), *f, MPFR_RNDN);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 1.8414709848078) == 1);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 1.8414709848079) == -1);
#ifdef HAVE_SYMENGINE_MPC
    mpc_class a(100);
    eval_mpc(a.get_mpc_t(), *f, MPFR_RNDN);
    mpc_abs(aa.get_mpfr_t(), a.get_mpc_t(), MPFR_RNDN);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 1.8414709848078) == 1);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 1.8414709848079) == -1);
#endif // HAVE_SYMENGINE_MPC
#endif // HAVE_SYMENGINE_MPFR
}
/* ---------------------------- */

TEST_CASE("MPFR and MPC: functions", "[functions]")
{
#ifdef HAVE_SYMENGINE_MPFR
    RCP<const Basic> r1, r2;
    RCP<const Basic> i2 = integer(2);
    integer_class p = 1000000000000000_z;
    integer_class q;

    mpfr_class a(60), b1(60), b2(60), b3(60), b4(60), b5(60), b6(60);
    mpfr_set_ui(b1.get_mpfr_t(), 1, MPFR_RNDN);
    mpfr_set_ui(b2.get_mpfr_t(), 2, MPFR_RNDN);
    mpfr_set_ui(b3.get_mpfr_t(), 3, MPFR_RNDN);
    mpfr_set_d(b4.get_mpfr_t(), 2.2, MPFR_RNDN);
    mpfr_set_d(b5.get_mpfr_t(), 0.5, MPFR_RNDN);
    mpfr_set_d(b6.get_mpfr_t(), 0.4, MPFR_RNDN);

    std::vector<std::tuple<RCP<const Basic>, integer_class, integer_class>>
        testvec = {
            std::make_tuple(sin(real_mpfr(b1)), 841470984807896_z,
                            841470984807897_z),
            std::make_tuple(sin(sub(div(pi, i2), real_mpfr(b2))),
                            -416146836547143_z, -416146836547142_z),
            std::make_tuple(cos(real_mpfr(b1)), 540302305868139_z,
                            540302305868140_z),
            std::make_tuple(tan(real_mpfr(b1)), 1557407724654902_z,
                            1557407724654903_z),
            std::make_tuple(cot(real_mpfr(b2)), -457657554360286_z,
                            -457657554360285_z),
            std::make_tuple(sec(real_mpfr(b3)), -1010108665907994_z,
                            -1010108665907993_z),
            std::make_tuple(csc(real_mpfr(b4)), 1236863881243858_z,
                            1236863881243859_z),

            std::make_tuple(asin(real_mpfr(b1)), 1570796326794896_z,
                            1570796326794897_z),
            std::make_tuple(acos(real_mpfr(b5)), 1047197551196597_z,
                            1047197551196598_z),
            std::make_tuple(atan(real_mpfr(b4)), 1144168833668020_z,
                            1144168833668021_z),
            std::make_tuple(acot(real_mpfr(b5)), 1107148717794090_z,
                            1107148717794091_z),
            std::make_tuple(asec(real_mpfr(b3)), 1230959417340774_z,
                            1230959417340775_z),
            std::make_tuple(acsc(real_mpfr(b3)), 339836909454121_z,
                            339836909454122_z),

            std::make_tuple(sinh(real_mpfr(b2)), 3626860407847018_z,
                            3626860407847019_z),
            std::make_tuple(cosh(real_mpfr(b2)), 3762195691083631_z,
                            3762195691083632_z),
            std::make_tuple(tanh(real_mpfr(b1)), 761594155955764_z,
                            761594155955765_z),
            std::make_tuple(coth(real_mpfr(b4)), 1024859893164471_z,
                            1024859893164472_z),
            std::make_tuple(sech(real_mpfr(b5)), 886818883970073_z,
                            886818883970074_z),
            std::make_tuple(csch(real_mpfr(b4)), 224360871403841_z,
                            2243608714038412_z),

            std::make_tuple(asinh(real_mpfr(b1)), 881373587019543_z,
                            881373587019544_z),
            std::make_tuple(acosh(real_mpfr(b2)), 1316957896924816_z,
                            1316957896924817_z),
            std::make_tuple(atanh(real_mpfr(b5)), 549306144334054_z,
                            549306144334055_z),
            std::make_tuple(acoth(real_mpfr(b4)), 490414626505863_z,
                            490414626505864_z),
            std::make_tuple(asech(real_mpfr(b6)), 1566799236972411_z,
                            1566799236972412_z),
            std::make_tuple(acsch(real_mpfr(b4)), 440191235352683_z,
                            440191235352684_z),

            std::make_tuple(log(real_mpfr(b4)), 788457360364270_z,
                            788457360364271_z),
            std::make_tuple(gamma(div(real_mpfr(b3), i2)), 886226925452758_z,
                            886226925452759_z),
            std::make_tuple(exp(real_mpfr(b4)), 9025013499434122_z,
                            9025013499434123_z),
            std::make_tuple(erf(real_mpfr(b2)), 995322265018952_z,
                            995322265018953_z),
            std::make_tuple(erfc(real_mpfr(b2)), 4677734981047_z,
                            4677734981048_z),
        };

    mpfr_set_ui(a.get_mpfr_t(), 1, MPFR_RNDN);
    r1 = asech(real_mpfr(a));
    REQUIRE(is_a<RealMPFR>(*r1));

    mpfr_mul_z(a.get_mpfr_t(), down_cast<const RealMPFR &>(*r1).i.get_mpfr_t(),
               get_mpz_t(p), MPFR_RNDN);
    q = 0_z;
    REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(q)) == 0);

    mpfr_set_si(a.get_mpfr_t(), -22, MPFR_RNDN);
    r1 = abs(real_mpfr(a));
    q = 22_z;
    REQUIRE(mpfr_cmp_z(down_cast<const RealMPFR &>(*r1).i.get_mpfr_t(),
                       get_mpz_t(q))
            == 0);

    mpfr_set_si(a.get_mpfr_t(), -3, MPFR_RNDN);
    CHECK_THROWS_AS(gamma(real_mpfr(a)), NotImplementedError);

    for (unsigned i = 0; i < testvec.size(); i++) {
        r1 = std::get<0>(testvec[i]);
        REQUIRE(is_a<RealMPFR>(*r1));
        mpfr_mul_z(a.get_mpfr_t(),
                   down_cast<const RealMPFR &>(*r1).i.get_mpfr_t(),
                   get_mpz_t(p), MPFR_RNDN);
        REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(std::get<1>(testvec[i])))
                > 0);
        REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(std::get<2>(testvec[i])))
                < 0);
    }
    testvec.clear();

#ifdef HAVE_SYMENGINE_MPC

    mpc_srcptr b;
    mpc_class c(60), d1(60), d2(60), d3(60);
    mpc_set_si_si(d1.get_mpc_t(), 1, 1, MPFR_RNDN);
    mpc_set_si_si(d2.get_mpc_t(), 2, 3, MPFR_RNDN);
    mpc_set_si_si(d3.get_mpc_t(), 4, 5, MPFR_RNDN);

    testvec = {
        std::make_tuple(asin(real_mpfr(b2)), 2049824188203704_z,
                        2049824188203705_z),
        std::make_tuple(acos(real_mpfr(b2)), 1316957896924816_z,
                        1316957896924817_z),
        std::make_tuple(asec(real_mpfr(b6)), 1566799236972411_z,
                        1566799236972412_z),
        std::make_tuple(acsc(real_mpfr(b6)), 2218616900064017_z,
                        2218616900064018_z),
        std::make_tuple(acosh(real_mpfr(b5)), 1047197551196597_z,
                        1047197551196598_z),
        std::make_tuple(atanh(real_mpfr(b2)), 1664072817059243_z,
                        1664072817059244_z),
        std::make_tuple(acoth(real_mpfr(b6)), 1626923328349102_z,
                        1626923328349103_z),
        std::make_tuple(asech(real_mpfr(b2)), 1047197551196597_z,
                        1047197551196598_z),
        std::make_tuple(log(sub(real_mpfr(b1), real_mpfr(b3))),
                        3217150511711809_z, 3217150511711810_z),

        std::make_tuple(sin(complex_mpc(d2)), 10059057603556098_z,
                        10059057603556099_z),
        std::make_tuple(cos(complex_mpc(d2)), 10026514661176940_z,
                        10026514661176941_z),
        std::make_tuple(tan(complex_mpc(d2)), 1003245688405081_z,
                        1003245688405082_z),
        std::make_tuple(csc(complex_mpc(d2)), 99412891287795_z,
                        99412891287796_z),
        std::make_tuple(sec(complex_mpc(d2)), 99735554556364_z,
                        99735554556365_z),
        std::make_tuple(cot(complex_mpc(d2)), 996764812007075_z,
                        996764812007076_z),

        std::make_tuple(asin(complex_mpc(d2)), 2063848034787096_z,
                        2063848034787097_z),
        std::make_tuple(acos(complex_mpc(d3)), 2706069014027540_z,
                        2706069014027541_z),
        std::make_tuple(atan(complex_mpc(d2)), 1428408786089582_z,
                        1428408786089583_z),
        std::make_tuple(acsc(complex_mpc(d2)), 275919504119167_z,
                        275919504119168_z),
        std::make_tuple(asec(complex_mpc(d2)), 1439125555072813_z,
                        1439125555072814_z),
        std::make_tuple(acot(complex_mpc(d3)), 156440457398915_z,
                        156440457398916_z),

        std::make_tuple(sinh(complex_mpc(d2)), 3629604837263012_z,
                        3629604837263013_z),
        std::make_tuple(cosh(complex_mpc(d3)), 27291391405744611_z,
                        27291391405744612_z),
        std::make_tuple(tanh(complex_mpc(d2)), 965436479673952_z,
                        965436479673953_z),
        std::make_tuple(csch(complex_mpc(d2)), 275512085980707_z,
                        275512085980708_z),
        std::make_tuple(sech(complex_mpc(d2)), 265989418396841_z,
                        265989418396842_z),
        std::make_tuple(coth(complex_mpc(d3)), 999437204152625_z,
                        999437204152626_z),

        std::make_tuple(asinh(complex_mpc(d2)), 2192282215636676_z,
                        2192282215636677_z),
        std::make_tuple(acosh(complex_mpc(d2)), 2221285937468018_z,
                        2221285937468019_z),
        std::make_tuple(atanh(complex_mpc(d3)), 1451512702064822_z,
                        1451512702064823_z),
        std::make_tuple(acsch(complex_mpc(d3)), 156308000814648_z,
                        156308000814649_z),
        std::make_tuple(acoth(complex_mpc(d3)), 155883315867942_z,
                        155883315867943_z),
        std::make_tuple(asech(complex_mpc(d2)), 1439125555072813_z,
                        1439125555072814_z),

        std::make_tuple(log(complex_mpc(d2)), 1615742802564794_z,
                        1615742802564795_z),
        std::make_tuple(exp(complex_mpc(d1)), 2718281828459045_z,
                        2718281828459046_z),
    };

    r1 = abs(complex_mpc(d2));
    REQUIRE(is_a<RealMPFR>(*r1));
    mpfr_mul_z(a.get_mpfr_t(), down_cast<const RealMPFR &>(*r1).i.get_mpfr_t(),
               get_mpz_t(p), MPFR_RNDN);
    REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(3605551275463989_z)) == 1);
    REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(3605551275463990_z)) == -1);

    for (unsigned i = 0; i < testvec.size(); i++) {
        r1 = std::get<0>(testvec[i]);
        REQUIRE(is_a<ComplexMPC>(*r1));
        b = down_cast<const ComplexMPC &>(*r1).as_mpc().get_mpc_t();
        mpc_abs(a.get_mpfr_t(), b, MPFR_RNDN);
        mpfr_mul_z(a.get_mpfr_t(), a.get_mpfr_t(), get_mpz_t(p), MPFR_RNDN);
        REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(std::get<1>(testvec[i])))
                > 0);
        REQUIRE(mpfr_cmp_z(a.get_mpfr_t(), get_mpz_t(std::get<2>(testvec[i])))
                < 0);
    }

    mpc_set_si_si(c.get_mpc_t(), 1, 1, MPFR_RNDN);
    CHECK_THROWS_AS(erf(complex_mpc(c)), NotImplementedError);
    CHECK_THROWS_AS(erfc(complex_mpc(c)), NotImplementedError);
    CHECK_THROWS_AS(gamma(complex_mpc(c)), NotImplementedError);
#else
    mpfr_set_si(a.get_mpfr_t(), 2, MPFR_RNDN);
    CHECK_THROWS_AS(asin(real_mpfr(a)), SymEngineException);
    CHECK_THROWS_AS(asech(real_mpfr(a)), SymEngineException);
#endif // HAVE_SYMENGINE_MPC
#endif // HAVE_SYMENGINE_MPFR
}

TEST_CASE("max: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r2_5 = Rational::from_two_ints(*integer(2), *integer(5));
    RCP<const Basic> rd = real_double(0.32);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> c = Complex::from_two_nums(*one, *one);

    RCP<const Basic> res, tmp;

    res = max({x, y}); // checking if elements stored in order
    tmp = down_cast<const Max &>(*res).get_args()[0];
    res = max({y, x});
    REQUIRE(eq(*(down_cast<const Max &>(*res).get_args()[0]), *tmp));

    res = max({x, y});
    REQUIRE(eq(*res, *max({y, x}))); // max(x, y) == max(y, x)
    REQUIRE(is_a<Max>(*res));        // max(x, y) is a Max

    res = max({x});
    REQUIRE(eq(*res, *x)); // max(x) == x

    res = max({Inf});
    REQUIRE(eq(*res, *Inf));

    res = max({NegInf, one});
    REQUIRE(eq(*res, *one));

    res = max({x, x});
    REQUIRE(eq(*res, *x)); // max(x, x) == x

    res = max({x, x, max({x, y})});
    REQUIRE(eq(*res, *max({x, y}))); // max(x, x, max(x, y)) == max(x,y)

    res = max({i2, rd, r2_5});
    REQUIRE(eq(*res, *i2)); // max(2, 2/5, 0.32) == 2

    res = max({x, max({i2, y})});
    REQUIRE(eq(*res, *max({x, i2, y}))); // max(x, max(2, y)) == max(x, 2, y)

    res = max({max({x, max({y, i2})}), max({r2_5, rd})});
    REQUIRE(eq(
        *res,
        *max({x, i2,
              y}))); // max(max(x, max(y, 2)), max(2/5, 0.32)) == max(x, 2, y)

    res = max({i2, r2_5, x});
    REQUIRE(eq(*res, *max({i2, x}))); // max(2, 2/5, x) == max(2, x)

    res = max({max({x, i2}), max({y, r2_5})});
    REQUIRE(eq(
        *res, *max({x, i2, y}))); // max(max(2, x), max(2/5, y)) == max(x, 2, y)

    CHECK_THROWS_AS(min({}), SymEngineException);

    CHECK_THROWS_AS(min({c}), SymEngineException);
}

TEST_CASE("min: functions", "[functions]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r2_5 = Rational::from_two_ints(*integer(2), *integer(5));
    RCP<const Basic> rd = real_double(0.32);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> c = Complex::from_two_nums(*one, *one);

    RCP<const Basic> res;

    res = min({x, y});
    REQUIRE(eq(*res, *min({y, x}))); // min(x, y) == min(y, x)
    REQUIRE(is_a<Min>(*res));        // min(x, y) is a min

    res = min({x});
    REQUIRE(eq(*res, *x)); // min(x) == x

    res = min({x, x});
    REQUIRE(eq(*res, *x)); // min(x, x) == x

    res = min({x, x, min({x, y})});
    REQUIRE(eq(*res, *min({x, y}))); // min(x, x, min(x, y)) == min(x,y)

    res = min({i2, rd, r2_5});
    REQUIRE(eq(*res, *rd)); // min(2, 2/5, 0.32) == 0.32

    res = min({i2, rd, max({x})});
    REQUIRE(eq(*res, *min({rd, x}))); // min(2, 0.32, max(x)) == min(0.32, x)

    res = min({x, min({i2, y})});
    REQUIRE(eq(*res, *min({x, i2, y}))); // min(x, min(2, y)) == min(x, 2, y)

    res = min({min({x, min({y, i2})}), min({r2_5, rd})});
    REQUIRE(eq(
        *res,
        *min(
            {x, rd,
             y}))); // min(min(x, min(y, 2)), min(2/5, 0.32)) == min(x, 0.32, y)

    res = min({min({x, i2}), min({y, r2_5})});
    REQUIRE(eq(
        *res,
        *min({x, r2_5, y}))); // min(min(2, x), min(2/5, y)) == min(x, 2/5, y)

    CHECK_THROWS_AS(min({}), SymEngineException);

    CHECK_THROWS_AS(min({c}), SymEngineException);
}

TEST_CASE("test_dummy", "[Dummy]")
{
    RCP<const Symbol> x1 = symbol("x");
    RCP<const Symbol> x2 = symbol("x");
    RCP<const Symbol> xdummy1 = dummy("x");
    RCP<const Symbol> xdummy2 = dummy("x");

    CHECK(eq(*x1, *x2));
    CHECK(neq(*x1, *xdummy1));
    CHECK(neq(*xdummy1, *xdummy2));
    CHECK(neq(*dummy(), *dummy()));
    CHECK(neq(*dummy("x"), *dummy("x")));

    xdummy1 = x1->as_dummy();
    CHECK(neq(*xdummy1, *x1));
    CHECK(neq(*xdummy1, *x1->as_dummy()));

    REQUIRE(xdummy1->compare(*xdummy1) == 0);
}

TEST_CASE("test_sign", "[Sign]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r = sign(one);
    RCP<const Basic> s;
    CHECK(eq(*r, *one));

    r = sign(minus_one);
    CHECK(eq(*r, *minus_one));

    r = sign(zero);
    CHECK(eq(*r, *zero));

    r = sign(minus_one);
    CHECK(eq(*r, *minus_one));

    r = sign(real_double(1.2));
    CHECK(eq(*r, *one));

    r = sign(real_double(0.0));
    CHECK(eq(*r, *zero));

    r = sign(real_double(-1.2));
    CHECK(eq(*r, *minus_one));

    r = sign(rational(-1, 2));
    CHECK(eq(*r, *minus_one));

    r = sign(rational(1, 2));
    CHECK(eq(*r, *one));

    r = sign(rational(0, 2));
    CHECK(eq(*r, *zero));

    r = sign(Inf);
    CHECK(eq(*r, *one));

    r = sign(NegInf);
    CHECK(eq(*r, *minus_one));

    r = sign(ComplexInf);
    CHECK(r->__str__() == "sign(zoo)");

    r = sign(Nan);
    CHECK(eq(*r, *Nan));

    r = sign(complex_double(std::complex<double>(0, 3)));
    CHECK(eq(*r, *I));

    r = sign(complex_double(std::complex<double>(0, -3)));
    CHECK(eq(*r, *mul(I, minus_one)));

    r = sign(pi);
    CHECK(eq(*r, *one));

    r = sign(E);
    CHECK(eq(*r, *one));

    r = sign(sign(x));
    CHECK(eq(*r, *sign(x)));

    r = sign(pow(I, rational(1, 2)));
    s = sign(sqrt(I));
    CHECK(eq(*r, *s));

    r = mul(mul(integer(2), x), pow(y, integer(3)));
    s = sign(mul(x, pow(y, integer(3))));
    CHECK(eq(*sign(r), *s));

    r = mul(mul(mul(integer(2), x), pow(y, integer(3))), I);
    s = mul(sign(mul(x, pow(y, integer(3)))), I);
    CHECK(eq(*sign(r), *s));

    r = mul(mul(mul(integer(2), x), pow(y, integer(3))),
            pow(mul(integer(3), I), integer(3)));
    s = mul(sign(mul(x, pow(y, integer(3)))), mul(I, minus_one));
    CHECK(eq(*sign(r), *s));

    r = sign(mul(mul(pow(Complex::from_two_nums(*integer(2), *integer(3)),
                         Rational::from_two_ints(3, 2)),
                     x),
                 pow(mul(integer(3), I), integer(3))));
    s = mul(mul(I, minus_one),
            sign(mul(x, pow(Complex::from_two_nums(*integer(2), *integer(3)),
                            Rational::from_two_ints(3, 2)))));
    CHECK(eq(*r, *s));
}

TEST_CASE("test_floor", "[Floor]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> r = floor(integer(1));
    CHECK(eq(*r, *one));

    r = floor(Complex::from_two_nums(*integer(2), *integer(1)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(2), *integer(1))));

    r = floor(Nan);
    CHECK(eq(*r, *Nan));

    r = floor(Inf);
    CHECK(eq(*r, *Inf));

    r = floor(NegInf);
    CHECK(eq(*r, *NegInf));

    r = floor(Rational::from_two_ints(3, 1));
    CHECK(eq(*r, *integer(3)));

    r = floor(Rational::from_two_ints(3, 2));
    CHECK(eq(*r, *integer(1)));

    r = floor(Rational::from_two_ints(-3, 2));
    CHECK(eq(*r, *integer(-2)));

    r = floor(real_double(2.65));
    CHECK(eq(*r, *integer(2)));

    r = floor(complex_double(std::complex<double>(2.86, 2.79)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(2), *integer(2))));

    r = floor(pi);
    CHECK(eq(*r, *integer(3)));

    r = floor(add(pi, integer(4)));
    CHECK(eq(*r, *integer(7)));

    r = floor(E);
    CHECK(eq(*r, *integer(2)));

    r = floor(floor(x));
    CHECK(eq(*r, *floor(x)));

    r = floor(ceiling(x));
    CHECK(eq(*r, *ceiling(x)));

    r = floor(truncate(x));
    CHECK(eq(*r, *truncate(x)));

    r = floor(add(add(integer(2), mul(integer(2), x)), mul(integer(3), y)));
    CHECK(eq(*r, *add(integer(2),
                      floor(add(mul(integer(2), x), mul(integer(3), y))))));

    r = floor(add(add(Rational::from_two_ints(2, 3), mul(integer(2), x)),
                  mul(integer(3), y)));
    CHECK(eq(*r, *floor(add(add(mul(integer(2), x), mul(integer(3), y)),
                            Rational::from_two_ints(2, 3)))));

    CHECK_THROWS_AS(floor(Eq(integer(2), integer(3))), SymEngineException);

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class a(100);
    mpfr_set_d(a.get_mpfr_t(), 10.65, MPFR_RNDN);
    r = floor(real_mpfr(std::move(a)));
    CHECK(eq(*r, *integer(10)));
#endif // HAVE_SYMENGINE_MPFR

#ifdef HAVE_SYMENGINE_MPC
    mpc_class c(100);
    mpc_set_d_d(c.get_mpc_t(), 10.65, 11.47, MPFR_RNDN);
    r = floor(complex_mpc(std::move(c)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(10), *integer(11))));
#endif // HAVE_SYMENGINE_MPC
}

TEST_CASE("test_ceiling", "[Ceiling]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> r = ceiling(integer(1));
    CHECK(eq(*r, *one));

    r = ceiling(Complex::from_two_nums(*integer(2), *integer(1)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(2), *integer(1))));

    r = ceiling(Nan);
    CHECK(eq(*r, *Nan));

    r = ceiling(Inf);
    CHECK(eq(*r, *Inf));

    r = ceiling(NegInf);
    CHECK(eq(*r, *NegInf));

    r = ceiling(Rational::from_two_ints(3, 1));
    CHECK(eq(*r, *integer(3)));

    r = ceiling(Rational::from_two_ints(3, 2));
    CHECK(eq(*r, *integer(2)));

    r = ceiling(Rational::from_two_ints(-3, 2));
    CHECK(eq(*r, *integer(-1)));

    r = ceiling(real_double(2.65));
    CHECK(eq(*r, *integer(3)));

    r = ceiling(complex_double(std::complex<double>(2.86, 2.79)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(3), *integer(3))));

    r = ceiling(pi);
    CHECK(eq(*r, *integer(4)));

    r = ceiling(E);
    CHECK(eq(*r, *integer(3)));

    r = ceiling(floor(x));
    CHECK(eq(*r, *floor(x)));

    r = ceiling(ceiling(x));
    CHECK(eq(*r, *ceiling(x)));

    r = ceiling(truncate(x));
    CHECK(eq(*r, *truncate(x)));

    r = ceiling(add(add(integer(2), mul(integer(2), x)), mul(integer(3), y)));
    CHECK(eq(*r, *add(integer(2),
                      ceiling(add(mul(integer(2), x), mul(integer(3), y))))));

    r = ceiling(add(add(Rational::from_two_ints(2, 3), mul(integer(2), x)),
                    mul(integer(3), y)));
    CHECK(eq(*r, *ceiling(add(add(mul(integer(2), x), mul(integer(3), y)),
                              Rational::from_two_ints(2, 3)))));

    CHECK_THROWS_AS(ceiling(Eq(integer(2), integer(3))), SymEngineException);

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class a(100);
    mpfr_set_d(a.get_mpfr_t(), 10.65, MPFR_RNDN);
    r = ceiling(real_mpfr(std::move(a)));
    CHECK(eq(*r, *integer(11)));
#endif // HAVE_SYMENGINE_MPFR

#ifdef HAVE_SYMENGINE_MPC
    mpc_class b(100);
    mpc_set_d_d(b.get_mpc_t(), 10.65, 11.47, MPFR_RNDN);
    r = ceiling(complex_mpc(std::move(b)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(11), *integer(12))));
#endif // HAVE_SYMENGINE_MPC
}

TEST_CASE("test_truncate", "[Truncate]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> r = truncate(integer(1));
    CHECK(eq(*r, *one));

    r = truncate(Complex::from_two_nums(*integer(2), *integer(1)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(2), *integer(1))));

    r = truncate(Nan);
    CHECK(eq(*r, *Nan));

    r = truncate(Inf);
    CHECK(eq(*r, *Inf));

    r = truncate(NegInf);
    CHECK(eq(*r, *NegInf));

    r = truncate(Rational::from_two_ints(3, 1));
    CHECK(eq(*r, *integer(3)));

    r = truncate(Rational::from_two_ints(3, 2));
    CHECK(eq(*r, *integer(1)));

    r = truncate(Rational::from_two_ints(-3, 2));
    CHECK(eq(*r, *integer(-1)));

    r = truncate(real_double(2.65));
    CHECK(eq(*r, *integer(2)));

    r = truncate(complex_double(std::complex<double>(2.86, 2.79)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(2), *integer(2))));

    r = truncate(pi);
    CHECK(eq(*r, *integer(3)));

    r = truncate(E);
    CHECK(eq(*r, *integer(2)));

    r = truncate(floor(x));
    CHECK(eq(*r, *floor(x)));

    r = truncate(ceiling(x));
    CHECK(eq(*r, *ceiling(x)));

    r = truncate(truncate(x));
    CHECK(eq(*r, *truncate(x)));

    r = truncate(add(add(integer(2), mul(integer(2), x)), mul(integer(3), y)));
    CHECK(eq(*r, *add(integer(2),
                      truncate(add(mul(integer(2), x), mul(integer(3), y))))));

    r = truncate(add(add(Rational::from_two_ints(2, 3), mul(integer(2), x)),
                     mul(integer(3), y)));
    CHECK(eq(*r, *truncate(add(add(mul(integer(2), x), mul(integer(3), y)),
                               Rational::from_two_ints(2, 3)))));

    CHECK_THROWS_AS(truncate(Eq(integer(2), integer(3))), SymEngineException);

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class a(100);
    mpfr_set_d(a.get_mpfr_t(), 10.65, MPFR_RNDN);
    r = truncate(real_mpfr(std::move(a)));
    CHECK(eq(*r, *integer(10)));
#endif // HAVE_SYMENGINE_MPFR

#ifdef HAVE_SYMENGINE_MPC
    mpc_class b(100);
    mpc_set_d_d(b.get_mpc_t(), 10.65, 11.47, MPFR_RNDN);
    r = truncate(complex_mpc(std::move(b)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(10), *integer(11))));
#endif // HAVE_SYMENGINE_MPC
}

TEST_CASE("test_conjugate", "[Conjugate]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const Basic> r = conjugate(x);
    REQUIRE(is_a<Conjugate>(*r));

    r = conjugate(ComplexInf);
    CHECK(r->__str__() == "conjugate(zoo)");

    r = conjugate(integer(2));
    CHECK(eq(*r, *integer(2)));

    r = conjugate(Rational::from_two_ints(2, 4));
    CHECK(eq(*r, *Rational::from_two_ints(2, 4)));

    r = conjugate(real_double(2.3));
    CHECK(eq(*r, *real_double(2.3)));

    r = conjugate(Inf);
    CHECK(eq(*r, *Inf));

    r = conjugate(NegInf);
    CHECK(eq(*r, *NegInf));

    r = conjugate(Nan);
    CHECK(eq(*r, *Nan));

    r = conjugate(E);
    CHECK(eq(*r, *E));

    r = conjugate(conjugate(x));
    CHECK(eq(*r, *x));

    r = conjugate(sign(x));
    CHECK(eq(*r, *sign(conjugate(x))));

    r = conjugate(erf(x));
    CHECK(eq(*r, *erf(conjugate(x))));

    r = conjugate(erfc(x));
    CHECK(eq(*r, *erfc(conjugate(x))));

    r = conjugate(gamma(x));
    CHECK(eq(*r, *gamma(conjugate(x))));

    r = conjugate(loggamma(x));
    CHECK(eq(*r, *loggamma(conjugate(x))));

    r = conjugate(abs(x));
    CHECK(eq(*r, *abs(x)));

    r = conjugate(sin(x));
    CHECK(eq(*r, *sin(conjugate(x))));

    r = conjugate(cos(x));
    CHECK(eq(*r, *cos(conjugate(x))));

    r = conjugate(tan(x));
    CHECK(eq(*r, *tan(conjugate(x))));

    r = conjugate(cot(x));
    CHECK(eq(*r, *cot(conjugate(x))));

    r = conjugate(csc(x));
    CHECK(eq(*r, *csc(conjugate(x))));

    r = conjugate(sec(x));
    CHECK(eq(*r, *sec(conjugate(x))));

    r = conjugate(sinh(x));
    CHECK(eq(*r, *sinh(conjugate(x))));

    r = conjugate(cosh(x));
    CHECK(eq(*r, *cosh(conjugate(x))));

    r = conjugate(tanh(x));
    CHECK(eq(*r, *tanh(conjugate(x))));

    r = conjugate(coth(x));
    CHECK(eq(*r, *coth(conjugate(x))));

    r = conjugate(csch(x));
    CHECK(eq(*r, *csch(conjugate(x))));

    r = conjugate(sech(x));
    CHECK(eq(*r, *sech(conjugate(x))));

    r = conjugate(kronecker_delta(x, y));
    CHECK(eq(*r, *kronecker_delta(x, y)));

    r = conjugate(levi_civita({x, y}));
    CHECK(eq(*r, *levi_civita({x, y})));

    r = conjugate(atan2(x, y));
    CHECK(eq(*r, *atan2(conjugate(x), conjugate(y))));

    r = conjugate(lowergamma(x, y));
    CHECK(eq(*r, *lowergamma(conjugate(x), conjugate(y))));

    r = conjugate(uppergamma(x, y));
    CHECK(eq(*r, *uppergamma(conjugate(x), conjugate(y))));

    r = conjugate(beta(x, y));
    CHECK(eq(*r, *beta(conjugate(x), conjugate(y))));

    r = conjugate(Complex::from_two_nums(*integer(2), *integer(3)));
    CHECK(eq(*r, *Complex::from_two_nums(*integer(2), *integer(-3))));

    r = conjugate(
        pow(Complex::from_two_nums(*integer(2), *integer(3)), integer(2)));
    CHECK(eq(*r, *pow(Complex::from_two_nums(*integer(2), *integer(-3)),
                      integer(2))));

    r = conjugate(complex_double(std::complex<double>(0.0, 1.0)));
    CHECK(eq(*r, *complex_double(std::complex<double>(0.0, -1.0))));

    r = conjugate(
        pow(complex_double(std::complex<double>(2.0, 3.0)), integer(-2)));
    CHECK(eq(*r, *pow(complex_double(std::complex<double>(2.0, -3.0)),
                      integer(-2))));

    r = conjugate(pow(y, integer(2)));
    CHECK(eq(*r, *pow(conjugate(y), integer(2))));

    r = conjugate(mul(mul(integer(2), x), pow(y, integer(3))));
    RCP<const Basic> s
        = mul(integer(2), mul(conjugate(x), pow(conjugate(y), integer(3))));
    CHECK(eq(*r, *s));

    r = conjugate(
        mul(mul(integer(2), x), pow(y, Rational::from_two_ints(3, 2))));
    s = mul(integer(2), mul(conjugate(x),
                            conjugate(pow(y, Rational::from_two_ints(3, 2)))));
    CHECK(eq(*r, *s));

    r = conjugate(
        mul(mul(complex_double(std::complex<double>(2.0, 3.0)), x), y));
    s = mul(mul(complex_double(std::complex<double>(2.0, -3.0)), conjugate(x)),
            conjugate(y));
    CHECK(eq(*r, *s));

#ifdef HAVE_SYMENGINE_MPC
    mpc_class a(100), b(100);
    mpc_set_d_d(a.get_mpc_t(), 10.65, 11.47, MPFR_RNDN);
    mpc_set_d_d(b.get_mpc_t(), 10.65, -11.47, MPFR_RNDN);
    r = conjugate(complex_mpc(std::move(a)));
    s = complex_mpc(std::move(b));
    CHECK(eq(*r, *s));
#endif // HAVE_SYMENGINE_MPC
}

TEST_CASE("test rewrite_as_exp", "[Functions]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> r1, r2;
    r1 = rewrite_as_exp(sin(x));
    r2 = mul({rational(-1, 2), I, sub(exp(mul(I, x)), exp(mul(neg(I), x)))});
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(add(sin(x), cos(x)));
    r2 = add(
        mul({rational(-1, 2), I, sub(exp(mul(I, x)), exp(mul(neg(I), x)))}),
        mul(rational(1, 2), add(exp(mul(I, x)), exp(mul(neg(I), x)))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(add(sin(x), mul(integer(2), cos(x))));
    r2 = add(
        mul({rational(-1, 2), I, sub(exp(mul(I, x)), exp(mul(neg(I), x)))}),
        add(exp(mul(I, x)), exp(mul(neg(I), x))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(cos(cos(x)));
    r2 = mul(
        rational(1, 2),
        add(exp(mul(I, mul(rational(1, 2),
                           add(exp(mul(I, x)), exp(mul(neg(I), x)))))),
            exp(mul(neg(I), mul(rational(1, 2),
                                add(exp(mul(I, x)), exp(mul(neg(I), x))))))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(pow(cos(x), integer(2)));
    r2 = mul(rational(1, 4),
             pow(add(exp(mul(I, x)), exp(mul(neg(I), x))), integer(2)));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(log(sin(x)));
    r2 = log(
        mul({rational(-1, 2), I, sub(exp(mul(I, x)), exp(mul(neg(I), x)))}));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(zeta(sin(x), cos(x)));
    r2 = zeta(
        mul({rational(-1, 2), I, sub(exp(mul(I, x)), exp(mul(neg(I), x)))}),
        mul(rational(1, 2), add(exp(mul(I, x)), exp(mul(neg(I), x)))));
    REQUIRE(eq(*r1, *r2));

    r1 = rewrite_as_exp(levi_civita({sin(x), symbol("y"), symbol("z")}));
    r2 = levi_civita(
        {mul({rational(-1, 2), I, sub(exp(mul(I, x)), exp(mul(neg(I), x)))}),
         symbol("y"), symbol("z")});
    REQUIRE(eq(*r1, *r2));
}
TEST_CASE("test UnevaluatedExpr", "[Functions]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = pow(add(x, one), integer(2));
    RCP<const Basic> z = unevaluated_expr(y);
    auto r1 = add(z, x);
    auto r2 = add(y, x);
    REQUIRE(neq(*r1, *r2));
    REQUIRE(eq(*expand(z), *z));

    y = add(x, one);
    z = unevaluated_expr(y);
    r1 = add(z, one);
    r2 = add(y, one);
    REQUIRE(neq(*r1, *r2));

    r1 = z->subs({{x, zero}});
    REQUIRE(neq(*r1, *one));
}
