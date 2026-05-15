#include "catch.hpp"

#include <symengine/mul.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/rational.h>
#include <symengine/polys/basic_conversions.h>
#include <symengine/symengine_exception.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::E;
using SymEngine::eq;
using SymEngine::Expression;
using SymEngine::from_basic;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::MExprPoly;
using SymEngine::MIntPoly;
using SymEngine::minus_one;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::Pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::Rational;
using SymEngine::rational_class;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::set_basic;
using SymEngine::sin;
using SymEngine::symbol;
using SymEngine::Symbol;
using SymEngine::SymEngineException;
using SymEngine::UExprPoly;
using SymEngine::UIntPoly;
using SymEngine::umap_basic_num;
using SymEngine::URatPoly;
using SymEngine::zero;

using namespace SymEngine::literals;
using rc = rational_class;

TEST_CASE("find_gen_poly", "[b2poly]")
{
    umap_basic_num gens, rgens;
    RCP<const Basic> basic;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Number> i2 = rcp_static_cast<const Number>(integer(2));
    RCP<const Number> i3 = rcp_static_cast<const Number>(integer(3));
    RCP<const Number> i6 = rcp_static_cast<const Number>(integer(6));
    RCP<const Number> hf = rcp_static_cast<const Number>(div(one, integer(2)));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);

    // x**2 + x**(1/2) -> (x**(1/2))
    basic = add(pow(x, hf), pow(x, i2));
    gens = _find_gens_poly(basic);
    rgens = {{x, hf}};
    REQUIRE(unified_eq(gens, rgens));

    // x**(-1/2) + x**2 + x**(-1) -> (x, x**(-1/2))
    basic = add(add(pow(x, neg(hf)), pow(x, i2)), pow(x, minus_one));
    gens = _find_gens_poly(basic);
    rgens = {{x, one}, {pow(x, minus_one), hf}};
    REQUIRE(unified_eq(gens, rgens));

    // x/2 + 1/2 -> (x)
    basic = add(xb2, hf);
    gens = _find_gens_poly(basic);
    rgens = {{x, one}};
    REQUIRE(unified_eq(gens, rgens));

    // x*y*z**2 -> (x, y, z)
    basic = mul(x, mul(y, pow(z, i2)));
    gens = _find_gens_poly(basic);
    rgens = {{x, one}, {y, one}, {z, one}};
    REQUIRE(unified_eq(gens, rgens));

    // 2**(2*x + 1) -> (2**x)
    basic = pow(i2, add(mul(i2, x), one));
    gens = _find_gens_poly(basic);
    rgens = {{twopx, one}};
    REQUIRE(unified_eq(gens, rgens));

    // 2**(x**(x+1))-> (2**(x**(x+1)))
    basic = pow(i2, pow(x, add(x, one)));
    gens = _find_gens_poly(basic);
    rgens = {{basic, one}};
    REQUIRE(unified_eq(gens, rgens));

    // sin(x)*sin(y) + sin(x)**2 + sin(y) -> (sin(x), sin(y))
    basic = add(mul(sin(x), sin(y)), add(pow(sin(x), i2), sin(y)));
    gens = _find_gens_poly(basic);
    rgens = {{sin(x), one}, {sin(y), one}};
    REQUIRE(unified_eq(gens, rgens));

    // 2**x + 2**(x+y) -> (2**x, 2**y)
    basic = add(twopx, pow(i2, add(x, y)));
    gens = _find_gens_poly(basic);
    rgens = {{pow(i2, y), one}, {twopx, one}};
    REQUIRE(unified_eq(gens, rgens));

    // x**x + x**(x/2) + x**(x/3) -> (x**(x/6))
    basic = add(pow(x, x), add(pow(x, div(x, i2)), pow(x, div(x, i3))));
    gens = _find_gens_poly(basic);
    rgens = {{pow(x, x), rcp_static_cast<const Number>(div(one, i6))}};
    REQUIRE(unified_eq(gens, rgens));

    // x + (1/(x**2)) -> (x, 1/x)
    basic = add(x, div(one, pow(x, i2)));
    gens = _find_gens_poly(basic);
    rgens = {{x, one}, {pow(x, minus_one), one}};
    REQUIRE(unified_eq(gens, rgens));

    // x + (1/(x**2)) -> (x, 1/x)
    basic = add(x, div(one, pow(x, i2)));
    gens = _find_gens_poly(basic);
    rgens = {{x, one}, {pow(x, minus_one), one}};

    // ((x+1)**6)*(y**2) + x*y**3 -> (x, y)
    basic = add(mul(x, pow(y, i3)), mul(pow(add(x, one), i6), pow(y, i2)));
    gens = _find_gens_poly(basic);
    rgens = {{x, one}, {y, one}};
    REQUIRE(unified_eq(gens, rgens));

    // ((x+1)**6)*(y**2) + x*y**3 -> (x, y)
    basic = add(mul(x, pow(y, i3)), mul(pow(add(x, one), i6), pow(y, i2)));
    gens = _find_gens_poly(basic);
    rgens = {{x, one}, {y, one}};
    REQUIRE(unified_eq(gens, rgens));

    // 2**(3x) + y/2 + z**-2 -> (2**x, y, 1/z)
    basic = add(pow(i2, mul(x, i3)), add(div(y, i2), pow(z, neg(i2))));
    gens = _find_gens_poly(basic);
    rgens = {{pow(i2, x), one}, {y, one}, {pow(z, minus_one), one}};
    REQUIRE(unified_eq(gens, rgens));

    // E**2 + E*pi -> (E, pi)
    basic = add(pow(E, i2), mul(E, pi));
    gens = _find_gens_poly(basic);
    rgens = {{E, one}, {pi, one}};
    REQUIRE(unified_eq(gens, rgens));

    // 3 + (1/2) -> ()
    basic = add(i3, div(one, i2));
    gens = _find_gens_poly(basic);
    rgens = {};
    REQUIRE(unified_eq(gens, rgens));

    // x**(3/2) -> (x**(1/2))
    basic = pow(x, div(i3, i2));
    gens = _find_gens_poly(basic);
    rgens = {{x, hf}};
    REQUIRE(unified_eq(gens, rgens));

    // 2**(-x + 3) + 2**(-2x) -> (2**(-x))
    basic = add(pow(i2, add(i3, neg(x))), pow(i2, mul(neg(i2), x)));
    gens = _find_gens_poly(basic);
    rgens = {{pow(i2, neg(x)), one}};
    REQUIRE(unified_eq(gens, rgens));
}

TEST_CASE("basic_to_poly UInt", "[b2poly]")
{
    RCP<const Basic> basic, gen;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, integer(2));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);
    RCP<const UIntPoly> poly1, poly2, poly3;

    // x**2 + x**(1/2)
    basic = add(pow(x, i2), pow(x, hf));
    gen = pow(x, hf);
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z, 1_z, 0_z, 0_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));

    // 3x + 2
    basic = add(mul(x, i3), i2);
    gen = x;
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {2_z, 3_z});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(2x + 1)
    basic = pow(i2, add(mul(i2, x), one));
    gen = pow(i2, x);
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z, 0_z, 2_z}});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(-x + 3) + 2**(-2x) -> (2**(-x))
    basic = add(pow(i2, add(i3, neg(x))), pow(i2, mul(neg(i2), x)));
    gen = pow(i2, neg(x));
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z, 8_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));

    // x**x + x**(x/2) + x**(x/3)
    basic = add(pow(x, x), add(pow(x, div(x, i2)), pow(x, div(x, i3))));
    gen = pow(x, div(x, i6));
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z, 0_z, 1_z, 1_z, 0_z, 0_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));

    // (x**(1/2)+1)**3 + (x+2)**6
    basic = add(pow(add(pow(x, hf), one), i3), pow(add(x, i2), i6));
    gen = pow(x, hf);
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = pow_upoly(*UIntPoly::from_vec(gen, {1_z, 1_z}), 3);
    poly3 = pow_upoly(*UIntPoly::from_vec(gen, {{2_z, 0_z, 1_z}}), 6);
    poly2 = add_upoly(*poly2, *poly3);
    REQUIRE(eq(*poly1, *poly2));

    // (2**x)**2 * (2**(3x + 2) + 1)
    basic = mul(pow(twopx, i2), add(one, pow(i2, add(i2, mul(x, i3)))));
    gen = twopx;
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z, 0_z, 1_z, 0_z, 0_z, 4_z}});
    REQUIRE(eq(*poly1, *poly2));

    // 9**(x+(1/2)) + 9**(2x +(3/2))
    basic = add(pow(i9, add(x, hf)), pow(i9, add(mul(i2, x), div(i3, i2))));
    gen = pow(i9, x);
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z, 3_z, 27_z}});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(2**x) + 2**(2**(x+1)) + 3
    basic = add(pow(i2, twopx), add(i3, pow(i2, pow(i2, add(x, one)))));
    gen = pow(i2, twopx);
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{3_z, 1_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));

    poly1 = from_basic<UIntPoly>(poly2, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(
        from_basic<UIntPoly>(URatPoly::from_vec(gen, {{3_q, 1_q, 1_q}}), gen),
        SymEngineException); // Rat->Int
    CHECK_THROWS_AS(
        from_basic<UIntPoly>(
            UExprPoly::from_vec(
                gen, {{Expression("y"), Expression(1), Expression(1)}}),
            gen),
        SymEngineException); // Expr->Int

#ifdef HAVE_SYMENGINE_FLINT
    auto fpoly = SymEngine::UIntPolyFlint::from_vec(gen, {{3_z, 1_z, 1_z}});
    poly1 = from_basic<UIntPoly>(fpoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(
        from_basic<UIntPoly>(
            SymEngine::URatPolyFlint::from_vec(gen, {{3_q, 1_q, 1_q}}), gen),
        SymEngineException);
#endif

#ifdef HAVE_SYMENGINE_PIRANHA
    auto ppoly = SymEngine::UIntPolyPiranha::from_vec(gen, {{3_z, 1_z, 1_z}});
    poly1 = from_basic<UIntPoly>(ppoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(
        from_basic<UIntPoly>(
            SymEngine::URatPolyPiranha::from_vec(gen, {{3_q, 1_q, 1_q}}), gen),
        SymEngineException);
#endif

    // 0
    basic = zero;
    gen = x;
    poly1 = from_basic<UIntPoly>(basic, gen);
    poly2 = UIntPoly::from_vec(gen, {{0_z}});
    REQUIRE(eq(*poly1, *poly2));

    // x + y
    basic = add(x, y);
    gen = x;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // x + 1/2
    basic = add(x, hf);
    gen = x;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // x/2 + 1
    basic = add(div(x, i2), one);
    gen = x;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // x + 1/x
    basic = add(x, div(one, x));
    gen = x;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // xy + 1
    basic = add(mul(x, y), one);
    gen = x;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // x**(1/2) + 1
    basic = add(pow(x, hf), one);
    gen = x;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // 3**x + 2**x
    basic = add(pow(i3, x), pow(i2, x));
    gen = twopx;
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // 2**(2**(2x + 1)) + 2**(2**x)
    basic = add(pow(i2, twopx), pow(i2, pow(i2, add(mul(i2, x), one))));
    gen = pow(i2, twopx);
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);

    // 9**(x + (1/3))
    basic = pow(i9, add(div(one, i3), x));
    gen = pow(i9, x);
    CHECK_THROWS_AS(from_basic<UIntPoly>(basic, gen), SymEngineException);
}

TEST_CASE("basic_to_poly URat", "[b2poly]")
{
    RCP<const Basic> basic, gen;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, i2);
    RCP<const Basic> i2bi3 = div(i2, i3);
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> yb2 = div(x, i3);
    RCP<const URatPoly> poly1, poly2, poly3;

    // x/2
    basic = xb2;
    gen = x;
    poly1 = from_basic<URatPoly>(basic, gen);
    poly2 = URatPoly::from_vec(gen, {0_q, rc(1_z, 2_z)});
    REQUIRE(eq(*poly1, *poly2));

    // 3x + 2
    basic = add(mul(x, i3), i2);
    gen = x;
    poly1 = from_basic<URatPoly>(basic, gen);
    poly2 = URatPoly::from_vec(gen, {2_q, 3_q});
    REQUIRE(eq(*poly1, *poly2));

    poly1 = from_basic<URatPoly>(poly2, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(
        from_basic<URatPoly>(
            UExprPoly::from_vec(
                gen, {{Expression("y"), Expression(1), Expression(1)}}),
            gen),
        SymEngineException); // Expr->Rat

#ifdef HAVE_SYMENGINE_FLINT
    auto fpoly = SymEngine::UIntPolyFlint::from_vec(gen, {{2_z, 3_z}});
    poly1 = from_basic<URatPoly>(fpoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    auto fpoly2 = SymEngine::URatPolyFlint::from_vec(gen, {{2_q, 3_q}});
    poly1 = from_basic<URatPoly>(fpoly2, gen);
    REQUIRE(eq(*poly1, *poly2));
#endif

#ifdef HAVE_SYMENGINE_PIRANHA
    auto ppoly = SymEngine::UIntPolyPiranha::from_vec(gen, {{2_z, 3_z}});
    poly1 = from_basic<URatPoly>(ppoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    auto ppoly2 = SymEngine::URatPolyPiranha::from_vec(gen, {{2_q, 3_q}});
    poly1 = from_basic<URatPoly>(ppoly2, gen);
    REQUIRE(eq(*poly1, *poly2));
#endif

    // 3/2 * (2**x)
    basic = mul(div(i3, i2), pow(i2, x));
    gen = pow(i2, x);
    poly1 = from_basic<URatPoly>(basic, gen);
    poly2 = URatPoly::from_vec(gen, {0_q, rc(3_z, 2_z)});
    REQUIRE(eq(*poly1, *poly2));

    // x + y
    basic = add(x, y);
    CHECK_THROWS_AS(from_basic<URatPoly>(basic), SymEngineException);

    // x + 1/x
    basic = add(x, div(one, x));
    CHECK_THROWS_AS(from_basic<URatPoly>(basic), SymEngineException);
}

TEST_CASE("basic_to_poly UExpr", "[b2poly]")
{
    RCP<const Basic> basic, gen;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, integer(2));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);
    RCP<const UExprPoly> poly1, poly2, poly3;

    // x + xy + (x**1/2)*z
    basic = add(x, add(mul(x, y), mul(z, pow(x, hf))));
    gen = pow(x, hf);
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(
        gen, {{Expression(0), Expression("z"), add(one, y)}});
    REQUIRE(eq(*poly1, *poly2));

    // 3*2**x + 2**(x+y)
    basic = add(mul(i3, twopx), pow(i2, add(x, y)));
    gen = twopx;
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(gen, {{Expression(0), add(i3, pow(i2, y))}});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(-x + (1/2)) + 2**(-2x)
    basic = add(pow(i2, add(neg(x), hf)), pow(i2, mul(neg(i2), x)));
    gen = pow(i2, neg(x));
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(gen,
                                {{Expression(0), pow(i2, hf), Expression(1)}});
    REQUIRE(eq(*poly1, *poly2));

    // xy + xz + yz
    basic = add(mul(x, y), add(mul(x, z), mul(y, z)));
    gen = x;
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(gen, {{mul(y, z), add(z, y)}});
    REQUIRE(eq(*poly1, *poly2));

    // (x+1)**2 + 2xy
    basic = add(mul(mul(i2, x), y), pow(add(x, one), i2));
    gen = x;
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(
        gen, {{Expression(1), add(mul(i2, y), i2), Expression(1)}});
    REQUIRE(eq(*poly1, *poly2));

    // x**x + x**(2x + y)
    basic = add(pow(x, x), pow(x, add(mul(i2, x), y)));
    gen = pow(x, x);
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2
        = UExprPoly::from_vec(gen, {{Expression(0), Expression(1), pow(x, y)}});
    REQUIRE(eq(*poly1, *poly2));

    // (1/2)*x**2 + 1/x
    basic = add(mul(hf, pow(x, i2)), div(one, x));
    gen = x;
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(
        gen, {{div(one, x), Expression(0), Expression(1) / 2}});
    REQUIRE(eq(*poly1, *poly2));

    // (x/2)**2 + xz
    basic = add(pow(div(x, i2), i2), mul(z, x));
    gen = x;
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(
        gen, {{Expression(0), Expression("z"), div(one, integer(4))}});
    REQUIRE(eq(*poly1, *poly2));

    // pi**2 + E*pi
    basic = add(pow(pi, i2), mul(pi, E));
    gen = pi;
    poly1 = from_basic<UExprPoly>(basic, gen);
    poly2 = UExprPoly::from_vec(
        gen, {{Expression(0), Expression(E), Expression(1)}});
    REQUIRE(eq(*poly1, *poly2));

    poly1 = from_basic<UExprPoly>(poly2, gen);
    REQUIRE(eq(*poly1, *poly2));

    gen = x;
    poly2 = UExprPoly::from_vec(
        gen, {{Expression(1), Expression(2), Expression(3)}});
    auto ipoly1 = UIntPoly::from_vec(gen, {{1_z, 2_z, 3_z}});
    poly1 = from_basic<UExprPoly>(ipoly1, gen);
    REQUIRE(eq(*poly1, *poly2));

    auto ratpoly1 = URatPoly::from_vec(gen, {{1_q, 2_q, 3_q}});
    poly1 = from_basic<UExprPoly>(ratpoly1, gen);
    REQUIRE(eq(*poly1, *poly2));

#ifdef HAVE_SYMENGINE_FLINT
    auto fpoly = SymEngine::UIntPolyFlint::from_vec(gen, {{1_z, 2_z, 3_z}});
    poly1 = from_basic<UExprPoly>(fpoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    auto fpoly2 = SymEngine::URatPolyFlint::from_vec(gen, {{1_q, 2_q, 3_q}});
    poly1 = from_basic<UExprPoly>(fpoly2, gen);
    REQUIRE(eq(*poly1, *poly2));
#endif

#ifdef HAVE_SYMENGINE_PIRANHA
    auto ppoly = SymEngine::UIntPolyPiranha::from_vec(gen, {{1_z, 2_z, 3_z}});
    poly1 = from_basic<UExprPoly>(ppoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    auto ppoly2 = SymEngine::URatPolyPiranha::from_vec(gen, {{1_q, 2_q, 3_q}});
    poly1 = from_basic<UExprPoly>(ppoly2, gen);
    REQUIRE(eq(*poly1, *poly2));
#endif
}

#ifdef HAVE_SYMENGINE_PIRANHA

using SymEngine::UIntPolyPiranha;
TEST_CASE("basic_to_poly UIntPiranha", "[b2poly]")
{
    RCP<const Basic> basic, gen;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, integer(2));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);
    RCP<const UIntPolyPiranha> poly1, poly2, poly3;

    // x**2 + x**(1/2)
    basic = add(pow(x, i2), pow(x, hf));
    gen = pow(x, hf);
    poly1 = from_basic<UIntPolyPiranha>(basic, gen);
    poly2 = UIntPolyPiranha::from_vec(gen, {{0_z, 1_z, 0_z, 0_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(2x + 1)
    basic = pow(i2, add(mul(i2, x), one));
    gen = pow(i2, x);
    poly1 = from_basic<UIntPolyPiranha>(basic, gen);
    poly2 = UIntPolyPiranha::from_vec(gen, {{0_z, 0_z, 2_z}});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(-x + 3) + 2**(-2x) -> (2**(-x))
    basic = add(pow(i2, add(i3, neg(x))), pow(i2, mul(neg(i2), x)));
    gen = pow(i2, neg(x));
    poly1 = from_basic<UIntPolyPiranha>(basic, gen);
    poly2 = UIntPolyPiranha::from_vec(gen, {{0_z, 8_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));
}
#endif

#ifdef HAVE_SYMENGINE_FLINT

using SymEngine::UIntPolyFlint;
TEST_CASE("basic_to_poly UIntFlint", "[b2poly]")
{
    RCP<const Basic> basic, gen;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, integer(2));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);
    RCP<const UIntPolyFlint> poly1, poly2, poly3;

    // x**x + x**(x/2) + x**(x/3)
    basic = add(pow(x, x), add(pow(x, div(x, i2)), pow(x, div(x, i3))));
    gen = pow(x, div(x, i6));
    poly1 = from_basic<UIntPolyFlint>(basic, gen);
    poly2 = UIntPolyFlint::from_vec(gen, {{0_z, 0_z, 1_z, 1_z, 0_z, 0_z, 1_z}});
    REQUIRE(eq(*poly1, *poly2));

    poly1 = from_basic<UIntPolyFlint>(poly2, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(from_basic<UIntPolyFlint>(
                        URatPoly::from_vec(gen, {{1_q, 2_q, 0_q, 4_q}}), x),
                    SymEngineException);

    // (x**(1/2)+1)**3 + (x+2)**6
    basic = add(pow(add(pow(x, hf), one), i3), pow(add(x, i2), i6));
    gen = pow(x, hf);
    poly1 = from_basic<UIntPolyFlint>(basic, gen);
    poly2 = pow_upoly(*UIntPolyFlint::from_vec(gen, {{1_z, 1_z}}), 3);
    poly3 = pow_upoly(*UIntPolyFlint::from_vec(gen, {{2_z, 0_z, 1_z}}), 6);
    poly2 = add_upoly(*poly2, *poly3);
    REQUIRE(eq(*poly1, *poly2));

    // (2**x)**2 * (2**(3x + 2) + 1)
    basic = mul(pow(twopx, i2), add(one, pow(i2, add(i2, mul(x, i3)))));
    gen = twopx;
    poly1 = from_basic<UIntPolyFlint>(basic, gen);
    poly2 = UIntPolyFlint::from_vec(gen, {{0_z, 0_z, 1_z, 0_z, 0_z, 4_z}});
    REQUIRE(eq(*poly1, *poly2));

    gen = x;
    poly1 = UIntPolyFlint::from_vec(gen, {{1_z, 2_z, 0_z, 4_z}});
    auto ipoly = UIntPoly::from_vec(gen, {{1_z, 2_z, 0_z, 4_z}});
    poly2 = from_basic<UIntPolyFlint>(ipoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(
        from_basic<UIntPolyFlint>(
            SymEngine::URatPolyFlint::from_vec(gen, {{1_q, 2_q, 0_q, 4_q}}), x),
        SymEngineException);

#ifdef HAVE_SYMENGINE_PIRANHA
    auto ppoly
        = SymEngine::UIntPolyPiranha::from_vec(gen, {{1_z, 2_z, 0_z, 4_z}});
    poly2 = from_basic<UIntPolyFlint>(ppoly, gen);
    REQUIRE(eq(*poly1, *poly2));

    CHECK_THROWS_AS(
        from_basic<UIntPolyFlint>(
            SymEngine::URatPolyPiranha::from_vec(gen, {{1_q, 2_q, 0_q, 4_q}}),
            x),
        SymEngineException);
#endif

    CHECK_THROWS_AS(
        from_basic<UIntPolyFlint>(
            UExprPoly::from_vec(gen, {{Expression(1), Expression(2),
                                       Expression(0), Expression(4)}}),
            x),
        SymEngineException);
}
#endif

TEST_CASE("basic_to_poly MInt", "[b2poly]")
{
    RCP<const Basic> basic, tt, yy;
    set_basic gens;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, integer(2));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);
    RCP<const Basic> twopy = pow(i2, y);
    RCP<const MIntPoly> poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8;

    // x + y
    basic = add(x, y);
    gens = {x, y};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict({x, y}, {{{0, 1}, 1_z}, {{1, 0}, 1_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // 3x + 2
    basic = add(mul(x, i3), i2);
    gens = {x};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict({x}, {{{0}, 2_z}, {{1}, 3_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // 2**(x + y)
    basic = pow(i2, add(x, y));
    gens = {twopx, twopy};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict({twopx, twopy}, {{{1, 1}, 1_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // 3*x*2**x - x**2 + 2**(2*x)
    basic = add({mul(i3, mul(x, twopx)), neg(pow(x, i2)), pow(i2, mul(x, i2))});
    gens = {twopx, x};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict({twopx, x},
                                {{{1, 1}, 3_z}, {{0, 2}, -1_z}, {{2, 0}, 1_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // (x+y)**3 + (2x+y)**6
    basic = add(pow(add(x, y), i3), pow(add(mul(i2, x), y), i6));
    gens = {x, y};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly2 = pow_mpoly(
        *MIntPoly::from_dict({x, y}, {{{0, 1}, 1_z}, {{1, 0}, 1_z}}), 3);
    poly3 = pow_mpoly(
        *MIntPoly::from_dict({x, y}, {{{0, 1}, 1_z}, {{1, 0}, 2_z}}), 6);
    poly2 = add_mpoly(*poly2, *poly3);
    REQUIRE(eq(*poly1, *poly2));

    // (2**x + 2**y) * (2**(3x + 1) + 2**y)
    basic = mul(add(twopx, twopy), add(twopy, pow(i2, add(one, mul(x, i3)))));
    gens = {twopx, twopy};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict(
        {twopx, twopy},
        {{{4, 0}, 2_z}, {{1, 1}, 1_z}, {{3, 1}, 2_z}, {{0, 2}, 1_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // x + 1/x + 1
    basic = add({x, div(one, x), one});
    gens = {x, div(one, x)};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict({x, div(one, x)},
                                {{{1, 0}, 1_z}, {{0, 1}, 1_z}, {{0, 0}, 1_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // 0
    basic = zero;
    gens = {x, y};
    poly1 = from_basic<MIntPoly>(basic, gens);
    poly3 = from_basic<MIntPoly>(basic);
    poly2 = MIntPoly::from_dict({x, y}, {{{0, 0}, 0_z}});
    REQUIRE(eq(*poly1, *poly2));
    REQUIRE(eq(*poly1, *poly3));

    // x + y
    basic = add(x, y);
    gens = {x};
    CHECK_THROWS_AS(from_basic<MIntPoly>(basic, gens), SymEngineException);

    // x + 1/2
    basic = add(x, hf);
    gens = {x};
    CHECK_THROWS_AS(from_basic<MIntPoly>(basic, gens), SymEngineException);

    // x**(1/2) + 1
    basic = add(pow(x, hf), one);
    gens = {x};
    CHECK_THROWS_AS(from_basic<MIntPoly>(basic, gens), SymEngineException);

    // x + y + x/y
    basic = add({x, y, div(x, y)});
    gens = {x, y};
    CHECK_THROWS_AS(from_basic<MIntPoly>(basic, gens), SymEngineException);
}

TEST_CASE("basic_to_poly MExpr", "[b2poly]")
{
    RCP<const Basic> basic;
    set_basic gens;
    RCP<const Integer> one = integer(1);
    RCP<const Integer> minus_one = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> hf = div(one, integer(2));
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> xb2 = div(x, i2);
    RCP<const Basic> twopx = pow(i2, x);
    RCP<const MExprPoly> poly1, poly2, poly3;

    // x + xyz
    basic = add(x, mul(x, mul(y, z)));
    gens = {x, y};
    poly1 = from_basic<MExprPoly>(basic, gens);
    poly2 = MExprPoly::from_dict(
        {x, y}, {{{1, 0}, Expression(1)}, {{1, 1}, Expression("z")}});
    REQUIRE(eq(*poly1, *poly2));

    // x*2**x + 2**(x+y)
    basic = add(mul(x, twopx), pow(i2, add(x, y)));
    gens = {twopx};
    poly1 = from_basic<MExprPoly>(basic, gens);
    poly2 = MExprPoly::from_dict({twopx}, {{{1}, add(x, pow(i2, y))}});
    REQUIRE(eq(*poly1, *poly2));

    // 2**(-x + (1/2)) + 2**(-2x)
    basic = add(pow(i2, add(neg(x), hf)), pow(i2, mul(neg(i2), x)));
    gens = {pow(i2, neg(x)), pow(i2, hf)};
    poly1 = from_basic<MExprPoly>(basic, gens);
    poly2 = MExprPoly::from_dict(
        {pow(i2, neg(x)), pow(i2, hf)},
        {{{1, 1}, Expression(1)}, {{2, 0}, Expression(1)}});
    REQUIRE(eq(*poly1, *poly2));

    // xy + xz + yz
    basic = add(mul(x, y), add(mul(x, z), mul(y, z)));
    gens = {x, y};
    poly1 = from_basic<MExprPoly>(basic, gens);
    poly2 = MExprPoly::from_dict({x, y}, {{{1, 1}, Expression(1)},
                                          {{0, 1}, Expression("z")},
                                          {{1, 0}, Expression("z")}});
    REQUIRE(eq(*poly1, *poly2));

    // x**(x + z) + x**(2x + y)
    basic = add(pow(x, add(x, z)), pow(x, add(mul(i2, x), y)));
    gens = {pow(x, x), pow(x, y)};
    poly1 = from_basic<MExprPoly>(basic, gens);
    poly2 = MExprPoly::from_dict(
        {pow(x, x), pow(x, y)}, {{{1, 0}, pow(x, z)}, {{2, 1}, Expression(1)}});
    REQUIRE(eq(*poly1, *poly2));

    // pi**2 + E*pi + E*pi*z
    basic = add(mul(E, mul(pi, z)), add(pow(pi, i2), mul(pi, E)));
    gens = {pi, E};
    poly1 = from_basic<MExprPoly>(basic, gens);
    poly2 = MExprPoly::from_dict(
        {pi, E}, {{{1, 1}, add(one, z)}, {{2, 0}, Expression(1)}});
    REQUIRE(eq(*poly1, *poly2));
}
