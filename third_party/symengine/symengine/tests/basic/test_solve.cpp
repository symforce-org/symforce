#include "catch.hpp"
#include <iostream>

#include <symengine/solve.h>
#include <symengine/mul.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/polys/basic_conversions.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::boolFalse;
using SymEngine::boolTrue;
using SymEngine::ConditionSet;
using SymEngine::DenseMatrix;
using SymEngine::down_cast;
using SymEngine::dummy;
using SymEngine::emptyset;
using SymEngine::Eq;
using SymEngine::Expression;
using SymEngine::finiteset;
using SymEngine::FiniteSet;
using SymEngine::Ge;
using SymEngine::I;
using SymEngine::imageset;
using SymEngine::Inf;
using SymEngine::integer;
using SymEngine::interval;
using SymEngine::Interval;
using SymEngine::is_a;
using SymEngine::linear_eqns_to_matrix;
using SymEngine::linsolve;
using SymEngine::logical_and;
using SymEngine::mul;
using SymEngine::Ne;
using SymEngine::neg;
using SymEngine::NegInf;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::pow;
using SymEngine::rational;
using SymEngine::rational_class;
using SymEngine::RCP;
using SymEngine::Set;
using SymEngine::set_union;
using SymEngine::solve;
using SymEngine::solve_poly_quartic;
using SymEngine::symbol;
using SymEngine::Symbol;
using SymEngine::SymEngineException;
using SymEngine::UIntPoly;
using SymEngine::Union;
using SymEngine::URatPoly;
using SymEngine::vec_basic;
using SymEngine::zero;
#ifdef HAVE_SYMENGINE_FLINT
using SymEngine::UIntPolyFlint;
using SymEngine::URatPolyFlint;
#endif
#ifdef HAVE_SYMENGINE_PIRANHA
using SymEngine::UIntPolyPiranha;
using SymEngine::URatPolyPiranha;
#endif
using namespace SymEngine::literals;

TEST_CASE("Trivial cases", "[Solve]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> poly;
    RCP<const Set> reals = interval(NegInf, Inf, true, true);
    RCP<const Set> soln;

    REQUIRE(eq(*solve(boolTrue, x, reals), *reals));
    REQUIRE(eq(*solve(boolFalse, x, reals), *emptyset()));

    // constants
    poly = one;
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *emptyset()));

    poly = zero;
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *reals));
}

TEST_CASE("linear and quadratic polynomials", "[Solve]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> poly;
    RCP<const Set> reals = interval(NegInf, Inf, true, true);
    RCP<const Set> soln;

    auto i2 = integer(2), i3 = integer(3), im2 = integer(-2), im3 = integer(-3);
    auto sqx = mul(x, x);

    // linear
    poly = add(x, i3);
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *finiteset({im3})));

    soln = solve(poly, x, interval(zero, Inf, false, true));
    REQUIRE(eq(*soln, *emptyset()));

    poly = add(div(x, i2), div(one, i3));
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({neg(rational(2, 3))})));

    poly = x;
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *finiteset({zero})));

    auto y = symbol("y");
    soln = solve(add(x, y), x);
    REQUIRE(eq(*soln, *finiteset({neg(y)})));

    CHECK_THROWS_AS(solve_poly_linear({one}, reals), SymEngineException);

    // Quadratic
    poly = add(sqx, one);
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({neg(I), I})));

    poly = add(add(sqx, mul(im2, x)), one);
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({one})));

    poly = sub(add(div(sqx, i3), x), div(i2, integer(5)));
    soln = solve(poly, x);
    REQUIRE(
        eq(*soln,
           *finiteset(
               {add(div(mul(sqrt(integer(69)), sqrt(integer(5))), integer(10)),
                    div(im3, i2)),
                sub(div(im3, i2), div(mul(sqrt(integer(69)), sqrt(integer(5))),
                                      integer(10)))})));

    poly = add(sqx, mul(x, i2));
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({zero, im2})));

    poly = add(sqx, sub(mul(integer(8), x), integer(5)));
    soln = solve(poly, x);
    REQUIRE(
        eq(*soln, *finiteset({add(integer(-4), div(sqrt(integer(84)), i2)),
                              sub(integer(-4), div(sqrt(integer(84)), i2))})));

    poly = add(sqx, sub(integer(50), mul(integer(8), x)));
    soln = solve(poly, x);
    REQUIRE(
        eq(*soln,
           *finiteset({add(integer(4), div(mul(sqrt(integer(136)), I), i2)),
                       sub(integer(4), div(mul(sqrt(integer(136)), I), i2))})));

    auto b = symbol("b"), c = symbol("c");
    soln = solve(add({sqx, mul(b, x), c}), x);
    REQUIRE(soln->__str__()
            == "{(-1/2)*b + (1/2)*sqrt(-4*c + b**2), (-1/2)*b "
               "+ (-1/2)*sqrt(-4*c + b**2)}");

    soln = solve(add({sqx, mul(i3, x), c}), x);
    REQUIRE(soln->__str__()
            == "{-3/2 + (-1/2)*sqrt(9 - 4*c), -3/2 + (1/2)*sqrt(9 - 4*c)}");

    soln = solve(add({sqx, mul({i3, b, x}), c}), x);
    REQUIRE(soln->__str__()
            == "{(-3/2)*b + (-1/2)*sqrt(-4*c + 9*b**2), "
               "(-3/2)*b + (1/2)*sqrt(-4*c + 9*b**2)}");

    CHECK_THROWS_AS(solve_poly_quadratic({one}, reals), SymEngineException);

    auto onebyx = div(one, x);
    poly = add(onebyx, one);
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({neg(one)})));

    poly = mul(onebyx, onebyx);
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *emptyset()));
}

TEST_CASE("cubic and quartic polynomials", "[Solve]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> poly;
    RCP<const Set> reals = interval(NegInf, Inf, true, true);
    RCP<const Set> soln;

    auto sqx = mul(x, x), cbx = mul(sqx, x), qx = mul(cbx, x);
    auto i2 = integer(2), i3 = integer(3), im2 = integer(-2);

    // cubic
    poly = Eq(add({cbx, mul(x, i3), mul(sqx, i3)}), neg(one));
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *finiteset({neg(one)})));

    poly = Ne(add({cbx, mul(x, i3), mul(sqx, i3)}), neg(one));
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *set_union({interval(NegInf, integer(-1), true, true),
                                  interval(integer(-1), Inf, true, true)})));

    poly = Ge(add({cbx, mul(x, i3), mul(sqx, i3)}), one);
    soln = solve(poly, x, reals);
    REQUIRE(is_a<ConditionSet>(*soln));

    REQUIRE(eq(*down_cast<const ConditionSet &>(*soln).get_condition(),
               *logical_and({Ge(add({cbx, mul(x, i3), mul(sqx, i3)}), one),
                             reals->contains(x)})));

    poly = mul(cbx, i3);
    soln = solve(poly, x, reals);
    REQUIRE(eq(*soln, *finiteset({zero})));

    poly = add(cbx, sub(add(mul(sqx, i3), mul(i3, x)), one));
    soln = solve(poly, x);
    auto r1 = neg(div(add(i3, pow(integer(-54), div(one, i3))), i3));
    auto r2 = neg(div(add(i3, mul(add(div(one, im2), div(mul(I, sqrt(i3)), i2)),
                                  pow(integer(-54), div(one, i3)))),
                      i3));
    auto r3 = neg(div(add(i3, mul(sub(div(one, im2), div(mul(I, sqrt(i3)), i2)),
                                  pow(integer(-54), div(one, i3)))),
                      i3));
    REQUIRE(eq(*soln, *finiteset({r1, r2, r3})));

    poly = sub(add(cbx, mul(integer(201), x)),
               add(integer(288), mul(sqx, integer(38))));
    soln = solve(poly, x);
    r1 = i3;
    r2 = integer(32);
    REQUIRE(eq(*soln, *finiteset({r1, r2})));

    poly = sub(cbx, x);
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({zero, one, neg(one)})));

    poly = Eq(cbx, neg(one));
    soln = solve(poly, x);
    r1 = neg(one);
    r2 = add(div(one, i2), div(mul(I, sqrt(i3)), i2));
    r3 = sub(div(one, i2), div(mul(I, sqrt(i3)), i2));
    // -(-1/2 - 1/2*I*sqrt(3)) != 1/2 + 1/2*I*sqrt(3) ?
    // REQUIRE(eq(*soln, *finiteset({r1, r2, r3})));

    CHECK_THROWS_AS(solve_poly_cubic({one}, reals), SymEngineException);
    CHECK_THROWS_AS(solve_poly_quartic({one}, reals), SymEngineException);

    // Quartic
    poly = qx;
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({zero})));

    poly = add(qx, cbx);
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({zero, neg(one)})));

    poly = add({qx, mul(cbx, i2), mul(integer(-41), sqx), mul(x, integer(-42)),
                integer(360)});
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({i3, integer(-4), integer(5), integer(-6)})));

    poly = add({qx, mul(cbx, rational(5, 2)), mul(integer(-19), sqx),
                mul(x, integer(14)), integer(12)});
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({i2, rational(-1, 2), integer(-6)})));

    poly = mul(add(x, i2), add(x, i3));
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({neg(i2), neg(i3)})));

    soln = solve_poly_quartic({rational(51, 256), one, one, one, one});
    REQUIRE(eq(*soln->contains(rational(-1, 4)), *boolTrue));
}

TEST_CASE("Higher order(degree >=5) polynomials", "[Solve]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> poly;
    RCP<const Set> soln;

    auto sqx = mul(x, x), cbx = mul(sqx, x), qx = mul(cbx, x);
    auto i2 = integer(2), i3 = integer(3);

#if defined(HAVE_SYMENGINE_FLINT) and __FLINT_RELEASE > 20502

    poly = add({mul({qx, x, integer(15)}), mul(qx, integer(-538)),
                mul(cbx, integer(5015)), mul(sqx, integer(-13436)),
                mul(x, integer(3700)), integer(14352)});
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({i2, i3, rational(26, 3), rational(-4, 5),
                                  integer(23)})));

    poly = add({mul(qx, qx), mul({i3, qx, cbx}), mul({i2, qx, sqx}), mul(qx, x),
                mul(i3, qx), mul(i2, cbx), sqx, mul(i3, x), i2});
    soln = solve(poly, x);
    REQUIRE(is_a<Union>(*soln));
    auto &temp = down_cast<const Union &>(*soln).get_container();
    REQUIRE(temp.size() == 2);
    REQUIRE(temp.find(finiteset({neg(one), neg(i2)})) != temp.end());
    REQUIRE(temp.find(conditionset(x, Eq(add({mul(cbx, cbx), cbx, one}), zero)))
            != temp.end());

#else

    poly = add({mul({qx, x, integer(15)}), mul(qx, integer(-538)),
                mul(cbx, integer(5015)), mul(sqx, integer(-13436)),
                mul(x, integer(3700)), integer(14352)});
    soln = solve(poly, x);
    REQUIRE(is_a<ConditionSet>(*soln));
    REQUIRE(eq(*down_cast<const ConditionSet &>(*soln).get_condition(),
               *Eq(poly, zero)));

    poly = add({mul(qx, qx), mul({i3, qx, cbx}), mul({i2, qx, sqx}), mul(qx, x),
                mul(i3, qx), mul(i2, cbx), sqx, mul(i3, x), i2});
    soln = solve(poly, x);
    REQUIRE(is_a<ConditionSet>(*soln));
    REQUIRE(eq(*down_cast<const ConditionSet &>(*soln).get_condition(),
               *Eq(poly, zero)));

#endif

    poly = add({mul(cbx, cbx), cbx, one});
    soln = solve(poly, x);
    REQUIRE(is_a<ConditionSet>(*soln));
    REQUIRE(eq(*down_cast<const ConditionSet &>(*soln).get_condition(),
               *Eq(poly, zero)));
}

TEST_CASE("solve_poly_rational", "[Solve]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> poly;
    RCP<const Set> soln;
    auto i2 = integer(2), i3 = integer(3);

    poly = div(add(x, i2), add(x, i3));
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({neg(i2)})));

    poly = div(mul(add(x, i2), add(x, i3)), add(x, i2));
    soln = solve(poly, x);
    REQUIRE(eq(*soln, *finiteset({neg(i3)})));
}

TEST_CASE("solve_poly", "[Solve]")
{
    auto x = symbol("x");
    RCP<const Set> soln;

    auto p1 = UIntPoly::from_dict(x, {{0, 2_z}, {1, 3_z}, {2, 1_z}});
    soln = solve_poly(p1, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));

    auto p2 = URatPoly::from_dict(x, {{0, 2_q}, {1, 3_q}, {2, 1_q}});
    soln = solve_poly(p2, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));

    auto P = uexpr_poly(
        x, {{0, Expression(2)}, {1, Expression(3)}, {2, Expression(1)}});
    soln = solve_poly(P, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));

#ifdef HAVE_SYMENGINE_FLINT
    auto p3 = UIntPolyFlint::from_dict(x, {{0, 2_z}, {1, 3_z}, {2, 1_z}});
    soln = solve_poly(p3, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));

    auto P3 = URatPolyFlint::from_dict(x, {{0, 1_q},
                                           {1, rational_class(3_z, 2_z)},
                                           {2, rational_class(1_z, 2_z)}});
    soln = solve_poly(P3, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));
#endif

#ifdef HAVE_SYMENGINE_PIRANHA
    auto p4 = UIntPolyPiranha::from_dict(x, {{0, 2_z}, {1, 3_z}, {2, 1_z}});
    soln = solve_poly(p4, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));

    auto P4 = URatPolyPiranha::from_dict(x, {{0, 1_q},
                                             {1, rational_class(3_z, 2_z)},
                                             {2, rational_class(1_z, 2_z)}});
    soln = solve_poly(P4, x);
    REQUIRE(eq(*soln, *finiteset({neg(one), neg(integer(2))})));
#endif
}

TEST_CASE("linsolve", "[Solve]")
{
    auto x = symbol("x"), y = symbol("y"), z = symbol("z"), t = symbol("t");
    // Augmented Matrix as input
    auto aug = DenseMatrix(
        4, 5, {integer(1), integer(2), integer(3), integer(4), integer(10),
               integer(2), integer(2), integer(3), integer(4), integer(11),
               integer(3), integer(3), integer(3), integer(4), integer(13),
               integer(9), integer(8), integer(7), integer(6), integer(30)});
    vec_basic solns = linsolve(aug, {x, y, z, t});
    REQUIRE(solns.size() == 4);
    REQUIRE(eq(*solns[0], *integer(1)));
    REQUIRE(eq(*solns[1], *integer(1)));
    REQUIRE(eq(*solns[2], *integer(1)));
    REQUIRE(eq(*solns[3], *integer(1)));

    // vector of equations as input
    solns
        = linsolve({add({x, mul(integer(2), y), mul(integer(3), z),
                         mul(integer(4), t), integer(10)}),
                    add({mul(integer(2), x), mul(integer(2), y),
                         mul(integer(3), z), mul(integer(4), t), integer(11)}),
                    add({mul(integer(3), x), mul(integer(3), y),
                         mul(integer(3), z), mul(integer(4), t), integer(13)}),
                    add({mul(integer(9), x), mul(integer(8), y),
                         mul(integer(7), z), mul(integer(6), t), integer(30)})},
                   {x, y, z, t});
    REQUIRE(solns.size() == 4);
    REQUIRE(eq(*solns[0], *integer(-1)));
    REQUIRE(eq(*solns[1], *integer(-1)));
    REQUIRE(eq(*solns[2], *integer(-1)));
    REQUIRE(eq(*solns[3], *integer(-1)));

    solns = linsolve({Eq(add({x, mul(integer(2), y), mul(integer(3), z),
                              mul(integer(4), t)}),
                         integer(10)),
                      Eq(add({mul(integer(2), x), mul(integer(2), y),
                              mul(integer(3), z), mul(integer(4), t)}),
                         integer(11)),
                      Eq(add({mul(integer(3), x), mul(integer(3), y),
                              mul(integer(3), z), mul(integer(4), t)}),
                         integer(13)),
                      Eq(add({mul(integer(9), x), mul(integer(8), y),
                              mul(integer(7), z), mul(integer(6), t)}),
                         integer(30))},
                     {x, y, z, t});
    REQUIRE(solns.size() == 4);
    REQUIRE(eq(*solns[0], *integer(1)));
    REQUIRE(eq(*solns[1], *integer(1)));
    REQUIRE(eq(*solns[2], *integer(1)));
    REQUIRE(eq(*solns[3], *integer(1)));

    solns = linsolve(
        {add({x, mul(integer(2), y), mul(integer(3), z)}),
         add({mul(integer(2), x), mul(integer(2), y), mul(integer(3), z)}),
         add({mul(integer(3), x), mul(integer(3), y), mul(integer(3), z)})},
        {x, y, z});

    REQUIRE(solns.size() == 3);
    REQUIRE(eq(*solns[0], *zero));
    REQUIRE(eq(*solns[1], *zero));
    REQUIRE(eq(*solns[2], *zero));

    solns = linsolve({sub(x, integer(2)), sub(y, integer(3))}, {x, y});

    REQUIRE(solns.size() == 2);
    REQUIRE(eq(*solns[0], *integer(2)));
    REQUIRE(eq(*solns[1], *integer(3)));

    auto a = symbol("a"), b = symbol("b"), c = symbol("c"), d = symbol("d"),
         e = symbol("e"), f = symbol("f");
    solns = linsolve(
        {add({mul(a, x), mul(b, y), e}), add({mul(c, x), mul(d, y), f})},
        {x, y});
    REQUIRE(solns.size() == 2);
    // solns[0] is in unsimplified form as `(-b*(-a*f + c*e) - e*(a*d -
    // b*c))/(a*(a*d - b*c))` instead of `(-d*e + b*f)/(a*d - b*c)`
    // REQUIRE(eq(*solns[0], *div(sub(mul(d, e), mul(b,f)), sub(mul(d, a),
    // mul(b,c)))));
    REQUIRE(eq(*solns[1],
               *div(sub(mul(c, e), mul(a, f)), sub(mul(d, a), mul(b, c)))));

    solns = linsolve({Eq(y, mul({integer(4), x, x}))}, {y});
    REQUIRE(solns.size() == 1);
    REQUIRE(eq(*solns[0], *mul({integer(4), x, x})));

    CHECK_THROWS_AS(
        linsolve({Eq(y, mul({integer(4), x, x})), add({x, y, integer(-10)})},
                 {x, y}),
        SymEngineException);

    // issue 1745
    {
        auto x = SymEngine::symbol("x"), y = SymEngine::symbol("y");
        auto a = SymEngine::symbol("a"), c = SymEngine::symbol("c");

        auto eqns = {add({mul(a, y), c}), add({mul(a, x), mul(a, y)})};
        auto s1 = div(c, a);
        auto s2 = neg(s1);

        auto solns1 = linsolve(eqns, {y, x}); ///< o.k. in issue 1745
        auto solns2
            = linsolve(eqns, {x, y}); ///< results in a nan in issue 1745

        REQUIRE(solns1.size() == 2);
        REQUIRE(solns2.size() == 2);

        REQUIRE(eq(*solns1[0], *s2));
        REQUIRE(eq(*solns1[1], *s1));

        REQUIRE(eq(*solns2[0], *s1));
        REQUIRE(eq(*solns2[1], *s2));
    }
}

TEST_CASE("linear_eqns_to_matrix", "[Solve]")
{
    auto x = symbol("x"), y = symbol("y"), z = symbol("z"), t = symbol("t");
    std::pair<DenseMatrix, DenseMatrix> solns = linear_eqns_to_matrix(
        {add({x, mul(integer(2), y), mul(integer(3), z)}),
         add({mul(integer(2), x), mul(integer(2), y), mul(integer(3), z)}),
         add({mul(integer(3), x), mul(integer(3), y), mul(integer(3), z)})},
        {x, y, z});
    REQUIRE(solns.first
            == DenseMatrix(3, 3,
                           {integer(1), integer(2), integer(3), integer(2),
                            integer(2), integer(3), integer(3), integer(3),
                            integer(3)}));
    REQUIRE(solns.second == DenseMatrix(3, 1, {zero, zero, zero}));

    solns = linear_eqns_to_matrix(
        {Eq(add({x, mul(integer(2), y), mul(integer(3), z)}), zero),
         add({mul(integer(2), x), mul(integer(2), y), mul(integer(3), z)}),
         add({mul(integer(3), x), mul(integer(3), y), mul(integer(3), z)})},
        {y, x, z});

    REQUIRE(solns.first
            == DenseMatrix(3, 3,
                           {integer(2), integer(1), integer(3), integer(2),
                            integer(2), integer(3), integer(3), integer(3),
                            integer(3)}));
    REQUIRE(solns.second == DenseMatrix(3, 1, {zero, zero, zero}));

    solns = linear_eqns_to_matrix(
        {Eq(add({x, mul(integer(2), y), mul(integer(3), z),
                 mul(integer(4), t)}),
            integer(10)),
         Eq(add({mul(integer(2), x), mul(integer(2), y), mul(integer(3), z),
                 mul(integer(4), t)}),
            integer(11)),
         Eq(add({mul(integer(3), x), mul(integer(3), y), mul(integer(3), z),
                 mul(integer(4), t)}),
            integer(13)),
         Eq(add({mul(integer(9), x), mul(integer(8), y), mul(integer(7), z),
                 mul(integer(6), t)}),
            integer(30))},
        {y, z, t, x});
    REQUIRE(solns.first
            == DenseMatrix(4, 4,
                           {integer(2), integer(3), integer(4), integer(1),
                            integer(2), integer(3), integer(4), integer(2),
                            integer(3), integer(3), integer(4), integer(3),
                            integer(8), integer(7), integer(6), integer(9)}));
    REQUIRE(solns.second
            == DenseMatrix(
                4, 1, {integer(10), integer(11), integer(13), integer(30)}));
}

TEST_CASE("is_a_LinearArgTrigEquation", "[Solve]")
{
    auto x = symbol("x");
    RCP<const Basic> r;

    REQUIRE(is_a_LinearArgTrigEquation(*tan(x), *x));
    REQUIRE(is_a_LinearArgTrigEquation(*sub(tan(x), one), *x));
    REQUIRE(is_a_LinearArgTrigEquation(*add(sin(x), tan(x)), *x));
    REQUIRE(
        is_a_LinearArgTrigEquation(*add(sin(add(x, symbol("y"))), sin(x)), *x));
    REQUIRE(
        is_a_LinearArgTrigEquation(*add(sin(x), sin(add(x, symbol("y")))), *x));
    REQUIRE(not is_a_LinearArgTrigEquation(*add(tan(x), x), *x));
    REQUIRE(not is_a_LinearArgTrigEquation(*add(x, tan(x)), *x));
    REQUIRE(not is_a_LinearArgTrigEquation(*mul(x, tan(x)), *x));
    REQUIRE(is_a_LinearArgTrigEquation(*mul(tan(x), tan(x)), *x));
    REQUIRE(not is_a_LinearArgTrigEquation(*tan(mul(x, x)), *x));
}

TEST_CASE("invertComplex", "[Solve]")
{
    auto x = symbol("x"), y = symbol("y");
    RCP<const Basic> r, fX;
    RCP<const Set> gY;
    auto i2 = integer(2);

    r = invertComplex(add(x, i2), finiteset({y}), x);
    REQUIRE(eq(*r, *finiteset({sub(y, i2)})));

    r = invertComplex(mul(x, i2), finiteset({y}), x);
    REQUIRE(eq(*r, *finiteset({div(y, i2)})));
}

TEST_CASE("trigonometric equations", "[Solve]")
{
    auto x = symbol("x");
    auto n = symbol("n");
    RCP<const Set> soln;
    RCP<const Basic> eqn;
    auto i2 = integer(2);

    eqn = sin(x);
    soln = solve(eqn, x);
    auto req = set_union(
        {imageset(n, add(mul({i2, n, pi}), pi),
                  interval(NegInf, Inf, true, true)),
         imageset(n, mul({i2, n, pi}), interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = cos(x);
    soln = solve(eqn, x);
    req = set_union({imageset(n, sub(mul({i2, n, pi}), div(pi, i2)),
                              interval(NegInf, Inf, true, true)),
                     imageset(n, add(mul({i2, n, pi}), div(pi, i2)),
                              interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = tan(x);
    soln = solve(eqn, x);
    req = set_union(
        {imageset(n, add(mul({i2, n, pi}), pi),
                  interval(NegInf, Inf, true, true)),
         imageset(n, mul({i2, n, pi}), interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = csc(x);
    soln = solve(eqn, x);
    REQUIRE(eq(*soln, *emptyset()));

    eqn = sec(x);
    soln = solve(eqn, x);
    REQUIRE(eq(*soln, *emptyset()));

    eqn = cot(x);
    soln = solve(eqn, x);
    req = set_union({imageset(n, add(mul({i2, n, pi}), div(pi, i2)),
                              interval(NegInf, Inf, true, true)),
                     imageset(n, sub(mul({i2, n, pi}), div(pi, i2)),
                              interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = Eq(sin(x), one);
    soln = solve(eqn, x);
    req = imageset(n, add(mul({i2, n, pi}), div(pi, i2)),
                   interval(NegInf, Inf, true, true));
    REQUIRE(eq(*soln, *req));

    eqn = add(sin(x), cos(x));
    soln = solve(eqn, x);
    req = set_union(
        {imageset(n, sub(mul({i2, n, pi}), div(pi, integer(4))),
                  interval(NegInf, Inf, true, true)),
         imageset(n,
                  add(mul({i2, n, pi}), div(mul(integer(3), pi), integer(4))),
                  interval(NegInf, Inf, true, true))});
    // REQUIRE(eq(*soln, *req)); // atan2(sqrt(2)/2, -sqrt(2)/2) is wrongly
    // computed as it can't idenfity `-sqrt(2)/2` as negative(should pass once
    // assumptions are implemented).

    eqn = Eq(add(sin(x), cos(x)), one);
    soln = solve(eqn, x);
    req = set_union(
        {imageset(n, mul({i2, n, pi}), interval(NegInf, Inf, true, true)),
         imageset(n, add(mul({i2, n, pi}), div(pi, i2)),
                  interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = add(mul(sin(x), sin(x)), mul(cos(x), cos(x)));
    soln = solve(eqn, x);
    REQUIRE(eq(*soln, *emptyset()));

    eqn = sub(cos(x), div(one, i2));
    soln = solve(eqn, x);
    req = set_union({imageset(n, add(mul({i2, n, pi}), div(pi, integer(3))),
                              interval(NegInf, Inf, true, true)),
                     imageset(n, sub(mul({i2, n, pi}), div(pi, integer(3))),
                              interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    auto y = symbol("y");
    eqn = sub(sin(add(x, y)), sin(x));
    soln = solve(eqn, y);
    req = conditionset(y, Eq(eqn, zero));
    REQUIRE(
        eq(*soln,
           *req)); // expand(exp(x + I*y)) stays as `exp(x + I*y)`. It should be
    // `exp(x)*exp(I*y)`.

    eqn = mul(sin(x), cos(x));
    soln = solve(eqn, x);
    req = set_union(
        {imageset(n, sub(mul({i2, n, pi}), div(pi, i2)),
                  interval(NegInf, Inf, true, true)),
         imageset(n, add(mul({i2, n, pi}), div(pi, i2)),
                  interval(NegInf, Inf, true, true)),
         imageset(n, add(mul({i2, n, pi}), pi),
                  interval(NegInf, Inf, true, true)),
         imageset(n, mul({i2, n, pi}), interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = add(mul(cos(x), cos(x)), cos(x));
    soln = solve(eqn, x);
    req = set_union({imageset(n, add(mul({i2, n, pi}), pi),
                              interval(NegInf, Inf, true, true)),
                     imageset(n, add(mul({i2, n, pi}), div(pi, i2)),
                              interval(NegInf, Inf, true, true)),
                     imageset(n, sub(mul({i2, n, pi}), div(pi, i2)),
                              interval(NegInf, Inf, true, true))});
    REQUIRE(eq(*soln, *req));

    eqn = add({mul(i2, cos(x)), cot(x), sin(x), mul(sin(x), cos(x))});
    soln = solve(eqn, x);
    req = conditionset(x, Eq(eqn, zero));
    REQUIRE(eq(*soln, *req));
}
