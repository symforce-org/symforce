#include "catch.hpp"
#include <chrono>

#include <symengine/printers/strprinter.h>
#include <symengine/symengine_exception.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Expression;
using SymEngine::integer;
using SymEngine::Integer;
using SymEngine::integer_class;
using SymEngine::make_rcp;
using SymEngine::map_int_Expr;
using SymEngine::MExprPoly;
using SymEngine::one;
using SymEngine::Pow;
using SymEngine::Precedence;
using SymEngine::PrecedenceEnum;
using SymEngine::print_stack_on_segfault;
using SymEngine::RCP;
using SymEngine::RCPBasicKeyLess;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::UExprPoly;
using SymEngine::vec_basic;
using SymEngine::vec_int;
using SymEngine::vec_uint;
using SymEngine::zero;

using namespace SymEngine::literals;

TEST_CASE("Constructing MExprPoly", "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression num1(2);                     // 2
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp2(2 - Expression("d"));  //(2 - d)
    Expression comp3(-3 + Expression("e")); //(-3 + e)
    Expression comp4(-4 - Expression("f")); //(-4 - f)
    vec_basic s;
    vec_int v;

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y},
        {{{1, 1}, a}, {{1, 2}, negB}, {{2, 1}, num1}, {{0, 1}, negNum}});
    RCP<const MExprPoly> pprime = MExprPoly::from_dict(
        {y, x},
        {{{1, 1}, a}, {{1, 2}, negB}, {{2, 1}, num1}, {{0, 1}, negNum}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {x, y},
        {{{1, 0}, comp1}, {{0, 0}, comp2}, {{2, 2}, comp3}, {{3, 4}, comp4}});
    RCP<const MExprPoly> p3
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(integer(0))}});
    RCP<const MExprPoly> p4 = MExprPoly::from_dict(s, {{v, Expression(0)}});
    RCP<const MExprPoly> p5 = MExprPoly::from_dict(s, {{v, comp1}});
    RCP<const MExprPoly> p6 = MExprPoly::from_dict({x, y}, {{{0, 0}, comp1},
                                                            {{0, -1}, comp2},
                                                            {{-2, 2}, comp3},
                                                            {{-3, -3}, comp4}});

    REQUIRE(eq(*p1->as_symbolic(),
               *add({mul(integer(2), mul(pow(x, integer(2)), y)),
                     mul(negB.get_basic(), mul(x, pow(y, integer(2)))),
                     mul(symbol("a"), mul(x, y)), mul(integer(-3), y)})));
    REQUIRE(eq(*pprime->as_symbolic(),
               *add({mul(negB.get_basic(), mul(pow(x, integer(2)), y)),
                     mul(integer(2), mul(x, pow(y, integer(2)))),
                     mul(symbol("a"), mul(x, y)), mul(integer(-3), x)})));
    REQUIRE(eq(*p2->as_symbolic(),
               *add({mul(comp4.get_basic(),
                         mul(pow(x, integer(3)), pow(y, integer(4)))),
                     mul(comp3.get_basic(),
                         mul(pow(x, integer(2)), pow(y, integer(2)))),
                     mul(comp1.get_basic(), x), comp2.get_basic()})));
    REQUIRE(eq(*p3->as_symbolic(), *zero));
    REQUIRE(eq(*p4->as_symbolic(), *zero));
    REQUIRE(eq(*p5->as_symbolic(), *comp1.get_basic()));
    REQUIRE(eq(*p6->as_symbolic(),
               *add({comp1.get_basic(),
                     mul(comp3.get_basic(),
                         mul(pow(x, integer(-2)), pow(y, integer(2)))),
                     mul(comp2.get_basic(), pow(y, integer(-1))),
                     mul(comp4.get_basic(),
                         mul(pow(x, integer(-3)), pow(y, integer(-3))))})));
}

TEST_CASE("Testing MExprPoly::__eq__(), __hash__, and compare", "[MExprPoly]")
{
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> i = symbol("i");
    RCP<const Symbol> j = symbol("j");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression sum(add(i, j));
    Expression difference(sub(mul(two, i), j));
    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y}, {{{2, 0}, sum}, {{1, -1}, Expression(a)}, {{0, 2}, sum}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {x, y}, {{{2, 0}, sum}, {{1, -1}, Expression(a) * -1}, {{0, 2}, sum}});
    RCP<const MExprPoly> p3 = MExprPoly::from_dict(
        {x, y}, {{{2, 0}, sum + sum}, {{0, 2}, sum + sum}});
    RCP<const MExprPoly> p4
        = MExprPoly::from_dict({a, b}, {{{2, 0}, sum * 2}, {{0, 2}, sum * 2}});
    vec_basic s;
    vec_int v;
    RCP<const MExprPoly> p5 = MExprPoly::from_dict(s, {{v, Expression(0)}});
    RCP<const MExprPoly> p6
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(0)}});
    RCP<const MExprPoly> p7 = MExprPoly::from_dict(s, {{v, sum}});
    RCP<const MExprPoly> p8 = MExprPoly::from_dict({x, y}, {{{0, 0}, sum}});

    REQUIRE(p1->__eq__(*p1));
    REQUIRE(!(p2->__eq__(*p1)));
    REQUIRE(p3->__eq__(*add_mpoly(*p1, *p2)));
    REQUIRE(p5->__eq__(*p6));
    REQUIRE(p7->__eq__(*p8));
    REQUIRE(!p6->__eq__(*p7));

    // Only requre that the same polynomial hash to the same value and that
    // different polynomials
    // hash to different values
    // Don't want to require a polynomial to have a particular hash in case
    // someone comes up with
    // a better hash function
    REQUIRE(p3->__hash__() == add_mpoly(*p1, *p2)->__hash__());
    REQUIRE(p1->__hash__() != p2->__hash__());
    REQUIRE(p3->__hash__() != p4->__hash__());

    // Same for compare.
    REQUIRE(0 == p3->compare(*add_mpoly(*p1, *p2)));
    REQUIRE(0 != p1->compare(*p2));
    REQUIRE(0 != p3->compare(*p4));
}

TEST_CASE("Testing MExprPoly::eval", "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(div(b, a));
    Expression ex(add(b, c));
    Expression why(mul(a, b));
    Expression zee(div(a, b));

    RCP<const MExprPoly> p
        = MExprPoly::from_dict({x, y, z}, {{{2, 0, 0}, expr1},
                                           {{0, 2, 0}, expr2},
                                           {{0, 0, 2}, expr3},
                                           {{1, 1, 1}, expr4},
                                           {{1, 1, 0}, expr1},
                                           {{0, 1, 1}, expr2},
                                           {{1, 0, 0}, expr1},
                                           {{0, 1, 0}, expr2},
                                           {{0, 0, 1}, expr3},
                                           {{0, 0, 0}, expr4},
                                           {{-1, -1, -1}, expr1},
                                           {{-2, -2, -2}, expr2},
                                           {{-2, 2, -2}, expr3}});
    std::map<RCP<const Basic>, Expression, RCPBasicKeyLess> m1
        = {{x, Expression(0)}, {y, Expression(0)}, {z, Expression(0)}};
    std::map<RCP<const Basic>, Expression, RCPBasicKeyLess> m2
        = {{x, ex}, {y, why}, {z, zee}};
    // CHECK_THROWS_AS(p->eval(m1), SymEngineException);
    REQUIRE(p->eval(m2)
            == expr1 * pow(ex, 2) + expr2 * pow(why, 2) + expr3 * pow(zee, 2)
                   + expr4 * ex * why * zee + expr1 * ex * why
                   + expr2 * why * zee + expr1 * ex + expr2 * why + expr3 * zee
                   + expr4 + expr1 * pow(ex, -1) * pow(why, -1) * pow(zee, -1)
                   + expr2 * pow(ex, -2) * pow(why, -2) * pow(zee, -2)
                   + expr3 * pow(ex, -2) * pow(why, 2) * pow(zee, -2));
}

TEST_CASE("Testing derivative of MExprPoly", "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(div(b, a));
    RCP<const MExprPoly> p = MExprPoly::from_dict({x, y}, {{{2, 1}, expr1},
                                                           {{1, 2}, expr2},
                                                           {{2, 0}, expr3},
                                                           {{0, 2}, expr4},
                                                           {{1, 0}, expr1},
                                                           {{0, 1}, expr2},
                                                           {{0, 0}, expr3},
                                                           {{-1, 0}, expr4},
                                                           {{0, -2}, expr1}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y}, {{{1, 1}, expr1 * 2},
                                        {{0, 2}, expr2},
                                        {{1, 0}, expr3 * 2},
                                        {{0, 0}, expr1},
                                        {{-2, 0}, expr4 * -1}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y}, {{{2, 0}, expr1},
                                        {{1, 1}, expr2 * 2},
                                        {{0, 1}, expr4 * 2},
                                        {{0, 0}, expr2},
                                        {{0, -3}, expr1 * -2}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(0)}});

    REQUIRE(eq(*(p->diff(x)), *q1));
    REQUIRE(eq(*(p->diff(y)), *q2));
    REQUIRE(eq(*(p->diff(z)), *q3));
}

TEST_CASE("Testing MExprPoly::get_args()", "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(pow(b, a));
    RCP<const MExprPoly> p1
        = MExprPoly::from_dict({x, y, z}, {{{0, 0, 0}, Expression(1)},
                                           {{1, 1, 1}, Expression(2)},
                                           {{0, 0, 2}, Expression(1)}});
    RCP<const MExprPoly> p2
        = MExprPoly::from_dict({x, y, z}, {{{0, 0, 0}, expr1},
                                           {{1, 1, 1}, expr2},
                                           {{0, 0, 2}, expr3},
                                           {{0, 2, 0}, expr4},
                                           {{-1, -1, -1}, expr2},
                                           {{0, 0, -2}, expr3},
                                           {{0, -2, 0}, expr4}});
    REQUIRE(eq(*p1->as_symbolic(), *add({mul(integer(2), mul(x, mul(y, z))),
                                         pow(z, integer(2)), one})));
    REQUIRE(
        eq(*p2->as_symbolic(),
           *add({mul(expr2.get_basic(), mul(x, mul(y, z))),
                 mul(expr4.get_basic(), pow(y, integer(2))),
                 mul(expr3.get_basic(), pow(z, integer(2))), expr1.get_basic(),
                 mul(expr3.get_basic(), pow(z, integer(-2))),
                 mul(expr4.get_basic(), pow(y, integer(-2))),
                 mul(expr2.get_basic(),
                     mul(pow(x, integer(-1)),
                         mul(pow(y, integer(-1)), pow(z, integer(-1)))))})));
}

TEST_CASE("Testing MExprPoly negation"
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression num1(2);                     // 2
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp2(2 - Expression("d"));  //(2 - d)
    Expression comp3(-3 + Expression("e")); //(-3 + e)
    Expression comp4(-4 - Expression("f")); //(-4 - f)

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y},
        {{{1, 1}, a}, {{1, -2}, negB}, {{-2, 1}, num1}, {{0, 1}, negNum}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {x, y},
        {{{1, 0}, comp1}, {{0, 0}, comp2}, {{2, 2}, comp3}, {{3, 4}, comp4}});
    RCP<const MExprPoly> p3
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(integer(0))}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y}, {{{1, 1}, a * -1},
                                        {{1, -2}, negB * -1},
                                        {{-2, 1}, num1 * -1},
                                        {{0, 1}, negNum * -1}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y}, {{{1, 0}, comp1 * -1},
                                        {{0, 0}, comp2 * -1},
                                        {{2, 2}, comp3 * -1},
                                        {{3, 4}, comp4 * -1}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(integer(0))}});

    REQUIRE(neg_mpoly(*p1)->__eq__(*q1));
    REQUIRE(neg_mpoly(*p2)->__eq__(*q2));
    REQUIRE(neg_mpoly(*p3)->__eq__(*q3));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPolys with the same set of variables",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression num1(2);                     // 2
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp4(-4 - Expression("f")); //(-4 - f)

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y},
        {{{1, 1}, a}, {{1, 0}, negB}, {{2, -1}, num1}, {{0, -1}, negNum}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {x, y}, {{{1, 0}, comp1}, {{0, 0}, comp4}, {{0, -1}, comp4}});
    RCP<const MExprPoly> p3
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(integer(0))}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y}, {{{1, 1}, a},
                                        {{1, 0}, negB + comp1},
                                        {{2, -1}, num1},
                                        {{0, 0}, comp4},
                                        {{0, -1}, comp4 + negNum}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y}, {{{2, -1}, num1},
                                        {{1, 1}, a},
                                        {{1, 0}, (-1 * comp1) + negB},
                                        {{0, -1}, negNum - comp4},
                                        {{0, 0}, comp4 * -1}});
    RCP<const MExprPoly> q22
        = MExprPoly::from_dict({x, y}, {{{2, -1}, -1 * num1},
                                        {{1, 1}, -1 * a},
                                        {{1, 0}, comp1 - negB},
                                        {{0, -1}, comp4 - negNum},
                                        {{0, 0}, comp4}});
    RCP<const MExprPoly> q3 = MExprPoly::from_dict(
        {x, y}, {{{2, 1}, a * comp1},
                 {{1, 1}, a * comp4},
                 {{1, 0}, a * comp4 + negB * comp4},
                 {{2, 0}, negB * comp1},
                 {{1, -1}, negB * comp4 + negNum * comp1},
                 {{3, -1}, num1 * comp1},
                 {{2, -1}, num1 * comp4},
                 {{2, -2}, num1 * comp4},
                 {{0, -1}, negNum * comp4},
                 {{0, -2}, negNum * comp4}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*add_mpoly(*p1, *p3), *p1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q22));
    REQUIRE(eq(*sub_mpoly(*p1, *p3), *p1));

    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q3));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p3), *p3));
    REQUIRE(eq(*mul_mpoly(*p3, *p1), *p3));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MultivaritePolynomials with disjoint sets of varables",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> n = symbol("n");
    RCP<const Symbol> m = symbol("m");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp4(-4 - Expression("f")); //(-4 - f)

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y}, {{{1, 1}, a}, {{-1, 0}, negB}, {{0, 0}, negNum}});
    RCP<const MExprPoly> p2
        = MExprPoly::from_dict({m, n}, {{{1, 0}, comp1}, {{2, -1}, comp4}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({m, n, x, y}, {{{0, 0, 1, 1}, a},
                                              {{0, 0, -1, 0}, negB},
                                              {{0, 0, 0, 0}, negNum},
                                              {{1, 0, 0, 0}, comp1},
                                              {{2, -1, 0, 0}, comp4}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({m, n, x, y}, {{{0, 0, 1, 1}, a},
                                              {{0, 0, -1, 0}, negB},
                                              {{0, 0, 0, 0}, negNum},
                                              {{1, 0, 0, 0}, comp1 * -1},
                                              {{2, -1, 0, 0}, comp4 * -1}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({m, n, x, y}, {{{0, 0, 1, 1}, a * -1},
                                              {{0, 0, -1, 0}, negB * -1},
                                              {{0, 0, 0, 0}, negNum * -1},
                                              {{1, 0, 0, 0}, comp1},
                                              {{2, -1, 0, 0}, comp4}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({m, n, x, y}, {{{2, -1, 1, 1}, a * comp4},
                                              {{2, -1, -1, 0}, negB * comp4},
                                              {{2, -1, 0, 0}, negNum * comp4},
                                              {{1, 0, 1, 1}, a * comp1},
                                              {{1, 0, -1, 0}, negB * comp1},
                                              {{1, 0, 0, 0}, negNum * comp1}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPolys with an overlapping set of variables",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp4(-4 - Expression("f")); //(-4 - f)

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y}, {{{1, -1}, a}, {{1, 0}, negB}, {{0, 0}, negNum}});
    RCP<const MExprPoly> p2
        = MExprPoly::from_dict({y, z}, {{{1, 0}, comp1}, {{-2, 1}, comp4}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y, z}, {{{1, -1, 0}, a},
                                           {{1, 0, 0}, negB},
                                           {{0, 0, 0}, negNum},
                                           {{0, 1, 0}, comp1},
                                           {{0, -2, 1}, comp4}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y, z}, {{{1, -1, 0}, a},
                                           {{1, 0, 0}, negB},
                                           {{0, 0, 0}, negNum},
                                           {{0, 1, 0}, comp1 * -1},
                                           {{0, -2, 1}, comp4 * -1}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({x, y, z}, {{{1, -1, 0}, a * -1},
                                           {{1, 0, 0}, negB * -1},
                                           {{0, 0, 0}, negNum * -1},
                                           {{0, 1, 0}, comp1},
                                           {{0, -2, 1}, comp4}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({x, y, z}, {{{1, 0, 0}, a * comp1},
                                           {{1, -3, 1}, a * comp4},
                                           {{1, 1, 0}, negB * comp1},
                                           {{1, -2, 1}, negB * comp4},
                                           {{0, 1, 0}, negNum * comp1},
                                           {{0, -2, 1}, comp4 * negNum}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPolys with the same variable",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(div(b, a));
    Expression expr5(add(b, c));
    Expression expr6(mul(a, b));
    Expression expr7(div(a, b));
    RCP<const MExprPoly> p1
        = MExprPoly::from_dict({x}, {{{1}, expr1}, {{2}, expr2}, {{0}, expr3}});
    RCP<const MExprPoly> p2
        = MExprPoly::from_dict({x}, {{{0}, expr4}, {{1}, expr1}});

    RCP<const MExprPoly> q1 = MExprPoly::from_dict(
        {x}, {{{1}, expr1 + expr1}, {{0}, expr4 + expr3}, {{2}, expr2}});
    RCP<const MExprPoly> q2 = MExprPoly::from_dict(
        {x}, {{{0}, expr3 - expr4}, {{1}, expr1 - expr1}, {{2}, expr2}});
    RCP<const MExprPoly> q3 = MExprPoly::from_dict(
        {x}, {{{0}, expr4 - expr3}, {{1}, expr1 - expr1}, {{2}, expr2 * -1}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({x}, {{{3}, expr2 * expr1},
                                     {{2}, expr2 * expr4 + expr1 * expr1},
                                     {{1}, expr1 * expr4 + expr1 * expr3},
                                     {{0}, expr3 * expr4}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPolys with the different variables",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(div(b, a));
    Expression expr5(add(b, c));
    RCP<const MExprPoly> p1
        = MExprPoly::from_dict({x}, {{{1}, expr1}, {{2}, expr2}, {{0}, expr3}});
    RCP<const MExprPoly> p2
        = MExprPoly::from_dict({y}, {{{0}, expr4}, {{1}, expr5}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y}, {{{1, 0}, expr1},
                                        {{2, 0}, expr2},
                                        {{0, 0}, expr3 + expr4},
                                        {{0, 0}, expr4},
                                        {{0, 1}, expr5}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y}, {{{1, 0}, expr1},
                                        {{2, 0}, expr2},
                                        {{0, 0}, expr3 - expr4},
                                        {{0, 1}, expr5 * -1}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({x, y}, {{{1, 0}, expr1 * -1},
                                        {{2, 0}, expr2 * -1},
                                        {{0, 0}, expr4 - expr3},
                                        {{0, 1}, expr5}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({x, y}, {{{2, 1}, expr2 * expr5},
                                        {{2, 0}, expr2 * expr4},
                                        {{1, 1}, expr1 * expr5},
                                        {{1, 0}, expr1 * expr4},
                                        {{0, 1}, expr3 * expr5},
                                        {{0, 0}, expr3 * expr4}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPolys with a MExprPoly whose variable "
          "are in the variable set",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp4(-4 - Expression("f")); //(-4 - f)

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y}, {{{1, 1}, a}, {{1, 0}, negB}, {{0, 0}, negNum}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {y}, {{{0}, comp4}, {{1}, Expression(integer(2))}, {{2}, comp1}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y}, {{{1, 1}, a},
                                        {{0, 2}, comp1},
                                        {{1, 0}, negB},
                                        {{0, 1}, Expression(2)},
                                        {{0, 0}, comp4 + negNum}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y}, {{{1, 1}, a},
                                        {{0, 2}, comp1 * -1},
                                        {{1, 0}, negB},
                                        {{0, 1}, Expression(-2)},
                                        {{0, 0}, -1 * comp4 + negNum}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({x, y}, {{{1, 1}, a * -1},
                                        {{0, 2}, comp1},
                                        {{1, 0}, negB * -1},
                                        {{0, 1}, Expression(2)},
                                        {{0, 0}, comp4 - negNum}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({x, y}, {{{1, 3}, a * comp1},
                                        {{1, 2}, 2 * a + negB * comp1},
                                        {{1, 1}, a * comp4 + negB * 2},
                                        {{0, 2}, negNum * comp1},
                                        {{1, 0}, negB * comp4},
                                        {{0, 1}, 2 * negNum},
                                        {{0, 0}, negNum * comp4}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPolys with a MExprPoly whose variables "
          "are not in the variable set",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    Expression a("a");                      // a
    Expression negB(-Expression("b"));      //-b
    Expression negNum(-3);                  //-3
    Expression comp1(1 + Expression("c"));  //(1+c)
    Expression comp4(-4 - Expression("f")); //(-4 - f)

    RCP<const MExprPoly> p1 = MExprPoly::from_dict(
        {x, y}, {{{1, 1}, a}, {{1, 0}, negB}, {{0, 0}, negNum}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {z}, {{{0}, comp4}, {{1}, Expression(integer(2))}, {{2}, comp1}});

    RCP<const MExprPoly> q1
        = MExprPoly::from_dict({x, y, z}, {{{1, 1, 0}, a},
                                           {{0, 0, 2}, comp1},
                                           {{1, 0, 0}, negB},
                                           {{0, 0, 1}, Expression(2)},
                                           {{0, 0, 0}, negNum + comp4}});
    RCP<const MExprPoly> q2
        = MExprPoly::from_dict({x, y, z}, {{{1, 1, 0}, a},
                                           {{0, 0, 2}, comp1 * -1},
                                           {{1, 0, 0}, negB},
                                           {{0, 0, 1}, Expression(-2)},
                                           {{0, 0, 0}, negNum - comp4}});
    RCP<const MExprPoly> q3
        = MExprPoly::from_dict({x, y, z}, {{{1, 1, 0}, a * -1},
                                           {{0, 0, 2}, comp1},
                                           {{1, 0, 0}, negB * -1},
                                           {{0, 0, 1}, Expression(2)},
                                           {{0, 0, 0}, -1 * negNum + comp4}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({x, y, z}, {{{1, 1, 2}, a * comp1},
                                           {{1, 1, 1}, 2 * a},
                                           {{1, 0, 2}, negB * comp1},
                                           {{1, 1, 0}, a * comp4},
                                           {{1, 0, 1}, 2 * negB},
                                           {{0, 0, 2}, negNum * comp1},
                                           {{1, 0, 0}, negB * comp4},
                                           {{0, 0, 1}, 2 * negNum},
                                           {{0, 0, 0}, negNum * comp4}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MExprPoly with empty set of variables ",
          "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(pow(b, a));

    vec_basic s;
    vec_int v;
    RCP<const MExprPoly> p1 = MExprPoly::from_dict(s, {{v, expr1}});
    RCP<const MExprPoly> p2 = MExprPoly::from_dict(
        {x, y}, {{{0, 0}, expr2}, {{0, 1}, expr3}, {{-1, 0}, expr4}});

    RCP<const MExprPoly> q1 = MExprPoly::from_dict(
        {x, y}, {{{0, 0}, expr2 + expr1}, {{0, 1}, expr3}, {{-1, 0}, expr4}});
    RCP<const MExprPoly> q2 = MExprPoly::from_dict(
        {x, y}, {{{0, 0}, expr2 - expr1}, {{0, 1}, expr3}, {{-1, 0}, expr4}});
    RCP<const MExprPoly> q3 = MExprPoly::from_dict(
        {x, y},
        {{{0, 0}, expr1 - expr2}, {{0, 1}, -1 * expr3}, {{-1, 0}, -1 * expr4}});
    RCP<const MExprPoly> q4
        = MExprPoly::from_dict({x, y}, {{{0, 0}, expr2 * expr1},
                                        {{0, 1}, expr3 * expr1},
                                        {{-1, 0}, expr4 * expr1}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q3));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q2));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing Precedence of MExprPoly", "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(add(a, b));
    Expression expr2(sub(mul(two, a), b));
    Expression expr3(mul(a, c));
    Expression expr4(pow(b, a));
    Expression expr5(a);
    Precedence Prec;
    RCP<const MExprPoly> p1
        = MExprPoly::from_dict({x, y}, {{{0, 0}, Expression(0)}});
    RCP<const MExprPoly> p2
        = MExprPoly::from_dict({x, y}, {{{1, 0}, expr1}, {{0, 0}, expr2}});
    RCP<const MExprPoly> p3 = MExprPoly::from_dict({x, y}, {{{0, 0}, expr5}});
    RCP<const MExprPoly> p4
        = MExprPoly::from_dict({x, y}, {{{1, 0}, Expression(1)}});
    RCP<const MExprPoly> p5 = MExprPoly::from_dict({x, y}, {{{1, 1}, expr4}});
    RCP<const MExprPoly> p6
        = MExprPoly::from_dict({x, y}, {{{2, 0}, Expression(1)}});
    RCP<const MExprPoly> p7 = MExprPoly::from_dict({x, y}, {{{1, 0}, expr1}});

    REQUIRE(Prec.getPrecedence(p1) == PrecedenceEnum::Atom);
    REQUIRE(Prec.getPrecedence(p2) == PrecedenceEnum::Add);
    REQUIRE(Prec.getPrecedence(p3) == PrecedenceEnum::Atom);
    REQUIRE(Prec.getPrecedence(p4) == PrecedenceEnum::Atom);
    REQUIRE(Prec.getPrecedence(p5) == PrecedenceEnum::Mul);
    REQUIRE(Prec.getPrecedence(p6) == PrecedenceEnum::Pow);
    REQUIRE(Prec.getPrecedence(p7) == PrecedenceEnum::Mul);
}

TEST_CASE("MExprPoly from_poly", "[MExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const MExprPoly> mpoly
        = MExprPoly::from_dict({x}, {{{1}, Expression("y")}, {{2}, 3_z}});
    RCP<const UExprPoly> upoly
        = UExprPoly::from_vec(x, {0_z, Expression("y"), 3_z});

    REQUIRE(eq(*MExprPoly::from_poly(*upoly), *mpoly));
}
