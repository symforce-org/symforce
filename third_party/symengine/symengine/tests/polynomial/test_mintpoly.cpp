#include "catch.hpp"
#include <chrono>

#include <symengine/printers/strprinter.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Expression;
using SymEngine::integer;
using SymEngine::Integer;
using SymEngine::integer_class;
using SymEngine::make_rcp;
using SymEngine::map_uint_mpz;
using SymEngine::MIntPoly;
using SymEngine::one;
using SymEngine::Pow;
using SymEngine::Precedence;
using SymEngine::PrecedenceEnum;
using SymEngine::print_stack_on_segfault;
using SymEngine::RCP;
using SymEngine::RCPBasicKeyLess;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::UIntPoly;
using SymEngine::vec_basic;
using SymEngine::vec_int;
using SymEngine::vec_uint;
using SymEngine::zero;

using namespace SymEngine::literals;

TEST_CASE("Constructing MIntPoly", "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const MIntPoly> P = MIntPoly::from_dict({x, y}, {{{1, 2}, 1_z},
                                                         {{1, 1}, 2_z},
                                                         {{0, 1}, 2_z},
                                                         {{1, 0}, 3_z},
                                                         {{0, 0}, 0_z}});

    REQUIRE(eq(*P->as_symbolic(),
               *add({mul(x, pow(y, integer(2))), mul(integer(2), mul(x, y)),
                     mul(integer(3), x), mul(integer(2), y)})));

    RCP<const MIntPoly> Pprime = MIntPoly::from_dict({y, x}, {{{1, 2}, 1_z},
                                                              {{1, 1}, 2_z},
                                                              {{0, 1}, 2_z},
                                                              {{1, 0}, 3_z},
                                                              {{0, 0}, 0_z}});

    REQUIRE(eq(*Pprime->as_symbolic(),
               *add({mul(pow(x, integer(2)), y), mul(integer(2), mul(x, y)),
                     mul(integer(2), x), mul(integer(3), y)})));

    RCP<const MIntPoly> P2 = MIntPoly::from_dict({x, y}, {{{0, 0}, 0_z}});

    vec_basic s;
    vec_uint v;
    RCP<const MIntPoly> P3 = MIntPoly::from_dict(s, {{v, 0_z}});

    REQUIRE(eq(*P2->as_symbolic(), *zero));
    REQUIRE(eq(*P3->as_symbolic(), *zero));

    RCP<const MIntPoly> P4 = MIntPoly::from_dict(s, {{v, 5_z}});
    REQUIRE(eq(*P4->as_symbolic(), *integer(5)));
}

TEST_CASE("Testing MIntPoly::__hash__() and compare", "[MIntPoly]")
{
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict(
        {x, y}, {{{2, 0}, 1_z}, {{1, 1}, 1_z}, {{0, 2}, 1_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict(
        {x, y}, {{{2, 0}, 1_z}, {{1, 1}, -1_z}, {{0, 2}, 1_z}});
    RCP<const MIntPoly> p3
        = MIntPoly::from_dict({x, y}, {{{2, 0}, 2_z}, {{0, 2}, 2_z}});
    RCP<const MIntPoly> p4
        = MIntPoly::from_dict({a, b}, {{{2, 0}, 2_z}, {{0, 2}, 2_z}});

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

TEST_CASE("Testing MIntPoly::__eq__(const Basic &o)", "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict(
        {x, y}, {{{2, 0}, 1_z}, {{1, 1}, 1_z}, {{0, 2}, 1_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict(
        {x, y}, {{{2, 0}, 1_z}, {{1, 1}, -1_z}, {{0, 2}, 1_z}});
    RCP<const MIntPoly> p3
        = MIntPoly::from_dict({x, y}, {{{2, 0}, 2_z}, {{0, 2}, 2_z}});
    RCP<const MIntPoly> p4 = MIntPoly::from_dict({x}, {{{0}, 5_z}});
    RCP<const MIntPoly> p5 = MIntPoly::from_dict({y}, {{{0}, 5_z}});
    RCP<const MIntPoly> p6 = MIntPoly::from_dict({x}, {{{0}, 0_z}});
    RCP<const MIntPoly> p7 = MIntPoly::from_dict({y}, {{{0}, 0_z}});

    REQUIRE(p1->__eq__(*p1));
    REQUIRE(!(p2->__eq__(*p1)));
    REQUIRE(p3->__eq__(*add_mpoly(*p1, *p2)));
    REQUIRE(p4->__eq__(*p5));
    REQUIRE(p6->__eq__(*p7));
    REQUIRE(!p5->__eq__(*p6));
}

TEST_CASE("Testing MIntPoly::eval((std::map<RCP<const "
          "Symbol>, integer_class, RCPSymbolCompare> &vals)",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p = MIntPoly::from_dict({x, y, z}, {{{2, 0, 0}, 1_z},
                                                            {{0, 2, 0}, 2_z},
                                                            {{0, 0, 2}, 3_z},
                                                            {{1, 1, 1}, 4_z},
                                                            {{1, 1, 0}, 1_z},
                                                            {{0, 1, 1}, 2_z},
                                                            {{1, 0, 0}, 1_z},
                                                            {{0, 1, 0}, 2_z},
                                                            {{0, 0, 1}, 3_z},
                                                            {{0, 0, 0}, 5_z}});
    std::map<RCP<const Basic>, integer_class, RCPBasicKeyLess> m1
        = {{x, 1_z}, {y, 2_z}, {z, 5_z}};
    std::map<RCP<const Basic>, integer_class, RCPBasicKeyLess> m2
        = {{x, 0_z}, {y, 0_z}, {z, 0_z}};
    std::map<RCP<const Basic>, integer_class, RCPBasicKeyLess> m3
        = {{x, -1_z}, {y, -2_z}, {z, -5_z}};

    REQUIRE(171_z == p->eval(m1));
    REQUIRE(5_z == p->eval(m2));
    REQUIRE(51_z == p->eval(m3));
}

TEST_CASE("Testing MIntPoly negation", "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p = MIntPoly::from_dict(
        {x, y, z}, {{{1, 0, 0}, 1_z}, {{0, 1, 0}, -2_z}, {{0, 0, 1}, 3_z}});
    RCP<const MIntPoly> p2 = neg_mpoly(*p);

    RCP<const MIntPoly> q = MIntPoly::from_dict(
        {x, y, z}, {{{1, 0, 0}, -1_z}, {{0, 1, 0}, 2_z}, {{0, 0, 1}, -3_z}});

    REQUIRE(p2->__eq__(*q));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MIntPolys with the same set of variables",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, 2_z},
                                                             {{4, 1, 0}, 3_z},
                                                             {{0, 0, 0}, 4_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 3_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 2_z},
                                                             {{4, 1, 0}, 3_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 7_z}});
    RCP<const MIntPoly> q2 = MIntPoly::from_dict({x, y, z}, {{{3, 2, 1}, 4_z},
                                                             {{4, 1, 0}, 3_z},
                                                             {{0, 1, 2}, -1_z},
                                                             {{0, 0, 0}, 1_z}});
    RCP<const MIntPoly> q3 = MIntPoly::from_dict({x, y, z}, {{{2, 4, 6}, 1_z},
                                                             {{5, 3, 3}, 3_z},
                                                             {{6, 4, 2}, -4_z},
                                                             {{7, 3, 1}, -6_z},
                                                             {{1, 3, 5}, 1_z},
                                                             {{3, 3, 3}, 2_z},
                                                             {{4, 2, 2}, 3_z},
                                                             {{0, 1, 2}, 4_z},
                                                             {{4, 1, 0}, 9_z},
                                                             {{0, 0, 0}, 12_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{1, 2, 3}, 7_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q3));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q3));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MIntPolys with disjoint sets of varables",
          "[MIntPoly]")
{
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict({a, b, c}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, 2_z},
                                                             {{4, 1, 0}, 3_z},
                                                             {{0, 0, 0}, 4_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 3_z}});

    RCP<const MIntPoly> q1
        = MIntPoly::from_dict({a, b, c, x, y, z}, {{{1, 2, 3, 0, 0, 0}, 1_z},
                                                   {{3, 2, 1, 0, 0, 0}, 2_z},
                                                   {{4, 1, 0, 0, 0, 0}, 3_z},
                                                   {{0, 0, 0, 0, 0, 0}, 7_z},
                                                   {{0, 0, 0, 1, 2, 3}, 1_z},
                                                   {{0, 0, 0, 3, 2, 1}, -2_z},
                                                   {{0, 0, 0, 0, 1, 2}, 1_z}});
    RCP<const MIntPoly> q2
        = MIntPoly::from_dict({a, b, c, x, y, z}, {{{1, 2, 3, 0, 0, 0}, 1_z},
                                                   {{3, 2, 1, 0, 0, 0}, 2_z},
                                                   {{4, 1, 0, 0, 0, 0}, 3_z},
                                                   {{0, 0, 0, 0, 0, 0}, 1_z},
                                                   {{0, 0, 0, 1, 2, 3}, -1_z},
                                                   {{0, 0, 0, 3, 2, 1}, 2_z},
                                                   {{0, 0, 0, 0, 1, 2}, -1_z}});
    RCP<const MIntPoly> q3
        = MIntPoly::from_dict({a, b, c, x, y, z}, {{{1, 2, 3, 0, 0, 0}, -1_z},
                                                   {{3, 2, 1, 0, 0, 0}, -2_z},
                                                   {{4, 1, 0, 0, 0, 0}, -3_z},
                                                   {{0, 0, 0, 0, 0, 0}, -1_z},
                                                   {{0, 0, 0, 1, 2, 3}, 1_z},
                                                   {{0, 0, 0, 3, 2, 1}, -2_z},
                                                   {{0, 0, 0, 0, 1, 2}, 1_z}});
    RCP<const MIntPoly> q4
        = MIntPoly::from_dict({a, b, c, x, y, z}, {{{1, 2, 3, 1, 2, 3}, 1_z},
                                                   {{3, 2, 1, 1, 2, 3}, 2_z},
                                                   {{4, 1, 0, 1, 2, 3}, 3_z},
                                                   {{0, 0, 0, 1, 2, 3}, 4_z},

                                                   {{1, 2, 3, 3, 2, 1}, -2_z},
                                                   {{3, 2, 1, 3, 2, 1}, -4_z},
                                                   {{4, 1, 0, 3, 2, 1}, -6_z},
                                                   {{0, 0, 0, 3, 2, 1}, -8_z},

                                                   {{1, 2, 3, 0, 1, 2}, 1_z},
                                                   {{3, 2, 1, 0, 1, 2}, 2_z},
                                                   {{4, 1, 0, 0, 1, 2}, 3_z},
                                                   {{0, 0, 0, 0, 1, 2}, 4_z},

                                                   {{1, 2, 3, 0, 0, 0}, 3_z},
                                                   {{3, 2, 1, 0, 0, 0}, 6_z},
                                                   {{4, 1, 0, 0, 0, 0}, 9_z},
                                                   {{0, 0, 0, 0, 0, 0}, 12_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MIntPolys with an overlapping set of variables",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict(
        {x, y}, {{{1, 2}, 1_z}, {{4, 0}, 3_z}, {{0, 3}, 4_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict(
        {y, z}, {{{2, 1}, -2_z}, {{0, 2}, 1_z}, {{1, 0}, 3_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 0}, 1_z},
                                                             {{4, 0, 0}, 3_z},
                                                             {{0, 3, 0}, 4_z},
                                                             {{0, 2, 1}, -2_z},
                                                             {{0, 0, 2}, 1_z},
                                                             {{0, 1, 0}, 3_z}});

    RCP<const MIntPoly> q2
        = MIntPoly::from_dict({x, y, z}, {{{1, 2, 0}, 1_z},
                                          {{4, 0, 0}, 3_z},
                                          {{0, 3, 0}, 4_z},
                                          {{0, 2, 1}, 2_z},
                                          {{0, 0, 2}, -1_z},
                                          {{0, 1, 0}, -3_z}});

    RCP<const MIntPoly> q3 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 0}, -1_z},
                                                             {{4, 0, 0}, -3_z},
                                                             {{0, 3, 0}, -4_z},
                                                             {{0, 2, 1}, -2_z},
                                                             {{0, 0, 2}, 1_z},
                                                             {{0, 1, 0}, 3_z}});

    RCP<const MIntPoly> q4
        = MIntPoly::from_dict({x, y, z}, {{{1, 4, 1}, -2_z},
                                          {{4, 2, 1}, -6_z},
                                          {{0, 5, 1}, -8_z},

                                          {{1, 2, 2}, 1_z},
                                          {{4, 0, 2}, 3_z},
                                          {{0, 3, 2}, 4_z},

                                          {{1, 3, 0}, 3_z},
                                          {{4, 1, 0}, 9_z},
                                          {{0, 4, 0}, 12_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing derivative of MIntPoly", "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p = MIntPoly::from_dict({x, y}, {{{2, 1}, 3_z},
                                                         {{1, 2}, 2_z},
                                                         {{2, 0}, 3_z},
                                                         {{0, 2}, 2_z},
                                                         {{1, 0}, 3_z},
                                                         {{0, 1}, 2_z},
                                                         {{0, 0}, 5_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict(
        {x, y}, {{{1, 1}, 6_z}, {{0, 2}, 2_z}, {{1, 0}, 6_z}, {{0, 0}, 3_z}});
    RCP<const MIntPoly> q2 = MIntPoly::from_dict(
        {x, y}, {{{2, 0}, 3_z}, {{1, 1}, 4_z}, {{0, 1}, 4_z}, {{0, 0}, 2_z}});
    RCP<const MIntPoly> q3 = MIntPoly::from_dict({x, y}, {{{0, 0}, 0_z}});
    REQUIRE(eq(*p->diff(x), *q1));
    REQUIRE(eq(*p->diff(y), *q2));
    REQUIRE(eq(*p->diff(z), *q3));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MIntPolys with a MIntPoly whose "
          "variable are in the variable set",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 3_z},
                                                             {{2, 0, 0}, 2_z},
                                                             {{1, 0, 0}, 1_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict({x}, {{{1}, 1_z}, {{2}, 1_z}});
    RCP<const MIntPoly> p3 = MIntPoly::from_dict({y}, {{{1}, 1_z}, {{2}, 1_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 3_z},
                                                             {{2, 0, 0}, 3_z},
                                                             {{1, 0, 0}, 2_z}});
    RCP<const MIntPoly> q2 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 3_z},
                                                             {{2, 0, 0}, 1_z}});
    RCP<const MIntPoly> q3
        = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, -1_z},
                                          {{3, 2, 1}, 2_z},
                                          {{0, 1, 2}, -1_z},
                                          {{0, 0, 0}, -3_z},
                                          {{2, 0, 0}, -1_z}});
    RCP<const MIntPoly> q4 = MIntPoly::from_dict({x, y, z}, {{{2, 2, 3}, 1_z},
                                                             {{4, 2, 1}, -2_z},
                                                             {{1, 1, 2}, 1_z},
                                                             {{1, 0, 0}, 3_z},
                                                             {{3, 0, 0}, 3_z},

                                                             {{3, 2, 3}, 1_z},
                                                             {{5, 2, 1}, -2_z},
                                                             {{2, 1, 2}, 1_z},
                                                             {{2, 0, 0}, 4_z},
                                                             {{4, 0, 0}, 2_z}});
    RCP<const MIntPoly> q5 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 3}, 1_z},
                                                             {{3, 2, 1}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 0}, 3_z},
                                                             {{2, 0, 0}, 2_z},
                                                             {{1, 0, 0}, 1_z},
                                                             {{0, 1, 0}, 1_z},
                                                             {{0, 2, 0}, 1_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
    REQUIRE(eq(*add_mpoly(*p1, *p3), *q5));
}

TEST_CASE("Testing addition, subtraction, multiplication of "
          "MIntPolys with a MIntPoly whose "
          "variables are not in the variable set",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const MIntPoly> p1 = MIntPoly::from_dict(
        {x, y}, {{{1, 2}, 1_z}, {{2, 1}, -2_z}, {{0, 1}, 1_z}, {{0, 0}, 3_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict({z}, {{{1}, 1_z}, {{2}, 1_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 0}, 1_z},
                                                             {{2, 1, 0}, -2_z},
                                                             {{0, 1, 0}, 1_z},
                                                             {{0, 0, 0}, 3_z},
                                                             {{0, 0, 1}, 1_z},
                                                             {{0, 0, 2}, 1_z}});
    RCP<const MIntPoly> q2
        = MIntPoly::from_dict({x, y, z}, {{{1, 2, 0}, 1_z},
                                          {{2, 1, 0}, -2_z},
                                          {{0, 1, 0}, 1_z},
                                          {{0, 0, 0}, 3_z},
                                          {{0, 0, 1}, -1_z},
                                          {{0, 0, 2}, -1_z}});
    RCP<const MIntPoly> q3 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 0}, -1_z},
                                                             {{2, 1, 0}, 2_z},
                                                             {{0, 1, 0}, -1_z},
                                                             {{0, 0, 0}, -3_z},
                                                             {{0, 0, 1}, 1_z},
                                                             {{0, 0, 2}, 1_z}});
    RCP<const MIntPoly> q4 = MIntPoly::from_dict({x, y, z}, {{{1, 2, 1}, 1_z},
                                                             {{2, 1, 1}, -2_z},
                                                             {{0, 1, 1}, 1_z},
                                                             {{0, 0, 1}, 3_z},

                                                             {{1, 2, 2}, 1_z},
                                                             {{2, 1, 2}, -2_z},
                                                             {{0, 1, 2}, 1_z},
                                                             {{0, 0, 2}, 3_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of two "
          "MIntPolys with different variables",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const MIntPoly> p1
        = MIntPoly::from_dict({x}, {{{1}, -1_z}, {{2}, 3_z}, {{0}, 0_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict({y}, {{{0}, 1_z}, {{1}, 1_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict(
        {x, y}, {{{1, 0}, -1_z}, {{2, 0}, 3_z}, {{0, 0}, 1_z}, {{0, 1}, 1_z}});
    RCP<const MIntPoly> q2 = MIntPoly::from_dict(
        {x, y},
        {{{1, 0}, -1_z}, {{2, 0}, 3_z}, {{0, 0}, -1_z}, {{0, 1}, -1_z}});
    RCP<const MIntPoly> q3 = MIntPoly::from_dict(
        {x, y}, {{{1, 0}, 1_z}, {{2, 0}, -3_z}, {{0, 0}, 1_z}, {{0, 1}, 1_z}});
    RCP<const MIntPoly> q4 = MIntPoly::from_dict(
        {x, y}, {{{2, 1}, 3_z}, {{2, 0}, 3_z}, {{1, 1}, -1_z}, {{1, 0}, -1_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing addition, subtraction, multiplication of two "
          "MIntPolys with the same variables",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const MIntPoly> p1
        = MIntPoly::from_dict({x}, {{{1}, -1_z}, {{2}, 3_z}, {{0}, 0_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict({x}, {{{0}, 1_z}, {{1}, 1_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict({x}, {{{0}, 1_z}, {{2}, 3_z}});
    RCP<const MIntPoly> q2
        = MIntPoly::from_dict({x}, {{{0}, -1_z}, {{1}, -2_z}, {{2}, 3_z}});
    RCP<const MIntPoly> q3
        = MIntPoly::from_dict({x}, {{{0}, 1_z}, {{1}, 2_z}, {{2}, -3_z}});
    RCP<const MIntPoly> q4
        = MIntPoly::from_dict({x}, {{{1}, -1_z}, {{2}, 2_z}, {{3}, 3_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
}

TEST_CASE("Testing addition, subtraction, and multiplication of "
          "MIntPolys with empty variable sets",
          "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    vec_basic s;
    vec_uint v;
    RCP<const MIntPoly> p1 = MIntPoly::from_dict(s, {{v, 2_z}});
    RCP<const MIntPoly> p2 = MIntPoly::from_dict(
        {x, y}, {{{0, 0}, 5_z}, {{0, 1}, 1_z}, {{1, 0}, 1_z}});

    RCP<const MIntPoly> q1 = MIntPoly::from_dict(
        {x, y}, {{{0, 0}, 7_z}, {{0, 1}, 1_z}, {{1, 0}, 1_z}});
    RCP<const MIntPoly> q2 = MIntPoly::from_dict(
        {x, y}, {{{0, 0}, -3_z}, {{0, 1}, -1_z}, {{1, 0}, -1_z}});
    RCP<const MIntPoly> q3 = MIntPoly::from_dict(
        {x, y}, {{{0, 0}, 3_z}, {{0, 1}, 1_z}, {{1, 0}, 1_z}});
    RCP<const MIntPoly> q4 = MIntPoly::from_dict(
        {x, y}, {{{0, 0}, 10_z}, {{0, 1}, 2_z}, {{1, 0}, 2_z}});

    REQUIRE(eq(*add_mpoly(*p1, *p2), *q1));
    REQUIRE(eq(*add_mpoly(*p2, *p1), *q1));
    REQUIRE(eq(*sub_mpoly(*p1, *p2), *q2));
    REQUIRE(eq(*sub_mpoly(*p2, *p1), *q3));
    REQUIRE(eq(*mul_mpoly(*p1, *p2), *q4));
    REQUIRE(eq(*mul_mpoly(*p2, *p1), *q4));
}

TEST_CASE("Testing Precedence of MIntPoly", "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> a = symbol("a");
    Precedence Prec;
    RCP<const MIntPoly> p1 = MIntPoly::from_dict({x, y}, {{{0, 0}, 0_z}});
    RCP<const MIntPoly> p2
        = MIntPoly::from_dict({x, y}, {{{1, 0}, 2_z}, {{0, 0}, 1_z}});
    RCP<const MIntPoly> p3 = MIntPoly::from_dict({x, y}, {{{0, 0}, 5_z}});
    RCP<const MIntPoly> p4 = MIntPoly::from_dict({x, y}, {{{1, 0}, 1_z}});
    RCP<const MIntPoly> p5 = MIntPoly::from_dict({x, y}, {{{1, 1}, 4_z}});
    RCP<const MIntPoly> p6 = MIntPoly::from_dict({x, y}, {{{2, 0}, 1_z}});
    REQUIRE(Prec.getPrecedence(p1) == PrecedenceEnum::Atom);
    REQUIRE(Prec.getPrecedence(p2) == PrecedenceEnum::Add);
    REQUIRE(Prec.getPrecedence(p3) == PrecedenceEnum::Atom);
    REQUIRE(Prec.getPrecedence(p4) == PrecedenceEnum::Atom);
    REQUIRE(Prec.getPrecedence(p5) == PrecedenceEnum::Mul);
    REQUIRE(Prec.getPrecedence(p6) == PrecedenceEnum::Pow);
}

TEST_CASE("MIntPoly from_poly", "[MIntPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const MIntPoly> mpoly
        = MIntPoly::from_dict({x}, {{{1}, -1_z}, {{2}, 3_z}, {{0}, 0_z}});
    RCP<const UIntPoly> upoly = UIntPoly::from_vec(x, {0_z, -1_z, 3_z});

    REQUIRE(eq(*MIntPoly::from_poly(*upoly), *mpoly));
}
/*
TEST_CASE("Testing equality of MultivariateExprPolynomials with Expressions",
          "[MultivariateExprPolynomial],[Expression]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Integer> two = make_rcp<const Integer>(integer_class(2));
    Expression expr1(mul(a, c));
    MultivariateExprPolynomial p1(
        MultivariatePolynomial::from_dict({x, y}, {{{0, 0}, Expression(0)}}));
    MultivariateExprPolynomial p2(
        MultivariatePolynomial::from_dict({x, y}, {{{0, 0}, expr1}}));
    REQUIRE(p1 == 0);
    REQUIRE(p2 == expr1);
}
*/
