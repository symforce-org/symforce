#include "catch.hpp"
#include <chrono>
#include <sstream>

#include <symengine/polys/uexprpoly.h>
#include <symengine/symengine_exception.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Expression;
using SymEngine::integer;
using SymEngine::make_rcp;
using SymEngine::map_int_Expr;
using SymEngine::one;
using SymEngine::Pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::RCP;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::uexpr_poly;
using SymEngine::UExprDict;
using SymEngine::UExprPoly;
using SymEngine::zero;

using namespace SymEngine::literals;

TEST_CASE("Constructor of UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> none = symbol("");
    Expression a(symbol("a"));
    Expression b(symbol("b"));
    Expression c(symbol("c"));
    Expression d(symbol("d"));
    Expression num2(integer(2));
    Expression num1(integer(1));

    RCP<const UExprPoly> P = uexpr_poly(x, {{0, num1}, {1, num2}, {2, num1}});
    REQUIRE(P->__str__() == "x**2 + 2*x + 1");

    RCP<const UExprPoly> Q = UExprPoly::from_vec(x, {1, 0, 2, 1});
    REQUIRE(Q->__str__() == "x**3 + 2*x**2 + 1");

    RCP<const UExprPoly> R = uexpr_poly(x, {{0, d}, {1, c}, {2, b}, {3, a}});
    REQUIRE(R->__str__() == "a*x**3 + b*x**2 + c*x + d");

    RCP<const UExprPoly> S = UExprPoly::from_vec(x, {1, 0, 2, 1});
    REQUIRE(S->__str__() == "x**3 + 2*x**2 + 1");

    R = uexpr_poly(x, {{-1, d}});
    REQUIRE(R->__str__() == "d*x**(-1)");
    REQUIRE(not(R->__str__() == "d*x**-1"));

    R = uexpr_poly(x, {{-2, d}, {-1, c}, {0, b}, {1, a}});
    REQUIRE(R->__str__() == "a*x + b + c*x**(-1) + d*x**(-2)");

    RCP<const UExprPoly> T = uexpr_poly(none, map_int_Expr{});
    REQUIRE(T->__str__() == "0");

    RCP<const UExprPoly> U = uexpr_poly(x, {{0, c}, {1, 0_z}, {2, d}});
    REQUIRE(U->__str__() == "d*x**2 + c");
}

TEST_CASE("Adding two UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    UExprDict adict_({{0, 1}, {1, 2}, {2, Expression("a")}});
    UExprDict bdict_({{0, 2}, {1, 3}, {2, Expression("b")}});
    const UExprPoly a(x, std::move(adict_));
    const UExprPoly b(x, std::move(bdict_));

    RCP<const Basic> c = add_upoly(a, b);
    REQUIRE(c->__str__() == "(a + b)*x**2 + 5*x + 3");

    RCP<const UExprPoly> d = uexpr_poly(x, {{0, Expression(2)}});
    REQUIRE(add_upoly(a, *d)->__str__() == "a*x**2 + 2*x + 3");
    REQUIRE(add_upoly(*d, a)->__str__() == "a*x**2 + 2*x + 3");

    d = uexpr_poly(y, {{0, 2}, {1, 4}});
    CHECK_THROWS_AS(add_upoly(a, *d), SymEngineException);
}

TEST_CASE("Negative of a UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    UExprDict adict_({{0, 1}, {1, Expression("a")}, {2, Expression("c")}});
    const UExprPoly a(x, std::move(adict_));
    {
        auto check_map_str
            = [](std::string to_chk, std::vector<std::string> key,
                 std::vector<std::string> val) {
                  if (key.size() != val.size())
                      return false;
                  for (unsigned i = 0; i < key.size(); i++) {
                      if (to_chk.find(key[i] + std::string(": " + val[i]))
                          == std::string::npos)
                          return false;
                  }
                  return true;
              };
        std::stringstream buffer;
        buffer << adict_;
        bool buffer_check
            = check_map_str(buffer.str(), {"0", "1", "2"}, {"1", "a", "c"});
        REQUIRE(buffer_check);
    }

    RCP<const UExprPoly> b = neg_upoly(a);
    REQUIRE(b->__str__() == "-c*x**2 - a*x - 1");

    RCP<const UExprPoly> c = uexpr_poly(x, map_int_Expr{});
    REQUIRE(neg_upoly(*c)->__str__() == "0");

    c = uexpr_poly(x, {{0, Expression(2)}});
    REQUIRE(neg_upoly(*c)->__str__() == "-2");
}

TEST_CASE("Subtracting two UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    UExprDict adict_({{0, 1}, {1, 2}, {2, 1}});
    UExprDict bdict_({{0, 2}, {1, Expression("b")}, {2, Expression("a")}});
    const UExprPoly a(x, std::move(adict_));
    const UExprPoly b(x, std::move(bdict_));

    RCP<const Basic> c = sub_upoly(b, a);
    REQUIRE(c->__str__() == "(-1 + a)*x**2 + (-2 + b)*x + 1");

    RCP<const UExprPoly> d = uexpr_poly(x, {{0, Expression(2)}});
    REQUIRE(sub_upoly(a, *d)->__str__() == "x**2 + 2*x - 1");
    REQUIRE(sub_upoly(*d, a)->__str__() == "-x**2 - 2*x + 1");

    d = uexpr_poly(y, {{0, 2}, {1, 4}});
    CHECK_THROWS_AS(sub_upoly(a, *d), SymEngineException);
}

TEST_CASE("Multiplication of two UExprPoly", "[UExprPoly]")
{

    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> none = symbol("");
    RCP<const UExprPoly> a
        = uexpr_poly(x, {{0, 1}, {1, Expression("b")}, {2, Expression("a")}});
    RCP<const UExprPoly> b
        = uexpr_poly(x, {{0, -1}, {1, -2}, {2, mul(integer(-1), symbol("a"))}});

    RCP<const UExprPoly> c = mul_upoly(*a, *a);
    RCP<const UExprPoly> d = mul_upoly(*a, *b);

    REQUIRE(c->__str__()
            == "a**2*x**4 + 2*a*b*x**3 + (2*a + b**2)*x**2 + 2*b*x + 1");
    REQUIRE(d->__str__()
            == "-a**2*x**4 + (-2*a - a*b)*x**3 + (-2*a - "
               "2*b)*x**2 + (-2 - b)*x - 1");

    RCP<const UExprPoly> f = uexpr_poly(x, {{0, Expression(2)}});
    REQUIRE(mul_upoly(*a, *f)->__str__() == "2*a*x**2 + 2*b*x + 2");
    REQUIRE(mul_upoly(*f, *a)->__str__() == "2*a*x**2 + 2*b*x + 2");

    f = uexpr_poly(y, {{0, 2}, {1, 4}});
    CHECK_THROWS_AS(mul_upoly(*a, *f), SymEngineException);

    f = uexpr_poly(x, map_int_Expr{});
    REQUIRE(mul_upoly(*a, *f)->__str__() == "0");

    a = uexpr_poly(x, {{-2, 5}, {-1, 3}, {0, 1}, {1, 2}});

    c = mul_upoly(*a, *b);
    REQUIRE(c->__str__()
            == "-2*a*x**3 + (-4 - a)*x**2 + (-4 - 3*a)*x + (-7 - "
               "5*a) - 13*x**(-1) - 5*x**(-2)");
}

TEST_CASE("Comparing two UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const UExprPoly> P = uexpr_poly(x, {{0, 1}, {1, 2}});
    RCP<const UExprPoly> Q
        = uexpr_poly(x, {{0, 1}, {1, Expression("b")}, {2, 1}});

    REQUIRE(P->compare(*Q) == -1);

    P = uexpr_poly(x, {{0, 1}, {1, Expression("k")}, {2, 3}, {3, 2}});
    REQUIRE(P->compare(*Q) == 1);

    P = uexpr_poly(x, {{0, 1}, {1, 2}, {3, 3}});
    REQUIRE(P->compare(*Q) == -1);

    P = uexpr_poly(y, {{0, 1}, {1, 2}, {3, 3}});
    REQUIRE(P->compare(*Q) == 1);

    P = uexpr_poly(x, {{0, 1}, {1, Expression("b")}, {2, 1}});
    REQUIRE(P->compare(*Q) == 0);
    REQUIRE(P->__eq__(*Q) == true);

    P = uexpr_poly(x, {{0, 1}, {1, Expression("a")}, {2, 1}});
    REQUIRE(P->compare(*Q) == -1);
    REQUIRE(P->__eq__(*Q) == false);
}

TEST_CASE("UExprPoly get_args", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UExprPoly> a = uexpr_poly(x, {{0, 1}, {1, 2}, {2, 1}});

    REQUIRE(eq(*a->as_symbolic(),
               *add({one, mul(integer(2), x), pow(x, integer(2))})));
    REQUIRE(not eq(*a->as_symbolic(),
                   *add({one, mul(integer(3), x), pow(x, integer(2))})));

    a = uexpr_poly(x, {{0, 1}, {1, 1}, {2, 2}});
    REQUIRE(eq(*a->as_symbolic(),
               *add({one, x, mul(integer(2), pow(x, integer(2)))})));

    a = uexpr_poly(x, map_int_Expr{});
    REQUIRE(eq(*a->as_symbolic(), *add({zero})));
}

TEST_CASE("UExprPoly max_coef", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UExprPoly> a = uexpr_poly(x, {{0, 1}, {1, 2}, {2, 4}});
    RCP<const UExprPoly> b
        = uexpr_poly(x, {{0, 2}, {1, 2}, {2, Expression("b")}});

    Expression c(symbol("a"));
    Expression d(symbol("b"));

    REQUIRE(a->max_coef() == 4);
    REQUIRE(not(a->max_coef() == 2));
    REQUIRE(b->max_coef() == d);
    REQUIRE(not(b->max_coef() == c));
    REQUIRE(not(b->max_coef() == 2));
}

TEST_CASE("Evaluation of UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UExprPoly> a
        = uexpr_poly(x, {{0, 1}, {1, 2}, {2, Expression("a")}});

    REQUIRE(a->eval(2).get_basic()->__str__() == "5 + 4*a");

    a = uexpr_poly(x, {{-2, 5}, {-1, 3}, {0, 1}, {1, 2}});
    REQUIRE(a->eval(2).get_basic()->__str__() == "31/4");
}

TEST_CASE("Convert UExprPoly to Symbolic", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UExprPoly> a
        = uexpr_poly(x, {{0, 1}, {1, 2}, {2, Expression("a")}});

    REQUIRE(eq(*a->get_poly().get_basic("x"),
               *add(integer(1),
                    add(mul(integer(2), x), mul(symbol("a"), mul(x, x))))));
}

TEST_CASE("Derivative of UExprPoly", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> none = symbol("");
    RCP<const UExprPoly> a
        = uexpr_poly(x, {{0, 1}, {1, 2}, {2, Expression("a")}});
    RCP<const UExprPoly> b = uexpr_poly(x, {{0, Expression(1)}});
    RCP<const UExprPoly> c = uexpr_poly(none, {{0, Expression(5)}});

    REQUIRE(a->diff(x)->__str__() == "2*a*x + 2");
    REQUIRE(a->diff(y)->__str__() == "0");
    REQUIRE(b->diff(y)->__str__() == "0");

    a = uexpr_poly(x, {{-2, 5}, {-1, 3}, {0, 1}, {1, 2}, {2, Expression("a")}});
    REQUIRE(a->diff(x)->__str__() == "2*a*x + 2 - 3*x**(-2) - 10*x**(-3)");

    REQUIRE(c->diff(x)->__str__() == "0");

    c = uexpr_poly(none, map_int_Expr{});
    REQUIRE(c->diff(x)->__str__() == "0");
}

TEST_CASE("Bool checks specific UExprPoly cases", "[UExprPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UExprPoly> z = uexpr_poly(x, {{0, Expression(0)}});
    RCP<const UExprPoly> o = uexpr_poly(x, {{0, Expression(1)}});
    RCP<const UExprPoly> mo = uexpr_poly(x, {{0, Expression(-1)}});
    RCP<const UExprPoly> i = uexpr_poly(x, {{0, Expression(6)}});
    RCP<const UExprPoly> s = uexpr_poly(x, {{1, Expression(1)}});
    RCP<const UExprPoly> m1 = uexpr_poly(x, {{1, Expression(6)}});
    RCP<const UExprPoly> m2 = uexpr_poly(x, {{3, Expression(5)}});
    RCP<const UExprPoly> po = uexpr_poly(x, {{5, Expression(1)}});
    RCP<const UExprPoly> poly = uexpr_poly(x, {{0, 1}, {1, 2}, {2, 1}});
    RCP<const UExprPoly> neg
        = uexpr_poly(x, {{-2, 5}, {-1, 3}, {0, 1}, {1, 2}});

    REQUIRE((z->is_zero() and not z->is_one() and not z->is_minus_one()
             and z->is_integer() and not z->is_symbol() and not z->is_mul()
             and not z->is_pow()));
    REQUIRE((not o->is_zero() and o->is_one() and not o->is_minus_one()
             and o->is_integer() and not o->is_symbol() and not o->is_mul()
             and not o->is_pow()));
    REQUIRE((not mo->is_zero() and not mo->is_one() and mo->is_minus_one()
             and mo->is_integer() and not mo->is_symbol() and not mo->is_mul()
             and not mo->is_pow()));
    REQUIRE((not i->is_zero() and not i->is_one() and not i->is_minus_one()
             and i->is_integer() and not i->is_symbol() and not i->is_mul()
             and not i->is_pow()));
    REQUIRE((not s->is_zero() and not s->is_one() and not s->is_minus_one()
             and not s->is_integer() and s->is_symbol() and not s->is_mul()
             and not s->is_pow()));
    REQUIRE((not m1->is_zero() and not m1->is_one() and not m1->is_minus_one()
             and not m1->is_integer() and not m1->is_symbol() and m1->is_mul()
             and not m1->is_pow()));
    REQUIRE((not m2->is_zero() and not m2->is_one() and not m2->is_minus_one()
             and not m2->is_integer() and not m2->is_symbol() and m2->is_mul()
             and not m2->is_pow()));
    REQUIRE((not po->is_zero() and not po->is_one() and not po->is_minus_one()
             and not po->is_integer() and not po->is_symbol()
             and not po->is_mul() and po->is_pow()));
    REQUIRE((not poly->is_zero() and not poly->is_one()
             and not poly->is_minus_one() and not poly->is_integer()
             and not poly->is_symbol() and not poly->is_mul()
             and not poly->is_pow()));
    REQUIRE((not neg->is_zero() and not neg->is_one()
             and not neg->is_minus_one() and not neg->is_integer()
             and not neg->is_symbol() and not neg->is_mul()
             and not neg->is_pow()));
}

TEST_CASE("UExprPoly expand", "[UExprPoly][expand]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UExprPoly> a
        = uexpr_poly(x, {{1, 1}, {2, 1}, {3, Expression("a")}});
    RCP<const Basic> b = make_rcp<const Pow>(a, integer(3));
    RCP<const Basic> c = expand(b);

    REQUIRE(b->__str__() == "(a*x**3 + x**2 + x)**3");
    REQUIRE(c->__str__()
            == "a**3*x**9 + 3*a**2*x**8 + (2*a + a*(1 + 2*a) + "
               "a**2)*x**7 + (1 + 6*a)*x**6 + (3 + 3*a)*x**5 + "
               "3*x**4 + x**3");
}
