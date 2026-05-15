#include "catch.hpp"
#include <chrono>

#include <symengine/polys/uintpoly_flint.h>
#include <symengine/polys/uintpoly_piranha.h>
#include <symengine/polys/uratpoly.h>
#include <symengine/polys/uintpoly.h>
#include <symengine/pow.h>
#include <symengine/symengine_exception.h>

#ifdef HAVE_SYMENGINE_PIRANHA
using SymEngine::URatPolyPiranha;
#endif
using SymEngine::add;
using SymEngine::Basic;
using SymEngine::integer;
using SymEngine::make_rcp;
using SymEngine::map_uint_mpq;
using SymEngine::one;
using SymEngine::Pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::rational_class;
using SymEngine::RCP;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::UIntPoly;
using SymEngine::URatPoly;
using SymEngine::URatPolyFlint;
using SymEngine::zero;

using namespace SymEngine::literals;
using rc = rational_class;

TEST_CASE("Constructor of URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");

    RCP<const URatPolyFlint> P
        = URatPolyFlint::from_dict(x, {{0, rc(1_z, 2_z)}, {2, rc(3_z, 2_z)}});
    REQUIRE(P->__str__() == "3/2*x**2 + 1/2");

    RCP<const URatPolyFlint> Q
        = URatPolyFlint::from_vec(x, {0_q, rc(1_z, 2_z), rc(1_z, 2_z)});
    REQUIRE(Q->__str__() == "1/2*x**2 + 1/2*x");

    RCP<const URatPolyFlint> S = URatPolyFlint::from_dict(x, {{0, 2_q}});
    REQUIRE(S->__str__() == "2");

    RCP<const URatPolyFlint> T = URatPolyFlint::from_dict(x, map_uint_mpq{});
    REQUIRE(T->__str__() == "0");
}

TEST_CASE("Adding two URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const URatPolyFlint> a = URatPolyFlint::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 3_z)}, {2, 1_q}});
    RCP<const URatPolyFlint> b
        = URatPolyFlint::from_dict(x, {{0, rc(2_z, 3_z)}, {1, 3_q}, {2, 2_q}});

    RCP<const Basic> c = add_upoly(*a, *b);
    REQUIRE(c->__str__() == "3*x**2 + 11/3*x + 7/6");

    RCP<const URatPolyFlint> d
        = URatPolyFlint::from_dict(x, {{0, rc(1_z, 2_z)}});
    RCP<const Basic> e = add_upoly(*a, *d);
    RCP<const Basic> f = add_upoly(*d, *a);
    REQUIRE(e->__str__() == "x**2 + 2/3*x + 1");
    REQUIRE(f->__str__() == "x**2 + 2/3*x + 1");

    RCP<const URatPolyFlint> g = URatPolyFlint::from_dict(
        y, {{0, 2_q}, {1, rc(-3_z, 2_z)}, {2, rc(1_z, 4_z)}});
    CHECK_THROWS_AS(add_upoly(*a, *g), SymEngineException);
}

TEST_CASE("Negative of a URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a
        = URatPolyFlint::from_dict(x, {{0, rc(-1_z, 2_z)}, {1, 2_q}, {2, 3_q}});

    RCP<const URatPolyFlint> b = neg_upoly(*a);
    REQUIRE(b->__str__() == "-3*x**2 - 2*x + 1/2");
}

TEST_CASE("Subtracting two URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const URatPolyFlint> a = URatPolyFlint::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 3_z)}, {2, 1_q}});
    RCP<const URatPolyFlint> b
        = URatPolyFlint::from_dict(x, {{0, rc(2_z, 3_z)}, {1, 3_q}, {2, 2_q}});
    RCP<const URatPolyFlint> c = URatPolyFlint::from_dict(
        x, {{0, 2_q}, {1, rc(-3_z, 2_z)}, {2, rc(1_z, 4_z)}});
    RCP<const URatPolyFlint> f = URatPolyFlint::from_dict(y, {{0, 2_q}});

    RCP<const Basic> d = sub_upoly(*b, *a);
    REQUIRE(d->__str__() == "x**2 + 7/3*x + 1/6");
    d = sub_upoly(*c, *a);
    REQUIRE(d->__str__() == "-3/4*x**2 - 13/6*x + 3/2");
    d = sub_upoly(*a, *c);
    REQUIRE(d->__str__() == "3/4*x**2 + 13/6*x - 3/2");
    CHECK_THROWS_AS(sub_upoly(*a, *f), SymEngineException);
}

TEST_CASE("Multiplication of two URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const URatPolyFlint> a = URatPolyFlint::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 3_z)}, {2, 1_q}});
    RCP<const URatPolyFlint> b
        = URatPolyFlint::from_dict(x, {{0, rc(2_z, 3_z)}, {1, 3_q}, {2, 2_q}});
    RCP<const URatPolyFlint> e = URatPolyFlint::from_dict(
        x, {{0, 2_q}, {1, rc(-3_z, 2_z)}, {2, rc(1_z, 4_z)}});
    RCP<const URatPolyFlint> f
        = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, rc(1_z, 2_z)}});

    RCP<const URatPolyFlint> c = mul_upoly(*a, *a);
    RCP<const URatPolyFlint> d = mul_upoly(*a, *b);
    RCP<const URatPolyFlint> g = mul_upoly(*e, *e);
    RCP<const URatPolyFlint> h = mul_upoly(*e, *f);
    RCP<const URatPolyFlint> i = mul_upoly(*f, *f);

    REQUIRE(c->__str__() == "x**4 + 4/3*x**3 + 13/9*x**2 + 2/3*x + 1/4");
    REQUIRE(d->__str__() == "2*x**4 + 13/3*x**3 + 11/3*x**2 + 35/18*x + 1/3");
    REQUIRE(g->__str__() == "1/16*x**4 - 3/4*x**3 + 13/4*x**2 - 6*x + 4");
    REQUIRE(h->__str__() == "1/8*x**3 - 1/2*x**2 - 1/2*x + 2");
    REQUIRE(i->__str__() == "1/4*x**2 + x + 1");

    c = URatPolyFlint::from_dict(x, {{0, rc(-1_z)}});
    REQUIRE(mul_upoly(*a, *c)->__str__() == "-x**2 - 2/3*x - 1/2");
    REQUIRE(mul_upoly(*c, *a)->__str__() == "-x**2 - 2/3*x - 1/2");

    c = URatPolyFlint::from_dict(y, {{0, rc(-1_z)}});
    CHECK_THROWS_AS(mul_upoly(*a, *c), SymEngineException);
}

TEST_CASE("Comparing two URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const URatPolyFlint> P
        = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, rc(2_z, 3_z)}});
    RCP<const URatPolyFlint> Q
        = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, 2_q}, {2, 1_q}});

    REQUIRE(P->compare(*Q) == -1);

    P = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, 2_q}, {2, 3_q}, {3, 2_q}});
    REQUIRE(P->compare(*Q) == 1);

    P = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, 2_q}, {2, 1_q}});
    REQUIRE(P->compare(*Q) == 0);

    P = URatPolyFlint::from_dict(y, {{0, 1_q}, {1, rc(2_z, 3_z)}});
    REQUIRE(P->compare(*Q) == -1);
}

TEST_CASE("URatPolyFlint as_symbolic", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a
        = URatPolyFlint::from_dict(x, {{0, rc(1_z, 2_z)}, {1, 2_q}, {2, 1_q}});

    REQUIRE(eq(
        *a->as_symbolic(),
        *add({div(one, integer(2)), mul(integer(2), x), pow(x, integer(2))})));

    RCP<const URatPolyFlint> b
        = URatPolyFlint::from_dict(x, {{1, rc(3_z, 2_z)}, {2, 2_q}});
    REQUIRE(eq(*b->as_symbolic(), *add(mul(x, div(integer(3), integer(2))),
                                       mul(integer(2), pow(x, integer(2))))));

    RCP<const URatPolyFlint> c = URatPolyFlint::from_dict(x, map_uint_mpq{});
    REQUIRE(eq(*c->as_symbolic(), *zero));
}

TEST_CASE("Evaluation of URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a
        = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, rc(2_z, 3_z)}});
    RCP<const URatPolyFlint> b = URatPolyFlint::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 5_z)}, {2, 1_q}});

    REQUIRE(a->eval(2_q) == rc(7_z, 3_z));
    REQUIRE(a->eval(10_q) == rc(23_z, 3_z));
    REQUIRE(b->eval(-1_q) == rc(11_z, 10_z));
    REQUIRE(b->eval(0_q) == rc(1_z, 2_z));

    std::vector<rational_class> resa = {rc(7_z, 3_z), rc(5_z, 3_z), 1_q};
    std::vector<rational_class> resb
        = {rc(53_z, 10_z), rc(19_z, 10_z), rc(1_z, 2_z)};
    REQUIRE(a->multieval({2_q, 1_q, 0_q}) == resa);
    REQUIRE(b->multieval({2_q, 1_q, 0_q}) == resb);
}

TEST_CASE("Derivative of URatPolyFlint", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const URatPolyFlint> a = URatPolyFlint::from_dict(
        x, {{0, 1_q}, {1, rc(2_z, 3_z)}, {2, rc(1_z, 2_z)}});
    RCP<const URatPolyFlint> b
        = URatPolyFlint::from_dict(y, {{2, rc(4_z, 3_z)}});

    REQUIRE(a->diff(x)->__str__() == "x + 2/3");
    REQUIRE(a->diff(y)->__str__() == "0");
    REQUIRE(b->diff(y)->__str__() == "8/3*y");
}

TEST_CASE("URatPolyFlint pow", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a
        = URatPolyFlint::from_dict(x, {{0, rc(1_z, 2_z)}, {1, 1_q}});
    RCP<const URatPolyFlint> b
        = URatPolyFlint::from_dict(x, {{0, 3_q}, {2, rc(3_z, 2_z)}});

    RCP<const URatPolyFlint> aaa = pow_upoly(*a, 3);
    RCP<const URatPolyFlint> bb = pow_upoly(*b, 2);

    REQUIRE(aaa->__str__() == "x**3 + 3/2*x**2 + 3/4*x + 1/8");
    REQUIRE(bb->__str__() == "9/4*x**4 + 9*x**2 + 9");
}

TEST_CASE("URatPolyFlint divides", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a
        = URatPolyFlint::from_dict(x, {{0, 1_q}, {1, 1_q}});
    RCP<const URatPolyFlint> b = URatPolyFlint::from_dict(x, {{0, 4_q}});
    RCP<const URatPolyFlint> c
        = URatPolyFlint::from_dict(x, {{0, 8_q}, {1, 8_q}});
    RCP<const URatPolyFlint> res;

    REQUIRE(divides_upoly(*a, *c, outArg(res)));
    REQUIRE(res->__str__() == "8");
    REQUIRE(divides_upoly(*b, *c, outArg(res)));
    REQUIRE(res->__str__() == "2*x + 2");
    REQUIRE(divides_upoly(*b, *a, outArg(res)));
    REQUIRE(res->__str__() == "1/4*x + 1/4");
    REQUIRE(!divides_upoly(*a, *b, outArg(res)));
}

TEST_CASE("URatPolyFlint gcd", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a = URatPolyFlint::from_dict(x, {{2, 2_q}});
    RCP<const URatPolyFlint> b = URatPolyFlint::from_dict(x, {{1, 3_q}});
    RCP<const URatPolyFlint> c
        = URatPolyFlint::from_dict(x, {{0, 6_q}, {1, 8_q}, {2, 2_q}});
    RCP<const URatPolyFlint> d
        = URatPolyFlint::from_dict(x, {{1, 4_q}, {2, 4_q}});

    RCP<const URatPolyFlint> ab = gcd_upoly(*a, *b);
    RCP<const URatPolyFlint> cd = gcd_upoly(*c, *d);
    RCP<const URatPolyFlint> ad = gcd_upoly(*a, *d);
    RCP<const URatPolyFlint> bc = gcd_upoly(*b, *c);

    REQUIRE(ab->__str__() == "x");
    REQUIRE(cd->__str__() == "x + 1");
    REQUIRE(ad->__str__() == "x");
    REQUIRE(bc->__str__() == "1");

    // https://github.com/wbhart/flint2/issues/276
}

TEST_CASE("URatPolyFlint lcm", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a = URatPolyFlint::from_dict(x, {{2, 6_q}});
    RCP<const URatPolyFlint> b = URatPolyFlint::from_dict(x, {{1, 8_q}});
    RCP<const URatPolyFlint> c
        = URatPolyFlint::from_dict(x, {{0, 8_q}, {1, 8_q}});

    RCP<const URatPolyFlint> ab = lcm_upoly(*a, *b);
    RCP<const URatPolyFlint> bc = lcm_upoly(*b, *c);
    RCP<const URatPolyFlint> ac = lcm_upoly(*a, *c);

    REQUIRE(ab->__str__() == "x**2");
    REQUIRE(bc->__str__() == "x**2 + x");
    REQUIRE(ac->__str__() == "x**3 + x**2");
}

TEST_CASE("URatPolyFlint from_poly symengine", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UIntPoly> a = UIntPoly::from_dict(x, {{0, 1_z}, {2, 1_z}});
    RCP<const URatPolyFlint> b = URatPolyFlint::from_poly(*a);

    RCP<const URatPoly> c
        = URatPoly::from_dict(x, {{0, rc(1_z, 2_z)}, {2, rc(3_z, 2_z)}});
    RCP<const URatPolyFlint> d = URatPolyFlint::from_poly(*c);

    REQUIRE(b->__str__() == "x**2 + 1");
    REQUIRE(d->__str__() == "3/2*x**2 + 1/2");
}

#ifdef HAVE_SYMENGINE_PIRANHA
TEST_CASE("URatPolyFlint from_poly piranha", "[URatPolyFlint]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyPiranha> a
        = URatPolyPiranha::from_dict(x, {{0, rc(1_z, 2_z)}, {2, rc(3_z, 2_z)}});
    RCP<const URatPolyFlint> b = URatPolyFlint::from_poly(*a);
    REQUIRE(b->__str__() == "3/2*x**2 + 1/2");
}
#endif