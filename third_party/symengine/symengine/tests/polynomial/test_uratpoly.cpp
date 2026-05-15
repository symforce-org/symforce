#include "catch.hpp"
#include <chrono>
#include <sstream>

#include <symengine/polys/uratpoly.h>
#include <symengine/polys/uintpoly.h>
#include <symengine/pow.h>
#include <symengine/symengine_exception.h>
#include <symengine/polys/uintpoly_piranha.h>
#include <symengine/polys/uintpoly_flint.h>

#ifdef HAVE_SYMENGINE_PIRANHA
using SymEngine::URatPolyPiranha;
#endif
#ifdef HAVE_SYMENGINE_FLINT
using SymEngine::URatPolyFlint;
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
using SymEngine::URatDict;
using SymEngine::URatPoly;
using SymEngine::zero;

using namespace SymEngine::literals;
using rc = rational_class;
using ic = SymEngine::integer_class;

TEST_CASE("Constructor of URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");

    RCP<const URatPoly> P
        = URatPoly::from_dict(x, {{0, rc(1_z, 2_z)}, {2, rc(3_z, 2_z)}});
    REQUIRE(P->__str__() == "3/2*x**2 + 1/2");

    RCP<const URatPoly> Q
        = URatPoly::from_vec(x, {0_q, rc(1_z, 2_z), rc(1_z, 2_z)});
    REQUIRE(Q->__str__() == "1/2*x**2 + 1/2*x");

    RCP<const URatPoly> S = URatPoly::from_dict(x, {{0, 2_q}});
    REQUIRE(S->__str__() == "2");

    RCP<const URatPoly> T = URatPoly::from_dict(x, map_uint_mpq{});
    REQUIRE(T->__str__() == "0");

    std::stringstream ss;
    ss << *T;
    REQUIRE(ss.str() == "0");
}

TEST_CASE("Adding two URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const URatPoly> a = URatPoly::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 3_z)}, {2, 1_q}});
    RCP<const URatPoly> b
        = URatPoly::from_dict(x, {{0, rc(2_z, 3_z)}, {1, 3_q}, {2, 2_q}});

    RCP<const Basic> c = add_upoly(*a, *b);
    REQUIRE(c->__str__() == "3*x**2 + 11/3*x + 7/6");

    RCP<const URatPoly> d = URatPoly::from_dict(x, {{0, rc(1_z, 2_z)}});
    RCP<const Basic> e = add_upoly(*a, *d);
    RCP<const Basic> f = add_upoly(*d, *a);
    REQUIRE(e->__str__() == "x**2 + 2/3*x + 1");
    REQUIRE(f->__str__() == "x**2 + 2/3*x + 1");

    RCP<const URatPoly> g = URatPoly::from_dict(
        // With expression templates on in boostmp, we cannot
        // use negated literal in constructor of rational_class.
        // rc(-3_z,2_z); //error
        // a literal (e.g. 2_z) returns an integer_class, but unary minus
        // applied to a literal (e.g. -3_z) returns an expression template,
        // and rational_class cannot be constructed from two args,
        // one of which is an expression template and one of which
        // is an integer_class.
        // So we must use the string constructor of integer_class directly
        y, {{0, 2_q}, {1, rc(ic(-3), 2_z)}, {2, rc(1_z, 4_z)}});
    CHECK_THROWS_AS(add_upoly(*a, *g), SymEngineException);
}

TEST_CASE("Negative of a URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPoly> a
        = URatPoly::from_dict(x, {{0, rc(ic(-1), 2_z)}, {1, 2_q}, {2, 3_q}});

    RCP<const URatPoly> b = neg_upoly(*a);
    REQUIRE(b->__str__() == "-3*x**2 - 2*x + 1/2");
}

TEST_CASE("Subtracting two URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const URatPoly> a = URatPoly::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 3_z)}, {2, 1_q}});
    RCP<const URatPoly> b
        = URatPoly::from_dict(x, {{0, rc(2_z, 3_z)}, {1, 3_q}, {2, 2_q}});
    RCP<const URatPoly> c = URatPoly::from_dict(
        x, {{0, 2_q}, {1, rc(ic(-3), 2_z)}, {2, rc(1_z, 4_z)}});
    RCP<const URatPoly> f = URatPoly::from_dict(y, {{0, 2_q}});

    RCP<const Basic> d = sub_upoly(*b, *a);
    REQUIRE(d->__str__() == "x**2 + 7/3*x + 1/6");
    d = sub_upoly(*c, *a);
    REQUIRE(d->__str__() == "-3/4*x**2 - 13/6*x + 3/2");
    d = sub_upoly(*a, *c);
    REQUIRE(d->__str__() == "3/4*x**2 + 13/6*x - 3/2");
    CHECK_THROWS_AS(sub_upoly(*a, *f), SymEngineException);
}

TEST_CASE("Multiplication of two URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    RCP<const URatPoly> a = URatPoly::from_dict(
        x, {{0, rc(1_z, 2_z)}, {1, rc(2_z, 3_z)}, {2, 1_q}});
    RCP<const URatPoly> b
        = URatPoly::from_dict(x, {{0, rc(2_z, 3_z)}, {1, 3_q}, {2, 2_q}});
    RCP<const URatPoly> e = URatPoly::from_dict(
        x, {{0, 2_q}, {1, rc(ic(-3), 2_z)}, {2, rc(1_z, 4_z)}});
    RCP<const URatPoly> f
        = URatPoly::from_dict(x, {{0, 1_q}, {1, rc(1_z, 2_z)}});

    RCP<const URatPoly> c = mul_upoly(*a, *a);
    RCP<const URatPoly> d = mul_upoly(*a, *b);
    RCP<const URatPoly> g = mul_upoly(*e, *e);
    RCP<const URatPoly> h = mul_upoly(*e, *f);
    RCP<const URatPoly> i = mul_upoly(*f, *f);

    REQUIRE(c->__str__() == "x**4 + 4/3*x**3 + 13/9*x**2 + 2/3*x + 1/4");
    REQUIRE(d->__str__() == "2*x**4 + 13/3*x**3 + 11/3*x**2 + 35/18*x + 1/3");
    REQUIRE(g->__str__() == "1/16*x**4 - 3/4*x**3 + 13/4*x**2 - 6*x + 4");
    REQUIRE(h->__str__() == "1/8*x**3 - 1/2*x**2 - 1/2*x + 2");
    REQUIRE(i->__str__() == "1/4*x**2 + x + 1");

    c = URatPoly::from_dict(x, {{0, rc(-1_z)}});
    REQUIRE(mul_upoly(*a, *c)->__str__() == "-x**2 - 2/3*x - 1/2");
    REQUIRE(mul_upoly(*c, *a)->__str__() == "-x**2 - 2/3*x - 1/2");

    c = URatPoly::from_dict(y, {{0, rc(-1_z)}});
    CHECK_THROWS_AS(mul_upoly(*a, *c), SymEngineException);
}

TEST_CASE("Comparing two URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const URatPoly> P
        = URatPoly::from_dict(x, {{0, 1_q}, {1, rc(2_z, 3_z)}});
    RCP<const URatPoly> Q
        = URatPoly::from_dict(x, {{0, 1_q}, {1, 2_q}, {2, 1_q}});

    REQUIRE(P->compare(*Q) == -1);

    P = URatPoly::from_dict(x, {{0, 1_q}, {1, 2_q}, {2, 3_q}, {3, 2_q}});
    REQUIRE(P->compare(*Q) == 1);

    P = URatPoly::from_dict(x, {{0, 1_q}, {1, 2_q}, {2, 1_q}});
    REQUIRE(P->compare(*Q) == 0);

    P = URatPoly::from_dict(y, {{0, 1_q}, {1, rc(2_z, 3_z)}});
    REQUIRE(P->compare(*Q) == -1);
}

TEST_CASE("URatPoly as_symbolic", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPoly> a
        = URatPoly::from_dict(x, {{0, rc(1_z, 2_z)}, {1, 2_q}, {2, 1_q}});

    REQUIRE(eq(
        *a->as_symbolic(),
        *add({div(one, integer(2)), mul(integer(2), x), pow(x, integer(2))})));

    RCP<const URatPoly> b
        = URatPoly::from_dict(x, {{1, rc(3_z, 2_z)}, {2, 2_q}});
    REQUIRE(eq(*b->as_symbolic(), *add(mul(x, div(integer(3), integer(2))),
                                       mul(integer(2), pow(x, integer(2))))));

    RCP<const URatPoly> c = URatPoly::from_dict(x, map_uint_mpq{});
    REQUIRE(eq(*c->as_symbolic(), *zero));
}

TEST_CASE("Evaluation of URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPoly> a
        = URatPoly::from_dict(x, {{0, 1_q}, {1, rc(2_z, 3_z)}});
    RCP<const URatPoly> b = URatPoly::from_dict(
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

TEST_CASE("Derivative of URatPoly", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const URatPoly> a = URatPoly::from_dict(
        x, {{0, 1_q}, {1, rc(2_z, 3_z)}, {2, rc(1_z, 2_z)}});
    RCP<const URatPoly> b = URatPoly::from_dict(y, {{2, rc(4_z, 3_z)}});

    REQUIRE(a->diff(x)->__str__() == "x + 2/3");
    REQUIRE(a->diff(y)->__str__() == "0");
    REQUIRE(b->diff(y)->__str__() == "8/3*y");
}

TEST_CASE("URatPoly pow", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPoly> a
        = URatPoly::from_dict(x, {{0, rc(1_z, 2_z)}, {1, 1_q}});
    RCP<const URatPoly> b
        = URatPoly::from_dict(x, {{0, 3_q}, {2, rc(3_z, 2_z)}});

    RCP<const URatPoly> aaa = pow_upoly(*a, 3);
    RCP<const URatPoly> bb = pow_upoly(*b, 2);

    REQUIRE(aaa->__str__() == "x**3 + 3/2*x**2 + 3/4*x + 1/8");
    REQUIRE(bb->__str__() == "9/4*x**4 + 9*x**2 + 9");
}

TEST_CASE("URatPoly divides", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPoly> a = URatPoly::from_dict(x, {{0, 1_q}, {1, 1_q}});
    RCP<const URatPoly> b = URatPoly::from_dict(x, {{0, 4_q}});
    RCP<const URatPoly> c = URatPoly::from_dict(x, {{0, 8_q}, {1, 8_q}});
    RCP<const URatPoly> res;

    REQUIRE(divides_upoly(*a, *c, outArg(res)));
    REQUIRE(res->__str__() == "8");
    REQUIRE(divides_upoly(*b, *c, outArg(res)));
    REQUIRE(res->__str__() == "2*x + 2");
    REQUIRE(divides_upoly(*b, *a, outArg(res)));
    REQUIRE(res->__str__() == "1/4*x + 1/4");
    REQUIRE(!divides_upoly(*a, *b, outArg(res)));
}

TEST_CASE("URatPoly from_poly uint", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const UIntPoly> a = UIntPoly::from_dict(x, {{0, 1_z}, {2, 1_z}});
    RCP<const URatPoly> b = URatPoly::from_poly(*a);
    REQUIRE(b->__str__() == "x**2 + 1");
}

#ifdef HAVE_SYMENGINE_PIRANHA
TEST_CASE("URatPoly from_poly piranha", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyPiranha> a
        = URatPolyPiranha::from_dict(x, {{0, rc(1_z, 2_z)}, {2, rc(3_z, 2_z)}});
    RCP<const URatPoly> b = URatPoly::from_poly(*a);
    REQUIRE(b->__str__() == "3/2*x**2 + 1/2");
}
#endif

#ifdef HAVE_SYMENGINE_FLINT
TEST_CASE("URatPoly from_poly flint", "[URatPoly]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const URatPolyFlint> a
        = URatPolyFlint::from_dict(x, {{0, rc(1_z, 2_z)}, {2, rc(3_z, 2_z)}});
    RCP<const URatPoly> b = URatPoly::from_poly(*a);
    REQUIRE(b->__str__() == "3/2*x**2 + 1/2");
}
#endif
