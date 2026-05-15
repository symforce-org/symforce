#include "catch.hpp"
#include <chrono>

#include <symengine/sets.h>
#include <symengine/series_generic.h>
#include <symengine/symengine_exception.h>

using SymEngine::Basic;
using SymEngine::DomainError;
using SymEngine::emptyset;
using SymEngine::EulerGamma;
using SymEngine::Expression;
using SymEngine::gamma;
using SymEngine::I;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::integer_class;
using SymEngine::make_rcp;
using SymEngine::map_int_Expr;
using SymEngine::NotImplementedError;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::Pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::rational;
using SymEngine::RCP;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::UExprDict;
using SymEngine::umap_int_basic;
using SymEngine::umap_short_basic;
using SymEngine::univariate_series;
using SymEngine::UnivariateSeries;
using SymEngine::vec_basic_eq_perm;
using SymEngine::zero;

using namespace SymEngine::literals;

TEST_CASE("Create UnivariateSeries", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    map_int_Expr adict_ = {{0, 1}, {1, 2}, {2, 1}};
    UExprDict apoly_(adict_);
    RCP<const UnivariateSeries> P = univariate_series(x, 2, apoly_);
    REQUIRE(P->__str__() == "x**2 + 2*x + 1 + O(x**2)");

    map_int_Expr bdict_ = {{0, 1}, {1, 0}, {2, 2}, {3, 1}};
    UExprDict bpoly_(bdict_);
    RCP<const UnivariateSeries> Q = UnivariateSeries::create(x, 5, bpoly_);
    REQUIRE(Q->__str__() == "x**3 + 2*x**2 + 1 + O(x**5)");

    map_int_Expr cdict_
        = {{0, Expression("c")}, {1, Expression("b")}, {2, Expression("a")}};
    UExprDict cpoly_(cdict_);
    RCP<const UnivariateSeries> R = UnivariateSeries::create(x, 3, cpoly_);
    REQUIRE(R->__str__() == "a*x**2 + b*x + c + O(x**3)");

    // check if unknown types are handled by visitor
    RCP<const UnivariateSeries> S
        = UnivariateSeries::series(emptyset(), "x", 2);
    REQUIRE(S->__str__() == "EmptySet + O(x**2)");
}

TEST_CASE("Adding two UnivariateSeries", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    map_int_Expr adict_ = {{0, 1}, {1, 2}, {2, 1}};
    UExprDict apoly_(adict_);
    map_int_Expr bdict_ = {{0, 2}, {1, 3}, {2, 4}};
    UExprDict bpoly_(bdict_);
    map_int_Expr ddict_ = {{0, 3}, {1, 5}, {2, 5}};
    UExprDict dpoly_(ddict_);

    RCP<const UnivariateSeries> a = UnivariateSeries::create(x, 5, apoly_);
    RCP<const UnivariateSeries> b = UnivariateSeries::create(x, 4, bpoly_);
    RCP<const Basic> c = add(a, b);
    RCP<const UnivariateSeries> d = UnivariateSeries::create(x, 4, dpoly_);
    REQUIRE(c->__cmp__(*d));

    RCP<const UnivariateSeries> e = UnivariateSeries::create(x, 2, bpoly_);
    RCP<const Basic> f = add(a, e);
    RCP<const Basic> g = UnivariateSeries::create(x, 2, dpoly_);
    REQUIRE(f->__cmp__(*g));
}

TEST_CASE("Negative of a UnivariateSeries", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    map_int_Expr adict_ = {{0, 1}, {1, 2}, {2, 1}};
    UExprDict apoly_(adict_);
    map_int_Expr bdict_ = {{0, -1}, {1, -2}, {2, -1}};
    UExprDict bpoly_(bdict_);
    map_int_Expr cdict_ = {{0, 1}, {1, Expression("a")}};
    UExprDict cpoly_(cdict_);
    map_int_Expr ddict_ = {{0, -1}, {1, mul(integer(-1), symbol("a"))}};
    UExprDict dpoly_(ddict_);

    RCP<const UnivariateSeries> a = UnivariateSeries::create(x, 5, apoly_);
    RCP<const Basic> b = neg(a);
    RCP<const UnivariateSeries> c = UnivariateSeries::create(x, 5, bpoly_);
    RCP<const UnivariateSeries> d = UnivariateSeries::create(x, 5, cpoly_);
    RCP<const Basic> e = neg(d);
    RCP<const UnivariateSeries> f = UnivariateSeries::create(x, 5, dpoly_);
    REQUIRE(b->__cmp__(*c));
    REQUIRE(e->__cmp__(*f));
}

TEST_CASE("Subtracting two UnivariateSeries", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    map_int_Expr adict_ = {{0, 1}, {1, 2}, {2, 1}};
    UExprDict apoly_(adict_);
    map_int_Expr bdict_ = {{0, 2}, {1, 3}, {2, 4}};
    UExprDict bpoly_(bdict_);
    map_int_Expr fdict_ = {{0, -1}, {1, -1}, {2, -3}};
    UExprDict fpoly_(fdict_);
    map_int_Expr gdict_ = {{0, -1}, {1, -1}};
    UExprDict gpoly_(gdict_);

    RCP<const UnivariateSeries> a = UnivariateSeries::create(x, 3, apoly_);
    RCP<const UnivariateSeries> b = UnivariateSeries::create(x, 4, bpoly_);
    RCP<const Basic> c = sub(a, b);
    RCP<const UnivariateSeries> f = UnivariateSeries::create(x, 4, fpoly_);
    REQUIRE(c->__cmp__(*f));

    RCP<const UnivariateSeries> d = UnivariateSeries::create(x, 2, bpoly_);
    RCP<const Basic> e = sub(a, d);
    RCP<const UnivariateSeries> g = UnivariateSeries::create(x, 2, gpoly_);
    REQUIRE(e->__cmp__(*f));
}

TEST_CASE("Multiplication of two UExprDict with precision",
          "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    UExprDict a({{0, 1}, {1, 2}, {2, 1}});
    UExprDict b({{0, -1}, {1, -2}, {2, -1}});
    UExprDict c({{0, 1}, {1, 4}, {2, 6}, {3, 4}});
    UExprDict d({{0, -1}, {1, -4}, {2, -6}, {3, -4}, {4, -1}});

    UExprDict e = UnivariateSeries::mul(a, a, 4);
    UExprDict f = UnivariateSeries::mul(a, b, 5);

    REQUIRE(e == c);
    REQUIRE(f == d);
}

TEST_CASE("Exponentiation of UExprDict with precision", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    UExprDict zero({{0, Expression(0)}});
    UExprDict one({{0, Expression(1)}});
    UExprDict a({{0, 1}, {1, 2}, {2, 1}});
    UExprDict b({{0, -1}, {1, -2}, {2, -1}});
    UExprDict c({{0, 1}, {1, 4}, {2, 6}, {3, 4}});
    UExprDict d({{0, -1}, {1, -6}, {2, -15}, {3, -20}, {4, -15}});

    UExprDict e = UnivariateSeries::pow(a, 2, 4);
    UExprDict f = UnivariateSeries::pow(b, 3, 5);
    UExprDict g = UnivariateSeries::pow(a, 0, 2);

    REQUIRE(e == c);
    REQUIRE(f == d);
    REQUIRE(g == one);
    REQUIRE_THROWS_AS(UnivariateSeries::pow(zero, 0, 1), DomainError);
}

TEST_CASE("Differentiation of UnivariateSeries", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    UExprDict a({{0, 1}, {1, 2}, {2, 1}});
    UExprDict b({{0, 2}, {1, 2}});
    REQUIRE(UnivariateSeries::diff(a, UnivariateSeries::var("x")) == b);
}

TEST_CASE("Integration of UnivariateSeries", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    UExprDict a({{-1, Expression(1)}});
    UExprDict b({{0, 1}, {1, 2}, {2, 3}});
    UExprDict c({{1, 1}, {2, 1}, {3, 1}});
    REQUIRE_THROWS_AS(
        UnivariateSeries::integrate(a, UnivariateSeries::var("x")),
        NotImplementedError);
    REQUIRE(UnivariateSeries::integrate(b, UnivariateSeries::var("x")) == c);
}

TEST_CASE("UnivariateSeries: compare, as_basic, as_dict", "[UnivariateSeries]")
{
    RCP<const Symbol> x = symbol("x");
    UExprDict P({{0, 1}, {1, 2}});
    UExprDict Q({{0, 1}, {1, Expression("b")}, {2, 1}});
    RCP<const UnivariateSeries> R = univariate_series(x, 4, P);
    RCP<const UnivariateSeries> S = univariate_series(x, 5, Q);
    umap_int_basic m = {{0, integer(1)}, {1, integer(2)}};

    REQUIRE(R->compare(*R) == 0);
    REQUIRE(R->compare(*S) == -1);
    REQUIRE(S->as_basic()->__eq__(*S->as_basic()) == true);
    REQUIRE(unified_eq(R->as_dict(), m) == true);
}

#define series_coeff(EX, SYM, PREC, COEFF)                                     \
    UnivariateSeries::series(EX, SYM->get_name(), PREC)->get_coeff(COEFF)
#define invseries_coeff(EX, SYM, PREC, COEFF)                                  \
    UnivariateSeries::series_reverse(                                          \
        UnivariateSeries::series(EX, SYM->get_name(), PREC)->get_poly(),       \
        UExprDict(SYM->get_name()), PREC)                                      \
        .find_cf(COEFF)                                                        \
        .get_basic()

static bool expand_check_pairs(const RCP<const Basic> &ex,
                               const RCP<const Symbol> &x, int prec,
                               const umap_short_basic &pairs)
{
    auto ser = SymEngine::UnivariateSeries::series(ex, x->get_name(), prec);
    for (auto it : pairs) {
        if (not it.second->__eq__(*(ser->get_coeff(it.first))))
            return false;
    }
    return true;
}

TEST_CASE("Expression series expansion: Add ", "[Expansion of Add]")
{
    RCP<const Symbol> x = symbol("x"), y = symbol("y");
    auto z = add(integer(1), x);
    z = sub(z, pow(x, integer(2)));
    z = add(z, pow(x, integer(4)));
    auto a = sub(integer(1), pow(symbol("x"), integer(2)));
    a = add(symbol("a"), a);

    auto vb = umap_short_basic{
        {0, integer(1)}, {1, integer(1)}, {2, integer(-1)}, {4, integer(1)}};
    REQUIRE(expand_check_pairs(z, x, 5, vb));
    auto vb1
        = umap_short_basic{{0, integer(1)}, {1, integer(1)}, {2, integer(-1)}};
    REQUIRE(expand_check_pairs(z, x, 3, vb1));
    auto vc = umap_short_basic{
        {0, add(integer(1), symbol("a"))}, {1, integer(0)}, {2, integer(-1)}};
    REQUIRE(expand_check_pairs(a, x, 5, vc));
}

TEST_CASE("Expression series expansion: sin, cos", "[Expansion of sin, cos]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> one = integer(1);
    auto z1 = sin(x);
    auto z2 = cos(x);
    auto z3 = add(sin(x), cos(x));
    auto z4 = mul(sin(x), cos(x));
    auto z5 = sin(atan(x));
    auto z5prime = sin(add(x, integer(5)));
    auto z6 = cos(div(x, sub(one, x)));
    auto z6prime = cos(add(x, integer(5)));
    auto z7 = sin(mul(symbol("a"), x));

    REQUIRE(series_coeff(z1, x, 10, 9)->__eq__(*rational(1, 362880)));
    auto res = umap_short_basic{{0, integer(1)}, {2, rational(-1, 2)}};
    REQUIRE(expand_check_pairs(z2, x, 3, res));
    REQUIRE(series_coeff(z3, x, 9, 8)->__eq__(*rational(1, 40320)));
    REQUIRE(series_coeff(z4, x, 12, 11)->__eq__(*rational(-4, 155925)));
    REQUIRE(series_coeff(z5, x, 12, 11)->__eq__(*rational(-63, 256)));
    REQUIRE(series_coeff(z5prime, x, 11, 10)
                ->__eq__(*mul(rational(-1, 3628800), sin(integer(5)))));
    REQUIRE(series_coeff(z6, x, 12, 11)->__eq__(*rational(-125929, 362880)));
    REQUIRE(series_coeff(z6prime, x, 15, 11)
                ->__eq__(*mul(rational(1, 39916800), sin(integer(5)))));
    REQUIRE(series_coeff(z7, x, 10, 9)
                ->__eq__(
                    *mul((pow(symbol("a"), integer(9))), rational(1, 362880))));
}

TEST_CASE("Expression series expansion: division, inversion ",
          "[Expansion of 1/ex]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> one = integer(1);
    RCP<const Integer> two = integer(2);
    RCP<const Integer> three = integer(3);
    auto ex1 = div(one, sub(one, x));                 // 1/(1-x)
    auto ex2 = div(x, sub(sub(one, x), pow(x, two))); // x/(1-x-x^2)
    auto ex3
        = div(pow(x, three), sub(one, mul(pow(x, two), two))); // x^3/(1-2x^2)
    auto ex4 = div(one, sub(one, sin(x)));                     // 1/(1-sin(x))
    auto ex5 = div(one, x);
    auto ex6 = div(one, mul(x, sub(one, x)));
    auto ex7 = div(one, mul(symbol("a"), x));
    auto res1 = umap_short_basic{{-1, integer(1)}};
    auto res2 = umap_short_basic{{-1, integer(1)}, {0, integer(1)}};
    auto res3 = umap_short_basic{{-1, div(integer(1), symbol("a"))}};

    REQUIRE(series_coeff(ex1, x, 100, 99)->__eq__(*integer(1)));
    REQUIRE(series_coeff(ex2, x, 11, 10)->__eq__(*integer(55)));
    REQUIRE(series_coeff(ex3, x, 12, 11)->__eq__(*integer(16)));
    REQUIRE(series_coeff(ex4, x, 11, 10)->__eq__(*rational(1382, 14175)));
    REQUIRE(expand_check_pairs(ex5, x, 8, res1));
    REQUIRE(expand_check_pairs(ex6, x, 8, res2));
    REQUIRE(expand_check_pairs(ex7, x, 8, res3));
}

TEST_CASE("Expression series expansion: roots", "[Expansion of root(ex)]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Number> q12 = rational(1, 2);
    RCP<const Number> qm23 = rational(-2, 3);
    RCP<const Integer> one = integer(1);
    RCP<const Integer> four = integer(4);
    auto ex1 = pow(sub(four, x), q12);
    auto ex2 = pow(sub(one, x), qm23);
    auto ex3 = sqrt(sub(one, x));
    auto ex4 = pow(cos(x), q12);
    auto ex5 = pow(cos(x), qm23);
    auto ex6 = sqrt(cos(x));

    REQUIRE(series_coeff(ex1, x, 8, 6)->__eq__(*rational(-21, 2097152)));
    REQUIRE(series_coeff(ex2, x, 12, 10)->__eq__(*rational(1621477, 4782969)));
    REQUIRE(series_coeff(ex3, x, 12, 10)->__eq__(*rational(-2431, 262144)));
    REQUIRE(series_coeff(ex4, x, 10, 8)->__eq__(*rational(-559, 645120)));
    REQUIRE(series_coeff(ex5, x, 20, 10)->__eq__(*rational(701, 127575)));
    REQUIRE(series_coeff(ex6, x, 10, 8)->__eq__(*rational(-559, 645120)));
}

TEST_CASE("Expression series expansion: log, exp ", "[Expansion of log, exp]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> one = integer(1);
    RCP<const Integer> two = integer(2);
    RCP<const Integer> three = integer(3);
    auto ex1 = log(add(one, x));
    auto ex2 = log(cos(x));
    auto ex3 = log(div(one, sub(one, x)));
    auto ex4 = exp(x);
    auto ex5 = exp(log(add(x, one)));
    auto ex5prime = exp(add(x, integer(5)));
    auto ex6 = log(exp(x));
    auto ex7 = exp(sin(x));
    auto ex8 = pow(cos(x), sin(x));

    REQUIRE(series_coeff(ex1, x, 10, 8)->__eq__(*rational(-1, 8)));
    REQUIRE(series_coeff(ex2, x, 13, 12)->__eq__(*rational(-691, 935550)));
    REQUIRE(series_coeff(ex3, x, 10, 8)->__eq__(*rational(1, 8)));
    REQUIRE(series_coeff(ex4, x, 10, 9)->__eq__(*rational(1, 362880)));
    auto res1 = umap_short_basic{{0, integer(1)}, {1, integer(1)}};
    auto res2 = umap_short_basic{{1, integer(1)}};
    REQUIRE(expand_check_pairs(ex5, x, 10, res1));
    REQUIRE(series_coeff(ex5prime, x, 12, 11)
                ->__eq__(*mul(exp(integer(5)), rational(1, 39916800))));
    REQUIRE(expand_check_pairs(ex6, x, 10, res2));
    REQUIRE(series_coeff(ex7, x, 12, 10)->__eq__(*rational(-2951, 3628800)));
    REQUIRE(series_coeff(ex8, x, 12, 11)->__eq__(*rational(-71, 57600)));
}

TEST_CASE("Expression series expansion: reversion ", "[Expansion of f^-1]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> two = integer(2);
    RCP<const Integer> three = integer(3);
    auto ex1 = sub(x, pow(x, two));
    auto ex2 = sub(x, pow(x, three));
    auto ex3 = sin(x);
    auto ex4 = mul(x, exp(x));

    REQUIRE(invseries_coeff(ex1, x, 9, 8)->__eq__(*integer(429)));
    REQUIRE(invseries_coeff(ex2, x, 10, 9)->__eq__(*integer(55)));
    REQUIRE(invseries_coeff(ex3, x, 10, 9)->__eq__(*rational(35, 1152)));
    REQUIRE(invseries_coeff(ex4, x, 11, 10)->__eq__(*rational(-156250, 567)));
}

TEST_CASE("Expression series expansion: atan, tan, asin, cot, sec, csc",
          "[Expansion of tan, atan, asin, cot, sec, csc]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> one = integer(1);
    auto ex1 = atan(x);
    auto ex2 = atan(div(x, sub(one, x)));
    auto ex3 = tan(x);
    auto ex4 = tan(div(x, sub(one, x)));
    auto ex4prime = tan(add(x, integer(5)));
    auto ans4 = add(add(pow(tan(integer(5)), integer(5)), // tanh(5)**5
                        mul(rational(5, 3), pow(tan(integer(5)),
                                                integer(3)))), // 5*tanh(5)**3/3
                    mul(rational(2, 3), tan(integer(5))));     // 2*tanh(5)/3
    auto ex5 = asin(x);
    auto ex6 = asin(div(x, sub(one, x)));
    auto ex6prime = asin(add(x, integer(5)));
    auto ex7 = cot(x);
    auto ex8 = cot(sin(x));
    auto res1 = umap_short_basic{{-1, integer(1)}, {1, rational(-1, 3)}};
    auto res2
        = umap_short_basic{{-1, integer(1)}, {7, rational(-1051, 1814400)}};
    auto ex9 = sec(x);
    auto ex10 = csc(x);

    auto s = UnivariateSeries::series(ex8, "x", 10);

    REQUIRE(series_coeff(ex1, x, 10, 7)->__eq__(*rational(-1, 7)));
    REQUIRE(series_coeff(ex2, x, 11, 10)->__eq__(*rational(16, 5)));
    REQUIRE(series_coeff(ex3, x, 10, 9)->__eq__(*rational(62, 2835)));
    REQUIRE(series_coeff(ex4, x, 8, 7)->__eq__(*rational(2537, 315)));
    REQUIRE(expand(series_coeff(ex4prime, x, 5, 4))->__eq__(*ans4));
    REQUIRE(series_coeff(ex5, x, 16, 15)->__eq__(*rational(143, 10240)));
    REQUIRE(series_coeff(ex6, x, 17, 16)->__eq__(*rational(1259743, 2048)));
    REQUIRE(series_coeff(ex6prime, x, 6, 5)
                ->__eq__(*mul(rational(-5603, 318504960),
                              mul(I, sqrt(integer(24))))));
    REQUIRE(expand_check_pairs(ex7, x, 5, res1));
    REQUIRE(expand_check_pairs(ex8, x, 10, res2));
    REQUIRE(series_coeff(ex9, x, 10, 8)->__eq__(*rational(277, 8064)));
    REQUIRE(series_coeff(ex10, x, 10, 7)->__eq__(*rational(127, 604800)));
}

TEST_CASE("Expression series expansion: sinh, cosh, tanh, asinh, atanh",
          "[Expansion of sinh, cosh, tanh, asinh, atanh]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Integer> one = integer(1);
    auto ex1 = sinh(x);
    auto ex2 = sinh(div(x, sub(one, x)));
    auto ex3 = cosh(x);
    auto ex4 = cosh(div(x, sub(one, x)));
    auto ex5 = tanh(x);
    auto ex6 = tanh(div(x, sub(one, x)));
    auto ex6prime = tanh(add(x, integer(5)));
    auto ans6
        = add(add(pow(tanh(integer(5)), integer(5)), // tanh(5)**5
                  mul(rational(-5, 3),
                      pow(tanh(integer(5)), integer(3)))), //- 5*tanh(5)**3/3
              mul(rational(2, 3), tanh(integer(5))));      // 2*tanh(5)/3

    auto ex7 = atanh(x);
    auto ex8 = atanh(div(x, sub(one, x)));
    auto ex9 = asinh(x);
    auto ex10 = asinh(div(x, sub(one, x)));

    REQUIRE(series_coeff(ex1, x, 10, 9)->__eq__(*rational(1, 362880)));
    REQUIRE(series_coeff(ex2, x, 11, 10)->__eq__(*rational(325249, 40320)));
    REQUIRE(series_coeff(ex3, x, 12, 10)->__eq__(*rational(1, 3628800)));
    REQUIRE(series_coeff(ex4, x, 12, 11)->__eq__(*rational(3756889, 362880)));
    REQUIRE(series_coeff(ex5, x, 14, 13)->__eq__(*rational(21844, 6081075)));
    REQUIRE(series_coeff(ex6, x, 15, 14)->__eq__(*rational(225979, 66825)));
    REQUIRE(expand(series_coeff(ex6prime, x, 5, 4))->__eq__(*ans6));
    REQUIRE(series_coeff(ex7, x, 10, 9)->__eq__(*rational(1, 9)));
    REQUIRE(series_coeff(ex8, x, 9, 8)->__eq__(*integer(16)));
    REQUIRE(series_coeff(ex9, x, 10, 9)->__eq__(*rational(35, 1152)));
    REQUIRE(series_coeff(ex10, x, 10, 9)->__eq__(*rational(419, 1152)));
}

TEST_CASE("Expression series expansion: lambertw ", "[Expansion of lambertw]")
{
    RCP<const Symbol> x = symbol("x");
    auto ex1 = lambertw(x);
    auto ex2 = lambertw(sin(x));

    REQUIRE(series_coeff(ex1, x, 10, 7)->__eq__(*rational(16807, 720)));
    REQUIRE(series_coeff(ex2, x, 12, 10)->__eq__(*rational(-2993294, 14175)));
}

TEST_CASE("Expression series expansion: gamma ", "[Expansion of gamma]")
{
    RCP<const Symbol> x = symbol("x");
    auto ex1 = gamma(x);

    REQUIRE(series_coeff(ex1, x, 10, -1)->__eq__(*one));
    REQUIRE(series_coeff(ex1, x, 10, 0)->__eq__(*neg(EulerGamma)));
}
