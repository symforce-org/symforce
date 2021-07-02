#include "catch.hpp"
#include <chrono>

#include <symengine/subs.h>

using SymEngine::Basic;
using SymEngine::Add;
using SymEngine::Mul;
using SymEngine::Pow;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::umap_basic_num;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::multinomial_coefficients;
using SymEngine::one;
using SymEngine::zero;
using SymEngine::sin;
using SymEngine::erf;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::map_basic_basic;
using SymEngine::print_stack_on_segfault;
using SymEngine::real_double;
using SymEngine::kronecker_delta;
using SymEngine::levi_civita;
using SymEngine::msubs;
using SymEngine::xreplace;
using SymEngine::ssubs;
using SymEngine::subs;
using SymEngine::function_symbol;
using SymEngine::gamma;
using SymEngine::ComplexInf;
using SymEngine::interval;
using SymEngine::imageset;
using SymEngine::dummy;
using SymEngine::Set;
using SymEngine::set_union;
using SymEngine::finiteset;
using SymEngine::E;
using SymEngine::is_a;
using SymEngine::down_cast;

TEST_CASE("Symbol: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = x;
    RCP<const Basic> r2 = y;
    map_basic_basic d;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *r2));
    REQUIRE(neq(*r1->subs(d), *r1));
}

TEST_CASE("Add: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = add(x, y);
    RCP<const Basic> r2 = mul(i2, y);
    map_basic_basic d;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *r2));

    d[x] = z;
    d[y] = w;
    r1 = add(x, y);
    r2 = add(z, w);
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[add(x, y)] = z;
    r1 = add(x, y);
    r2 = z;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[pow(x, y)] = z;
    d[pow(x, i2)] = y;
    d[pow(i2, y)] = x;
    r1 = add(add(pow(x, y), pow(x, i2)), pow(i2, y));
    r2 = add(add(x, y), z);
    REQUIRE(eq(*r1->subs(d), *r2));

    r1 = add(add(add(add(pow(x, y), pow(x, i2)), pow(i2, y)), x), i3);
    r2 = add(add(add(mul(i2, x), y), z), i3);
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[x] = integer(5);
    r1 = add(mul(integer(12), add(integer(3), sin(x))), sin(integer(4)));
    r2 = add(mul(integer(12), add(integer(3), sin(integer(5)))),
             sin(integer(4)));
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[x] = integer(1);
    r1 = add(mul(x, add(integer(3), sin(integer(4)))), sin(integer(1)));
    r2 = add(add(integer(3), sin(integer(4))), sin(integer(1)));
    REQUIRE(eq(*r1->subs(d), *r2));

    // (2+2*x).subs({2: y}) -> y+x*y
    d.clear();
    d[i2] = y;
    r1 = add(i2, mul(i2, x));
    r2 = add(y, mul(x, y));
    REQUIRE(eq(*r1->subs(d), *r2));

    // (1+2*x*y).subs({2*x*y: z}) -> 1+z
    d.clear();
    d[mul(mul(i2, x), y)] = z;
    r1 = add(one, mul(i2, mul(x, y)));
    r2 = add(one, z);
    REQUIRE(eq(*r1->subs(d), *r2));
}

TEST_CASE("Mul: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = mul(x, y);
    RCP<const Basic> r2 = pow(y, i2);
    map_basic_basic d;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *r2));

    d[x] = z;
    d[y] = w;
    r1 = mul(x, y);
    r2 = mul(z, w);
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[mul(x, y)] = z;
    r1 = mul(x, y);
    r2 = z;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[pow(x, y)] = z;
    r1 = mul(i2, pow(x, y));
    r2 = mul(i2, z);
    REQUIRE(eq(*r1->subs(d), *r2));

    r1 = mul(x, y)->subs({{x, real_double(0.0)}});
    r2 = real_double(0.0);
    REQUIRE(eq(*r1, *r2));

    d.clear();
    r1 = div(one, x);
    d[x] = zero;
    REQUIRE(eq(*r1->subs(d), *ComplexInf));

    d.clear();
    r1 = div(i2, x);
    d[x] = zero;
    REQUIRE(eq(*r1->subs(d), *ComplexInf));

    d.clear();
    r1 = div(one, mul(x, y));
    d[x] = zero;
    REQUIRE(eq(*r1->subs(d), *div(ComplexInf, y)));

    d.clear();
    r1 = mul(i2, x);
    d[i2] = one;
    REQUIRE(eq(*r1->subs(d), *x));
}

TEST_CASE("Pow: subs", "[subs]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = pow(x, y);
    RCP<const Basic> r2 = pow(y, y);
    map_basic_basic d;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *r2));

    d[x] = z;
    d[y] = w;
    r1 = pow(x, y);
    r2 = pow(z, w);
    REQUIRE(eq(*r1->subs(d), *r2));

    r1 = pow(x, i2);
    r2 = pow(z, i2);
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[pow(x, y)] = z;
    r1 = pow(x, y);
    r2 = z;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[pow(E, x)] = z;
    r1 = pow(E, mul(x, x));
    r2 = r1->subs(d);
    REQUIRE(is_a<Pow>(*r2));
    REQUIRE(eq(*down_cast<const Pow &>(*r2).get_base(), *E));
    REQUIRE(eq(*down_cast<const Pow &>(*r2).get_exp(), *mul(x, x)));

    r2 = r1->xreplace(d);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(E, mul(i2, x));
    r2 = pow(z, i2);
    REQUIRE(eq(*r1->subs(d), *r2));

    r2 = r1->xreplace(d);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(E, add(i2, x));
    r2 = r1->subs(d); // TODO : Ideally, `r1->subs(d)` should be (E**2)*z.
    REQUIRE(eq(*r1, *r2));

    r2 = r1->xreplace(d);
    REQUIRE(eq(*r1, *r2));

    d.clear();
    d[x] = y;
    r1 = function_symbol("f", mul(i2, x))->diff(x);
    r2 = function_symbol("f", mul(i2, y))->diff(y);
    REQUIRE(eq(*r1->xreplace(d), *r2));
}

TEST_CASE("Erf: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1 = erf(add(x, y));
    RCP<const Basic> r2 = erf(mul(i2, x));
    RCP<const Basic> r3 = erf(add(y, x));

    map_basic_basic d;
    d[y] = x;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[z] = x;
    REQUIRE(eq(*r1->subs(d), *r3));

    d.clear();
    d[y] = zero;
    REQUIRE(eq(*r1->subs(d), *erf(x)));
}

TEST_CASE("Trig: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = sin(x);
    RCP<const Basic> r2 = zero;
    map_basic_basic d;
    d[x] = zero;
    REQUIRE(eq(*r1->subs(d), *r2));

    r1 = cos(x);
    r2 = one;
    REQUIRE(eq(*r1->subs(d), *r2));

    d[x] = y;
    r1 = sin(pow(x, i2));
    r2 = sin(pow(y, i2));
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[sin(x)] = z;
    r1 = sin(x);
    r2 = z;
    REQUIRE(eq(*r1->subs(d), *r2));

    r1 = mul(i2, sin(x));
    r2 = mul(i2, z);
    REQUIRE(eq(*r1->subs(d), *r2));
}

TEST_CASE("KroneckerDelta: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = kronecker_delta(x, y);
    RCP<const Basic> r2 = kronecker_delta(y, y);
    RCP<const Basic> r3;
    map_basic_basic d;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[y] = x;
    r2 = kronecker_delta(x, x);
    REQUIRE(eq(*r1->subs(d), *r2));

    r1 = kronecker_delta(add(x, i2), y);
    r2 = kronecker_delta(i4, y);
    r3 = kronecker_delta(add(x, i2), i2);
    d.clear();
    d[x] = i2;
    REQUIRE(eq(*r1->subs(d), *r2));
    d.clear();
    d[y] = i2;
    REQUIRE(eq(*r1->subs(d), *r3));
}

TEST_CASE("Gamma: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1 = gamma(add(x, y));
    RCP<const Basic> r2 = gamma(mul(i2, x));
    RCP<const Basic> r3 = gamma(add(y, x));

    map_basic_basic d;
    d[y] = x;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[z] = x;
    REQUIRE(eq(*r1->subs(d), *r3));

    d.clear();
    d[y] = zero;
    REQUIRE(eq(*r1->subs(d), *gamma(x)));
}

TEST_CASE("LowerGamma: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1 = lowergamma(add(x, z), y);
    RCP<const Basic> r2 = lowergamma(x, y);
    RCP<const Basic> r3 = lowergamma(add(x, x), add(y, one));

    map_basic_basic d;
    d[x] = one;
    REQUIRE(eq(*lowergamma(x, y)->subs(d), *lowergamma(one, y)));

    d.clear();
    d[z] = x;
    d[y] = add(y, one);
    REQUIRE(eq(*r1->subs(d), *r3));

    d.clear();
    d[z] = zero;
    d[y] = one;
    REQUIRE(eq(*r1->subs(d), *lowergamma(x, one)));

    d.clear();
    d[w] = one;
    d[z] = i2;
    d[y] = add(add(x, y), one);
    r2 = lowergamma(add(x, i2), add(add(x, y), one));
    REQUIRE(eq(*r1->subs(d), *r2));
}

TEST_CASE("FunctionSymbol: subs", "[subs]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> f = function_symbol("f", x);

    auto t = subs(f->diff(x), {{f, mul(x, x)}});
    REQUIRE(eq(*t, *mul(integer(2), x)));
}

TEST_CASE("UpperGamma: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1 = uppergamma(add(x, z), y);
    RCP<const Basic> r2 = uppergamma(x, y);
    RCP<const Basic> r3 = uppergamma(add(x, x), add(y, one));

    map_basic_basic d;
    d[x] = one;
    REQUIRE(eq(*uppergamma(x, y)->subs(d), *uppergamma(one, y)));

    d.clear();
    d[z] = x;
    d[y] = add(y, one);
    REQUIRE(eq(*r1->subs(d), *r3));

    d.clear();
    d[z] = zero;
    d[y] = one;
    REQUIRE(eq(*r1->subs(d), *uppergamma(x, one)));

    d.clear();
    d[w] = one;
    d[z] = i2;
    d[y] = add(add(x, y), one);
    r2 = uppergamma(add(x, i2), add(add(x, y), one));
    REQUIRE(eq(*r1->subs(d), *r2));
}

TEST_CASE("PolyGamma: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1 = polygamma(add(x, z), y);
    RCP<const Basic> r2 = polygamma(x, y);
    RCP<const Basic> r3 = polygamma(add(x, x), add(y, one));

    map_basic_basic d;
    d[x] = one;
    REQUIRE(eq(*polygamma(x, y)->subs(d), *polygamma(one, y)));

    d.clear();
    d[z] = x;
    d[y] = add(y, one);
    REQUIRE(eq(*r1->subs(d), *r3));

    d.clear();
    d[z] = zero;
    d[y] = one;
    REQUIRE(eq(*r1->subs(d), *polygamma(x, one)));

    d.clear();
    d[w] = one;
    d[z] = i2;
    d[y] = add(add(x, y), one);
    r2 = polygamma(add(x, i2), add(add(x, y), one));
    REQUIRE(eq(*r1->subs(d), *r2));
}

TEST_CASE("Beta: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> r1 = beta(add(x, z), y);
    RCP<const Basic> r2 = beta(x, y);
    RCP<const Basic> r3 = beta(add(x, x), add(y, one));

    map_basic_basic d;
    d[x] = one;
    REQUIRE(eq(*beta(x, y)->subs(d), *beta(y, one)));

    d.clear();
    d[z] = zero;
    d[y] = x;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *beta(x, y)));

    d.clear();
    d[z] = zero;
    d[y] = one;
    REQUIRE(eq(*r1->subs(d), *beta(x, one)));

    d.clear();
    d[w] = one;
    d[z] = i2;
    d[y] = i2;
    REQUIRE(eq(*r1->subs(d), *beta(i2, add(i2, x))));
}

TEST_CASE("Sets: subs", "[subs]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    auto n = dummy("n");
    RCP<const Basic> i2 = integer(2);
    auto interval1 = interval(integer(-2), integer(2));

    RCP<const Set> r1 = imageset(x, mul(x, x), interval1);
    RCP<const Set> r2 = imageset(y, mul(y, y), interval1);
    RCP<const Set> r3 = imageset(n, mul(n, n), interval1);

    map_basic_basic d;
    d[x] = y;
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[n] = x;
    REQUIRE(eq(*r3->subs(d), *r1));

    d.clear();
    d[x] = y;
    r1 = set_union({r1, imageset(x, add(x, i2), interval1)});
    r2 = set_union({r2, imageset(y, add(y, i2), interval1)});
    REQUIRE(eq(*r1->subs(d), *r2));

    d.clear();
    d[x] = n;
    r1 = finiteset({x, y});
    r2 = finiteset({n, y});
    REQUIRE(eq(*r1->subs(d), *r2));
}

TEST_CASE("MSubs: subs", "[subs]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> f = function_symbol("f", x);

    auto t = msubs(f->diff(x), {{f, y}});
    REQUIRE(eq(*t, *f->diff(x)));

    t = msubs(f->diff(x), {{f->diff(x), y}});
    REQUIRE(eq(*t, *y));
}

TEST_CASE("SSubs: subs", "[ssubs]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> f = function_symbol("f", x);
    RCP<const Basic> g = function_symbol("g", x);

    auto t = ssubs(f->diff(x), {{f, g}});
    REQUIRE(eq(*t, *g->diff(x)));
}

TEST_CASE("Cache: subs", "[subs]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> f = pow(x, integer(2));
    RCP<const Basic> g = add(f, sin(f));
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> s, t;

    s = add(i3, sin(i3));

    t = xreplace(g, {{f, i3}}, false);
    REQUIRE(eq(*t, *s));

    t = xreplace(g, {{f, i3}}, true);
    REQUIRE(eq(*t, *s));

    t = subs(g, {{f, i3}}, false);
    REQUIRE(eq(*t, *s));

    t = subs(g, {{f, i3}}, true);
    REQUIRE(eq(*t, *s));

    t = ssubs(g, {{f, i3}}, false);
    REQUIRE(eq(*t, *s));

    t = ssubs(g, {{f, i3}}, true);
    REQUIRE(eq(*t, *s));

    t = msubs(g, {{f, i3}}, false);
    REQUIRE(eq(*t, *s));

    t = msubs(g, {{f, i3}}, true);
    REQUIRE(eq(*t, *s));
}
