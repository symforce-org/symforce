#include "catch.hpp"

#include <symengine/basic.h>
#include <symengine/pow.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/functions.h>
#include <symengine/logic.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::boolTrue;
using SymEngine::cse;
using SymEngine::div;
using SymEngine::Gt;
using SymEngine::integer;
using SymEngine::mul;
using SymEngine::neg;
using SymEngine::one;
using SymEngine::piecewise;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::sin;
using SymEngine::sqrt;
using SymEngine::sub;
using SymEngine::symbol;
using SymEngine::unified_eq;
using SymEngine::vec_basic;
using SymEngine::vec_pair;

TEST_CASE("CSE: simple", "[cse]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> x0 = symbol("x0");
    RCP<const Basic> x1 = symbol("x1");
    RCP<const Basic> x2 = symbol("x2");
    RCP<const Basic> x3 = symbol("x3");
    RCP<const Basic> x4 = symbol("x4");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    {
        auto e = add({mul({i2, x, y}), mul({i2, x, z}), mul({i2, y, z})});
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, mul(x, i2)}}));
        REQUIRE(unified_eq(reduced,
                           {add({mul(x0, y), mul(x0, z), mul({i2, y, z})})}));
    }
    {
        auto e = add(pow(add(x, y), i2), sqrt(add(x, y)));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, add(x, y)}}));
        REQUIRE(unified_eq(reduced, {add(sqrt(x0), pow(x0, i2))}));
    }
    {
        auto e = add(x, y);
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {}));
        REQUIRE(unified_eq(reduced, {e}));
    }
    {
        auto e = add(pow(add(mul(w, x), y), i2), sqrt(add(mul(w, x), y)));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, add(mul(w, x), y)}}));
        REQUIRE(unified_eq(reduced, {add(sqrt(x0), pow(x0, i2))}));
    }
    {
        auto e1 = mul(add(x, y), z);
        auto e2 = mul(add(x, y), w);
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2});
        REQUIRE(unified_eq(substs, {{x0, add(x, y)}}));
        REQUIRE(unified_eq(reduced, {mul(x0, z), mul(x0, w)}));
    }
    {
        auto e1 = add(mul(mul(w, x), y), z);
        auto e2 = mul(y, w);
        vec_pair substs;
        vec_pair rsubsts;
        vec_basic reduced;
        vec_basic rreduced;
        cse(substs, reduced, {e1, e2});
        cse(rsubsts, rreduced, {e2, e1});
        REQUIRE(unified_eq(substs, rsubsts));
        REQUIRE(unified_eq(reduced, {add(mul(x0, x), z), x0}));
    }
    {
        auto e1 = mul(mul(w, x), y);
        auto e2 = add(mul(mul(w, x), y), z);
        auto e3 = mul(y, w);
        vec_pair substs;
        vec_pair rsubsts;
        vec_basic reduced;
        vec_basic rreduced;
        cse(substs, reduced, {e1, e2, e3});
        cse(rsubsts, rreduced, {e3, e2, e1});
        REQUIRE(unified_eq(substs, rsubsts));
        REQUIRE(unified_eq(reduced, {x1, add(x1, z), x0}));
    }
    {
        auto e2 = sub(x, z);
        auto e3 = sub(y, z);
        auto e1 = mul(e2, e3);
        vec_pair substs;
        vec_pair rsubsts;
        vec_basic reduced;
        vec_basic rreduced;
        cse(substs, reduced, {e1, e2, e3});
        cse(rsubsts, rreduced, {e3, e2, e1});
        REQUIRE(unified_eq(substs,
                           {{x0, neg(z)}, {x1, add(x, x0)}, {x2, add(x0, y)}}));
        REQUIRE(unified_eq(rsubsts,
                           {{x0, neg(z)}, {x1, add(y, x0)}, {x2, add(x0, x)}}));
        REQUIRE(unified_eq(reduced, {mul(x1, x2), x1, x2}));
    }
    {
        auto e2 = sub(x, z);
        auto e3 = sub(y, z);
        auto e1 = mul(e2, e3);
        vec_pair substs;
        vec_pair rsubsts;
        vec_basic reduced;
        vec_basic rreduced;
        cse(substs, reduced, {e1, e2, e3});
        cse(rsubsts, rreduced, {e3, e2, e1});
        REQUIRE(unified_eq(substs,
                           {{x0, neg(z)}, {x1, add(x, x0)}, {x2, add(x0, y)}}));
        REQUIRE(unified_eq(rsubsts,
                           {{x0, neg(z)}, {x1, add(y, x0)}, {x2, add(x0, x)}}));
        REQUIRE(unified_eq(reduced, {mul(x1, x2), x1, x2}));
    }
    {
        auto e1 = add(x, add(w, add(y, add(z, mul(w, y)))));
        auto e2 = mul(w, mul(x, y));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2});
        REQUIRE(unified_eq(substs, {{x0, mul(w, y)}}));
        REQUIRE(unified_eq(reduced,
                           {add(x, add(w, add(y, add(z, x0)))), mul(x0, x)}));
    }
    {
        auto e1 = add(x, y);
        auto e2 = add(x, add(y, z));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2});
        REQUIRE(unified_eq(substs, {{x0, add(x, y)}}));
        REQUIRE(unified_eq(reduced, {x0, add(z, x0)}));
    }
    {
        auto e1 = add(x, y);
        auto e2 = add(x, z);
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2});
        REQUIRE(unified_eq(substs, {}));
        REQUIRE(unified_eq(reduced, {e1, e2}));
    }
    {
        auto e1 = mul(x, y);
        auto e2 = add(z, mul(x, y));
        auto e3 = add(i3, mul(z, mul(x, y)));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2, e3});
        REQUIRE(unified_eq(substs, {{x0, mul(x, y)}}));
        REQUIRE(unified_eq(reduced, {x0, add(z, x0), add(i3, mul(x0, z))}));
    }
    {
        auto e = div(sin(pow(x, x)), pow(x, x));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, x)}}));
        REQUIRE(unified_eq(reduced, {div(sin(x0), x0)}));
    }
    {
        auto e = add(pow(x, im2), pow(x, i2));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, mul(x, x)}}));
        REQUIRE(unified_eq(reduced, {add(pow(x0, im1), x0)}));
    }
    {
        auto e = add(div(add(one, pow(x, im2)), pow(x, i2)), pow(x, i2));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, mul(x, x)}, {x1, pow(x0, im1)}}));
        REQUIRE(unified_eq(reduced, {add(mul(x1, add(one, x1)), x0)}));
    }
    {
        auto e = add(mul(add(one, pow(x, im2)), pow(x, i2)), pow(x, im2));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, mul(x, x)}, {x1, pow(x0, im1)}}));
        REQUIRE(unified_eq(reduced, {add(mul(x0, add(one, x1)), x1)}));
    }
    {
        auto e = add(cos(pow(x, im2)), sin(pow(x, im2)));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, im2)}}));
        REQUIRE(unified_eq(reduced, {add(sin(x0), cos(x0))}));
    }
    {
        auto e = add(cos(pow(x, i2)), sin(pow(x, i2)));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, i2)}}));
        REQUIRE(unified_eq(reduced, {add(sin(x0), cos(x0))}));
    }
    {
        auto e = add(div(y, add(i2, pow(x, i2))), div(div(z, pow(x, i2)), y));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, i2)}}));
        REQUIRE(unified_eq(reduced,
                           {add(div(y, add(x0, i2)), div(z, mul(x0, y)))}));
    }
    {
        auto e = add(exp(pow(x, i2)), mul(pow(x, i2), cos(pow(x, im2))));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, i2)}}));
        REQUIRE(
            unified_eq(reduced, {add(exp(x0), mul(x0, cos(pow(x0, im1))))}));
    }
    {
        auto e = div(add(one, pow(x, im2)), pow(x, i2));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, im2)}}));
        REQUIRE(unified_eq(reduced, {mul(x0, add(x0, one))}));
    }
    {
        auto e = add(pow(x, mul(i2, y)), pow(x, mul(im2, y)));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {{x0, pow(x, mul(i2, y))}}));
        REQUIRE(unified_eq(reduced, {add(x0, div(one, x0))}));
    }
    {
        auto z1 = add(x0, y);
        auto z2 = add(x2, x3);
        auto e1 = add(cos(z1), z1);
        auto e2 = add(cos(z2), z2);
        auto e3 = add(x0, x2);
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2, e3});
        REQUIRE(unified_eq(substs, {{x1, add(x0, y)}, {x4, add(x2, x3)}}));
        REQUIRE(unified_eq(reduced,
                           {add(x1, cos(x1)), add(x4, cos(x4)), add(x0, x2)}));
    }
    {
        auto e1 = add(x, i3);
        auto e2 = add(x, i4);
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2});
        REQUIRE(unified_eq(substs, {}));
        REQUIRE(unified_eq(reduced, {e1, e2}));
    }
    {
        auto e = div(x, sub(pow(y, i2), mul(i4, pow(x, i2))));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e});
        REQUIRE(unified_eq(substs, {}));
        REQUIRE(unified_eq(reduced, {e}));
    }
    {
        auto e1 = add(x, y);
        auto e2 = add(add(x, y), i2);
        auto e3 = add(add(x, y), z);
        auto e4 = add(add(add(x, y), z), i3);
        // x + y, 2 + x + y, x + y + z, 3 + x + y + z
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e2, e3, e4});
        REQUIRE(unified_eq(substs, {{x0, add(x, y)}, {x1, add(x0, z)}}));
        REQUIRE(unified_eq(reduced, {x0, add(i2, x0), x1, add(i3, x1)}));
    }
    {
        auto pw1 = piecewise(
            {{pow(add(x, y), i2), Gt(x, y)}, {sqrt(add(x, y)), boolTrue}});

        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {pw1});
        REQUIRE(unified_eq(substs, {{x0, add(x, y)}}));
        REQUIRE(unified_eq(reduced, {piecewise({{pow(x0, i2), Gt(x, y)},
                                                {sqrt(x0), boolTrue}})}));
    }
    {
        auto pw2 = piecewise({{pow(x, i2), Gt(add(x, y), i3)},
                              {sqrt(y), Gt(add(x, y), i2)},
                              {sqrt(x), boolTrue}});

        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {pw2});
        REQUIRE(unified_eq(substs, {{x0, add(x, y)}}));
        REQUIRE(unified_eq(reduced, {piecewise({{pow(x, i2), Gt(x0, i3)},
                                                {sqrt(y), Gt(x0, i2)},
                                                {sqrt(x), boolTrue}})}));
    }
}

TEST_CASE("CSE: regression test gh-1463", "[cse]")
{
    RCP<const Basic> x1 = symbol("x1");
    RCP<const Basic> x2 = symbol("x2");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    {
        auto z32 = pow(z, div(i3, i2));
        auto x1w = mul(x1, w);
        auto x2x1w = mul(x2, x1w);
        auto z_x2y = add(z, mul(x2, y));
        auto x1wSQRTz = mul(x1, mul(w, sqrt(z)));
        auto m3_2 = div(neg(i3), i2);

        auto e1 = div(mul(neg(x1), z32), z_x2y);
        auto e3 = div(mul(i2, mul(x1, z32)), z_x2y);
        auto e4 = add(div(mul(x1w, z32), pow(z_x2y, i2)),
                      div(mul(m3_2, x1wSQRTz), z_x2y));
        auto e6 = add(div(mul(im2, mul(x1w, z32)), pow(z_x2y, i2)),
                      mul(i3, div(x1wSQRTz, z_x2y)));
        auto e7 = div(mul(x2, mul(x1w, z32)), pow(z_x2y, i2));
        auto e9 = div(mul(im2, mul(x2, mul(x1w, z32))), pow(z_x2y, i2));
        vec_pair substs;
        vec_basic reduced;
        cse(substs, reduced, {e1, e1, e3, e4, e4, e6, e7, e7, e9});
    }
}
