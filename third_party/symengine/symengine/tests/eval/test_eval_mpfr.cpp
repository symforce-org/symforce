#include "catch.hpp"
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/eval_mpfr.h>
#include <symengine/constants.h>
#include <symengine/functions.h>
#include <symengine/logic.h>
#include <symengine/symengine_exception.h>
#include <symengine/pow.h>

using SymEngine::abs;
using SymEngine::acos;
using SymEngine::acosh;
using SymEngine::acot;
using SymEngine::acoth;
using SymEngine::acsc;
using SymEngine::acsch;
using SymEngine::asec;
using SymEngine::asech;
using SymEngine::asin;
using SymEngine::asinh;
using SymEngine::atan;
using SymEngine::atanh;
using SymEngine::Basic;
using SymEngine::beta;
using SymEngine::Catalan;
using SymEngine::constant;
using SymEngine::cos;
using SymEngine::cosh;
using SymEngine::cot;
using SymEngine::coth;
using SymEngine::csc;
using SymEngine::csch;
using SymEngine::E;
using SymEngine::Eq;
using SymEngine::erf;
using SymEngine::erfc;
using SymEngine::EulerGamma;
using SymEngine::eval_mpfr;
using SymEngine::gamma;
using SymEngine::GoldenRatio;
using SymEngine::integer;
using SymEngine::integer_class;
using SymEngine::lambertw;
using SymEngine::Le;
using SymEngine::loggamma;
using SymEngine::lowergamma;
using SymEngine::Lt;
using SymEngine::max;
using SymEngine::min;
using SymEngine::minus_one;
using SymEngine::mul;
using SymEngine::Ne;
using SymEngine::NotImplementedError;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::RCP;
using SymEngine::sec;
using SymEngine::sech;
using SymEngine::sin;
using SymEngine::sinh;
using SymEngine::sub;
using SymEngine::SymEngineException;
using SymEngine::tan;
using SymEngine::tanh;
using SymEngine::uppergamma;
using SymEngine::zero;

TEST_CASE("precision: eval_mpfr", "[eval_mpfr]")
{
    mpfr_t a;
    mpfr_init2(a, 53);
    RCP<const Basic> s = mul(pi, integer(1963319607));
    RCP<const Basic> t = integer(integer_class("6167950454"));
    RCP<const Basic> r = sub(s, t);
    RCP<const Basic> u = Eq(r, integer(2));
    // value of `r` is approximately 0.000000000149734291

    eval_mpfr(a, *u, MPFR_RNDN);
    CHECK(mpfr_cmp_si(a, 0) == 0);

    u = Ne(r, integer(2));
    eval_mpfr(a, *u, MPFR_RNDN);
    CHECK(mpfr_cmp_si(a, 1) == 0);

    u = Le(r, integer(2));
    eval_mpfr(a, *u, MPFR_RNDN);
    CHECK(mpfr_cmp_si(a, 1) == 0);

    u = Lt(r, integer(2));
    eval_mpfr(a, *u, MPFR_RNDN);
    CHECK(mpfr_cmp_si(a, 1) == 0);

    eval_mpfr(a, *r, MPFR_RNDN);
    // `eval_mpfr` was done with a precision of 53 bits (precision of `a`) and
    // rounding mode `MPFR_RNDN`
    // With 53 bit precision, `s` and `t` have the same value.
    // Hence value of `r` was  rounded down to `0.000000000000000`
    REQUIRE(mpfr_cmp_si(a, 0) == 0);

    mpfr_set_prec(a, 100);
    eval_mpfr(a, *r, MPFR_RNDN);
    // `eval_mpfr` was done with a precision of 100 bits (precision of `a`) and
    // rounding mode `MPFR_RNDN`
    // With 100 bit precision, `s` and `t` are not equal in value.
    // Value of `r` is a positive quantity with value 0.000000000149734291.....
    REQUIRE(mpfr_cmp_si(a, 0) == 1);

    // Check that value of `r` (`a`) starts with 0.000000000149734291
    REQUIRE(mpfr_cmp_d(a, 0.000000000149734291) == 1);
    REQUIRE(mpfr_cmp_d(a, 0.000000000149734292) == -1);

    s = mul(EulerGamma, integer(100000000));
    t = integer(57721566);
    r = div(sub(s, t), integer(100000000));
    // value of `r` is approximately 0.0000000049015328606065120900824024...

    eval_mpfr(a, *r, MPFR_RNDN);
    // Check that value of `r` (`a`) starts with 0.00000000490153
    REQUIRE(mpfr_cmp_d(a, 0.00000000490153) == 1);
    REQUIRE(mpfr_cmp_d(a, 0.00000000490154) == -1);

    s = mul(E, integer(100000));
    t = integer(271828);
    r = div(sub(s, t), integer(100000000));

    eval_mpfr(a, *r, MPFR_RNDN);
    // Check that value of `r` (`a`) starts with 0.00000000182845
    REQUIRE(mpfr_cmp_d(a, 0.00000000182845) == 1);
    REQUIRE(mpfr_cmp_d(a, 0.00000000182846) == -1);

    s = mul(Catalan, integer(100000000));
    t = integer(91596559);
    r = div(sub(s, t), integer(100000000));

    eval_mpfr(a, *r, MPFR_RNDN);
    // Check that value of `r` (`a`) starts with 0.000000004177219
    REQUIRE(mpfr_cmp_d(a, 0.000000004177219) == 1);
    REQUIRE(mpfr_cmp_d(a, 0.000000004177220) == -1);

    s = mul(GoldenRatio, integer(100000000));
    t = integer(161803398);
    r = div(sub(s, t), integer(100000000));

    eval_mpfr(a, *r, MPFR_RNDN);
    // Check that value of `r` (`a`) starts with 0.0000000087498948482
    REQUIRE(mpfr_cmp_d(a, 0.00000000874989) == 1);
    REQUIRE(mpfr_cmp_d(a, 0.00000000874990) == -1);

    RCP<const Basic> arg1 = integer(2);
    RCP<const Basic> arg2 = div(one, integer(4));
    RCP<const Basic> arg3 = div(minus_one, integer(4));

    std::vector<std::tuple<RCP<const Basic>, double, double>> testvec = {
        std::make_tuple(pow(E, integer(2)), 7.3890560989306, 7.38905609893066),
        std::make_tuple(max({integer(3), integer(2)}), 2.99999999999999,
                        3.0000000000001),
        std::make_tuple(max({sqrt(integer(3)), sqrt(integer(2))}),
                        1.73205080756, 1.73205080758),
        std::make_tuple(min({sqrt(integer(3)), sqrt(integer(2))}),
                        1.41421356236, 1.41421356238),
        std::make_tuple(loggamma(E), 0.44946174181, 0.44946174183),
        std::make_tuple(loggamma(integer(5)), 3.17805383033, 3.17805383035),
        std::make_tuple(log(arg1), 0.693147180559945, 0.693147180559946),
        std::make_tuple(erf(integer(2)), 0.995322265017, 0.995322265019),
        std::make_tuple(erf(div(E, pi)), 0.778918254986, 0.778918254988),
        std::make_tuple(erfc(integer(2)), 0.004677734981, 0.004677734983),
        std::make_tuple(floor(arg2), -0.000000000001, 0.000000000001),
        std::make_tuple(floor(arg3), -1.000000000001, -0.999999999999),
        std::make_tuple(ceiling(arg2), 0.999999999999, 1.000000000001),
        std::make_tuple(ceiling(arg3), -0.000000000001, 0.000000000001),
        std::make_tuple(truncate(arg2), -0.000000000001, 0.000000000001),
        std::make_tuple(truncate(arg3), -0.000000000001, 0.000000000001),
        std::make_tuple(sin(arg1), 0.90929742682568, 0.90929742682569),
        std::make_tuple(cos(arg1), -0.41614683654715, -0.41614683654714),
        std::make_tuple(tan(arg1), -2.1850398632616, -2.1850398632615),
        std::make_tuple(csc(arg1), 1.0997501702946, 1.0997501702947),
        std::make_tuple(sec(arg1), -2.4029979617224, -2.4029979617223),
        std::make_tuple(cot(arg1), -0.45765755436029, -0.45765755436028),
        std::make_tuple(asin(arg2), 0.25268025514207, 0.25268025514208),
        std::make_tuple(acos(arg2), 1.3181160716528, 1.3181160716529),
        std::make_tuple(atan(arg1), 1.1071487177940, 1.1071487177941),
        std::make_tuple(acsc(arg1), 0.52359877559829, 0.5235987755983),
        std::make_tuple(asec(arg1), 1.04719755119659, 1.04719755119660),
        std::make_tuple(acot(arg1), 0.4636476090008, 0.46364760900081),
        std::make_tuple(sinh(arg1), 3.6268604078470, 3.6268604078471),
        std::make_tuple(cosh(arg2), 1.0314130998795, 1.0314130998796),
        std::make_tuple(tanh(arg1), 0.96402758007581, 0.96402758007582),
        std::make_tuple(csch(arg1), 0.27572056477178, 0.27572056477179),
        std::make_tuple(sech(arg1), 0.2658022288340, 0.2658022288341),
        std::make_tuple(coth(arg2), 4.082988165073, 4.082988165074),
        std::make_tuple(asinh(arg1), 1.4436354751788, 1.4436354751789),
        std::make_tuple(acosh(arg1), 1.3169578969248, 1.3169578969249),
        std::make_tuple(atanh(arg2), 0.255412811882995, 0.255412811883),
        std::make_tuple(acsch(arg2), 2.094712547261, 2.094712547262),
        std::make_tuple(acoth(arg1), 0.54930614433405, 0.54930614433406),
        std::make_tuple(asech(arg2), 2.0634370688955, 2.0634370688956),
        std::make_tuple(atan2(arg1, arg2), 1.4464413322481, 1.4464413322482),
        std::make_tuple(asech(div(one, integer(3))), 1.76274717403908,
                        1.76274717403909),
        std::make_tuple(gamma(div(arg1, integer(3))), 1.35411793942640,
                        1.35411793942641),
        std::make_tuple(gamma(arg2), 3.62560990822190, 3.62560990822191),
        std::make_tuple(gamma(add(arg1, arg2)), 1.13300309631934,
                        1.13300309631935),
        std::make_tuple(gamma(add(add(arg1, arg2), arg1)), 8.28508514183522,
                        8.28508514183523),
#if MPFR_VERSION_MAJOR > 3
        std::make_tuple(uppergamma(sqrt(integer(2)), arg2), 0.80040012955715,
                        0.80040012955716),
        std::make_tuple(lowergamma(sqrt(integer(2)), arg2), 0.08618129916210,
                        0.08618129916211),
#endif
        std::make_tuple(beta(add(arg1, arg2), arg1), 0.13675213675213,
                        0.13675213675214),
        std::make_tuple(abs((neg(sqrt(add(arg2, arg2))))), 0.70710678118654,
                        0.70710678118655)
    };

    for (unsigned i = 0; i < testvec.size(); i++) {
        eval_mpfr(a, *std::get<0>(testvec[i]), MPFR_RNDN);
        REQUIRE(mpfr_cmp_d(a, std::get<1>(testvec[i])) == 1);
        REQUIRE(mpfr_cmp_d(a, std::get<2>(testvec[i])) == -1);
    }

    CHECK_THROWS_AS(eval_mpfr(a, *constant("dummy_constant"), MPFR_RNDN),
                    NotImplementedError);

    CHECK_THROWS_AS(eval_mpfr(a, *lambertw(arg1), MPFR_RNDN),
                    NotImplementedError);

    mpfr_clear(a);
}
