#include "catch.hpp"
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/eval_mpc.h>
#include <symengine/eval_mpfr.h>
#include <symengine/symengine_exception.h>
#include <symengine/real_double.h>
#include <symengine/constants.h>

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
using SymEngine::Catalan;
using SymEngine::constant;
using SymEngine::cos;
using SymEngine::cosh;
using SymEngine::cot;
using SymEngine::coth;
using SymEngine::csc;
using SymEngine::csch;
using SymEngine::E;
using SymEngine::EulerGamma;
using SymEngine::eval_mpc;
using SymEngine::gamma;
using SymEngine::GoldenRatio;
using SymEngine::I;
using SymEngine::integer;
using SymEngine::mul;
using SymEngine::NotImplementedError;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::pow;
using SymEngine::print_stack_on_segfault;
using SymEngine::RCP;
using SymEngine::real_double;
using SymEngine::sec;
using SymEngine::sech;
using SymEngine::sin;
using SymEngine::sinh;
using SymEngine::sub;
using SymEngine::SymEngineException;
using SymEngine::tan;
using SymEngine::tanh;

TEST_CASE("eval: eval_mpc", "[eval_mpc]")
{
    mpc_t a, b;
    mpfr_t real, imag;
    mpc_init2(a, 100);
    mpc_init2(b, 100);
    mpfr_init2(real, 100);
    mpfr_init2(imag, 100);
    RCP<const Basic> s = add(one, cos(integer(2)));
    RCP<const Basic> t = sin(integer(2));
    RCP<const Basic> r = add(pow(E, mul(integer(2), I)), one);
    RCP<const Basic> arg1 = add(integer(2), mul(integer(3), I));
    RCP<const Basic> arg2 = add(integer(4), mul(integer(5), I));
    eval_mpc(a, *r, MPFR_RNDN);
    eval_mpfr(real, *s, MPFR_RNDN);
    eval_mpfr(imag, *t, MPFR_RNDN);

    mpc_set_fr_fr(b, real, imag, MPFR_RNDN);

    REQUIRE(mpc_cmp(a, b) == 0);

    std::vector<std::tuple<RCP<const Basic>, double, double>> testvec = {
        std::make_tuple(sin(arg1), 10.0590576035560, 10.0590576035561),
        std::make_tuple(cos(arg1), 10.0265146611769, 10.0265146611770),
        std::make_tuple(tan(arg1), 1.00324568840508, 1.00324568840509),
        std::make_tuple(csc(arg1), 0.09941289128779, 0.09941289128780),
        std::make_tuple(sec(arg1), 0.09973555455636, 0.09973555455637),
        std::make_tuple(cot(arg1), 0.99676481200707, 0.99676481200708),
        std::make_tuple(asin(arg1), 2.06384803478709, 2.06384803478710),
        std::make_tuple(acos(arg2), 2.70606901402754, 2.70606901402755),
        std::make_tuple(atan(arg1), 1.42840878608958, 1.42840878608959),
        std::make_tuple(acsc(arg1), 0.275919504119167, 0.275919504119168),
        std::make_tuple(asec(arg1), 1.439125555072813, 1.439125555072814),
        std::make_tuple(acot(arg2), 0.156440457398915, 0.156440457398916),
        std::make_tuple(sinh(arg1), 3.629604837263012, 3.629604837263013),
        std::make_tuple(cosh(arg2), 27.2913914057446, 27.2913914057447),
        std::make_tuple(tanh(arg1), 0.9654364796739529, 0.9654364796739530),
        std::make_tuple(csch(arg1), 0.275512085980707, 0.275512085980708),
        std::make_tuple(sech(arg1), 0.2659894183968419, 0.2659894183968420),
        std::make_tuple(coth(arg2), 0.999437204152625, 0.999437204152626),
        std::make_tuple(asinh(arg1), 2.19228221563667, 2.19228221563668),
        std::make_tuple(acosh(arg1), 2.22128593746801, 2.22128593746802),
        std::make_tuple(atanh(arg2), 1.45151270206482, 1.45151270206483),
        std::make_tuple(acsch(arg2), 0.156308000814648, 0.156308000814649),
        std::make_tuple(acoth(arg2), 0.155883315867942, 0.155883315867943),
        std::make_tuple(asech(arg1), 1.43912555507281, 1.43912555507282),
        std::make_tuple(log(arg1), 1.615742802564794, 1.615742802564795),
        std::make_tuple(abs(mul(arg1, arg2)), 23.086792761230, 23.086792761231),
        std::make_tuple(abs(arg1), 3.605551275463989, 3.605551275463990),
    };

    for (unsigned i = 0; i < testvec.size(); i++) {
        eval_mpc(a, *std::get<0>(testvec[i]), MPFR_RNDN);
        mpc_abs(real, a, MPFR_RNDN);
        REQUIRE(mpfr_cmp_d(real, std::get<1>(testvec[i])) == 1);
        REQUIRE(mpfr_cmp_d(real, std::get<2>(testvec[i])) == -1);
    }

    r = add(one, mul(EulerGamma, I));
    s = one;
    t = EulerGamma;

    eval_mpc(a, *r, MPFR_RNDN);
    eval_mpfr(real, *s, MPFR_RNDN);
    eval_mpfr(imag, *t, MPFR_RNDN);

    mpc_set_fr_fr(b, real, imag, MPFR_RNDN);

    REQUIRE(mpc_cmp(a, b) == 0);

    r = add(one, mul(Catalan, I));
    s = one;
    t = Catalan;

    eval_mpc(a, *r, MPFR_RNDN);
    eval_mpfr(real, *s, MPFR_RNDN);
    eval_mpfr(imag, *t, MPFR_RNDN);

    mpc_set_fr_fr(b, real, imag, MPFR_RNDN);

    REQUIRE(mpc_cmp(a, b) == 0);

    r = add(one, mul(GoldenRatio, I));
    s = one;
    t = GoldenRatio;

    eval_mpc(a, *r, MPFR_RNDN);
    eval_mpfr(real, *s, MPFR_RNDN);
    eval_mpfr(imag, *t, MPFR_RNDN);

    mpc_set_fr_fr(b, real, imag, MPFR_RNDN);

    REQUIRE(mpc_cmp(a, b) == 0);

    r = add(one, mul(E, I));
    s = one;
    t = E;

    eval_mpc(a, *r, MPFR_RNDN);
    eval_mpfr(real, *s, MPFR_RNDN);
    eval_mpfr(imag, *t, MPFR_RNDN);

    mpc_set_fr_fr(b, real, imag, MPFR_RNDN);

    REQUIRE(mpc_cmp(a, b) == 0);

    CHECK_THROWS_AS(eval_mpc(a, *constant("dummy_constant"), MPFR_RNDN),
                    NotImplementedError);
    CHECK_THROWS_AS(eval_mpc(a, *gamma(arg1), MPFR_RNDN), NotImplementedError);

    r = erf(add(one, mul(integer(2), I)));
    CHECK_THROWS_AS(eval_mpc(a, *r, MPFR_RNDN), NotImplementedError);

    r = erfc(add(one, mul(integer(2), I)));
    CHECK_THROWS_AS(eval_mpc(a, *r, MPFR_RNDN), NotImplementedError);

    mpfr_clear(real);
    mpfr_clear(imag);
    mpc_clear(a);
    mpc_clear(b);
}
