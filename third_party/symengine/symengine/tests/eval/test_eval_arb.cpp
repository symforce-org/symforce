#include "catch.hpp"
#include <limits>
#include <cstdlib>
#include <ctime>

#include <symengine/rational.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/eval_arb.h>
#include <symengine/eval_mpfr.h>
#include <symengine/eval_double.h>
#include <symengine/symengine_exception.h>

using SymEngine::SymEngineException;
using SymEngine::RCP;
using SymEngine::Basic;
using SymEngine::integer;
using SymEngine::eval_mpfr;
using SymEngine::div;
using SymEngine::Rational;
using SymEngine::add;
using SymEngine::sqrt;
using SymEngine::mul;
using SymEngine::pow;
using SymEngine::sin;
using SymEngine::cos;
using SymEngine::tan;
using SymEngine::csc;
using SymEngine::sec;
using SymEngine::cot;
using SymEngine::log;
using SymEngine::asin;
using SymEngine::acos;
using SymEngine::atan;
using SymEngine::acsc;
using SymEngine::asec;
using SymEngine::acot;
using SymEngine::asinh;
using SymEngine::acosh;
using SymEngine::atanh;
using SymEngine::acoth;
using SymEngine::asech;
using SymEngine::gamma;
using SymEngine::zeta;
using SymEngine::abs;
using SymEngine::E;
using SymEngine::EulerGamma;
using SymEngine::eval_arb;
using SymEngine::print_stack_on_segfault;
using SymEngine::min;
using SymEngine::max;
using SymEngine::loggamma;
using SymEngine::eval_double;

TEST_CASE("Integer: eval_arb", "[eval_arb]")
{
    arb_t a, b;
    arb_init(a);
    arb_init(b);

    eval_arb(a, *integer(3));
    arb_set_ui(b, 3);

    REQUIRE(arb_equal(a, b));

    eval_arb(a, *integer(-45));
    arb_set_si(b, -45);

    REQUIRE(arb_equal(a, b));

    srand(time(nullptr));
    unsigned int ui = rand();

    RCP<const Basic> i = integer(ui);
    eval_arb(a, *i);
    arb_set_ui(b, ui);

    REQUIRE(arb_equal(a, b));
    arb_clear(a);
    arb_clear(b);
}

TEST_CASE("Rational: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> q = div(integer(22), integer(7));
    eval_arb(a, *q, 4);

    REQUIRE(arb_contains_si(a, 3) != 0);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, 22.0 / 7, MPFR_RNDN);

    eval_arb(a, *q, 10);

    REQUIRE(arb_contains_mpfr(a, f));

    srand(time(nullptr));
    int x_ = rand();
    srand(time(nullptr) + 1000);
    int y_ = rand() - RAND_MAX / 2;
    if (y_ == 0) {
        y_ = RAND_MAX / 2;
    }

    RCP<const Basic> x = integer(x_);
    RCP<const Basic> y = integer(y_);
    q = div(x, y);
    eval_arb(a, *q, 10);

    mpfr_set_d(f, (double)x_ / y_, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Add: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> q = div(integer(223), integer(71));
    RCP<const Basic> r1 = add(sqrt(integer(3)), q);
    eval_arb(a, *r1, 12);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, sqrt(3) + 223.0 / 71, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    q = div(integer(34), integer(7100));
    r1 = add(sqrt(integer(7)), q);
    eval_arb(a, *r1, 10);

    mpfr_set_d(f, sqrt(7) + 34.0 / 7100, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Mul: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> q = div(integer(377), integer(120));
    RCP<const Basic> r1 = mul(sqrt(integer(15)), q);
    eval_arb(a, *r1, 11);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, sqrt(15) * 377.0 / 120, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    q = div(integer(9001), integer(71));
    r1 = mul(sqrt(integer(2317)), q);
    eval_arb(a, *r1, 8);

    mpfr_set_d(f, sqrt(2317) * 9001.0 / 71, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Pow: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> b = div(integer(377), integer(120));
    RCP<const Basic> e = div(integer(34), integer(1257));
    RCP<const Basic> r1 = pow(b, e);
    eval_arb(a, *r1, 8);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, pow(377.0 / 120, 34.0 / 1257), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    b = div(integer(120), integer(377));
    e = div(integer(97), integer(34));
    r1 = pow(b, e);
    eval_arb(a, *r1, 10);

    mpfr_set_d(f, pow(120.0 / 377, 97.0 / 34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = pow(E, e);
    eval_arb(a, *r1, 12);

    mpfr_set_d(f, exp(97.0 / 34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Sin: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = sin(integer(2));
    eval_arb(a, *r1, 10);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, sin(2), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    RCP<const Basic> q = div(integer(3), integer(5));
    r1 = add(sin(integer(2)), sin(q)); // r1 = sin(2) + sin(3/5)
    eval_arb(a, *r1, 10);

    mpfr_set_d(f, sin(2) + sin(3.0 / 5), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Cos: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = cos(integer(-1));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, cos(-1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    RCP<const Basic> q = div(integer(7), integer(3));
    r1 = mul(cos(integer(-1)), cos(q)); // r1 = cos(-1)*cos(7/3)
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, cos(-1) * cos(7.0 / 3), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Tan: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = tan(integer(1239));
    eval_arb(a, *r1, 10);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, tan(1239), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    // r1 = tan(2) + tan(3)*tan(7)
    r1 = add(tan(integer(2)), mul(tan(integer(3)), tan(integer(7))));
    eval_arb(a, *r1, 14);

    mpfr_set_d(f, tan(2) + tan(3) * tan(7), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Csc: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = csc(integer(2));
    eval_arb(a, *r1, 10);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, 1 / sin(2), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    RCP<const Basic> q = div(integer(3), integer(5));
    r1 = add(csc(integer(2)), csc(q)); // r1 = csc(2) + csc(3/5)
    eval_arb(a, *r1, 10);

    mpfr_set_d(f, 1 / sin(2) + 1 / sin(3.0 / 5), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Sec: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = sec(integer(-1));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, 1 / cos(-1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    RCP<const Basic> q = div(integer(7), integer(3));
    r1 = mul(sec(integer(-1)), sec(q)); // r1 = sec(-1)*sec(7/3)
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, 1 / cos(-1) * 1 / cos(7.0 / 3), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Cot: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = cot(integer(1239));
    eval_arb(a, *r1, 10);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, 1 / tan(1239), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    // r1 = cot(2) + cot(3)*cot(7)
    r1 = add(cot(integer(2)), mul(cot(integer(3)), cot(integer(7))));
    eval_arb(a, *r1, 14);

    mpfr_set_d(f, 1 / tan(2) + 1 / tan(3) * 1 / tan(7), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Log: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = log(integer(3498));
    eval_arb(a, *r1, 10);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, log(3498), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = add(log(integer(2)), log(div(integer(3), integer(73))));
    eval_arb(a, *r1, 14);

    mpfr_set_d(f, log(2) + log(3.0 / 73), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Asin: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = asin(div(integer(1), integer(2)));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, asin(0.5), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = asin(integer(1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, asin(1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Acos: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = acos(div(sqrt(integer(3)), integer(2)));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, acos(sqrt(3) / 2), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acos(integer(-1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, acos(-1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Asec: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = asec(integer(23));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, acos(1.0 / 23), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = asec(integer(-1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, acos(1.0 / -1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = asec(integer(1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, acos(1.0 / 1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("ACsc: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = acsc(integer(-34));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, asin(1.0 / -34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acsc(integer(-1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, asin(1.0 / -1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acsc(integer(16));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, asin(1.0 / 16), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("ATan: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = atan(sqrt(integer(3)));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, atan(sqrt(3)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = atan(integer(-1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, atan(-1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("ACot: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = acot(integer(-34));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, atan(1.0 / -34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acot(integer(591));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, atan(1.0 / 591), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acot(integer(16));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, atan(1.0 / 16), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("ATan2: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = atan2(sqrt(integer(3)), integer(1));
    eval_arb(a, *r1, 15);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, atan2(sqrt(3), 1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = atan2(integer(-1), sqrt(integer(3)));
    eval_arb(a, *r1, 15);

    mpfr_set_d(f, atan2(-1, sqrt(3)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Sinh: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = sinh(integer(3));
    eval_arb(a, *r1, 15);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, sinh(3), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = sinh(mul(integer(-5), sqrt(integer(3))));
    eval_arb(a, *r1, 15);

    mpfr_set_d(f, sinh(-5 * sqrt(3)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Cosh: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = cosh(integer(123));
    eval_arb(a, *r1, 15);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, cosh(123), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = cosh(div(integer(5), sqrt(integer(3))));
    eval_arb(a, *r1, 15);

    mpfr_set_d(f, cosh(5.0 / sqrt(3)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Tanh: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = tanh(integer(-34));
    eval_arb(a, *r1, 15);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, tanh(-34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = tanh(add(integer(5), sqrt(integer(3))));
    eval_arb(a, *r1, 15);

    mpfr_set_d(f, tanh(5 + sqrt(3)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Coth: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = coth(integer(40));
    eval_arb(a, *r1, 15);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, -sinh(2 * 40) / (1 - cosh(2 * 40)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = coth(integer(5));
    eval_arb(a, *r1, 15);

    mpfr_set_d(f, -sinh(2 * 5) / (1 - cosh(2 * 5)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Constants: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = mul(EulerGamma, integer(100000000));
    RCP<const Basic> r2 = div(sub(r1, integer(57721566)), integer(100000000));

    eval_arb(a, *r1, 45);

    mpfr_t f;
    mpfr_init2(f, 57);
    eval_mpfr(f, *r1, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Gamma: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = gamma(integer(4));
    RCP<const Basic> r2 = gamma(div(integer(5), integer(2)));

    eval_arb(a, *r1, 45);

    mpfr_t f;
    mpfr_init2(f, 57);
    eval_mpfr(f, *r1, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    eval_arb(a, *r2, 45);
    eval_mpfr(f, *r2, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("LogGamma: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = loggamma(E);
    RCP<const Basic> r2 = loggamma(EulerGamma);

    eval_arb(a, *r1, 45);

    mpfr_t f;
    mpfr_init2(f, 57);
    eval_mpfr(f, *r1, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    eval_arb(a, *r2, 45);
    eval_mpfr(f, *r2, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Min/Max: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> res;
    RCP<const Basic> r1 = div(integer(3), integer(5));
    RCP<const Basic> r2 = sub(r1, div(integer(57721566), integer(100000000)));

    res = max({r1, r2});
    eval_arb(a, *res, 45);

    mpfr_t f;
    mpfr_init2(f, 57);
    eval_mpfr(f, *res, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    res = min({r1, r2});
    eval_arb(a, *res, 45);

    mpfr_init2(f, 57);
    eval_mpfr(f, *res, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Asinh: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = asinh(div(integer(1), integer(2)));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, asinh(0.5), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = asinh(integer(1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, asinh(1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Acosh: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = acosh(div(integer(1), integer(2)));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, acosh(2), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acosh(integer(1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, acosh(1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Atanh: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = atanh(div(integer(1), integer(2)));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, atanh(0.5), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = atanh(integer(1));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, atanh(1), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("ACoth: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = acoth(integer(-34));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, atanh(1.0 / -34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acoth(integer(591));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, atanh(1.0 / 591), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = acoth(integer(16));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, atanh(1.0 / 16), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Asech: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = asech(integer(1));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, acosh(1.0), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = asech(div(integer(1), integer(2)));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, acosh(2.0), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Abs: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = abs(integer(-34));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, abs(-34), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = abs(integer(16));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, abs(16), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}

TEST_CASE("Zeta: eval_arb", "[eval_arb]")
{
    arb_t a;
    arb_init(a);

    RCP<const Basic> r1 = zeta(integer(-34));
    eval_arb(a, *r1, 13);

    mpfr_t f;
    mpfr_init2(f, 17);
    mpfr_set_d(f, eval_double(*integer(0)), MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = zeta(div(integer(1), integer(678)));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, -0.501357552105367, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));

    r1 = zeta(div(integer(1), integer(-247)));
    eval_arb(a, *r1, 13);

    mpfr_set_d(f, -0.496295978053852, MPFR_RNDN);

    REQUIRE(arb_contains_mpfr(a, f));
    mpfr_clear(f);
    arb_clear(a);
}