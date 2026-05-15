#include "catch.hpp"
#include "symengine/constants.h"
#include "symengine/nan.h"
#include <chrono>
#include <array>

#include <symengine/lambda_double.h>
#include <symengine/symengine_exception.h>
#include <symengine/eval.h>
#include <symengine/rational.h>

#ifdef HAVE_SYMENGINE_LLVM
#include <symengine/llvm_double.h>
#include <symengine/eval_mpfr.h>
using SymEngine::LLVMDoubleVisitor;
using SymEngine::LLVMFloatVisitor;

#ifdef HAVE_SYMENGINE_MPFR
using SymEngine::RealMPFR;
#endif
#endif

using SymEngine::acos;
using SymEngine::acosh;
using SymEngine::acot;
using SymEngine::acoth;
using SymEngine::acsc;
using SymEngine::acsch;
using SymEngine::add;
using SymEngine::asec;
using SymEngine::asech;
using SymEngine::asin;
using SymEngine::asinh;
using SymEngine::atan;
using SymEngine::atan2;
using SymEngine::atanh;
using SymEngine::Basic;
using SymEngine::boolTrue;
using SymEngine::Catalan;
using SymEngine::ceiling;
using SymEngine::complex_double;
using SymEngine::cos;
using SymEngine::cosh;
using SymEngine::cot;
using SymEngine::coth;
using SymEngine::csc;
using SymEngine::csch;
using SymEngine::down_cast;
using SymEngine::E;
using SymEngine::Eq;
using SymEngine::evalf;
using SymEngine::floor;
using SymEngine::gamma;
using SymEngine::Inf;
using SymEngine::integer;
using SymEngine::LambdaComplexDoubleVisitor;
using SymEngine::LambdaRealDoubleVisitor;
using SymEngine::Le;
using SymEngine::log;
using SymEngine::loggamma;
using SymEngine::logical_and;
using SymEngine::Lt;
using SymEngine::map_basic_basic;
using SymEngine::max;
using SymEngine::min;
using SymEngine::mul;
using SymEngine::Nan;
using SymEngine::Ne;
using SymEngine::NegInf;
using SymEngine::NotImplementedError;
using SymEngine::pi;
using SymEngine::piecewise;
using SymEngine::pow;
using SymEngine::rational;
using SymEngine::RCP;
using SymEngine::real_double;
using SymEngine::sec;
using SymEngine::sech;
using SymEngine::sign;
using SymEngine::sin;
using SymEngine::sinh;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::tan;
using SymEngine::tanh;
using SymEngine::truncate;
using SymEngine::vec_basic;
using SymEngine::zeta;

TEST_CASE("Evaluate to double", "[lambda_double]")
{
    RCP<const Basic> x, y, z, r;
    double d;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    r = add(x, add(mul(y, z), pow(x, integer(2))));

    LambdaRealDoubleVisitor v;
    v.init({x, y, z}, *r);

    d = v.call({1.5, 2.0, 3.0});
    REQUIRE(::fabs(d - 9.75) < 1e-12);

    d = v.call({1.5, -1.0, 2.0});
    REQUIRE(::fabs(d - 1.75) < 1e-12);

    r = max({x, add(mul(y, z), integer(3))});
    v.init({x, y, z}, *r);

    d = v.call({4.0, 1.0, 2.5});
    REQUIRE(::fabs(d - 5.5) < 1e-12);

    r = min({pow(x, y), add(mul(y, z), integer(3))});
    v.init({x, y, z}, *r);

    d = v.call({4.0, 2.0, 2.5});
    REQUIRE(::fabs(d - 8.0) < 1e-12);

    // Evaluating to double when there are complex doubles raise an exception
    CHECK_THROWS_AS(
        v.init({x}, *add(complex_double(std::complex<double>(1, 2)), x)),
        NotImplementedError);

    // Undefined symbols raise an exception
    CHECK_THROWS_AS(v.init({x}, *r), SymEngineException);

    // Piecewise
    auto int1 = interval(NegInf, integer(2), true, false);
    auto int2 = interval(integer(2), integer(5), true, false);
    r = piecewise({{x, contains(x, int1)},
                   {y, contains(x, int2)},
                   {add(x, y), boolTrue}});
    v.init({x, y}, *r);
    d = v.call({1.1, 3.3});
    REQUIRE(::fabs(d - 1.1) < 1e-12);
    d = v.call({2.2, 3.3});
    REQUIRE(::fabs(d - 3.3) < 1e-12);
    d = v.call({5.5, 3.3});
    REQUIRE(::fabs(d - 8.8) < 1e-12);
}

TEST_CASE("Evaluate double cse", "[lambda_double_cse]")
{
    RCP<const Basic> x, y, z, r, s;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    r = add(x, add(mul(y, z), pow(mul(y, z), integer(2))));
    s = add(mul(integer(2), x), add(mul(y, z), pow(mul(y, z), integer(2))));

    LambdaRealDoubleVisitor v;
    v.init({x, y, z}, {r, s}, true);

    double d[2];
    double inps[] = {1.5, 2.0, 3.0};
    v.call(d, inps);
    REQUIRE(::fabs(d[0] - 43.5) < 1e-12);
    REQUIRE(::fabs(d[1] - 45.0) < 1e-12);
}

TEST_CASE("LambdaRealDoubleVisitor with cse can be moved",
          "[lambda_double_cse]")
{
    RCP<const Basic> x, y, z, r, s;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    r = add(x, add(mul(y, z), pow(mul(y, z), integer(2))));
    s = add(mul(integer(2), x), add(mul(y, z), pow(mul(y, z), integer(2))));

    double d[2];
    double inps[] = {1.5, 2.0, 3.0};
    LambdaRealDoubleVisitor v2;
    {
        LambdaRealDoubleVisitor v;
        v.init({x, y, z}, {r, s}, true);
        v.call(d, inps);
        v.call(d, inps);
        REQUIRE(::fabs(d[0] - 43.5) < 1e-12);
        REQUIRE(::fabs(d[1] - 45.0) < 1e-12);
        v2 = std::move(v); // Move-construct another visitor
    }
    v2.call(d, inps);
    REQUIRE(::fabs(d[0] - 43.5) < 1e-12);
    REQUIRE(::fabs(d[1] - 45.0) < 1e-12);
}

TEST_CASE("Evaluate to std::complex<double>", "[lambda_complex_double]")
{
    RCP<const Basic> x, y, z, r;
    std::complex<double> d;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    r = add(x,
            add(mul(y, z), pow(x, complex_double(std::complex<double>(3, 4)))));

    LambdaComplexDoubleVisitor v;
    v.init({x, y, z}, *r);

    d = v.call({std::complex<double>(1.5, 1.0), std::complex<double>(2.5, 4.0),
                std::complex<double>(-8.3, 3.2)});
    REQUIRE(::fabs(d.real() + 32.360749607381) < 1e-12);
    REQUIRE(::fabs(d.imag() + 24.6630395370884) < 1e-12);

    v.init({x, y, z}, *add(x, add(mul(y, z), pow(x, integer(2)))));
    d = v.call({std::complex<double>(1.5, 0.0), std::complex<double>(-1.0, 0.0),
                std::complex<double>(2.0, 0.0)});
    REQUIRE(::fabs(d.real() - 1.75) < 1e-12);
    REQUIRE(::fabs(d.imag() - 0.0) < 1e-12);

    // Undefined symbols raise an exception
    CHECK_THROWS_AS(v.init({x}, *r), SymEngineException);
}

TEST_CASE("Evaluate functions", "[lambda_gamma]")
{
    RCP<const Basic> x, y, z, r;
    double d;
    x = symbol("x");

    LambdaRealDoubleVisitor v;

    std::vector<std::tuple<RCP<const Basic>, double, double>> testvec = {
        std::make_tuple(pow(E, cos(x)), 1.3, 1.30669209920819),
        std::make_tuple(add(sin(x), cos(x)), 1.2, 1.29439684044390),
        std::make_tuple(mul(tan(x), sec(x)), 2.2, 2.33444426269917),
        std::make_tuple(add(csc(x), cot(x)), 0.5, 3.91631736464594),
        std::make_tuple(asin(x), 0.5, 0.523598775598299),
        std::make_tuple(acos(x), 0.9, 0.451026811796262),
        std::make_tuple(add(atan(x), asec(x)), 1.1, 1.26268093282586),
        std::make_tuple(add(acot(x), acsc(x)), 3, 0.661587463850764),
        std::make_tuple(add(sinh(x), csch(x)), 2.2, 4.68146604193974),
        std::make_tuple(mul(tanh(x), cosh(x)), 0.2, 0.201336002541094),
        std::make_tuple(sech(x), 0.5, 0.886818883970074),
        std::make_tuple(coth(x), 0.9, 1.39606725303001),
        std::make_tuple(mul(asinh(x), acosh(x)), 1.2, 0.632303583495024),
        std::make_tuple(mul(acsch(x), asech(x)), 0.3, 3.59566705273267),
        std::make_tuple(acoth(x), 3.3, 0.312852949882206),
        std::make_tuple(atanh(x), 0.7, 0.867300527694053),
        std::make_tuple(Eq(real_double(2.0), x), 2.0, 1.000000000000),
        std::make_tuple(Eq(real_double(3.0), x), 2.0, 0.000000000000),
        std::make_tuple(Ne(real_double(2.0), x), 2.0, 0.000000000000),
        std::make_tuple(Ne(real_double(3.0), x), 2.0, 1.000000000000),
        std::make_tuple(Le(real_double(2.0), x), 2.0, 1.000000000000),
        std::make_tuple(Le(real_double(3.0), x), 2.0, 0.000000000000),
        std::make_tuple(Le(real_double(2.0), x), 2.0000000000001,
                        1.000000000000),
        std::make_tuple(Lt(real_double(3.0), x), 2.0, 0.000000000000),
        std::make_tuple(Lt(real_double(2.0), x), 2.0000000000001,
                        1.000000000000),
        std::make_tuple(atan2(x, add(x, integer(1))), 2.1, 0.595409875478733),
        std::make_tuple(log(x), 1.2, 0.182321556793955),
        std::make_tuple(gamma(x), 1.1, 0.9513507698668),
        std::make_tuple(add(Catalan, x), 1.1, 2.01596559417722),
        std::make_tuple(loggamma(x), 1.3, -0.10817480950786047),
        std::make_tuple(add(gamma(x), loggamma(x)), 1.1, 0.901478328607033459),
        std::make_tuple(abs(x), -1.112111321, 1.112111321),
        std::make_tuple(erf(x), 1.1, 0.88020506957408169),
        std::make_tuple(erfc(x), 2.1, 0.00297946665633298),
    };

    for (unsigned i = 0; i < testvec.size(); i++) {
        RCP<const Basic> expr1 = std::get<0>(testvec[i]);
        RCP<const Basic> expr2
            = Basic::loads(expr1->dumps()); // test serialization
        const auto arg = std::get<1>(testvec[i]);
        const auto ref = std::get<2>(testvec[i]);
        std::array<RCP<const Basic>, 2> exprs{{expr1, expr2}};
        for (auto expr : exprs) {
            v.init({x}, *expr);
            d = v.call({arg});
            REQUIRE(::fabs(d - ref) < 1e-12);
        }
    }
    v.init({}, *Nan);
    REQUIRE(std::isnan(v.call({})));
    v.init({}, *Inf);
    std::isinf(v.call({}));
    v.init({}, *NegInf);
    REQUIRE(std::isinf(v.call({})));
}

#ifdef HAVE_SYMENGINE_LLVM
TEST_CASE("Check llvm and lambda are equal", "[llvm_double]")
{

    RCP<const Basic> x, y, z, r, a, b;
    double d, d2, d3;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    a = add(x, z);
    b = add(y, z);

    vec_basic exprs = {
        log(a),   abs(a),      tan(a),      sinh(a),     cosh(a),    tanh(a),
        asinh(b), acosh(b),    atanh(a),    asin(a),     acos(a),    atan(a),
        gamma(a), loggamma(a), erf(a),      erfc(a),     floor(a),   ceiling(a),
        sign(a),  max({a, b}), min({a, b}), atan2(a, b), truncate(a)};

    for (unsigned i = 0; i < exprs.size(); i++) {
        exprs[i] = add(exprs[i], z);
    }

    r = add(sin(x), add(mul(pow(y, integer(4)), mul(z, integer(2))),
                        pow(sin(x), integer(2))));
    exprs.push_back(r);
    exprs.push_back(neg(abs(z)));

    // Piecewise
    auto int1 = interval(NegInf, integer(2), true, false);
    auto int2 = interval(integer(2), integer(5), true, false);

    SymEngine::set_boolean s = {Lt(x, integer(6)), Gt(x, integer(5))};
    r = add(z, piecewise({{x, contains(x, int1)},
                          {y, contains(x, int2)},
                          {z, Ge(x, integer(7))},
                          {a, logical_and(s)},
                          {add(x, y), boolTrue}}));
    exprs.push_back(r);

    for (auto &expr : exprs) {
        LambdaRealDoubleVisitor v;
        v.init({x, y, z}, *expr);

        LLVMDoubleVisitor v2;
        v2.init({x, y, z}, *expr);

        LLVMDoubleVisitor v3;
        bool symbolic_cse = true;
        int opt_level = 3;
        v3.init({x, y, z}, *expr, symbolic_cse, opt_level);

        d = v.call({1.4, 3.0, -1.0});
        d2 = v2.call({1.4, 3.0, -1.0});
        d3 = v3.call({1.4, 3.0, -1.0});
        REQUIRE(::fabs((d - d2)) < 1e-12);
        REQUIRE(::fabs((d - d3)) < 1e-12);
    }
    {
        double out[2];
        LLVMDoubleVisitor v;
        bool symbolic_cse = false;
        int opt_level = 0;
        v.init({}, {Nan, Inf}, symbolic_cse, opt_level);
        v.call(out, {});
        REQUIRE(std::isnan(out[0]));
        REQUIRE(std::isinf(out[1]));
    }
}

TEST_CASE("Check llvm with opt_level 0-3 is equal to llvm without opt_level",
          "[llvm_double]")
{

    RCP<const Basic> x, y, z, r, a, b;
    double d, d2;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    a = add(x, z);
    b = add(y, z);

    vec_basic exprs = {
        log(a),   abs(a),      tan(a),      sinh(a),     cosh(a),    tanh(a),
        asinh(b), acosh(b),    atanh(a),    asin(a),     acos(a),    atan(a),
        gamma(a), loggamma(a), erf(a),      erfc(a),     floor(a),   ceiling(a),
        sign(a),  max({a, b}), min({a, b}), atan2(a, b), truncate(a)};

    for (unsigned i = 0; i < exprs.size(); i++) {
        exprs[i] = add(exprs[i], z);
    }

    r = add(sin(x), add(mul(pow(y, integer(4)), mul(z, integer(2))),
                        pow(sin(x), integer(2))));
    exprs.push_back(r);
    exprs.push_back(neg(abs(z)));

    // Piecewise
    auto int1 = interval(NegInf, integer(2), true, false);
    auto int2 = interval(integer(2), integer(5), true, false);

    SymEngine::set_boolean s = {Lt(x, integer(6)), Gt(x, integer(5))};
    r = add(z, piecewise({{x, contains(x, int1)},
                          {y, contains(x, int2)},
                          {z, Ge(x, integer(7))},
                          {a, logical_and(s)},
                          {add(x, y), boolTrue}}));
    exprs.push_back(r);

    bool symbolic_cse = true;
    for (auto &expr : exprs) {
        LLVMDoubleVisitor v;
        v.init({x, y, z}, *expr, symbolic_cse);

        for (int opt_level = 0; opt_level < 4; ++opt_level) {
            LLVMDoubleVisitor v2;
            v2.init({x, y, z}, *expr, symbolic_cse, opt_level);

            d = v.call({1.4, 3.0, -1.0});
            d2 = v2.call({1.4, 3.0, -1.0});
            REQUIRE(::fabs((d - d2)) < 1e-12);
        }
    }
}

TEST_CASE("Check llvm save and load", "[llvm_double]")
{
    RCP<const Basic> x, y, z, r;
    double d, d2, d3;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    r = add(sin(x), add(mul(pow(y, integer(4)), mul(z, integer(2))),
                        pow(sin(x), integer(2))));

    for (int i = 0; i < 4; ++i) {
        r = mul(add(pow(integer(2), E), add(r, pow(x, pow(E, cos(x))))), r);
    }

    LLVMDoubleVisitor v;
    auto t1 = std::chrono::high_resolution_clock::now();
    v.init({x, y, z}, *r);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Initializing "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count()
              << "us" << std::endl;

    LLVMDoubleVisitor v2;

    t1 = std::chrono::high_resolution_clock::now();
    auto &s = v.dumps();
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Saving "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count()
              << "us" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    v2.loads(s);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Loading "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                     .count()
              << "us" << std::endl;

    d = v.call({0.4, 2.0, 3.0});
    d2 = v2.call({0.4, 2.0, 3.0});
    REQUIRE(::fabs((d - d2) / d) < 1e-12);

    // Test that dumping and loading on a loaded object also works
    auto &s2 = v2.dumps();
    LLVMDoubleVisitor v3;
    v3.loads(s2);

    d3 = v3.call({0.4, 2.0, 3.0});
    REQUIRE(::fabs((d - d3) / d) < 1e-12);
}

TEST_CASE("LLVMDoubleVisitor Exceptions", "[llvm_double]")
{
    RCP<const Basic> x, y, r;
    x = symbol("x");
    y = symbol("y");
    r = add(x, zeta(x, y));
    LLVMDoubleVisitor v;
    CHECK_THROWS_AS(v.init({x, y}, *r), NotImplementedError);
    CHECK_THROWS_WITH(v.init({x, y}, *r), "zeta(x, y)");

    r = add(x, pow(x, y));
    CHECK_THROWS_AS(v.init({x}, *r), SymEngineException);
    CHECK_THROWS_WITH(v.init({x}, *r), "Symbol y not in the symbols vector.");
}

TEST_CASE("Check that our default LLVM passes give correct results",
          "[llvm_double]")
{
    RCP<const Basic> x, y, z, r;
    double d, d2;
    float d4;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");

    vec_basic vec = {log(x),
                     abs(x),
                     tan(x),
                     sinh(x),
                     cosh(x),
                     tanh(x),
                     asinh(y),
                     acosh(y),
                     atanh(x),
                     asin(x),
                     acos(x),
                     atan(x),
                     gamma(x),
                     loggamma(x),
                     erf(x),
                     erfc(x),
                     add(pi, div(integer(1), integer(3)))};

    r = mul(add(sin(x), add(mul(pow(y, integer(4)), mul(z, integer(2))),
                            pow(sin(x), integer(2)))),
            add(vec));
    for (int i = 0; i < 4; ++i) {
        r = mul(add(pow(integer(2), E), add(r, pow(x, pow(E, cos(x))))), r);
    }

    // r = add(add(x, y), pow(add(x, y), integer(2)));

    LambdaRealDoubleVisitor v;
    v.init({x, y, z}, *r);
    LLVMDoubleVisitor v2;
    for (int opt_level = 0; opt_level < 4; ++opt_level) {
        v2.init({x, y, z}, *r, false,
                LLVMDoubleVisitor::create_default_passes(opt_level), opt_level);
        d = v.call({0.4, 2.0, 3.0});
        d2 = v2.call({0.4, 2.0, 3.0});
        // Check for 12 digits with doubles
        REQUIRE(::fabs((d - d2) / d) < 1e-12);
    }
#ifdef SYMENGINE_HAVE_LLVM_LONG_DOUBLE
    SymEngine::LLVMLongDoubleVisitor v3;
#endif
    LLVMFloatVisitor v4;
    for (auto &arg : vec) {
        v.init({x, y, z}, *arg);
        d = v.call({0.4, 2.0, 3.0});
        v4.init({x, y, z}, *arg);
        d4 = v4.call({0.4f, 2.0f, 3.0f});
        // Check only for 6 digits with floats
        REQUIRE(::fabs((d - d4) / d) < 1e-6);
#if defined(SYMENGINE_HAVE_LLVM_LONG_DOUBLE) && defined(HAVE_SYMENGINE_MPFR)
        long double d3, mpfr_d;
        v3.init({x, y, z}, *arg);
        d3 = v3.call({0.4l, 2.0l, 3.0l});
        map_basic_basic subs_dict = {
            {x, evalf(*rational(4, 10), 128, SymEngine::EvalfDomain::Real)},
            {y, evalf(*integer(2), 128, SymEngine::EvalfDomain::Real)},
            {z, evalf(*integer(3), 128, SymEngine::EvalfDomain::Real)},
        };
        SymEngine::mpfr_class mc
            = down_cast<const RealMPFR &>(*evalf(*arg->subs(subs_dict), 128,
                                                 SymEngine::EvalfDomain::Real))
                  .as_mpfr();
        mpfr_d = mpfr_get_ld(mc.get_mpfr_t(), MPFR_RNDN);
        // Check for 16 digits with long doubles
        REQUIRE(::fabsl((mpfr_d - d3) / mpfr_d) < 1e-16);
#endif
    }
    v4.init({x, y, z}, *r);
    REQUIRE(std::isinf(v4.call({0.4f, 2.0f, 3.0f})));
}
#endif
