#include "catch.hpp"
#include <chrono>

#include <symengine/expression.h>

using SymEngine::complex_double;
using SymEngine::cos;
using SymEngine::eq;
using SymEngine::Expression;
using SymEngine::integer;
using SymEngine::pi;
using SymEngine::real_double;
using SymEngine::sin;
using SymEngine::symbol;

TEST_CASE("Constructors of Expression", "[Expression]")
{
    Expression e0("x");
    REQUIRE(eq(*e0.get_basic(), *symbol("x")));

    e0 = e0 + sin(e0);
    e0 = cos(e0);
    e0 = e0 + integer(1);
    REQUIRE(eq(*e0.get_basic(),
               *add(cos(add(symbol("x"), sin(symbol("x")))), integer(1))));

    Expression e1 = 20;
    REQUIRE(eq(*e1.get_basic(), *integer(20)));

    Expression e2 = 10.0;
    REQUIRE(eq(*e2.get_basic(), *real_double(10.0)));

    Expression e3 = std::complex<double>(1.0, 2.0);
    REQUIRE(
        eq(*e3.get_basic(), *complex_double(std::complex<double>(1.0, 2.0))));
}

TEST_CASE("Printing of Expression", "[Expression]")
{
    Expression e0("x");
    std::stringstream s;
    s << e0;
    REQUIRE(s.str() == "x");
}

TEST_CASE("Arithmetic of Expression", "[Expression]")
{
    Expression x("x"), y("y");
    auto z = x + y;
    std::cout << z << std::endl;
    z += y;
    std::cout << z << std::endl;
    REQUIRE(z == x + y + y);
    REQUIRE(z == x + 2 * y);
    std::cout << pow(z, z) << std::endl;
    std::cout << pow(z, 45) << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto res = expand(pow(z, 45) * pow(z, 45));
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
        << "ns" << std::endl;
    std::cout << res << std::endl;
}

TEST_CASE("Substitution of Expression", "[Expression]")
{
    const Expression x("x");
    const Expression f_x = 2 * x * x;
    const Expression f_x_subs = f_x.subs({{x, integer(2)}});
    REQUIRE(f_x_subs == 8);
}

TEST_CASE("Conversion of Expression", "[Expression]")
{
    const Expression x("x");
    const Expression f_x = x * x;

    REQUIRE(static_cast<int>(f_x.subs({{x, integer(2)}})) == 4);
    REQUIRE(static_cast<double>(f_x.subs({{x, real_double(3.5)}})) == 12.25);
    REQUIRE(static_cast<float>(f_x.subs({{x, real_double(3.5)}})) == 12.25f);
    REQUIRE(std::abs(static_cast<std::complex<double>>(
                         f_x.subs({{x, complex_double({0.0, 2.0})}}))
                     - std::complex<double>(-4, 0))
            < 1e-12);
    REQUIRE(std::abs(static_cast<std::complex<float>>(
                         f_x.subs({{x, complex_double({0.0f, 2.0f})}}))
                     - std::complex<float>(-4, 0))
            < 1e-12);
}

TEST_CASE("Differentiation of Expression", "[Expression]")
{
    const Expression x("x");
    const Expression f_x = x * x;
    const Expression df_dx = f_x.diff(x);
    REQUIRE(df_dx == 2 * x);

    const auto symb_x = symbol("x");
    const Expression df_dsx = f_x.diff(symb_x);
    REQUIRE(df_dsx == 2 * Expression(symb_x));
}
