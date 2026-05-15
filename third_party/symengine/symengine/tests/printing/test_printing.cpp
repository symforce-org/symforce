#include "catch.hpp"
#include <chrono>

#include <symengine/matrix.h>
#include <symengine/printers/strprinter.h>
#include <symengine/printers/stringbox.h>
#include <symengine/printers.h>
#include <symengine/parser.h>
#include <symengine/logic.h>

using SymEngine::add;
using SymEngine::BaseVisitor;
using SymEngine::Basic;
using SymEngine::Boolean;
using SymEngine::ceiling;
using SymEngine::Complex;
using SymEngine::complex_double;
using SymEngine::complexes;
using SymEngine::ComplexInf;
using SymEngine::conditionset;
using SymEngine::DenseMatrix;
using SymEngine::Derivative;
using SymEngine::diff;
using SymEngine::div;
using SymEngine::down_cast;
using SymEngine::emptyset;
using SymEngine::erf;
using SymEngine::erfc;
using SymEngine::Expression;
using SymEngine::finiteset;
using SymEngine::floor;
using SymEngine::function_symbol;
using SymEngine::I;
using SymEngine::imageset;
using SymEngine::Inf;
using SymEngine::Infty;
using SymEngine::infty;
using SymEngine::integer;
using SymEngine::Integer;
using SymEngine::integer_class;
using SymEngine::integers;
using SymEngine::interval;
using SymEngine::julia_str;
using SymEngine::lambertw;
using SymEngine::latex;
using SymEngine::loggamma;
using SymEngine::logical_and;
using SymEngine::logical_or;
using SymEngine::logical_xor;
using SymEngine::map_uint_mpz;
using SymEngine::mul;
using SymEngine::NaN;
using SymEngine::naturals;
using SymEngine::naturals0;
using SymEngine::NegInf;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::parse;
using SymEngine::pi;
using SymEngine::piecewise;
using SymEngine::pow;
using SymEngine::primepi;
using SymEngine::print_stack_on_segfault;
using SymEngine::Rational;
using SymEngine::rationals;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::real_double;
using SymEngine::reals;
using SymEngine::Set;
using SymEngine::set_complement;
using SymEngine::set_intersection;
using SymEngine::set_union;
using SymEngine::Sin;
using SymEngine::StringBox;
using SymEngine::StrPrinter;
using SymEngine::Subs;
using SymEngine::symbol;
using SymEngine::Symbol;
using SymEngine::truncate;
using SymEngine::tuple;
using SymEngine::uexpr_poly;
using SymEngine::UIntPoly;
using SymEngine::unicode;
using SymEngine::universalset;
using SymEngine::zero;
using SymEngine::zeta;

using namespace SymEngine::literals;

// Macro to let string literals be unicode const char in all C++ standards
// Otherwise u8"" would be char8_t in C++20
#define U8(x) reinterpret_cast<const char *>(u8##x)

namespace SymEngine
{
class MyStrPrinter : public BaseVisitor<MyStrPrinter, StrPrinter>
{
public:
    using StrPrinter::bvisit;

    void bvisit(const Sin &x)
    {
        str_ = "MySin(" + this->apply(x.get_arg()) + ")";
    }
};
} // namespace SymEngine

TEST_CASE("test_printing(): printing", "[printing]")
{
    RCP<const Basic> r, r1, r2;
    RCP<const Integer> i = integer(-1);
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");

    r = div(integer(12), pow(integer(195), div(integer(1), integer(2))));
    REQUIRE(r->__str__() == "(4/65)*sqrt(195)");

    r = mul(integer(12), pow(integer(195), div(integer(1), integer(2))));
    REQUIRE(r->__str__() == "12*sqrt(195)");

    r = mul(integer(23), mul(pow(integer(5), div(integer(1), integer(2))),
                             pow(integer(7), div(integer(1), integer(2)))));
    REQUIRE(r->__str__() == "23*sqrt(5)*sqrt(7)");

    r = mul(integer(2), pow(symbol("x"), integer(2)));
    REQUIRE(r->__str__() == "2*x**2");

    r = pow(zero, x);
    REQUIRE(r->__str__() == "0**x");

    r = mul(integer(23),
            mul(pow(div(integer(5), integer(2)), div(integer(1), integer(2))),
                pow(div(integer(7), integer(3)), div(integer(1), integer(2)))));
    REQUIRE(r->__str__() == "(23/6)*sqrt(2)*sqrt(3)*sqrt(5)*sqrt(7)");

    r = exp(symbol("x"));
    REQUIRE(r->__str__() == "exp(x)");
    r = mul(exp(symbol("x")), integer(10));
    REQUIRE(r->__str__() == "10*exp(x)");
    r = exp(mul(symbol("x"), integer(-1)));
    REQUIRE(r->__str__() == "exp(-x)");
    r = exp(integer(-1));
    REQUIRE(r->__str__() == "exp(-1)");
    r = mul(exp(integer(-1)), integer(2));
    REQUIRE(r->__str__() == "2*exp(-1)");
    r = mul(exp(integer(-3)), integer(2));
    REQUIRE(r->__str__() == "2*exp(-3)");
    r = mul(exp(integer(1)), integer(2));
    REQUIRE(r->__str__() == "2*E");
    r = div(exp(integer(-1)), symbol("x"));
    REQUIRE(r->__str__() == "exp(-1)/x");

    r = pow(div(symbol("x"), integer(2)), div(integer(1), integer(2)));
    REQUIRE(r->__str__() == "(1/2)*sqrt(2)*sqrt(x)");

    r = pow(div(integer(3), integer(2)), div(integer(1), integer(2)));
    REQUIRE(r->__str__() == "(1/2)*sqrt(2)*sqrt(3)");

    r1 = mul(integer(12), pow(integer(196), div(integer(-1), integer(2))));
    r2 = mul(integer(294), pow(integer(196), div(integer(-1), integer(2))));
    r = add(integer(-51), mul(r1, r2));
    REQUIRE(r->__str__() == "-33");

    r1 = mul(x, i);
    r2 = mul(r1, y);
    REQUIRE(r1->__str__() == "-x");
    REQUIRE(r1->__str__() != "-1x");
    REQUIRE(r2->__str__() == "-x*y");
    REQUIRE(r2->__str__() != "-1x*y");

    r = mul(integer(-1), pow(integer(195), div(integer(1), integer(3))));
    REQUIRE(r->__str__() == "-195**(1/3)");
    r = pow(integer(-6), div(integer(1), integer(2)));
    REQUIRE(r->__str__() == "I*sqrt(6)");

    RCP<const Number> rn1, rn2, rn3, c1, c2;
    rn1 = Rational::from_two_ints(*integer(2), *integer(4));
    rn2 = Rational::from_two_ints(*integer(5), *integer(7));
    rn3 = Rational::from_two_ints(*integer(-5), *integer(7));

    c1 = Complex::from_two_rats(down_cast<const Rational &>(*rn1),
                                down_cast<const Rational &>(*rn2));
    c2 = Complex::from_two_rats(down_cast<const Rational &>(*rn1),
                                down_cast<const Rational &>(*rn3));
    r1 = mul(c1, x);
    r2 = mul(c2, x);
    REQUIRE(c1->__str__() == "1/2 + 5/7*I");
    REQUIRE(c2->__str__() == "1/2 - 5/7*I");
    REQUIRE(r1->__str__() == "(1/2 + 5/7*I)*x");
    REQUIRE(r2->__str__() == "(1/2 - 5/7*I)*x");
    r1 = pow(x, c1);
    r2 = pow(x, c2);
    REQUIRE(r1->__str__() == "x**(1/2 + 5/7*I)");
    REQUIRE(r2->__str__() == "x**(1/2 - 5/7*I)");

    c1 = Complex::from_two_nums(*rn1, *rn2);
    c2 = Complex::from_two_nums(*rn1, *rn3);
    REQUIRE(c1->__str__() == "1/2 + 5/7*I");
    REQUIRE(c2->__str__() == "1/2 - 5/7*I");

    rn1 = Rational::from_two_ints(*integer(0), *integer(4));
    c1 = Complex::from_two_nums(*rn1, *rn2);
    c2 = Complex::from_two_nums(*rn1, *rn3);
    r1 = mul(c1, x);
    r2 = mul(c2, x);
    REQUIRE(c1->__str__() == "5/7*I");
    REQUIRE(c2->__str__() == "-5/7*I");
    REQUIRE(r1->__str__() == "5/7*I*x");
    REQUIRE(r2->__str__() == "-5/7*I*x");
    r1 = pow(x, c1);
    r2 = pow(x, c2);
    REQUIRE(r1->__str__() == "x**(5/7*I)");
    REQUIRE(r2->__str__() == "x**(-5/7*I)");

    c1 = Complex::from_two_nums(*rn2, *rn1);
    c2 = Complex::from_two_nums(*rn3, *rn1);
    r1 = mul(c1, x);
    r2 = mul(c2, x);
    REQUIRE(c1->__str__() == "5/7");
    REQUIRE(c2->__str__() == "-5/7");
    REQUIRE(r1->__str__() == "(5/7)*x");
    REQUIRE(r2->__str__() == "(-5/7)*x");
    r1 = pow(x, c1);
    r2 = pow(x, c2);
    REQUIRE(r1->__str__() == "x**(5/7)");
    REQUIRE(r2->__str__() == "x**(-5/7)");

    rn1 = Rational::from_two_ints(*integer(1), *integer(1));
    c1 = Complex::from_two_nums(*rn2, *rn1);
    REQUIRE(c1->__str__() == "5/7 + I");
    rn1 = Rational::from_two_ints(*integer(-1), *integer(1));
    c1 = Complex::from_two_nums(*rn2, *rn1);
    REQUIRE(c1->__str__() == "5/7 - I");

    r1 = mul(c1, x);
    REQUIRE(r1->__str__() == "(5/7 - I)*x");

    r1 = mul(integer(2), x);
    REQUIRE(r1->__str__() == "2*x");

    r1 = mul(mul(integer(2), pow(symbol("x"), div(integer(2), integer(3)))), y);
    REQUIRE(r1->__str__() == "2*x**(2/3)*y");

    r1 = mul(x, y);
    REQUIRE(r1->__str__() == "x*y");

    r = div(x, add(x, y));
    r1 = div(x, pow(add(x, y), div(integer(2), integer(3))));
    r2 = div(x, pow(add(x, y), div(integer(-2), integer(3))));
    REQUIRE(r->__str__() == "x/(x + y)");
    REQUIRE(r1->__str__() == "x/(x + y)**(2/3)");
    REQUIRE(r2->__str__() == "x*(x + y)**(2/3)");

    r = div(integer(1), mul(x, add(x, y)));
    r1 = div(mul(y, integer(-1)), mul(x, add(x, y)));
    r2 = mul(pow(y, x), pow(x, y));
    REQUIRE(r->__str__() == "1/(x*(x + y))");
    REQUIRE(r1->__str__() == "-y/(x*(x + y))");
    REQUIRE(r2->__str__() == "x**y*y**x");

    r = pow(y, pow(x, integer(2)));
    r1 = pow(integer(3), mul(integer(2), x));
    r2 = pow(integer(3), mul(integer(-1), x));
    REQUIRE(r->__str__() == "y**(x**2)");
    REQUIRE(r1->__str__() == "3**(2*x)");
    REQUIRE(r2->__str__() == "3**(-x)");

    r1 = pow(mul(integer(2), x), y);
    r2 = pow(mul(x, y), z);
    REQUIRE(r1->__str__() == "(2*x)**y");
    REQUIRE(r2->__str__() == "(x*y)**z");

    r1 = pow(pow(integer(2), x), y);
    r2 = pow(pow(x, y), z);
    REQUIRE(r1->__str__() == "(2**x)**y");
    REQUIRE(r2->__str__() == "(x**y)**z");

    r = pow(I, x);
    r1 = sub(sub(integer(2), x), y);
    REQUIRE(r->__str__() == "I**x");
    REQUIRE(r1->__str__() == "2 - x - y");

    RCP<const Basic> f = function_symbol("f", x);
    RCP<const Basic> g = function_symbol("g", x);
    r = f->diff(x);
    r1 = Derivative::create(f, {x});
    r2 = Derivative::create(g, {x});

    REQUIRE(r->__str__() == "Derivative(f(x), x)");
    REQUIRE(r1->__str__() == "Derivative(f(x), x)");
    REQUIRE(r2->__str__() == "Derivative(g(x), x)");
    REQUIRE(r1->compare(*r2) == -1);

    r1 = f->diff(x)->diff(x);
    REQUIRE(r1->__str__() == "Derivative(f(x), x, x)");

    f = function_symbol("f", {x, y});
    r = f->diff(x)->diff(y);
    REQUIRE(r->__str__() == "Derivative(f(x, y), x, y)");
    r1 = Subs::create(Derivative::create(function_symbol("f", {y, x}), {x}),
                      {{x, add(x, y)}});
    REQUIRE(r1->__str__() == "Subs(Derivative(f(y, x), x), (x), (x + y))");
}

TEST_CASE("test_matrix(): printing", "[printing]")
{
    DenseMatrix A
        = DenseMatrix(2, 2, {integer(1), integer(0), integer(0), integer(1)});
    REQUIRE(A.__str__() == "[1, 0]\n[0, 1]\n");
    REQUIRE(str(A) == "[1, 0]\n[0, 1]\n");
}

TEST_CASE("test_UIntPoly::from_dict(): printing", "[printing]")
{
    RCP<const Basic> p;
    RCP<const Symbol> x = symbol("x");

    p = UIntPoly::from_dict(x, {{0, 0_z}});
    REQUIRE(p->__str__() == "0");

    p = UIntPoly::from_dict(x, {{0, 1_z}});
    REQUIRE(p->__str__() == "1");

    p = UIntPoly::from_dict(x, {{1, 1_z}});
    REQUIRE(p->__str__() == "x");

    p = UIntPoly::from_dict(x, {{0, 1_z}, {1, 2_z}});
    REQUIRE(p->__str__() == "2*x + 1");

    p = UIntPoly::from_dict(x, {{0, -1_z}, {1, 2_z}});
    REQUIRE(p->__str__() == "2*x - 1");

    p = UIntPoly::from_dict(x, {{0, -1_z}});
    REQUIRE(p->__str__() == "-1");

    p = UIntPoly::from_dict(x, {{1, -1_z}});
    REQUIRE(p->__str__() == "-x");

    p = UIntPoly::from_dict(x, {{0, -1_z}, {1, 1_z}});
    REQUIRE(p->__str__() == "x - 1");

    p = UIntPoly::from_dict(x, {{0, 1_z}, {1, 1_z}, {2, 1_z}});
    REQUIRE(p->__str__() == "x**2 + x + 1");

    p = UIntPoly::from_dict(x, {{0, 1_z}, {1, -1_z}, {2, 1_z}});
    REQUIRE(p->__str__() == "x**2 - x + 1");

    p = UIntPoly::from_dict(x, {{0, 1_z}, {1, 2_z}, {2, 1_z}});
    REQUIRE(p->__str__() == "x**2 + 2*x + 1");

    p = UIntPoly::from_dict(x, {{1, 2_z}, {2, 1_z}});
    REQUIRE(p->__str__() == "x**2 + 2*x");

    p = UIntPoly::from_dict(x, {{0, -1_z}, {1, -2_z}, {2, -1_z}});

    REQUIRE(p->__str__() == "-x**2 - 2*x - 1");
}

TEST_CASE("test_uexpr_poly(): printing", "[printing]")
{
    RCP<const Basic> p;
    RCP<const Symbol> x = symbol("x");
    Expression a(symbol("a"));
    Expression b(symbol("b"));
    Expression c(symbol("c"));
    Expression d(symbol("d"));

    p = uexpr_poly(x, {{0, Expression(0)}});
    REQUIRE(p->__str__() == "0");
    p = uexpr_poly(x, {{0, Expression(1)}});
    REQUIRE(p->__str__() == "1");
    p = uexpr_poly(x, {{1, Expression(1)}});
    REQUIRE(p->__str__() == "x");
    p = uexpr_poly(x, {{0, 1}, {1, 2}});
    REQUIRE(p->__str__() == "2*x + 1");
    p = uexpr_poly(x, {{0, -1}, {1, 2}});
    REQUIRE(p->__str__() == "2*x - 1");
    p = uexpr_poly(x, {{0, Expression(-1)}});
    REQUIRE(p->__str__() == "-1");
    p = uexpr_poly(x, {{1, Expression(-1)}});
    REQUIRE(p->__str__() == "-x");
    p = uexpr_poly(x, {{0, -1}, {1, 1}});
    REQUIRE(p->__str__() == "x - 1");
    p = uexpr_poly(x, {{0, 1}, {1, 1}, {2, 1}});
    REQUIRE(p->__str__() == "x**2 + x + 1");
    p = uexpr_poly(x, {{0, 1}, {1, -1}, {2, 1}});
    REQUIRE(p->__str__() == "x**2 - x + 1");
    p = uexpr_poly(x, {{0, 1}, {1, 2}, {2, 1}});
    REQUIRE(p->__str__() == "x**2 + 2*x + 1");
    p = uexpr_poly(x, {{1, 2}, {2, 1}});
    REQUIRE(p->__str__() == "x**2 + 2*x");
    p = uexpr_poly(x, {{0, -1}, {1, -2}, {2, -1}});
    REQUIRE(p->__str__() == "-x**2 - 2*x - 1");
    p = uexpr_poly(x, {{-1, d}});

    REQUIRE(p->__str__() == "d*x**(-1)");
    REQUIRE(not(p->__str__() == "d*x**-1"));

    p = uexpr_poly(x, {{-2, d}, {-1, c}, {0, b}, {1, a}});
    REQUIRE(p->__str__() == "a*x + b + c*x**(-1) + d*x**(-2)");
}

TEST_CASE("test_infinity(): printing", "[printing]")
{
    RCP<const Basic> a;

    a = infty(1);
    REQUIRE(a->__str__() == "oo");
    a = infty(-1);
    REQUIRE(a->__str__() == "-oo");
    a = infty(0);
    REQUIRE(a->__str__() == "zoo");
}

TEST_CASE("test_floats(): printing", "[printing]")
{
    RCP<const Basic> p;
    ;
    RCP<const Basic> x = symbol("x");

    p = real_double(11111.11);
    p = pow(p, x);
    REQUIRE(p->__str__() == "11111.11**x");

    p = real_double(123456.0);
    p = pow(p, x);
    REQUIRE(p->__str__() == "123456.0**x");

    p = real_double(123456789123456.0);
    p = pow(p, x);
    REQUIRE(p->__str__() == "123456789123456.**x");

    p = real_double(0.00001);
    p = pow(p, x);
    bool pr = p->__str__() == "1e-05**x" or p->__str__() == "1e-005**x";
    REQUIRE(pr == true);

    p = real_double(0.00000011);
    p = mul(p, x);
    pr = (p->__str__() == "1.1e-07*x") or (p->__str__() == "1.1e-007*x");
    REQUIRE(pr == true);

    p = complex_double(std::complex<double>(0.1, 0.2));
    p = mul(p, x);
    REQUIRE(p->__str__() == "(0.1 + 0.2*I)*x");

    p = real_double(123);
    p = sub(p, x);
    REQUIRE(p->__str__() == "123.0 - x");

    p = complex_double(std::complex<double>(1, 2));
    p = add(p, x);
    REQUIRE(p->__str__() == "1.0 + 2.0*I + x");

    p = complex_double(std::complex<double>(1, -2));
    p = add(p, x);
    REQUIRE(p->__str__() == "1.0 - 2.0*I + x");

    p = complex_double(std::complex<double>(1, 0.00000000000000001));
    p = add(p, x);
    pr = (p->__str__() == "1.0 + 1e-17*I + x")
         or (p->__str__() == "1.0 + 1e-017*I + x");
    REQUIRE(pr == true);

    p = complex_double(
        std::complex<double>(0.00000000000000001, 0.00000000000000001));
    p = add(p, x);
    pr = (p->__str__() == "1e-17 + 1e-17*I + x")
         or (p->__str__() == "1e-017 + 1e-017*I + x");
    REQUIRE(pr == true);

#ifdef HAVE_SYMENGINE_MPFR
    SymEngine::mpfr_class m1(75);
    mpfr_set_ui(m1.get_mpfr_t(), 123, MPFR_RNDN);
    p = SymEngine::real_mpfr(m1);
    p = add(p, x);
    REQUIRE(p->__str__() == "123.0000000000000000000 + x");
#ifdef HAVE_SYMENGINE_MPC
    SymEngine::mpc_class m2(75);
    mpc_set_si_si(m2.get_mpc_t(), -10, 10, MPC_RNDNN);
    p = SymEngine::complex_mpc(m2);
    p = div(p, x);
    REQUIRE(p->__str__()
            == "(-10.00000000000000000000 + 10.00000000000000000000*I)/x");
#endif
#endif
}

TEST_CASE("test_functions(): printing", "[printing]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> p = loggamma(x);
    REQUIRE(p->__str__() == "loggamma(x)");

    p = erf(x);
    REQUIRE(p->__str__() == "erf(x)");

    p = erf(add(x, y));
    REQUIRE(p->__str__() == "erf(x + y)");

    p = erfc(x);
    REQUIRE(p->__str__() == "erfc(x)");
}

TEST_CASE("test custom printing", "[printing]")
{
    SymEngine::MyStrPrinter printer;
    RCP<const Basic> p;
    RCP<const Symbol> x = symbol("x");
    p = sin(x);
    CHECK(printer.apply(p) == "MySin(x)");
    p = cos(sin(x));
    CHECK(printer.apply(p) == "cos(MySin(x))");
}

TEST_CASE("Ascii Art", "[basic]")
{
    std::cout << SymEngine::ascii_art() << std::endl;
}

TEST_CASE("test_sets(): printing", "[printing]")
{
    RCP<const Set> r1;
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");

    r1 = set_complement(interval(NegInf, Inf, true, true),
                        finiteset({symbol("y")}));
    REQUIRE(r1->__str__() == "(-oo, oo) \\ {y}");

    RCP<const Set> i1 = interval(integer(3), integer(10));

    r1 = conditionset(
        {x}, logical_and({i1->contains(x), Ge(mul(x, x), integer(9))}));
    REQUIRE(r1->__str__() == "{x | And(9 <= x**2, Contains(x, [3, 10]))}");

    r1 = imageset(x, mul(x, x), interval(zero, one));
    REQUIRE(r1->__str__() == "{x**2 | x in [0, 1]}");
}

TEST_CASE("test_sign(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> r
        = sign(mul(mul(pow(Complex::from_two_nums(*integer(2), *integer(3)),
                           Rational::from_two_ints(3, 2)),
                       x),
                   pow(mul(integer(3), I), integer(3))));
    CHECK(r->__str__() == "-I*sign(x*(2 + 3*I)**(3/2))");
}

TEST_CASE("test_floor(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r = floor(mul(pow(x, integer(3)), y));
    CHECK(r->__str__() == "floor(x**3*y)");

    r = floor(add(add(integer(2), mul(integer(2), x)), mul(integer(3), y)));
    CHECK(r->__str__() == "2 + floor(2*x + 3*y)");
}

TEST_CASE("test_ceiling(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r = ceiling(mul(pow(x, integer(3)), y));
    CHECK(r->__str__() == "ceiling(x**3*y)");

    r = ceiling(add(add(integer(2), mul(integer(2), x)), mul(integer(3), y)));
    CHECK(r->__str__() == "2 + ceiling(2*x + 3*y)");
}

TEST_CASE("test_truncate(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r = truncate(mul(pow(x, integer(3)), y));
    CHECK(r->__str__() == "truncate(x**3*y)");

    r = truncate(add(add(integer(2), mul(integer(2), x)), mul(integer(3), y)));
    CHECK(r->__str__() == "2 + truncate(2*x + 3*y)");
}

TEST_CASE("test_conjugate(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r = conjugate(
        mul(mul(complex_double(std::complex<double>(2.0, 3.0)), x), y));
    CHECK(r->__str__() == "(2.0 - 3.0*I)*conjugate(y)*conjugate(x)");

    r = conjugate(pow(y, Rational::from_two_ints(3, 2)));
    CHECK(r->__str__() == "conjugate(y**(3/2))");
}

TEST_CASE("test_logical(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> r1;
    r1 = logical_and({Ge(y, integer(2)), Ge(mul(x, x), integer(9))});
    REQUIRE(r1->__str__() == "And(2 <= y, 9 <= x**2)");
    r1 = logical_or({Ge(y, integer(2)), Ge(mul(x, x), integer(9))});
    REQUIRE(r1->__str__() == "Or(2 <= y, 9 <= x**2)");
    r1 = logical_xor({Ge(y, integer(2)), Ge(mul(x, x), integer(9))});
    REQUIRE(r1->__str__() == "Xor(2 <= y, 9 <= x**2)");
}

TEST_CASE("test_mathml()", "[mathml]")
{
    RCP<const Basic> x = parse("x^2");
    REQUIRE(mathml(*x)
            == "<apply><power/><ci>x</ci><cn type=\"integer\">2</cn></apply>");
    RCP<const Basic> y = parse("3/2 * y");
    REQUIRE(mathml(*y)
            == "<apply><times/><cn "
               "type=\"rational\">3<sep/>2</cn><ci>y</ci></apply>");
    RCP<const Basic> z = parse("x^(y^(5/3))");
    REQUIRE(mathml(*z)
            == "<apply><power/><ci>x</ci><apply><power/><ci>y</"
               "ci><cn "
               "type=\"rational\">5<sep/>3</cn></apply></apply>");
    RCP<const Basic> w = parse("1 + 4 * x * y");
    REQUIRE(mathml(*w)
            == "<apply><plus/><cn type=\"integer\">1</cn><apply><times/><cn "
               "type=\"integer\">4</cn><ci>x</ci><ci>y</ci></apply></apply>");
    RCP<const Basic> v = parse("1 + 4 * x - y");

    std::string s1 = "<apply><plus/><cn "
                     "type=\"integer\">1</cn><apply><times/><cn "
                     "type=\"integer\">4</cn><ci>x</ci></apply><apply><times/"
                     "><cn type=\"integer\">-1</cn><ci>y</ci></apply></apply>";
    std::string s2 = "<apply><plus/><cn "
                     "type=\"integer\">1</cn><apply><times/><cn "
                     "type=\"integer\">-1</cn><ci>y</ci></apply><apply><times/"
                     "><cn type=\"integer\">4</cn><ci>x</ci></apply></apply>";
    auto m = mathml(*v);
    auto b = (m == s1 or m == s2);
    REQUIRE(b);
    RCP<const Basic> u = parse("sin(x)");
    REQUIRE(mathml(*u) == "<apply><sin/><ci>x</ci></apply>");
    RCP<const Basic> b0 = complexes();
    REQUIRE(mathml(*b0) == "<complexes/>");
    RCP<const Basic> b1 = reals();
    REQUIRE(mathml(*b1) == "<reals/>");
    RCP<const Basic> b2 = rationals();
    REQUIRE(mathml(*b2) == "<rationals/>");
    RCP<const Basic> b3 = integers();
    REQUIRE(mathml(*b3) == "<integers/>");
}

TEST_CASE("test_relational(): printing", "[printing]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> r1;
    r1 = add(x, Lt(y, z));
    REQUIRE(r1->__str__() == "x + (y < z)");
    r1 = add(Lt(y, z), x);
    REQUIRE(r1->__str__() == "x + (y < z)");
    r1 = mul(x, Lt(y, z));
    REQUIRE(r1->__str__() == "x*(y < z)");
    r1 = mul(Lt(y, z), x);
    REQUIRE(r1->__str__() == "x*(y < z)");
}

TEST_CASE("test_julia(): printing", "[printing]")
{
    std::string r = julia_str(*parse("2 + 3*I"));
    CHECK(r == "2 + 3*im");
    r = julia_str(*parse("2.0 + 3.0*I"));
    CHECK(r == "2.0 + 3.0*im");
#ifdef HAVE_SYMENGINE_MPC
    r = julia_str(*parse("2.00000000000000000000000000000000 + 3*I"));
    CHECK(r
          == "2.00000000000000000000000000000000 + "
             "3.00000000000000000000000000000000*im");
#endif
}

TEST_CASE("test_latex_printing()", "[latex]")
{
    RCP<const Basic> l1 = parse("3/2");
    RCP<const Basic> l2 = parse("3/2 + 4*I/2");
    RCP<const Basic> l3 = parse("1.123123123123 + 1.123123123123*I");
    RCP<const Basic> l4 = parse("Eq(x, y)");
    RCP<const Basic> l5 = parse("Ne(x, y)");
    RCP<const Basic> l6 = parse("a <= 6");
    RCP<const Set> l7 = interval(integer(-3), integer(3), true, true);
    RCP<const Set> l8 = interval(integer(-3), integer(3), true, false);
    RCP<const Set> l9 = interval(integer(-3), integer(3), false, true);
    RCP<const Set> l10 = interval(integer(-3), integer(3), false, false);
    RCP<const Basic> l11 = parse("5 == 5");
    RCP<const Basic> l12 = parse("5 == 6");
    RCP<const Symbol> a = symbol("a");
    RCP<const Symbol> b = symbol("b");
    RCP<const Symbol> c = symbol("c");
    RCP<const Basic> l13 = logical_and({Ge(a, integer(2)), Ge(b, integer(5))});
    RCP<const Basic> l14
        = logical_and({logical_or({Eq(a, b), Ne(a, c)}), {Ge(a, b)}});
    RCP<const Basic> l15 = parse("f(a, b)")->diff(a);
    RCP<const Basic> l16 = parse("f(a, 2)")->diff(a);
    RCP<const Basic> l17 = parse("f(a, 2)")->diff(a)->diff(a)->diff(a);
    RCP<const Basic> l18 = parse("f(a, b)")->diff(a)->diff(a)->diff(b);
    RCP<const Basic> l19 = parse("pi^2 + e*2 + asin(sqrt(2)) + sin(2^(1/10))");
    RCP<const Basic> l20 = parse("f(2*a, 2*b)")->diff(a)->diff(b);
    RCP<const Basic> l21 = parse("alpha + _xi_1 + xi2");
    RCP<const Basic> l22 = parse("2 + 3 * x^10");
    RCP<const Basic> l23 = parse("exp(x-y)");
    RCP<const Basic> l24 = reals();
    RCP<const Basic> l25 = integers();
    RCP<const Basic> l26 = rationals();
    RCP<const Basic> l27 = primepi(symbol("x"));
    RCP<const Basic> l28 = complexes();

    CHECK(latex(*l1) == "\\frac{3}{2}");
    CHECK(latex(*l2) == "\\frac{3}{2} + 2j");
    CHECK(latex(*l3) == "1.123123123123 + 1.123123123123j");
    CHECK(latex(*l4) == "x = y");
    CHECK(latex(*l5) == "x \\neq y");
    CHECK(latex(*l6) == "a \\leq 6");
    CHECK(latex(*l7) == "\\left(-3, 3\\right)");
    CHECK(latex(*l8) == "\\left(-3, 3\\right]");
    CHECK(latex(*l9) == "\\left[-3, 3\\right)");
    CHECK(latex(*l10) == "\\left[-3, 3\\right]");
    CHECK(latex(*l11) == "\\mathrm{True}");
    CHECK(latex(*l12) == "\\mathrm{False}");
    // CHECK(latex(*l13) == "5 \\leq b \\wedge 2 \\leq a");
    //    CHECK(latex(*l14)
    //          == "b \\leq a \\wedge \\left(a \\neq c \\vee a = b\\right)");
    CHECK(latex(*l15) == "\\frac{\\partial}{\\partial a} f\\left(a, b\\right)");
    CHECK(latex(*l16) == "\\frac{d}{d a} f\\left(a, 2\\right)");
    CHECK(latex(*l17)
          == "\\frac{\\partial^3}{\\partial a^3 } f\\left(a, 2\\right)");
    CHECK(latex(*l18)
          == "\\frac{\\partial^3}{\\partial a^2 \\partial b } "
             "f\\left(a, b\\right)");
    CHECK(latex(*l19)
          == "\\pi^2 + 2 e + \\sin{\\left(\\sqrt[10]{2}\\right)} + "
             "\\operatorname{asin}{\\left(\\sqrt{2}\\right)}");
    CHECK(latex(*l20)
          == "4 \\left. \\frac{\\partial^2}{\\partial \\xi_1 "
             "\\partial \\xi_2 } f\\left(\\xi_1, "
             "\\xi_2\\right)\\right|_{\\substack{\\xi_1=2 a \\\\ "
             "\\xi_2=2 b}}");
    CHECK(latex(*l21) == "\\xi_1 + \\alpha + xi2");
    CHECK(latex(*l22) == "2 + 3 x^{10}");
    CHECK(latex(*l23) == "e^{x - y}");
    CHECK(latex(*l24) == "\\mathbb{R}");
    CHECK(latex(*l25) == "\\mathbb{Z}");
    CHECK(latex(*l26) == "\\mathbb{Q}");
    CHECK(latex(*l27) == "\\pi{\\left(x\\right)}");
    CHECK(latex(*l28) == "\\mathbb{C}");

    RCP<const Basic> l = naturals();
    CHECK(latex(*l) == "\\mathbb{N}");
    l = naturals0();
    CHECK(latex(*l) == "\\mathbb{N}_0");

    RCP<const Basic> i1 = integer(1);
    RCP<const Basic> i2 = integer(2);
    l = tuple({i1, i2});
    CHECK(latex(*l) == "\\left(1, 2\\right)");

    auto s1 = reals();
    auto s2 = finiteset({symbol("x")});
    auto s3 = set_intersection({s1, s2});
    CHECK(latex(*s3) == "\\mathbb{R} \\cap \\left{x\\right}");
}

TEST_CASE("test_latex_matrix_printing()", "[latex]")
{
    Expression x("x");
    DenseMatrix d(3, 1, {integer(1), integer(2), x});
    CHECK(latex(d)
          == "\\left[\\begin{matrix}\n1 \\\\\n2 \\\\\nx "
             "\\\\\n\\end{matrix}\\right]\n");
    CHECK(latex(d, 2)
          == "\\left[\\begin{matrix}\n1 \\\\\n\\vdots "
             "\\\\\n\\end{matrix}\\right]\n");

    DenseMatrix m(1, 3, {x, integer(1), integer(2)});
    CHECK(latex(m, 4, 2)
          == "\\left[\\begin{matrix}\nx & \\cdots \\\\\n"
             "\\end{matrix}\\right]\n");

    DenseMatrix d2(1, 1);
    try {
        latex(d2);
        throw "displaying unitialized matrix failed to generate exception";
    } catch (std::exception &e) {
        CHECK(std::string(e.what()) == "cannot display uninitialized element");
    }
}

TEST_CASE("test_unicode()", "[unicode]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    std::string s;

    s = unicode(*complexes());
    CHECK(s == U8("\u2102"));

    s = unicode(*reals());
    CHECK(s == U8("\u211D"));

    s = unicode(*rationals());
    CHECK(s == U8("\u211A"));

    s = unicode(*integers());
    CHECK(s == U8("\u2124"));

    s = unicode(*naturals());
    CHECK(s == U8("\u2115"));

    s = unicode(*naturals0());
    CHECK(s == U8("\u2115\u2080"));

    s = unicode(*emptyset());
    CHECK(s == U8("\u2205"));

    s = unicode(*universalset());
    CHECK(s == U8("\U0001D54C"));

    s = unicode(*finiteset({integer(1), integer(2)}));
    CHECK(s == U8("{1, 2}"));

    s = unicode(*finiteset(
        {Rational::from_two_ints(*integer(1), *integer(2)), integer(-1)}));
    CHECK(
        s
        == U8("\u23A71    \u23AB\n\u23A8\u2015, -1\u23AC\n\u23A92    \u23AD"));

    s = unicode(*reals()->contains(x));
    CHECK(s == U8("x \u220A \u211D"));

    s = unicode(*interval(integer(0), integer(1)));
    CHECK(s == U8("[0, 1]"));

    s = unicode(*interval(integer(-1), integer(1), true, false));
    CHECK(s == U8("(-1, 1]"));

    auto rat = Rational::from_two_ints(*integer(2), *integer(21));
    s = unicode(*interval(rat, integer(23), false, true));
    CHECK(s
          == U8("\u23A1 2    \u239E\n\u23A2\u2015\u2015, 23\u239F\n\u23A321    "
                u8"\u23A0"));

    s = unicode(*set_union({integers(), finiteset({Rational::from_two_ints(
                                            *integer(1), *integer(3))})}));
    CHECK(s
          == U8("    \u23A71\u23AB\n\u2124 \u222A \u23A8\u2015\u23AC\n    "
                u8"\u23A93\u23AD"));

    s = unicode(*set_intersection({reals(), finiteset({symbol("x")})}));
    CHECK(s == U8("\u211D \u2229 {x}"));

    s = unicode(*set_complement(reals(), rationals()));
    CHECK(s == U8("\u211D \\ \u211A"));

    s = unicode(*imageset(x, add(x, integer(1)), interval(zero, one)));
    CHECK(s == U8("{1 + x | x \u220A [0, 1]}"));

    s = unicode(*conditionset(
        {x}, logical_and({reals()->contains(x), Ge(x, integer(9))})));
    CHECK(s == U8("{x | 9 \u2264 x \u2227 x \u220A \u211D}"));

    s = unicode(NaN());
    CHECK(s == U8("NaN"));

    s = unicode(*pi);
    CHECK(s == U8("\U0001D70B"));

    s = unicode(*parse("e"));
    CHECK(s == U8("\U0001D452"));

    s = unicode(*parse("EulerGamma"));
    CHECK(s == U8("\U0001D6FE"));

    s = unicode(*parse("Catalan"));
    CHECK(s == U8("\U0001D43A"));

    s = unicode(*parse("GoldenRatio"));
    CHECK(s == U8("\U0001D719"));

    s = unicode(*lambertw(x));
    CHECK(s == U8("W(x)"));

    s = unicode(*zeta(x));
    CHECK(s == U8("\U0001D701(x, 1)"));

    s = unicode(*gamma(x));
    CHECK(s == U8("\u0393(x)"));

    s = unicode(*lowergamma(x, y));
    CHECK(s == U8("\U0001D6FE(x, y)"));

    s = unicode(*uppergamma(x, y));
    CHECK(s == U8("\u0393(x, y)"));

    s = unicode(*beta(x, y));
    CHECK(s == U8("B(y, x)"));

    s = unicode(*dirichlet_eta(x));
    CHECK(s == U8("\U0001D702(x)"));

    s = unicode(*loggamma(x));
    CHECK(s == U8("log \u0393(x)"));

    s = unicode(*primepi(x));
    CHECK(s == U8("\U0001D70B(x)"));

    s = unicode(*abs(x));
    CHECK(s == U8("\u2502x\u2502"));

    s = unicode(*floor(x));
    CHECK(s == U8("\u230Ax\u230B"));

    s = unicode(*ceiling(x));
    CHECK(s == U8("\u2308x\u2309"));

    auto c1 = Complex::from_two_nums(*integer(1), *integer(2));
    s = unicode(*c1);
    CHECK(s == U8("1 + 2\u22C5\U0001D456"));

    auto c2 = Complex::from_two_nums(*integer(-5), *integer(6));
    s = unicode(*c2);
    CHECK(s == U8("-5 + 6\u22C5\U0001D456"));

    auto c3 = Complex::from_two_nums(*integer(5), *integer(-6));
    s = unicode(*c3);
    CHECK(s == U8("5 - 6\u22C5\U0001D456"));

    auto c4 = Complex::from_two_nums(*integer(5), *integer(-1));
    s = unicode(*c4);
    CHECK(s == U8("5 - \U0001D456"));

    auto c5 = Complex::from_two_nums(*integer(0), *integer(-1));
    s = unicode(*c5);
    CHECK(s == U8("-\U0001D456"));

    auto c6 = Complex::from_two_nums(*integer(0), *integer(-2));
    s = unicode(*c6);
    CHECK(s == U8("-2\u22C5\U0001D456"));

    s = unicode(*infty());
    CHECK(s == U8("\u221E"));

    s = unicode(*mul(integer(-1), infty()));
    CHECK(s == U8("-\u221E"));

    s = unicode(*ComplexInf);
    CHECK(s == U8("\U0001D467\u221E"));

    s = unicode(*Le(x, integer(0)));
    CHECK(s == U8("x \u2264 0"));

    s = unicode(*Lt(x, integer(0)));
    CHECK(s == U8("x < 0"));

    s = unicode(*Ne(x, integer(0)));
    CHECK(s == U8("0 \u2260 x"));

    s = unicode(*Eq(x, integer(0)));
    CHECK(s == U8("0 = x"));

    s = unicode(*parse("5 == 5"));
    CHECK(s == U8("true"));

    s = unicode(*parse("5 != 5"));
    CHECK(s == U8("false"));

    s = unicode(*logical_and(
        {Lt(x, integer(0)), Ne(integer(-1), x), Lt(x, integer(2))}));
    CHECK(s == U8("-1 \u2260 x \u2227 x < 0 \u2227 x < 2"));

    s = unicode(*logical_or(
        {Lt(x, integer(0)), Ne(integer(-1), x), Lt(x, integer(2))}));
    CHECK(s == U8("-1 \u2260 x \u2228 x < 0 \u2228 x < 2"));

    s = unicode(*logical_xor(
        {Lt(x, integer(0)), Ne(integer(-1), x), Lt(x, integer(2))}));
    CHECK(s == U8("-1 \u2260 x \u22BB x < 0 \u22BB x < 2"));

    s = unicode(*logical_not(logical_xor({Ne(x, y), Lt(x, y)})));
    CHECK(s == U8("\u00AC(x \u2260 y \u22BB x < y)"));

    s = unicode(*integer(2));
    CHECK(s == U8("2"));

    s = unicode(*real_double(2.25));
    CHECK(s == U8("2.25"));

    s = unicode(*complex_double(std::complex<double>(2.25, -23)));
    CHECK(s == U8("2.25 - 23.0\u22C5\U0001D456"));

    s = unicode(*Rational::from_two_ints(*integer(1), *integer(3)));
    CHECK(s == U8("1\n\u2015\n3"));

    s = unicode(*Rational::from_two_ints(*integer(3), *integer(187)));
    CHECK(s == U8(" 3 \n\u2015\u2015\u2015\n187"));

    s = unicode(*Rational::from_two_ints(*integer(3), *integer(17)));
    CHECK(s == U8(" 3\n\u2015\u2015\n17"));

    s = unicode(*x);
    CHECK(s == U8("x"));

    s = unicode(*add(integer(2), x));
    CHECK(s == U8("2 + x"));

    s = unicode(*add(integer(-1), x));
    CHECK(s == U8("-1 + x"));

    s = unicode(*add(Rational::from_two_ints(*integer(1), *integer(3)), x));
    CHECK(s == U8("1    \n\u2015 + x\n3    "));

    s = unicode(*mul(integer(2), x));
    CHECK(s == U8("2\u22C5x"));

    s = unicode(*mul(integer(2), x));
    CHECK(s == U8("2\u22C5x"));

    s = unicode(*mul(mul(integer(2), x), pow(y, integer(2))));
    CHECK(s == U8("     2\n2\u22C5x\u22C5y "));

    s = unicode(*mul(Rational::from_two_ints(*integer(2), *integer(5)), x));
    CHECK(s == U8("2\u22C5x\n\u2015\u2015\u2015\n 5 "));

    s = unicode(*div(y, x));
    CHECK(s == U8("y\n\u2015\nx"));

    s = unicode(*sqrt(x));
    CHECK(s == U8("  _\n\u2572\u2571x"));

    s = unicode(*pow(x, y));
    CHECK(s == U8(" y\nx "));

    auto p = piecewise({{x, contains(x, reals())}});
    s = unicode(*p);
    CHECK(s == U8("{x if x \u220A \u211D"));

    p = piecewise(
        {{integer(1), Lt(x, integer(0))}, {integer(0), Eq(x, integer(0))}});
    s = unicode(*p);
    CHECK(s == U8("\u23A71 if x < 0\n\u23A8          \n\u23A90 if 0 = x"));

    p = piecewise({{Rational::from_two_ints(*integer(1), *integer(123)),
                    Lt(x, integer(0))},
                   {integer(0), Eq(x, integer(0))}});
    s = unicode(*p);
    CHECK(s
          == U8("\u23A7 1          \n\u23AA\u2015\u2015\u2015 if x "
                u8"< 0\n\u23A8123 "
                u8"        \n\u23A9 0 if 0 = x "));
    // FIXME: Test default

    s = unicode(*function_symbol("f", x));
    CHECK(s == U8("f(x)"));

    s = unicode(*tuple({integer(1), integer(2)}));
    CHECK(s == U8("(1, 2)"));

    s = unicode(*tuple({}));
    CHECK(s == U8("()"));
}

TEST_CASE("test_stringbox()", "[stringbox]")
{
    StringBox a("x");
    CHECK(a.get_string() == "x");
    StringBox b("-");
    a.add_below(b);
    CHECK(a.get_string() == "x\n-");
    StringBox c("13");
    a.add_below(c);
    CHECK(a.get_string() == " x\n -\n13");
    StringBox op("*");
    a.add_right(op);
    CHECK(a.get_string() == " x \n -*\n13 ");

    StringBox s1("abcd");
    StringBox s2("1234567890");
    s1.add_below(s2);
    CHECK(s1.get_string() == "   abcd   \n1234567890");

    StringBox s3("12");
    StringBox op2(" * ");
    s3.add_right(op2);
    CHECK(s3.get_string() == "12 * ");
}
