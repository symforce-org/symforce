#include "catch.hpp"
#include <chrono>

#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/complex_double.h>
#include <symengine/symengine_exception.h>
#include <symengine/symengine_casts.h>

using SymEngine::Basic;
using SymEngine::Add;
using SymEngine::Mul;
using SymEngine::Pow;
using SymEngine::Log;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::umap_basic_num;
using SymEngine::map_vec_uint;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::multinomial_coefficients;
using SymEngine::one;
using SymEngine::zero;
using SymEngine::sin;
using SymEngine::RCP;
using SymEngine::sqrt;
using SymEngine::pow;
using SymEngine::add;
using SymEngine::mul;
using SymEngine::div;
using SymEngine::sub;
using SymEngine::exp;
using SymEngine::E;
using SymEngine::Rational;
using SymEngine::Complex;
using SymEngine::Number;
using SymEngine::I;
using SymEngine::rcp_dynamic_cast;
using SymEngine::print_stack_on_segfault;
using SymEngine::RealDouble;
using SymEngine::ComplexDouble;
using SymEngine::real_double;
using SymEngine::complex_double;
using SymEngine::rational_class;
using SymEngine::is_a;
using SymEngine::set_basic;
using SymEngine::SymEngineException;
using SymEngine::Inf;
using SymEngine::NegInf;
using SymEngine::ComplexInf;
using SymEngine::down_cast;
using SymEngine::pi;
using SymEngine::minus_one;
using SymEngine::Nan;
using SymEngine::make_rcp;

TEST_CASE("Add: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1 = add(x, x);
    RCP<const Basic> r2 = mul(i2, x);
    RCP<const Basic> r3 = mul(i3, x);
    REQUIRE(eq(*r1, *r2));
    REQUIRE(neq(*r1, *r3));

    r3 = mul(i2, y);
    REQUIRE(neq(*r1, *r3));
    REQUIRE(neq(*r2, *r3));

    r1 = add(mul(y, x), mul(mul(i2, x), y));
    r2 = mul(mul(i3, x), y);
    REQUIRE(eq(*r1, *r2));

    r1 = add({mul(y, x), mul({i2, x, y})});
    r2 = mul({i3, x, y});
    REQUIRE(eq(*r1, *r2));

    r1 = add(add(x, x), x);
    r2 = mul(i3, x);
    REQUIRE(eq(*r1, *r2));

    r1 = add(add(x, x), x);
    r2 = mul(x, i3);
    REQUIRE(eq(*r1, *r2));
    r1 = add({x, x, x});
    REQUIRE(eq(*r1, *r2));

    r1 = add(x, one);
    r2 = add(one, x);
    REQUIRE(eq(*r1, *r2));

    r1 = add(pow(x, y), z);
    r2 = add(z, pow(x, y));
    REQUIRE(eq(*r1, *r2));
    r2 = add({z, pow(x, y)});
    REQUIRE(eq(*r1, *r2));

    r1 = add(x, I);
    r2 = add(I, x);
    REQUIRE(eq(*r1, *r2));
    r2 = add({I, x});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(x, I);
    r2 = mul(mul(I, i2), x);
    r3 = mul(mul(I, i3), x);
    r2 = add(r1, r2);
    REQUIRE(eq(*r3, *r2));

    r1 = real_double(0.1);
    r2 = Rational::from_mpq(rational_class(1, 2));
    r3 = add(add(add(r1, r2), integer(1)), real_double(0.2));
    REQUIRE(is_a<RealDouble>(*r3));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r3).i - 1.8) < 1e-12);
    r3 = add({r1, r2, integer(1), real_double(0.2)});
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r3).i - 1.8) < 1e-12);

    r1 = complex_double(std::complex<double>(0.1, 0.2));
    r2 = Complex::from_two_nums(*Rational::from_mpq(rational_class(1, 2)),
                                *Rational::from_mpq(rational_class(7, 5)));
    r3 = add(add(add(r1, r2), integer(1)), real_double(0.4));
    REQUIRE(is_a<ComplexDouble>(*r3));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.real() - 2.0)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.imag() - 1.6)
            < 1e-12);
    r3 = add({r1, r2, integer(1), real_double(0.4)});
    REQUIRE(is_a<ComplexDouble>(*r3));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.real() - 2.0)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.imag() - 1.6)
            < 1e-12);

    r1 = add({i2, i3, i4});
    REQUIRE(eq(*r1, *integer(9)));

    r1 = add({});
    REQUIRE(eq(*r1, *zero));

    r1 = add({i2});
    REQUIRE(eq(*r1, *i2));

    r1 = add(x, real_double(0.0));
    REQUIRE(eq(*r1, *x));
}

TEST_CASE("Mul: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> im2 = integer(-2);
    RCP<const Integer> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> i6 = integer(6);

    RCP<const Basic> r1, r2, mhalf;
    r1 = mul(pow(x, y), z);
    r2 = mul(z, pow(x, y));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(x, y), mul(y, x));
    r2 = mul(pow(x, i2), pow(y, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = mul({mul(x, y), mul(y, x)});
    r2 = mul({pow(x, i2), pow(y, i2)});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(pow(x, add(y, z)), z);
    r2 = mul(z, pow(x, add(z, y)));
    REQUIRE(eq(*r1, *r2));
    r1 = mul({pow(x, add({y, z})), z});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(pow(x, y), pow(x, z));
    r2 = pow(x, add(y, z));
    REQUIRE(eq(*r1, *r2));
    r1 = mul({pow(x, y), pow(x, z)});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(pow(x, y), pow(x, z)), pow(x, x));
    r2 = pow(x, add(add(x, y), z));
    REQUIRE(eq(*r1, *r2));
    r1 = mul({pow(x, y), pow(x, z), pow(x, x)});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(mul(pow(x, y), pow(x, z)), pow(x, x)), y);
    r2 = mul(pow(x, add(add(x, y), z)), y);
    REQUIRE(eq(*r1, *r2));
    r1 = mul({pow(x, y), pow(x, z), pow(x, x), y});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(i2, pow(y, mul(im2, pow(x, i2)))),
             mul(i3, pow(y, mul(i2, pow(x, i2)))));
    r2 = i6;
    REQUIRE(eq(*r1, *r2));
    r1 = mul({mul({i2, pow(y, mul({im2, pow(x, i2)}))}),
              mul({i3, pow(y, mul({i2, pow(x, i2)}))})});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(mul(mul(div(i3, i2), pow(cos(pow(x, i2)), im2)), x),
                 sin(pow(x, i2))),
             cos(div(mul(i3, x), i4)));
    r2 = mul(mul(mul(mul(div(i3, i2), pow(cos(pow(x, i2)), im2)), x),
                 sin(pow(x, i2))),
             cos(div(mul(i3, x), i4)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));
    r1 = mul({div(i3, i2), pow(cos(pow(x, i2)), im2), x, sin(pow(x, i2)),
              cos(div(mul(i3, x), i4))});
    REQUIRE(eq(*r1, *r2));

    mhalf = div(integer(-1), i2);
    r1 = mul(integer(12), pow(integer(196), mhalf));
    r2 = mul(integer(294), pow(integer(196), mhalf));
    REQUIRE(eq(*integer(18), *mul(r1, r2)));
    r1 = mul({integer(12), pow(integer(196), mhalf), integer(294),
              pow(integer(196), mhalf)});
    REQUIRE(eq(*integer(18), *r1));

    r1 = mul(mul(integer(12), pow(integer(196), mhalf)), pow(i3, mhalf));
    r2 = mul(mul(integer(294), pow(integer(196), mhalf)), pow(i3, mhalf));
    REQUIRE(eq(*i6, *mul(r1, r2)));

    r1 = mul(add(x, mul(y, I)), sub(x, mul(y, I)));
    r2 = mul(sub(x, mul(y, I)), add(x, mul(y, I)));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));
    r1 = mul({add(x, mul(y, I)), sub(x, mul(y, I))});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(x, I), mul(y, I));
    r2 = mul(integer(-1), mul(x, y));
    REQUIRE(eq(*r1, *r2));
    r1 = mul({x, I, y, I});
    REQUIRE(eq(*r1, *r2));

    RCP<const Number> rc1, rc2, c1, c2;
    rc1 = Rational::from_two_ints(*integer(2), *integer(1));
    rc2 = Rational::from_two_ints(*integer(3), *integer(1));
    c1 = Complex::from_two_nums(*rc1, *rc2);
    rc1 = Rational::from_two_ints(*integer(-5), *integer(1));
    rc2 = Rational::from_two_ints(*integer(12), *integer(1));
    c2 = Complex::from_two_nums(*rc1, *rc2);

    r1 = mul(x, c1);
    r2 = mul(x, c1);
    r1 = mul(r1, r2);
    r2 = mul(pow(x, i2), c2);
    REQUIRE(eq(*r1, *r2));
    r1 = mul({x, c1, x, c1});
    REQUIRE(eq(*r1, *r2));

    r1 = mul(sqrt(x), x);
    r2 = pow(x, div(i3, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(pow(i2, x), pow(i2, sub(div(i3, i2), x)));
    r2 = mul(i2, pow(i2, div(one, i2)));
    std::cout << r1->__str__() << std::endl;
    REQUIRE(eq(*r1, *r2));

    RCP<const Basic> r3;
    rc1 = Complex::from_two_nums(*one, *one);
    r1 = pow(rc1, x);
    r3 = mul(r1, pow(rc1, sub(div(i3, i2), x)));
    r2 = pow(rc1, div(i3, i2));
    REQUIRE(eq(*r3, *r2));
    r3 = mul({r1, pow(rc1, sub(div(i3, i2), x))});
    REQUIRE(eq(*r3, *r2));

    r1 = real_double(0.1);
    r2 = Rational::from_mpq(rational_class(1, 2));
    r2 = mul(mul(mul(r1, r2), integer(3)), real_double(0.2));
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 0.03) < 1e-12);
    r2 = mul({r1, Rational::from_mpq(rational_class(1, 2)), integer(3),
              real_double(0.2)});
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 0.03) < 1e-12);

    r1 = complex_double(std::complex<double>(0.1, 0.2));
    r2 = Complex::from_two_nums(*Rational::from_mpq(rational_class(1, 2)),
                                *Rational::from_mpq(rational_class(7, 5)));
    r3 = mul(mul(mul(r1, r2), integer(5)), real_double(0.7));
    REQUIRE(is_a<ComplexDouble>(*r3));
    REQUIRE(down_cast<const ComplexDouble &>(*r3).is_complex());
    REQUIRE(not down_cast<const ComplexDouble &>(*r3).is_minus_one());
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.real() + 0.805)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.imag() - 0.84)
            < 1e-12);
    r3 = mul({r1, r2, integer(5), real_double(0.7)});
    REQUIRE(is_a<ComplexDouble>(*r3));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.real() + 0.805)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r3).i.imag() - 0.84)
            < 1e-12);

    r1 = real_double(0.0);
    r2 = mul(r1, x);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(i2, mul(pow(x, i2), y));
    r1 = mul(r1, pow(x, real_double(-2.0)));
    r2 = mul(y, real_double(2.0));
    // (2*x**2*y) * (x**(-2.0)) == 2.0 * y
    REQUIRE(eq(*r1, *r2));
    r1 = mul({i2, pow(x, i2), y, pow(x, real_double(-2.0))});
    REQUIRE(eq(*r1, *r2));

    std::set<RCP<const Basic>, SymEngine::RCPBasicKeyLess> s;
    rc1 = Complex::from_two_nums(*one, *one);
    s.insert(rc1);
    rc1 = Complex::from_two_nums(*i2, *one);
    s.insert(rc1);
    rc1 = Complex::from_two_nums(*one, *one);
    s.insert(rc1);
    REQUIRE(s.size() == 2);

    CHECK_THROWS_AS(Complex::from_two_nums(*one, *real_double(1.0)),
                    SymEngineException &);

    r1 = mul({});
    REQUIRE(eq(*r1, *one));

    r1 = mul({i2});
    REQUIRE(eq(*r1, *i2));

    RCP<const Number> s1, s2, s3;

    s1 = complex_double(std::complex<double>(1.0, 2.0));
    rc1 = Rational::from_two_ints(*integer(2), *integer(1));
    s2 = complex_double(std::complex<double>(3.0, 2.0));
    s3 = s1->add(*rc1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(2.0, 4.0));
    s3 = s1->add(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(2.0, 2.0));
    s3 = s1->sub(*integer(-1));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-1.0, 2.0));
    s3 = s1->sub(*Rational::from_two_ints(2, 1));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(0.0, 0.0));
    s3 = s1->sub(*Complex::from_two_nums(*integer(1), *integer(2)));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(0.0, 2.0));
    s3 = s1->sub(*real_double(1.0));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(0.0, 0.0));
    s3 = s1->sub(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-2.0, -2.0));
    s3 = integer(-1)->sub(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(1.0, -2.0));
    s3 = Rational::from_two_ints(2, 1)->sub(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(0.0, 0.0));
    s3 = Complex::from_two_nums(*integer(1), *integer(2))->sub(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(0.0, -2.0));
    s3 = real_double(1.0)->sub(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-3.0, 4.0));
    s3 = s1->mul(*s1);
    REQUIRE(eq(*s2, *s3));

    s1 = complex_double(std::complex<double>(4.0, 4.0));
    s2 = complex_double(std::complex<double>(-2.0, -2.0));
    s3 = s1->div(*integer(-2));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-2.0, -2.0));
    s3 = s1->div(*Rational::from_two_ints(-2, 1));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-2.0, -2.0));
    s3 = s1->div(*real_double(-2.0));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(4.0, 0.0));
    s3 = s1->div(*Complex::from_two_nums(*integer(1), *integer(1)));
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(1.0, 0.0));
    s3 = s1->div(*s1);
    REQUIRE(eq(*s2, *s3));

    s1 = complex_double(std::complex<double>(1.0, -1.0));
    s2 = complex_double(std::complex<double>(1.0, 1.0));
    s3 = integer(2)->div(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-1.0, -1.0));
    s3 = Rational::from_two_ints(-2, 1)->div(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(-1.0, -1.0));
    s3 = real_double(-2.0)->div(*s1);
    REQUIRE(eq(*s2, *s3));

    s2 = complex_double(std::complex<double>(1.0, 0.0));
    s3 = Complex::from_two_nums(*integer(1), *integer(-1))->div(*s1);
    REQUIRE(eq(*s2, *s3));

    s1 = complex_double(std::complex<double>(1.0, 1.0));
    s2 = complex_double(std::complex<double>(0.0, 2.0));
    s3 = s1->pow(*integer(2));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(std::complex<double>(0.0, 2.0));
    s3 = s1->pow(*Rational::from_two_ints(2, 1));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(std::complex<double>(0.0, 2.0));
    s3 = s1->pow(*real_double(2.0));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(
        std::complex<double>(0.27395725383012, 0.58370075875861));
    s3 = s1->pow(*Complex::from_two_nums(*integer(1), *integer(1)));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(
        std::complex<double>(0.27395725383012, 0.58370075875861));
    s3 = s1->pow(*s1);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(
        std::complex<double>(1.53847780272794, 1.27792255262727));
    s3 = Rational::from_two_ints(2, 1)->pow(*s1);
    std::cout << *s3 << std::endl;
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(
        std::complex<double>(1.53847780272794, 1.27792255262727));
    s3 = real_double(2.0)->pow(*s1);
    std::cout << *s3 << std::endl;
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(
        std::complex<double>(0.27395725383012, 0.58370075875861));
    s3 = Complex::from_two_nums(*integer(1), *integer(1))->pow(*s1);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.real()
                     - down_cast<const ComplexDouble &>(*s2).i.real())
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*s3).i.imag()
                     - down_cast<const ComplexDouble &>(*s2).i.imag())
            < 1e-12);

    s2 = complex_double(std::complex<double>(0.0, 0.0));
    s3 = complex_double(std::complex<double>(1.0, 0.0));
    REQUIRE(eq(*exp(s2), *s3));
}

TEST_CASE("Sub: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    RCP<const Basic> r1, r2;

    r1 = sub(i3, i2);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = sub(x, x);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = sub(mul(i2, x), x);
    r2 = x;
    REQUIRE(eq(*r1, *r2));

    r1 = add(mul(mul(i2, x), y), mul(x, y));
    r2 = mul(i3, mul(x, y));
    REQUIRE(eq(*r1, *r2));

    r1 = add(mul(mul(i2, x), y), mul(x, y));
    r2 = mul(mul(x, y), i3);
    REQUIRE(eq(*r1, *r2));

    r1 = sub(mul(mul(i2, x), y), mul(x, y));
    r2 = mul(x, y);
    REQUIRE(eq(*r1, *r2));

    r1 = sub(add(x, one), x);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = add(add(x, one), add(x, i2));
    r2 = add(mul(i2, x), i3);
    REQUIRE(eq(*r1, *r2));

    r1 = sub(add(x, one), add(x, i2));
    r1 = expand(r1);
    r2 = im1;
    REQUIRE(eq(*r1, *r2));

    r1 = add(add(x, y), real_double(0.0));
    REQUIRE(eq(*sub(r1, y), *x));

    RCP<const Number> rc1, rc2, rc3, c1, c2;
    rc1 = Rational::from_two_ints(*integer(1), *integer(2));
    rc2 = Rational::from_two_ints(*integer(3), *integer(4));
    rc3 = Rational::from_two_ints(*integer(-5), *integer(6));

    c1 = Complex::from_two_nums(*rc1, *rc2);
    c2 = Complex::from_two_nums(*rc1, *rc3);

    r1 = mul(x, c1);
    r2 = mul(x, c2);
    r1 = sub(r1, r2);
    r2 = mul(div(mul(I, integer(19)), integer(12)), x);
    REQUIRE(eq(*r1, *r2));

    r1 = real_double(0.1);
    r2 = Rational::from_mpq(rational_class(1, 2));
    r2 = sub(r1, sub(sub(sub(r1, r2), integer(3)), real_double(0.2)));
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 3.7) < 1e-12);

    r1 = real_double(0.1);
    r2 = Complex::from_two_nums(*Rational::from_mpq(rational_class(1, 2)),
                                *Rational::from_mpq(rational_class(7, 5)));
    r2 = sub(sub(sub(r1, r2), integer(1)), real_double(0.4));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r2).i.real() + 1.8)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r2).i.imag() + 1.4)
            < 1e-12);
}

TEST_CASE("Div: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);

    REQUIRE(integer(2)->is_positive());
    REQUIRE(integer(0)->is_zero());
    REQUIRE(integer(1)->is_one());
    REQUIRE(not(integer(-1)->is_positive()));
    REQUIRE(integer(-1)->is_negative());

    RCP<const Basic> r1, r2;

    r1 = div(i4, integer(1));
    r2 = mul(integer(1), i4);
    std::cout << "r1: " << *r1 << std::endl;
    std::cout << "r2: " << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = pow(i3, i2);
    r2 = integer(9);
    REQUIRE(eq(*r1, *r2));

    r1 = div(i4, i2);
    r2 = i2;
    REQUIRE(eq(*r1, *r2));

    r1 = div(x, x);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = div(mul(i2, x), x);
    r2 = i2;
    REQUIRE(eq(*r1, *r2));

    r1 = div(pow(x, i2), x);
    r2 = x;
    REQUIRE(eq(*r1, *r2));

    r1 = div(zero, zero);
    REQUIRE(eq(*r1, *Nan));

    r1 = div(mul(mul(i2, x), y), mul(x, y));
    r2 = i2;
    std::cout << "r1: " << *r1 << std::endl;
    std::cout << "r2: " << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = div(mul(mul(y, x), i2), mul(x, y));
    r2 = i2;
    REQUIRE(eq(*r1, *r2));

    r1 = div(mul(x, i2), x);
    r2 = i2;
    REQUIRE(eq(*r1, *r2));

    r1 = div(mul(x, i4), mul(x, i2));
    r2 = i2;
    REQUIRE(eq(*r1, *r2));

    r1 = div(i2, div(i3, mul(i2, im1)));
    r2 = mul(im1, div(i4, i3));
    REQUIRE(eq(*r1, *r2));

    r1 = div(i4, mul(im1, i2));
    r2 = mul(im1, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = div(i4, im1);
    r2 = mul(im1, i4);
    REQUIRE(eq(*r1, *r2));

    r1 = div(integer(5), div(integer(1), integer(3)));
    REQUIRE(eq(*r1, *integer(15)));

    RCP<const Number> rc1, rc2, rc3, c1, c2;
    rc1 = Rational::from_two_ints(*integer(1), *integer(2));
    rc2 = Rational::from_two_ints(*integer(3), *integer(4));
    rc3 = Rational::from_two_ints(*integer(12), *integer(13));

    c1 = Complex::from_two_nums(*rc1, *rc2);
    c2 = Complex::from_two_nums(*rc1, *rc1);

    r1 = div(x, c1);
    r2 = mul(sub(div(integer(8), integer(13)), mul(I, rc3)), x);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(c2, div(x, c1));
    rc3 = Rational::from_two_ints(*integer(2), *integer(13));
    r2 = mul(sub(div(integer(10), integer(13)), mul(I, rc3)), x);
    REQUIRE(eq(*r1, *r2));

    r1 = real_double(0.1);
    r2 = Rational::from_mpq(rational_class(1, 2));
    r2 = div(div(div(r1, r2), integer(3)), real_double(0.2));
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 0.333333333333)
            < 1e-12);

    r1 = real_double(0.1);
    r2 = Complex::from_two_nums(*Rational::from_mpq(rational_class(1, 2)),
                                *Rational::from_mpq(rational_class(7, 5)));
    r2 = div(div(div(r1, r2), integer(2)), real_double(0.4));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r2).i.real()
                     - 0.0282805429864253)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r2).i.imag()
                     + 0.0791855203619909)
            < 1e-12);
}

TEST_CASE("Pow: arit", "[arit]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im3 = integer(-3);
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> i27 = integer(27);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = mul(x, x);
    r2 = pow(x, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(x, x), x);
    r2 = pow(x, i3);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(mul(x, x), x), x);
    r2 = pow(x, i4);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(add(x, y), add(x, y)), add(x, y));
    r2 = pow(add(x, y), i3);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(mul(add(x, y), add(y, x)), add(x, y));
    r2 = pow(add(x, y), i3);
    REQUIRE(eq(*r1, *r2));

    r1 = sub(pow(x, y), pow(x, y));
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = pow(zero, zero);
    REQUIRE(eq(*r1, *one));

    r1 = pow(zero, i2);
    REQUIRE(eq(*r1, *zero));

    r1 = pow(zero, im1);
    REQUIRE(eq(*r1, *ComplexInf));

    /* Test (x*y)**2 -> x**2*y**2 type of simplifications */

    r1 = pow(mul(x, y), i2);
    r2 = mul(pow(x, i2), pow(y, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(i2, mul(x, y)), i2);
    r2 = mul(i4, mul(pow(x, i2), pow(y, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(i3, mul(x, y)), i2);
    r2 = mul(i9, mul(pow(x, i2), pow(y, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(i3, mul(x, y)), im1);
    r2 = mul(div(one, i3), mul(pow(x, im1), pow(y, im1)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(i3, mul(pow(x, i2), pow(y, i3))), i2);
    r2 = mul(i9, mul(pow(x, i4), pow(y, i6)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(i3, mul(pow(x, i2), pow(y, im1))), i3);
    r2 = mul(i27, mul(pow(x, i6), pow(y, im3)));
    REQUIRE(eq(*r1, *r2));

    /*    */
    r1 = sqrt(x);
    r1 = r1->diff(x)->diff(x);
    r2 = mul(div(im1, i4), pow(x, div(im3, i2)));
    REQUIRE(eq(*r1, *r2));

    // Just test that it works:
    r2 = sin(r1)->diff(x)->diff(x);

    r1 = div(one, sqrt(i2));
    r2 = mul(pow(i2, pow(i2, im1)), pow(i2, im1));
    REQUIRE(eq(*r1, *r2));

    r1 = div(one, sqrt(i2));
    r2 = div(sqrt(i2), i2);
    REQUIRE(eq(*r1, *r2));

    r1 = exp(pow(x, i3));
    r1 = r1->diff(x);
    r2 = mul(mul(i3, exp(pow(x, i3))), pow(x, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(x, x);
    r1 = r1->diff(x);
    r2 = mul(pow(x, x), add(log(x), one));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(x, y);
    r1 = r1->diff(x);
    r2 = mul(pow(x, sub(y, one)), y);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(y, x);
    r1 = r1->diff(x);
    r2 = mul(pow(y, x), log(y));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(div(i4, i6), i2);
    REQUIRE(eq(*r1, *div(integer(4), integer(9))));

    r1 = pow(i2, div(im1, i2));
    r2 = div(sqrt(i2), i2);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(div(i3, i2), div(integer(7), i2));
    r2 = mul(div(integer(27), integer(16)),
             mul(pow(i2, div(integer(1), i2)), pow(i3, div(integer(1), i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(div(i2, i3), div(integer(7), i2));
    r2 = mul(div(integer(8), integer(81)),
             mul(pow(i2, div(integer(1), i2)), pow(i3, div(integer(1), i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(i6, div(integer(7), i2));
    r2 = mul(integer(216), pow(i6, div(integer(1), i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(div(i3, i2), div(integer(-7), i2));
    r2 = mul(div(integer(8), integer(81)),
             mul(pow(i2, div(integer(1), i2)), pow(i3, div(integer(1), i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(i6, div(integer(-7), i2));
    r2 = mul(div(one, integer(1296)), pow(i6, div(integer(1), i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(pow(i3, div(i27, i4)), pow(i2, div(integer(-13), i6)));
    r2 = mul(mul(div(integer(729), integer(8)), pow(i3, div(i3, i4))),
             pow(i2, div(integer(5), i6)));
    REQUIRE(eq(*r1, *r2));

    r1 = div(integer(12), pow(integer(196), div(integer(1), integer(2))));
    r2 = mul(div(i3, integer(49)), sqrt(integer(196)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(div(sqrt(integer(12)), sqrt(integer(6))), integer(2));
    r2 = integer(2);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(pow(x, y), z);
    r2 = pow(x, mul(y, z));
    REQUIRE(neq(*r1, *r2));

    r1 = pow(mul(x, y), z);
    r2 = mul(pow(x, z), pow(y, z));
    REQUIRE(neq(*r1, *r2));

    RCP<const Number> rc1, rc2, rc3, c1, c2;
    rc1 = Rational::from_two_ints(*integer(1), *integer(2));
    rc2 = Rational::from_two_ints(*integer(3), *integer(4));
    rc3 = Rational::from_two_ints(*integer(12), *integer(13));

    c1 = Complex::from_two_nums(*rc1, *rc2);
    c2 = Complex::from_two_nums(*rc1, *rc1);

    r1 = pow(x, c1);
    r2 = mul(pow(x, div(one, i2)), pow(x, mul(I, div(i3, i4))));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(c1, x);
    r2 = pow(c1, y);
    r1 = mul(r1, r2);
    r2 = pow(c1, add(x, y));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(I, integer(3));
    r2 = mul(im1, I);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(I, i2), i2);
    r2 = mul(im1, i4);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(I, im3), integer(5));
    r2 = mul(integer(243), mul(I, im1));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(I, im3), integer(4));
    r2 = integer(81);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(im1, div(one, i2));
    r2 = I;
    REQUIRE(eq(*r1, *r2));

    r1 = pow(im1, div(i6, i4));
    r2 = mul(im1, I);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(im1, div(integer(9), i6));
    r2 = mul(im1, I);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(im3, div(integer(9), i6));
    r2 = mul(mul(im3, I), pow(i3, div(one, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(im3, div(i4, i3));
    r2 = pow(r1, i3);
    REQUIRE(eq(*r2, *integer(81)));

    r1 = sqrt(div(one, i4));
    r2 = div(one, i2);
    REQUIRE(eq(*r1, *r2));

    r1 = sqrt(div(i3, i4));
    r2 = div(sqrt(i3), i2);
    REQUIRE(eq(*r1, *r2));

    r1 = sqrt(div(i4, i3));
    r2 = div(mul(i2, sqrt(i3)), i3);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(integer(8), div(i2, i3));
    r2 = i4;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(pow(integer(8), x), pow(integer(8), sub(div(i2, i3), x)));
    r2 = i4;
    REQUIRE(eq(*r1, *r2));

    r1 = real_double(0.1);
    r2 = Rational::from_mpq(rational_class(1, 2));
    r2 = pow(r1, r2);
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 0.316227766016)
            < 1e-12);
    r2 = pow(pow(r2, integer(3)), real_double(0.2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 0.501187233627)
            < 1e-12);
    r2 = pow(E, real_double(0.2));
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 1.22140275816017)
            < 1e-12);
    r2 = exp(x)->subs({{x, real_double(1.0)}});
    REQUIRE(is_a<RealDouble>(*r2));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r2).i - 2.71828182845905)
            < 1e-12);

    r1 = real_double(-0.01);
    r2 = pow(r1, Rational::from_mpq(rational_class(1, 2)));
    r2 = pow(integer(2), r2);
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r2).i.real()
                     - 0.997598696589298)
            < 1e-12);
    REQUIRE(std::abs(down_cast<const ComplexDouble &>(*r2).i.imag()
                     - 0.069259227279362)
            < 1e-12);

    r1 = pow(x, real_double(0.0));
    r2 = real_double(1.0);
    REQUIRE(eq(*r1, *r2));

    r1 = sqrt(mul(i2, x));
    r2 = mul(sqrt(i2), sqrt(x));
    REQUIRE(eq(*r1, *r2));

    r1 = sqrt(mul(neg(i2), x));
    r2 = mul(sqrt(i2), sqrt(neg(x)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(mul(sqrt(mul(y, x)), x), i2);
    r2 = mul(pow(x, i3), y);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(one, Inf);
    REQUIRE(eq(*r1, *Nan));

    r1 = pow(one, NegInf);
    REQUIRE(eq(*r1, *Nan));

    r1 = pow(one, ComplexInf);
    REQUIRE(eq(*r1, *Nan));

    r1 = pow(one, Nan);
    REQUIRE(eq(*r1, *Nan));
}

TEST_CASE("Log: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> im3 = integer(-3);
    RCP<const Basic> q = Complex::from_two_nums(*zero, *zero);
    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = log(E);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = log(one);
    r2 = zero;
    REQUIRE(eq(*r1, *r2));

    r1 = log(zero);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = log(q);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = log(Inf);
    REQUIRE(eq(*r1, *Inf));

    r1 = log(NegInf);
    REQUIRE(eq(*r1, *Inf));

    r1 = log(ComplexInf);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = log(i2, zero);
    REQUIRE(eq(*r1, *zero));

    r1 = log(i3);
    REQUIRE(is_a<Log>(*r1));

    r1 = log(im3);
    r2 = add(log(i3), mul(I, pi));
    REQUIRE(eq(*r1, *r2));

    RCP<const Number> c1;

    c1 = Complex::from_two_nums(*integer(0), *integer(2));
    r1 = log(c1);
    r2 = add(log(i2), mul(mul(I, pi), div(one, i2)));
    REQUIRE(eq(*r1, *r2));

    c1 = Complex::from_two_nums(*integer(0), *integer(-2));
    r1 = log(c1);
    r2 = sub(log(i2), mul(mul(I, pi), div(one, i2)));
    REQUIRE(eq(*r1, *r2));

    c1 = Complex::from_two_nums(*integer(0), *integer(-1));
    r1 = log(c1);
    r2 = mul(minus_one, mul(mul(I, pi), div(one, i2)));
    REQUIRE(eq(*r1, *r2));

    c1 = Complex::from_two_nums(*integer(2), *integer(-2));
    r1 = log(c1);
    REQUIRE(is_a<Log>(*r1));

    c1 = Complex::from_two_nums(*integer(0), *integer(0));
    r1 = log(c1);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = log(div(i2, i3));
    r2 = sub(log(i2), log(i3));
    REQUIRE(eq(*r1, *r2));

    r1 = log(E, i2);
    r2 = div(one, log(i2));
    REQUIRE(eq(*r1, *r2));

    r1 = log(x);
    r1 = r1->subs({{x, E}});
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = log(real_double(2.0));
    REQUIRE(is_a<RealDouble>(*r1));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*r1).i - 0.693147180559945)
            < 1e-12);

    r1 = log(complex_double(std::complex<double>(1, 2)));
    r2 = log(real_double(-3.0));
    REQUIRE(is_a<ComplexDouble>(*r1));
    REQUIRE(is_a<ComplexDouble>(*r2));
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r1).i)
                     - 1.36870408847499)
            < 1e-12);
    REQUIRE(std::abs(std::abs(down_cast<const ComplexDouble &>(*r2).i)
                     - 3.32814563411849)
            < 1e-12);

    // Test is_canonical()
    RCP<const Log> r4 = make_rcp<Log>(i2);
    REQUIRE(not(r4->is_canonical(zero)));
    REQUIRE(not(r4->is_canonical(one)));
    REQUIRE(not(r4->is_canonical(E)));
    REQUIRE(not(r4->is_canonical(minus_one)));
    REQUIRE(not(r4->is_canonical(im3)));
    REQUIRE(not(r4->is_canonical(c1)));
    REQUIRE(r4->is_canonical(i2));
    REQUIRE(not(r4->is_canonical(real_double(2.0))));
    REQUIRE(not(r4->is_canonical(div(one, i2))));
}

TEST_CASE("Multinomial: arit", "[arit]")
{
    map_vec_uint r;
    auto t1 = std::chrono::high_resolution_clock::now();
    multinomial_coefficients(4, 20, r);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
}

TEST_CASE("Expand1: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(10);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = pow(add(add(add(x, y), z), w), i4);

    std::cout << *r1 << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    r2 = expand(r1);
    auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << *r2 << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    std::cout << "number of terms: "
              << rcp_dynamic_cast<const Add>(r2)->get_dict().size()
              << std::endl;
}

TEST_CASE("Expand2: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> im1 = integer(-1);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i4 = integer(4);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> i6 = integer(6);
    RCP<const Basic> i9 = integer(9);
    RCP<const Basic> i10 = integer(10);
    RCP<const Basic> i12 = integer(12);
    RCP<const Basic> i16 = integer(16);
    RCP<const Basic> i24 = integer(24);
    RCP<const Basic> i25 = integer(25);
    RCP<const Basic> i30 = integer(30);

    RCP<const Basic> r1;
    RCP<const Basic> r2;

    r1 = mul(w, add(add(x, y), z)); // w*(x+y+z)
    std::cout << *r1 << std::endl;

    r2 = expand(r1);
    std::cout << *r2 << std::endl;

    REQUIRE(eq(*r2, *add(add(mul(w, x), mul(w, y)), mul(w, z))));
    REQUIRE(neq(*r2, *add(add(mul(w, x), mul(w, w)), mul(w, z))));

    r1 = mul(add(x, y), add(z, w)); // (x+y)*(z+w)
    std::cout << *r1 << std::endl;

    r2 = expand(r1);
    std::cout << *r2 << std::endl;

    REQUIRE(
        eq(*r2, *add(add(add(mul(x, z), mul(y, z)), mul(x, w)), mul(y, w))));
    REQUIRE(
        neq(*r2, *add(add(add(mul(y, z), mul(y, z)), mul(x, w)), mul(y, w))));

    r1 = pow(add(x, y), im1); // 1/(x+y)
    std::cout << *r1 << std::endl;

    r2 = expand(r1);
    std::cout << *r2 << std::endl;

    REQUIRE(eq(*r2, *r1));

    r1 = pow(add(x, y), im2); // 1/(x+y)^2
    std::cout << *r1 << std::endl;

    r2 = expand(r1);
    std::cout << *r2 << std::endl;

    REQUIRE(eq(
        *r2, *pow(add(add(pow(x, i2), mul(mul(i2, x), y)), pow(y, i2)), im1)));
    REQUIRE(neq(*r2, *r1));

    r1 = mul(im1, add(x, i2));
    r1 = expand(r1);
    r2 = add(mul(im1, x), im2);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(x, one), i2);
    r1 = expand(r1);
    r2 = add(add(pow(x, i2), mul(i2, x)), one);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(x, i2), i2);
    r1 = expand(r1);
    r2 = add(add(pow(x, i2), mul(i4, x)), i4);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(x, i3), i2);
    r1 = expand(r1);
    r2 = add(add(pow(x, i2), mul(i6, x)), i9);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(mul(i3, x), i5), i2);
    r1 = expand(r1);
    r2 = add(add(mul(i9, pow(x, i2)), mul(i30, x)), i25);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(mul(i2, pow(x, i2)), mul(i3, y)), i2);
    r1 = expand(r1);
    r2 = add(add(mul(i4, pow(x, i4)), mul(i12, mul(pow(x, i2), y))),
             mul(i9, pow(y, i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(add(pow(x, i3), pow(x, i2)), x), i2);
    r1 = expand(r1);
    r2 = add(add(add(add(pow(x, i6), mul(i2, pow(x, i5))), mul(i3, pow(x, i4))),
                 mul(i2, pow(x, i3))),
             pow(x, i2));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(x, pow(x, i5)), i2);
    r1 = expand(r1);
    r2 = add(add(pow(x, i10), mul(i2, pow(x, i6))), pow(x, i2));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(add(i2, x), add(i3, y));
    r1 = expand(r1);
    r2 = add(add(add(i6, mul(i2, y)), mul(i3, x)), mul(x, y));
    std::cout << *r1 << std::endl;
    std::cout << *r2 << std::endl;
    REQUIRE(eq(*r1, *r2));

    r1 = mul(i3, pow(i5, div(im1, i2)));
    r2 = mul(i4, pow(i5, div(im1, i2)));
    r2 = expand(pow(add(add(r1, r2), integer(1)), i2));
    REQUIRE(eq(*r2, *add(div(integer(54), i5),
                         mul(integer(14), pow(i5, div(im1, i2))))));

    r1 = pow(add(mul(I, x), i2), i2);
    r1 = expand(r1);
    r2 = add(sub(mul(mul(I, x), i4), pow(x, i2)), i4);
    REQUIRE(eq(*r1, *r2));

    r1 = mul(add(sqrt(i3), one), add(sqrt(i3), i2));
    r1 = expand(r1);
    r2 = add(i5, mul(i3, sqrt(i3)));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(add(mul(i2, x), sqrt(i2)), add(x, mul(i2, sqrt(i2))));
    r1 = expand(r1);
    r2 = add(mul(i2, mul(x, x)), add(mul(x, mul(sqrt(i2), i5)), i4));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(sqrt(i2), add(mul(x, i2), mul(i2, sqrt(i2))));
    r1 = expand(r1);
    r2 = add(mul(i2, mul(sqrt(i2), x)), i4);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(pow(add(one, sqrt(i2)), i2), one), i2);
    r1 = expand(r1);
    r2 = add(i24, mul(i16, sqrt(i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = mul(add(mul(sqrt(i3), x), one), sub(mul(sqrt(i3), x), one));
    r1 = expand(r1);
    r2 = sub(mul(mul(i3, x), x), one);
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(i2, mul(y, x)), i2);
    r1 = expand(r1);
    r2 = add(i4, add(mul(mul(i4, x), y), pow(mul(x, y), i2)));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(sub(sub(pow(add(x, one), i2), pow(x, i2)), mul(x, i2)), i2);
    r1 = expand(r1);
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    // The following test that the expand method outputs canonical objects
    r1 = pow(add(y, mul(sqrt(i3), z)), i2);
    r1 = expand(mul(r1, add(r1, one)));
    std::cout << r1->__str__() << std::endl;

    r1 = pow(add(y, mul(sqrt(i3), z)), i3);
    r1 = expand(mul(r1, add(r1, one)));
    std::cout << r1->__str__() << std::endl;

    r1 = pow(mul(sqrt(i3), mul(y, add(one, pow(i3, div(one, i3))))), i3);
    r1 = expand(mul(r1, add(r1, one)));
    std::cout << r1->__str__() << std::endl;

    r1 = expand(pow(add(sqrt(i2), mul(sqrt(i2), x)), i2));
    r2 = add(i2, add(mul(i4, x), mul(i2, pow(x, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = real_double(0.2);
    r1 = add(r1, x);
    r2 = expand(pow(r1, i2));
    REQUIRE(is_a<Add>(*r2));
    auto it = down_cast<const Add &>(*r2).get_dict().find(x);
    REQUIRE(it != down_cast<const Add &>(*r2).get_dict().end());
    REQUIRE(is_a<RealDouble>(*it->second));
    REQUIRE(std::abs(down_cast<const RealDouble &>(*(it->second)).i - 0.4)
            < 1e-12);

    r1 = expand(pow(add(real_double(0.0), x), i2));
    r2 = add(real_double(0.0), pow(x, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = expand(add(mul(i2, add(x, one)), mul(i3, mul(x, add(x, one)))));
    r2 = add(i2, add(mul(i5, x), mul(i3, pow(x, i2))));
    REQUIRE(eq(*r1, *r2));

    r1 = expand(mul(i3, add(x, one)));
    r2 = add(mul(i3, x), i3);
    REQUIRE(eq(*r1, *r2));

    r1 = expand(pow(add(sqrt(add(x, y)), one), i2));
    r2 = add(x, add(y, add(one, mul(i2, sqrt(add(x, y))))));
    REQUIRE(eq(*r1, *r2));

    r1 = expand(pow(add(mul(I, y), x), i3));
    r2 = add(
        sub(pow(x, i3), mul(pow(y, i3), I)),
        sub(mul(mul(i3, I), mul(pow(x, i2), y)), mul(i3, mul(pow(y, i2), x))));
    REQUIRE(eq(*r1, *r2));

    // Test that deep=False doesn't expand expression two levels deep.
    r1 = expand(mul(i2, add(x, mul(i2, add(y, z)))), false);
    r2 = add(mul(i2, x), mul(i4, add(y, z)));
    REQUIRE(eq(*r1, *r2));

    r1 = add(x, mul(i2, add(y, z)));
    r2 = expand(r1, false);
    REQUIRE(eq(*r1, *r2));

    r1 = expand(pow(add(one, mul(i2, sqrt(add(y, z)))), i2));
    r2 = expand(add(one, mul(i4, add(y, add(z, sqrt(add(y, z)))))));
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("Expand3: arit", "[arit]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Basic> z = symbol("z");
    RCP<const Basic> w = symbol("w");
    RCP<const Basic> i2 = integer(2);

    RCP<const Basic> e, f, r;

    e = pow(add(add(add(x, y), z), w), i2);
    f = mul(e, add(e, w));

    std::cout << *f << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    r = expand(f);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << *r << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    std::cout << "number of terms: "
              << rcp_dynamic_cast<const Add>(r)->get_dict().size() << std::endl;

    RCP<const Number> rc1, rc2, c1;
    rc1 = Rational::from_two_ints(*integer(2), *integer(1));
    rc2 = Rational::from_two_ints(*integer(3), *integer(1));

    c1 = Complex::from_two_nums(*rc1, *rc2);
    e = pow(add(x, c1), integer(40));

    t1 = std::chrono::high_resolution_clock::now();
    r = expand(e);
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << *r << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
    std::cout << "number of terms: "
              << rcp_dynamic_cast<const Add>(r)->get_dict().size() << std::endl;

    e = pow(c1, integer(-40));

    t1 = std::chrono::high_resolution_clock::now();
    r = expand(e);
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << *r << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << "ms" << std::endl;
}
