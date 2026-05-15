#include "catch.hpp"

#include <symengine/ntheory_funcs.h>
#include <symengine/basic.h>
#include <symengine/rational.h>
#include <symengine/real_double.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Complex;
using SymEngine::DomainError;
using SymEngine::down_cast;
using SymEngine::Inf;
using SymEngine::Infty;
using SymEngine::integer;
using SymEngine::Integer;
using SymEngine::is_a;
using SymEngine::make_rcp;
using SymEngine::Nan;
using SymEngine::NaN;
using SymEngine::pi;
using SymEngine::polygonal_number;
using SymEngine::PrimePi;
using SymEngine::Primorial;
using SymEngine::principal_polygonal_root;
using SymEngine::Rational;
using SymEngine::RCP;
using SymEngine::real_double;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::SymEngineException;

TEST_CASE("test_primepi(): ntheory_funcs", "[ntheory_funcs]")
{
    REQUIRE(eq(*primepi(integer(2)), *integer(1)));
    REQUIRE(eq(*primepi(pi), *integer(2)));
    REQUIRE(eq(*primepi(integer(5)), *integer(3)));
    REQUIRE(eq(*primepi(integer(28)), *integer(9)));
    REQUIRE(eq(*primepi(integer(19876)), *integer(2248)));
    REQUIRE(eq(*primepi(integer(1000000)), *integer(78498)));
    REQUIRE(eq(*primepi(integer(-1)), *integer(0)));
    REQUIRE(eq(*primepi(integer(-1000)), *integer(0)));
    REQUIRE(eq(*primepi(Rational::from_two_ints(1, 2)), *integer(0)));
    REQUIRE(eq(*primepi(Rational::from_two_ints(-1, 2)), *integer(0)));
    REQUIRE(eq(*primepi(Rational::from_two_ints(900000, 8999)), *integer(25)));
    REQUIRE(eq(*primepi(real_double(-23.5)), *integer(0)));
    REQUIRE(eq(*primepi(real_double(23.5)), *integer(9)));
    REQUIRE(eq(*primepi(real_double(16545.234)), *integer(1914)));
    REQUIRE(is_a<NaN>(*primepi(Nan)));
    REQUIRE(is_a<Infty>(*primepi(Inf)));
    REQUIRE(eq(*primepi(mul(integer(-1), Inf)), *integer(0)));
    CHECK_THROWS_AS(*primepi(Complex::from_two_nums(*integer(1), *integer(1))),
                    SymEngineException);
    REQUIRE(eq(*primepi(symbol("x")), *make_rcp<PrimePi>(symbol("x"))));
    REQUIRE(primepi(symbol("x"))->__str__() == "primepi(x)");
    REQUIRE(eq(*PrimePi(symbol("x")).create(integer(2)), *integer(1)));
    REQUIRE(not(PrimePi(symbol("x")).is_canonical(integer(1))));
}

TEST_CASE("test_primorial(): ntheory_funcs", "[ntheory_funcs]")
{
    CHECK_THROWS_AS(*primorial(integer(0)), SymEngineException);
    REQUIRE(eq(*primorial(integer(1)), *integer(1)));
    REQUIRE(eq(*primorial(integer(2)), *integer(2)));
    REQUIRE(eq(*primorial(integer(3)), *integer(6)));
    REQUIRE(eq(*primorial(integer(9)), *integer(210)));
    REQUIRE(eq(*primorial(real_double(23.9)), *integer(223092870)));
    REQUIRE(eq(*primorial(pi), *integer(6)));
    REQUIRE(is_a<NaN>(*primorial(Nan)));
    REQUIRE(is_a<Infty>(*primorial(Inf)));
    REQUIRE(eq(*primorial(symbol("x")), *make_rcp<Primorial>(symbol("x"))));
    REQUIRE(primorial(symbol("x"))->__str__() == "primorial(x)");
    REQUIRE(eq(*Primorial(symbol("x")).create(integer(1)), *integer(1)));
    REQUIRE(not(Primorial(symbol("x")).is_canonical(integer(1))));
}

TEST_CASE("test_polygonal_number(): ntheory", "[ntheory]")
{
    RCP<const Integer> i1 = integer(1);
    RCP<const Integer> i2 = integer(2);
    RCP<const Integer> i3 = integer(3);
    RCP<const Integer> i4 = integer(4);
    RCP<const Integer> i5 = integer(5);
    RCP<const Integer> i6 = integer(6);
    RCP<const Integer> i16 = integer(16);
    RCP<const Integer> i35 = integer(35);
    RCP<const Symbol> s1 = symbol("n");

    REQUIRE(eq(*polygonal_number(i3, i1), *i1));
    REQUIRE(eq(*polygonal_number(i3, i2), *i3));
    REQUIRE(eq(*polygonal_number(i3, i3), *i6));
    REQUIRE(eq(*polygonal_number(i4, i2), *i4));
    REQUIRE(eq(*polygonal_number(i4, i4), *i16));
    REQUIRE(eq(*polygonal_number(i5, i5), *i35));
    REQUIRE(eq(*polygonal_number(i3, s1), *div(add(s1, pow(s1, i2)), i2)));
    REQUIRE(eq(*polygonal_number(i4, s1), *mul(s1, s1)));
    CHECK_THROWS_AS(*polygonal_number(integer(0), integer(2)), DomainError);
    CHECK_THROWS_AS(*polygonal_number(integer(3), integer(0)), DomainError);
}

TEST_CASE("test_principal_polygonal_root(): ntheory", "[ntheory]")
{
    RCP<const Integer> m1 = integer(-1);
    RCP<const Integer> i1 = integer(1);
    RCP<const Integer> i2 = integer(2);
    RCP<const Integer> i3 = integer(3);
    RCP<const Integer> i4 = integer(4);
    RCP<const Integer> i5 = integer(5);
    RCP<const Integer> i6 = integer(6);
    RCP<const Integer> i8 = integer(8);
    RCP<const Integer> i16 = integer(16);
    RCP<const Integer> i35 = integer(35);
    RCP<const Symbol> s1 = symbol("x");

    REQUIRE(eq(*principal_polygonal_root(i3, i1), *i1));
    REQUIRE(eq(*principal_polygonal_root(i3, i3), *i2));
    REQUIRE(eq(*principal_polygonal_root(i3, i6), *i3));
    REQUIRE(eq(*principal_polygonal_root(i4, i4), *i2));
    REQUIRE(eq(*principal_polygonal_root(i4, i16), *i4));
    REQUIRE(eq(*principal_polygonal_root(i5, i35), *i5));
    REQUIRE(eq(*principal_polygonal_root(i3, s1),
               *div(add(m1, sqrt(add(i1, mul(i8, s1)))), i2)));
    REQUIRE(eq(*principal_polygonal_root(i4, s1), *sqrt(s1)));
    CHECK_THROWS_AS(*principal_polygonal_root(integer(0), integer(2)),
                    DomainError);
    CHECK_THROWS_AS(*principal_polygonal_root(integer(3), integer(0)),
                    DomainError);
}
