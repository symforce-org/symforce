#include "catch.hpp"
#include <symengine/assumptions.h>
#include <symengine/sets.h>

using SymEngine::Assumptions;
using SymEngine::Basic;
using SymEngine::complexes;
using SymEngine::integer;
using SymEngine::integers;
using SymEngine::Number;
using SymEngine::Rational;
using SymEngine::rationals;
using SymEngine::RCP;
using SymEngine::reals;
using SymEngine::Set;
using SymEngine::symbol;
using SymEngine::SymEngineException;

TEST_CASE("Test assumptions", "[assumptions]")
{
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Set> s1 = reals();
    RCP<const Set> s2 = integers();
    RCP<const Set> s3 = rationals();
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Basic> rel1 = Le(x, integer(0));
    RCP<const Basic> rel2 = Ge(x, integer(0));
    RCP<const Basic> rel3 = Le(x, integer(1));
    RCP<const Basic> rel4 = Ge(x, integer(1));
    RCP<const Basic> rel5 = Lt(x, integer(0));
    RCP<const Basic> rel6 = Gt(x, integer(0));
    RCP<const Basic> rel7 = Lt(x, integer(1));
    RCP<const Basic> rel8 = Ge(x, rat1);
    RCP<const Basic> rel9 = Le(x, integer(-2));
    RCP<const Basic> rel10 = Ge(x, integer(-2));
    RCP<const Basic> rel11 = Lt(x, integer(-2));
    RCP<const Basic> rel12 = Gt(x, integer(-2));
    RCP<const Basic> rel13 = Eq(x, integer(0));
    RCP<const Basic> rel14 = Ne(x, integer(0));
    RCP<const Basic> rel;

    Assumptions a = Assumptions({complexes()->contains(x)});
    REQUIRE(is_true(a.is_complex(x)));
    REQUIRE(is_indeterminate(a.is_real(x)));

    auto a1 = Assumptions({s1->contains(x)});
    REQUIRE(is_true(a1.is_real(x)));
    REQUIRE(is_indeterminate(a1.is_integer(x)));
    REQUIRE(is_indeterminate(a1.is_real(y)));

    auto a2 = Assumptions({s2->contains(x)});
    REQUIRE(is_true(a2.is_real(x)));
    REQUIRE(is_true(a2.is_integer(x)));
    REQUIRE(is_indeterminate(a2.is_real(y)));
    REQUIRE(is_indeterminate(a2.is_integer(y)));

    auto a3 = Assumptions({s1->contains(x), s2->contains(y)});
    REQUIRE(is_indeterminate(a3.is_integer(x)));
    REQUIRE(is_true(a3.is_integer(y)));
    REQUIRE(is_true(a3.is_real(x)));
    REQUIRE(is_true(a3.is_real(y)));

    auto a4 = Assumptions({s3->contains(x)});
    REQUIRE(is_true(a4.is_rational(x)));
    REQUIRE(is_indeterminate(a4.is_rational(y)));

    auto a5 = Assumptions({rel1});
    REQUIRE(is_true(a5.is_real(x)));
    REQUIRE(is_true(a5.is_nonpositive(x)));
    REQUIRE(is_indeterminate(a5.is_negative(x)));
    REQUIRE(is_indeterminate(a5.is_nonnegative(x)));
    REQUIRE(is_false(a5.is_positive(x)));
    REQUIRE(is_indeterminate(a5.is_nonzero(x)));
    REQUIRE(is_indeterminate(a5.is_zero(x)));
    REQUIRE(is_indeterminate(a5.is_rational(x)));

    auto a6 = Assumptions({rel2});
    REQUIRE(is_true(a6.is_real(x)));
    REQUIRE(is_indeterminate(a6.is_nonpositive(x)));
    REQUIRE(is_false(a6.is_negative(x)));
    REQUIRE(is_true(a6.is_nonnegative(x)));
    REQUIRE(is_indeterminate(a6.is_positive(x)));
    REQUIRE(is_indeterminate(a6.is_nonzero(x)));
    REQUIRE(is_indeterminate(a6.is_zero(x)));
    REQUIRE(is_indeterminate(a6.is_rational(x)));

    auto a7 = Assumptions({rel3});
    REQUIRE(is_true(a7.is_real(x)));
    REQUIRE(is_indeterminate(a7.is_nonpositive(x)));
    REQUIRE(is_indeterminate(a7.is_negative(x)));
    REQUIRE(is_indeterminate(a7.is_nonnegative(x)));
    REQUIRE(is_indeterminate(a7.is_positive(x)));
    REQUIRE(is_indeterminate(a7.is_nonzero(x)));
    REQUIRE(is_indeterminate(a7.is_zero(x)));
    REQUIRE(is_indeterminate(a7.is_rational(x)));

    auto a8 = Assumptions({rel4});
    REQUIRE(is_true(a8.is_real(x)));
    REQUIRE(is_false(a8.is_nonpositive(x)));
    REQUIRE(is_false(a8.is_negative(x)));
    REQUIRE(is_true(a8.is_nonnegative(x)));
    REQUIRE(is_true(a8.is_positive(x)));
    REQUIRE(is_true(a8.is_nonzero(x)));
    REQUIRE(is_indeterminate(a8.is_rational(x)));

    auto a9 = Assumptions({rel5});
    REQUIRE(is_true(a9.is_real(x)));
    REQUIRE(is_true(a9.is_nonpositive(x)));
    REQUIRE(is_true(a9.is_negative(x)));
    REQUIRE(is_false(a9.is_nonnegative(x)));
    REQUIRE(is_false(a9.is_positive(x)));
    REQUIRE(is_true(a9.is_nonzero(x)));
    REQUIRE(is_false(a9.is_zero(x)));
    REQUIRE(is_indeterminate(a9.is_rational(x)));

    auto a10 = Assumptions({rel6});
    REQUIRE(is_true(a10.is_real(x)));
    REQUIRE(is_false(a10.is_nonpositive(x)));
    REQUIRE(is_false(a10.is_negative(x)));
    REQUIRE(is_true(a10.is_nonnegative(x)));
    REQUIRE(is_true(a10.is_positive(x)));
    REQUIRE(is_true(a10.is_nonzero(x)));
    REQUIRE(is_false(a10.is_zero(x)));
    REQUIRE(is_indeterminate(a10.is_rational(x)));

    auto a11 = Assumptions({rel7});
    REQUIRE(is_true(a11.is_real(x)));
    REQUIRE(is_indeterminate(a11.is_nonpositive(x)));
    REQUIRE(is_indeterminate(a11.is_negative(x)));
    REQUIRE(is_indeterminate(a11.is_nonnegative(x)));
    REQUIRE(is_indeterminate(a11.is_positive(x)));
    REQUIRE(is_indeterminate(a11.is_nonzero(x)));
    REQUIRE(is_indeterminate(a11.is_zero(x)));
    REQUIRE(is_indeterminate(a11.is_rational(x)));

    auto a12 = Assumptions({rel8});
    REQUIRE(is_true(a12.is_real(x)));
    REQUIRE(is_false(a12.is_nonpositive(x)));
    REQUIRE(is_false(a12.is_negative(x)));
    REQUIRE(is_true(a12.is_nonnegative(x)));
    REQUIRE(is_true(a12.is_positive(x)));
    REQUIRE(is_true(a12.is_nonzero(x)));
    REQUIRE(is_false(a12.is_zero(x)));
    REQUIRE(is_indeterminate(a12.is_rational(x)));

    auto a13 = Assumptions({rel9});
    REQUIRE(is_true(a13.is_real(x)));
    REQUIRE(is_true(a13.is_nonpositive(x)));
    REQUIRE(is_true(a13.is_negative(x)));
    REQUIRE(is_false(a13.is_nonnegative(x)));
    REQUIRE(is_false(a13.is_positive(x)));
    REQUIRE(is_true(a13.is_nonzero(x)));
    REQUIRE(is_false(a13.is_zero(x)));
    REQUIRE(is_indeterminate(a13.is_rational(x)));

    auto a14 = Assumptions({rel10});
    REQUIRE(is_true(a14.is_real(x)));
    REQUIRE(is_indeterminate(a14.is_nonpositive(x)));
    REQUIRE(is_indeterminate(a14.is_negative(x)));
    REQUIRE(is_indeterminate(a14.is_nonnegative(x)));
    REQUIRE(is_indeterminate(a14.is_positive(x)));
    REQUIRE(is_indeterminate(a14.is_nonzero(x)));
    REQUIRE(is_indeterminate(a14.is_zero(x)));
    REQUIRE(is_indeterminate(a14.is_rational(x)));

    auto a15 = Assumptions({rel11});
    REQUIRE(is_true(a15.is_real(x)));
    REQUIRE(is_true(a15.is_nonpositive(x)));
    REQUIRE(is_true(a15.is_negative(x)));
    REQUIRE(is_false(a15.is_nonnegative(x)));
    REQUIRE(is_false(a15.is_positive(x)));
    REQUIRE(is_true(a15.is_nonzero(x)));
    REQUIRE(is_false(a15.is_zero(x)));
    REQUIRE(is_indeterminate(a15.is_rational(x)));

    auto a16 = Assumptions({rel12});
    REQUIRE(is_true(a16.is_real(x)));
    REQUIRE(is_indeterminate(a16.is_nonpositive(x)));
    REQUIRE(is_indeterminate(a16.is_negative(x)));
    REQUIRE(is_indeterminate(a16.is_nonnegative(x)));
    REQUIRE(is_indeterminate(a16.is_positive(x)));
    REQUIRE(is_indeterminate(a16.is_nonzero(x)));
    REQUIRE(is_indeterminate(a16.is_zero(x)));
    REQUIRE(is_indeterminate(a16.is_rational(x)));

    auto a17 = Assumptions({rel13});
    REQUIRE(is_true(a17.is_real(x)));
    REQUIRE(is_true(a17.is_rational(x)));
    REQUIRE(is_true(a17.is_complex(x)));
    REQUIRE(is_true(a17.is_integer(x)));
    REQUIRE(is_true(a17.is_nonpositive(x)));
    REQUIRE(is_false(a17.is_negative(x)));
    REQUIRE(is_true(a17.is_nonnegative(x)));
    REQUIRE(is_false(a17.is_positive(x)));
    REQUIRE(is_false(a17.is_nonzero(x)));
    REQUIRE(is_true(a17.is_zero(x)));

    auto a18 = Assumptions({rel14});
    REQUIRE(is_indeterminate(a18.is_real(x)));
    REQUIRE(is_indeterminate(a18.is_rational(x)));
    REQUIRE(is_indeterminate(a18.is_complex(x)));
    REQUIRE(is_indeterminate(a18.is_integer(x)));
    REQUIRE(is_indeterminate(a18.is_nonpositive(x)));
    REQUIRE(is_indeterminate(a18.is_negative(x)));
    REQUIRE(is_indeterminate(a18.is_nonnegative(x)));
    REQUIRE(is_indeterminate(a18.is_positive(x)));
    REQUIRE(is_true(a18.is_nonzero(x)));
    REQUIRE(is_false(a18.is_zero(x)));

    rel = Eq(x, integer(1));
    a = Assumptions({rel});
    REQUIRE(is_true(a.is_nonzero(x)));
    REQUIRE(is_false(a.is_zero(x)));

    CHECK_THROWS_AS(Assumptions({rel5, rel6}), SymEngineException);
}
