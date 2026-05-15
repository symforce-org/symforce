#include "catch.hpp"

#include <symengine/sets.h>
#include <symengine/logic.h>
#include <symengine/infinity.h>
#include <symengine/real_double.h>
#include <symengine/symengine_exception.h>
#include <symengine/add.h>
#include <symengine/pow.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::boolean;
using SymEngine::Boolean;
using SymEngine::boolFalse;
using SymEngine::boolTrue;
using SymEngine::Complement;
using SymEngine::Complex;
using SymEngine::Complexes;
using SymEngine::complexes;
using SymEngine::ConditionSet;
using SymEngine::conditionset;
using SymEngine::Contains;
using SymEngine::down_cast;
using SymEngine::dummy;
using SymEngine::EmptySet;
using SymEngine::emptyset;
using SymEngine::Eq;
using SymEngine::FiniteSet;
using SymEngine::finiteset;
using SymEngine::Gt;
using SymEngine::ImageSet;
using SymEngine::imageset;
using SymEngine::Inf;
using SymEngine::infty;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::Integers;
using SymEngine::integers;
using SymEngine::Interval;
using SymEngine::interval;
using SymEngine::is_a;
using SymEngine::Le;
using SymEngine::logical_and;
using SymEngine::make_rcp;
using SymEngine::max;
using SymEngine::min;
using SymEngine::mul;
using SymEngine::Naturals;
using SymEngine::naturals;
using SymEngine::Naturals0;
using SymEngine::naturals0;
using SymEngine::NegInf;
using SymEngine::Not;
using SymEngine::NotImplementedError;
using SymEngine::Number;
using SymEngine::one;
using SymEngine::pi;
using SymEngine::Rational;
using SymEngine::rational_class;
using SymEngine::Rationals;
using SymEngine::rationals;
using SymEngine::RCP;
using SymEngine::rcp_dynamic_cast;
using SymEngine::real_double;
using SymEngine::Reals;
using SymEngine::reals;
using SymEngine::Set;
using SymEngine::set_complement;
using SymEngine::set_intersection;
using SymEngine::set_set;
using SymEngine::set_union;
using SymEngine::sin;
using SymEngine::sup;
using SymEngine::symbol;
using SymEngine::Symbol;
using SymEngine::SymEngineException;
using SymEngine::Union;
using SymEngine::UniversalSet;
using SymEngine::universalset;
using SymEngine::zero;

TEST_CASE("Interval : Basic", "[basic]")
{
    RCP<const Set> r1, r2, r3, r4, reals;
    RCP<const Number> i2 = integer(2);
    RCP<const Number> i20 = integer(20);
    RCP<const Number> im5 = integer(-5);
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Number> rat2
        = Rational::from_two_ints(*integer(500), *integer(6));

    r1 = interval(zero, i20);                  // [0, 20]
    r2 = interval(im5, i2);                    // [-5, 2]
    reals = interval(NegInf, Inf, true, true); // (-oo,oo)

    REQUIRE(is_a<Interval>(*r1));
    REQUIRE(not is_a<EmptySet>(*r1));
    REQUIRE(not is_a<UniversalSet>(*r1));

    r3 = r1->set_intersection(r2); // [0, 2]
    REQUIRE(eq(*r1->contains(one), *boolTrue));
    r4 = interval(zero, i2); // [0, 2]
    REQUIRE(eq(*r3, *r4));

    r3 = r1->set_complement(reals);
    r4 = interval(NegInf, zero, true, true)
             ->set_union(interval(i20, Inf, true, true));
    REQUIRE(eq(*r3, *r4));
    r3 = (interval(im5, i2, true, false))->set_complement(reals);
    r4 = interval(NegInf, im5, true, false)
             ->set_union(interval(i2, Inf, true, true));
    REQUIRE(eq(*r3, *r4));
    r3 = (interval(im5, i2, false, true))->set_complement(reals);
    r4 = interval(NegInf, im5, true, true)
             ->set_union(interval(i2, Inf, false, true));
    REQUIRE(eq(*r3, *r4));
    r3 = (interval(im5, i2, true, true))->set_complement(reals);
    r4 = interval(NegInf, im5, true, false)
             ->set_union(interval(i2, Inf, false, true));
    REQUIRE(eq(*r3, *r4));

    r3 = (interval(im5, i2))->set_complement(interval(im5, i2, true, true));
    REQUIRE(eq(*r3, *emptyset()));

    r3 = (interval(im5, i2, true, true))->set_complement(interval(im5, i2));
    r4 = finiteset({im5, i2});
    REQUIRE(eq(*r3, *r4));

    r3 = (interval(im5, i2, true, false))->set_complement(interval(im5, i20));
    r4 = finiteset({im5})->set_union(interval(i2, i20, true, false));
    REQUIRE(eq(*r3, *r4));

    r3 = finiteset({symbol("x"), symbol("y"), symbol("z")});
    r4 = interval(integer(5), integer(10))->set_complement(r3);
    REQUIRE(is_a<Complement>(*r4));
    auto &r6 = down_cast<const Complement &>(*r4);
    REQUIRE(eq(*r6.get_container(), *interval(integer(5), integer(10))));
    REQUIRE(eq(*r6.get_universe(), *r3));

    r3 = finiteset({symbol("x"), integer(7), symbol("z")});
    r4 = interval(integer(5), integer(10))->set_complement(r3);
    REQUIRE(is_a<Complement>(*r4));
    auto &r7 = down_cast<const Complement &>(*r4);
    REQUIRE(eq(*r7.get_container(), *interval(integer(5), integer(10))));
    REQUIRE(eq(*r7.get_universe(), *finiteset({symbol("x"), symbol("z")})));

    r3 = interval(im5, i2, true, true); // (-5, 2)
    REQUIRE(eq(*r3->contains(i2), *boolFalse));
    REQUIRE(eq(*r3->contains(im5), *boolFalse));
    REQUIRE(eq(*r3->contains(rat2), *boolFalse));
    REQUIRE(eq(*r3->contains(integer(-7)), *boolFalse));
    r4 = r3->set_intersection(r2);
    REQUIRE(eq(*r3, *r4));
    r3 = r1->set_union(r2); // [-5, 20]
    REQUIRE(eq(*r3, *set_union({r1, r2})));
    r4 = interval(im5, i20);
    REQUIRE(eq(*r3, *r4));
    r3 = r2->set_union(r1); // [-5, 20]
    REQUIRE(eq(*r3, *set_union({r1, r2})));
    REQUIRE(eq(*r3, *r4));
    r3 = interval(integer(21), integer(22));
    r4 = r1->set_intersection(r3);
    REQUIRE(eq(*r4, *emptyset()));
    r3 = interval(im5, i2, false, false); // [-5, 2]
    r4 = interval(integer(3), i20, false, false);
    REQUIRE(r3->compare(*r4) == -1);

    REQUIRE(eq(*r3->set_union(r4), *set_union({r3, r4})));
    REQUIRE(eq(*r4->set_union(r3), *set_union({r3, r4})));

    r3 = interval(zero, i2, true, true); // (0, 2)
    REQUIRE(eq(*r3->contains(sqrt(i2)), *make_rcp<Contains>(sqrt(i2), r3)));

    r3 = interval(im5, i2, false, false); // [-5, 2]
    REQUIRE(r3->is_subset(r2));
    REQUIRE(not r3->is_subset(emptyset()));
    REQUIRE(not r3->is_proper_subset(emptyset()));
    REQUIRE(not r3->is_proper_subset(r2));
    REQUIRE(not r3->is_proper_superset(r2));
    r3 = interval(im5, i20);
    r4 = interval(zero, i2);
    REQUIRE(r3->is_superset(r4));
    REQUIRE(r3->is_proper_superset(r4));

    r1 = interval(rat1, rat2); // [5/6, 500/6]
    r2 = interval(im5, i2);    // [-5, 2]
    r3 = r1->set_intersection(r2);
    r4 = interval(rat1, i2);
    REQUIRE(eq(*r3, *r4));
    r3 = r2->set_intersection(r1);
    REQUIRE(eq(*r3, *r4));
    REQUIRE(eq(*emptyset(), *r1->set_intersection(emptyset())));
    REQUIRE(eq(*r1, *r1->set_union(emptyset())));
    REQUIRE(eq(*r1, *set_union({r1, emptyset()})));

    REQUIRE(r4->__str__() == "[5/6, 2]");
    REQUIRE(r4->compare(*r3) == 0);
    r4 = interval(rat1, i2, true, true);
    REQUIRE(r4->__str__() == "(5/6, 2)");

    r1 = interval(one, zero);
    REQUIRE(eq(*r1, *emptyset()));
    r1 = interval(one, one, true, true);
    REQUIRE(eq(*r1, *emptyset()));

    r1 = interval(zero, one);
    RCP<const Interval> r5 = rcp_dynamic_cast<const Interval>(r1);

    r2 = interval(zero, one, false, false);
    REQUIRE(eq(*r5->close(), *r1));
    r2 = interval(zero, one, true, false);
    REQUIRE(eq(*r5->Lopen(), *r2));
    r2 = interval(zero, one, false, true);
    REQUIRE(eq(*r5->Ropen(), *r2));
    r2 = interval(zero, one, true, true);
    REQUIRE(eq(*r5->open(), *r2));

    r1 = interval(zero, Inf, false, true);
    r2 = interval(NegInf, one, true, true);
    r3 = interval(zero, one, false, true);
    REQUIRE(eq(*r3, *r1->set_intersection(r2)));

    REQUIRE(not r5->__eq__(*r2));
    REQUIRE(r5->__hash__() != emptyset()->__hash__());
    REQUIRE(not r5->__eq__(*emptyset()));

    REQUIRE(r5->Lopen()->compare(*r5) == -1);
    REQUIRE(r5->compare(*r5->Lopen()) == 1);
    REQUIRE(r5->Ropen()->compare(*r5) == 1);
    REQUIRE(r5->compare(*r5->Ropen()) == -1);

    REQUIRE(eq(*r5->get_args()[0], *r5->get_start()));
    REQUIRE(eq(*r5->get_args()[1], *r5->get_end()));
    REQUIRE(eq(*r5->get_args()[2], *boolean(r5->get_left_open())));
    REQUIRE(eq(*r5->get_args()[3], *boolean(r5->get_right_open())));
    RCP<const Number> c1 = Complex::from_two_nums(*i2, *i20);
    CHECK_THROWS_AS(interval(c1, one), NotImplementedError);
    CHECK_THROWS_AS(r5->diff(symbol("x")), SymEngineException);
}

TEST_CASE("Complexes : Basic", "[basic]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Set> r0 = complexes();
    RCP<const Set> r1 = reals();
    RCP<const Set> r2 = interval(zero, one);
    RCP<const Set> r3 = finiteset({zero, one, integer(2)});
    RCP<const Number> i1 = integer(3);
    RCP<const Number> i2 = integer(5);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Set> r4 = finiteset({zero, one, integer(2), c1});
    RCP<const Set> r5 = finiteset({c1});
    RCP<const Set> r6 = set_union({r1, r5});
    RCP<const Set> r7 = finiteset({real_double(2.0), c1, x});
    RCP<const Set> r8 = finiteset({x});
    RCP<const Set> r9 = set_union({r0, r8});
    RCP<const Set> r12 = universalset();
    RCP<const Set> r13 = emptyset();
    RCP<const Set> r14 = set_complement(universalset(), complexes());

    REQUIRE(is_a<Complexes>(*r0));
    REQUIRE(not is_a<UniversalSet>(*r0));
    REQUIRE(r1->is_subset(r0));
    REQUIRE(r2->is_subset(r0));
    REQUIRE(r1->is_proper_subset(r0));
    REQUIRE(r2->is_proper_subset(r0));
    REQUIRE(r0->is_superset(r1));
    REQUIRE(r0->is_superset(r2));
    REQUIRE(eq(*r0, *r0->set_intersection(r0)));
    REQUIRE(eq(*r2, *r0->set_intersection(r2)));
    REQUIRE(eq(*r2, *r2->set_intersection(r0)));
    REQUIRE(eq(*r3, *r0->set_intersection(r3)));
    REQUIRE(eq(*r3, *r3->set_intersection(r0)));
    REQUIRE(eq(*r4, *r4->set_intersection(r0)));
    REQUIRE(eq(*r4, *r0->set_intersection(r4)));
    REQUIRE(eq(*r0, *r0->set_intersection(r12)));
    REQUIRE(eq(*r0, *r12->set_intersection(r0)));
    REQUIRE(eq(*r0, *r0->set_union(r0)));
    REQUIRE(eq(*r0, *r0->set_union(r1)));
    REQUIRE(eq(*r0, *r0->set_union(r2)));
    REQUIRE(eq(*r0, *r2->set_union(r0)));
    REQUIRE(eq(*r0, *r3->set_union(r0)));
    REQUIRE(eq(*r0, *r0->set_union(r3)));
    REQUIRE(eq(*r0, *r0->set_union(r4)));
    REQUIRE(eq(*r0, *r4->set_union(r0)));
    REQUIRE(eq(*r9, *r7->set_union(r0)));
    REQUIRE(eq(*r12, *r0->set_union(r12)));
    REQUIRE(eq(*r12, *r12->set_union(r0)));
    REQUIRE(eq(*r0->set_complement(r2), *emptyset()));
    REQUIRE(eq(*r0->set_complement(r0), *emptyset()));
    REQUIRE(eq(*r0->set_complement(r13), *emptyset()));
    REQUIRE(eq(*r0->set_complement(r12), *r14));
    REQUIRE(eq(*r0->set_complement(r3), *emptyset()));
    REQUIRE(eq(*r0->set_complement(r4), *emptyset()));
    REQUIRE(eq(*r12->set_complement(r0), *emptyset()));
    REQUIRE(eq(*r13->set_complement(r0), *r0));
    REQUIRE(r0->__str__() == "Complexes");
    REQUIRE(r0->__hash__() == complexes()->__hash__());
    REQUIRE(not r0->is_proper_subset(r0));
    REQUIRE(not r0->__eq__(*r2));
    REQUIRE(r0->__eq__(*r0));
    REQUIRE(r0->compare(*complexes()) == 0);
    REQUIRE(eq(*r0->contains(zero), *boolTrue));
    REQUIRE(eq(*r0->contains(c1), *boolTrue));
    REQUIRE(eq(*r0->contains(r0), *boolFalse));
    REQUIRE(eq(*r0->contains(x), *make_rcp<Contains>(x, r0)));
    REQUIRE(r0->get_args().empty());
    CHECK_THROWS_AS(r0->diff(symbol("x")), SymEngineException);
}

TEST_CASE("Reals : Basic", "[basic]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Set> r1 = reals();
    RCP<const Set> r2 = interval(zero, one);
    RCP<const Set> r3 = finiteset({zero, one, integer(2)});
    RCP<const Number> i1 = integer(3);
    RCP<const Number> i2 = integer(5);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Set> r4 = finiteset({zero, one, integer(2), c1});
    RCP<const Set> r5 = finiteset({c1});
    RCP<const Set> r6 = set_union({r1, r5});
    RCP<const Set> r7 = finiteset({real_double(2.0), c1, x});
    RCP<const Set> r8 = finiteset({c1, x});
    RCP<const Set> r9 = set_union({r1, r8});
    RCP<const Set> r10 = finiteset({real_double(2.0), x});
    RCP<const Set> r11 = set_intersection({r10, r1});
    RCP<const Set> r12 = universalset();
    RCP<const Set> r13 = emptyset();
    RCP<const Set> r14 = set_complement(universalset(), reals());

    REQUIRE(is_a<Reals>(*r1));
    REQUIRE(not is_a<UniversalSet>(*r1));
    REQUIRE(r2->is_subset(r1));
    REQUIRE(r2->is_proper_subset(r1));
    REQUIRE(r1->is_superset(r2));
    REQUIRE(eq(*r1, *r1->set_intersection(r1)));
    REQUIRE(eq(*r2, *r1->set_intersection(r2)));
    REQUIRE(eq(*r2, *r2->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r3)));
    REQUIRE(eq(*r3, *r3->set_intersection(r1)));
    REQUIRE(eq(*r3, *r4->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r4)));
    REQUIRE(eq(*r1, *r1->set_intersection(r12)));
    REQUIRE(eq(*r1, *r12->set_intersection(r1)));
    REQUIRE(eq(*r13, *r1->set_intersection(r5)));
    REQUIRE(eq(*r13, *r5->set_intersection(r1)));
    REQUIRE(eq(*r11, *r1->set_intersection(r7)));
    REQUIRE(eq(*r1, *r1->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r2)));
    REQUIRE(eq(*r1, *r2->set_union(r1)));
    REQUIRE(eq(*r1, *r3->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r3)));
    REQUIRE(eq(*r6, *r1->set_union(r4)));
    REQUIRE(eq(*r6, *r4->set_union(r1)));
    REQUIRE(eq(*r9, *r7->set_union(r1)));
    REQUIRE(eq(*r12, *r1->set_union(r12)));
    REQUIRE(eq(*r12, *r12->set_union(r1)));
    REQUIRE(eq(*r1->set_complement(r2), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r1), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r13), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r12), *r14));
    REQUIRE(eq(*r1->set_complement(r3), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r4), *r5));
    REQUIRE(eq(*r12->set_complement(r1), *emptyset()));
    REQUIRE(eq(*r13->set_complement(r1), *r1));
    REQUIRE(r1->__str__() == "Reals");
    REQUIRE(r1->__hash__() == reals()->__hash__());
    REQUIRE(not r1->is_proper_subset(r1));
    REQUIRE(not r1->__eq__(*r2));
    REQUIRE(r1->__eq__(*r1));
    REQUIRE(r1->compare(*reals()) == 0);
    REQUIRE(eq(*r1->contains(zero), *boolTrue));
    REQUIRE(eq(*r1->contains(c1), *boolFalse));
    REQUIRE(eq(*r1->contains(r1), *boolFalse));
    REQUIRE(eq(*r1->contains(zero), *boolTrue));
    REQUIRE(eq(*r1->contains(x), *make_rcp<Contains>(x, r1)));
    REQUIRE(r1->get_args().empty());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("Rationals : Basic", "[basic]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Number> i1 = integer(3);
    RCP<const Number> i2 = integer(5);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(1), *integer(3));
    RCP<const Number> real1 = real_double(2.0);
    RCP<const Set> r1 = rationals();
    RCP<const Set> r2 = finiteset({zero, one, integer(2)});
    RCP<const Set> r3 = emptyset();
    RCP<const Set> r4 = finiteset({c1});
    RCP<const Set> r5 = finiteset({zero, one, integer(2), c1});
    RCP<const Set> r6 = universalset();
    RCP<const Set> r7 = set_union({r1, r4});
    RCP<const Set> r8 = finiteset({rat1, real1});
    RCP<const Set> r9 = finiteset({rat1});
    RCP<const Set> r10 = finiteset({real1});
    RCP<const Set> r11 = set_union({r1, r10});
    RCP<const Set> r12 = integers();
    RCP<const Set> r13 = reals();
    RCP<const Set> r14 = set_complement(reals(), rationals());

    REQUIRE(is_a<Rationals>(*r1));
    REQUIRE(not is_a<UniversalSet>(*r1));
    REQUIRE(r2->is_subset(r1));
    REQUIRE(r2->is_proper_subset(r1));
    REQUIRE(r1->is_superset(r2));
    REQUIRE(not r1->is_proper_subset(r1));
    REQUIRE(eq(*r1, *r1->set_intersection(r1)));
    REQUIRE(eq(*r2, *r1->set_intersection(r2)));
    REQUIRE(eq(*r2, *r2->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r4)));
    REQUIRE(eq(*r3, *r4->set_intersection(r1)));
    REQUIRE(eq(*r2, *r5->set_intersection(r1)));
    REQUIRE(eq(*r2, *r1->set_intersection(r5)));
    REQUIRE(eq(*r1, *r1->set_intersection(r6)));
    REQUIRE(eq(*r1, *r6->set_intersection(r1)));
    REQUIRE(eq(*r9, *r8->set_intersection(r1)));
    REQUIRE(eq(*r9, *r1->set_intersection(r8)));
    REQUIRE(eq(*r12, *r1->set_intersection(r12)));
    REQUIRE(eq(*r12, *r12->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r1)));
    REQUIRE(eq(*r6, *r1->set_union(r6)));
    REQUIRE(eq(*r6, *r6->set_union(r1)));
    REQUIRE(eq(*r7, *r1->set_union(r4)));
    REQUIRE(eq(*r7, *r4->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r2)));
    REQUIRE(eq(*r1, *r2->set_union(r1)));
    REQUIRE(eq(*r11, *r1->set_union(r8)));
    REQUIRE(eq(*r11, *r8->set_union(r1)));
    REQUIRE(eq(*r1, *r12->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r12)));
    REQUIRE(eq(*r13, *r1->set_union(r13)));
    REQUIRE(eq(*r13, *r13->set_union(r1)));
    REQUIRE(eq(*r1->set_complement(r1), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r2), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r3), *emptyset()));
    REQUIRE(eq(*r3->set_complement(r1), *r1));
    REQUIRE(eq(*r6->set_complement(r1), *emptyset()));
    REQUIRE(eq(*r1->set_complement(r13), *r14));
    REQUIRE(eq(*r13->set_complement(r1), *emptyset()));
    REQUIRE(eq(*r1->contains(zero), *boolTrue));
    REQUIRE(eq(*r1->contains(c1), *boolFalse));
    REQUIRE(eq(*r1->contains(x), *make_rcp<Contains>(x, r1)));
    REQUIRE(eq(*r1->contains(r1), *boolFalse));
    REQUIRE(eq(*r1->contains(rat1), *boolTrue));
    REQUIRE(eq(*r1->contains(real1), *boolFalse));
    REQUIRE(r1->__eq__(*r1));
    REQUIRE(not r1->__eq__(*r2));
    REQUIRE(r1->compare(*rationals()) == 0);
    REQUIRE(r1->__str__() == "Rationals");
    REQUIRE(r1->__hash__() == rationals()->__hash__());
    REQUIRE(r1->get_args().empty());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("Integers : Basic", "[basic]")
{
    RCP<const Number> i1 = integer(3);
    RCP<const Number> i2 = integer(5);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Symbol> x = symbol("x");
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Number> rat2 = Rational::from_two_ints(*integer(17), *integer(8));
    RCP<const Set> r1 = integers();
    RCP<const Set> r2 = interval(zero, one);
    RCP<const Set> r3 = finiteset({zero, one});
    RCP<const Set> r4 = reals();
    RCP<const Set> r5 = set_union({r1, r2});
    RCP<const Set> r6 = universalset();
    RCP<const Set> r7 = finiteset({zero, one, c1, x});
    RCP<const Set> r8 = finiteset({c1, x});
    RCP<const Set> r9 = set_union({r1, r8});
    RCP<const Set> r10 = emptyset();
    RCP<const Set> r11 = interval(zero, rat2);
    RCP<const Set> r12 = finiteset({zero, one, integer(2)});
    RCP<const Set> r13 = interval(zero, one, true, true);
    RCP<const Set> r14 = interval(zero, rat2, true, false);
    RCP<const Set> r15 = finiteset({one, integer(2)});
    RCP<const Set> r16 = finiteset({zero, one, c1});
    RCP<const Set> r17 = set_complement(reals(), integers());
    RCP<const Set> r18 = set_complement(universalset(), integers());
    RCP<const Set> r19 = set_complement(integers(), r3);
    RCP<const Set> r20 = finiteset({c1});
    RCP<const Set> r21 = set_complement(integers(), r16);
    RCP<const Set> r22 = set_complement(integers(), r2);

    REQUIRE(is_a<Integers>(*r1));
    REQUIRE(r1->is_subset(r4));
    REQUIRE(r1->is_proper_subset(r4));
    REQUIRE(r4->is_superset(r1));
    REQUIRE(eq(*r10, *r1->set_complement(r1)));
    REQUIRE(eq(*r17, *r1->set_complement(r4)));
    REQUIRE(eq(*r10, *r4->set_complement(r1)));
    REQUIRE(eq(*r10, *r4->set_complement(r10)));
    REQUIRE(eq(*r1, *r10->set_complement(r1)));
    REQUIRE(eq(*r10, *r6->set_complement(r1)));
    REQUIRE(eq(*r18, *r1->set_complement(r6)));
    REQUIRE(eq(*r10, *r1->set_complement(r3)));
    REQUIRE(eq(*r19, *r3->set_complement(r1)));
    REQUIRE(eq(*r20, *r1->set_complement(r16)));
    REQUIRE(eq(*r21, *r16->set_complement(r1)));
    REQUIRE(eq(*r22, *r2->set_complement(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r1)));
    REQUIRE(eq(*r5, *r1->set_union(r2)));
    REQUIRE(eq(*r5, *r2->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r3)));
    REQUIRE(eq(*r1, *r3->set_union(r1)));
    REQUIRE(eq(*r4, *r1->set_union(r4)));
    REQUIRE(eq(*r4, *r4->set_union(r1)));
    REQUIRE(eq(*r6, *r1->set_union(r6)));
    REQUIRE(eq(*r6, *r6->set_union(r1)));
    REQUIRE(eq(*r9, *r1->set_union(r7)));
    REQUIRE(eq(*r9, *r7->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r10)));
    REQUIRE(eq(*r1, *r10->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r1)));
    REQUIRE(eq(*r1, *r4->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r4)));
    REQUIRE(eq(*r10, *r1->set_intersection(r10)));
    REQUIRE(eq(*r10, *r10->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r6)));
    REQUIRE(eq(*r1, *r6->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r2)));
    REQUIRE(eq(*r3, *r2->set_intersection(r1)));
    REQUIRE(eq(*r12, *r1->set_intersection(r11)));
    REQUIRE(eq(*r12, *r11->set_intersection(r1)));
    REQUIRE(eq(*r10, *r1->set_intersection(r13)));
    REQUIRE(eq(*r10, *r13->set_intersection(r1)));
    REQUIRE(eq(*r15, *r1->set_intersection(r14)));
    REQUIRE(eq(*r15, *r14->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r16)));
    REQUIRE(eq(*r3, *r16->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r3)));
    REQUIRE(eq(*r3, *r3->set_intersection(r1)));
    REQUIRE(eq(*r1->contains(zero), *boolTrue));
    REQUIRE(eq(*r1->contains(c1), *boolFalse));
    REQUIRE(eq(*r1->contains(r1), *boolFalse));
    REQUIRE(eq(*r1->contains(rat1), *boolFalse));
    REQUIRE(eq(*r1->contains(x), *make_rcp<Contains>(x, r1)));
    REQUIRE(r1->__eq__(*r1));
    REQUIRE(!r1->__eq__(*r2));
    REQUIRE(r1->compare(*integers()) == 0);
    REQUIRE(r1->get_args().empty());
    REQUIRE(r1->__str__() == "Integers");
    REQUIRE(r1->__hash__() == integers()->__hash__());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("Naturals : Basic", "[basic]")
{
    RCP<const Number> i1 = integer(3);
    RCP<const Number> i2 = integer(5);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Symbol> x = symbol("x");
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Number> rat2 = Rational::from_two_ints(*integer(17), *integer(8));
    RCP<const Set> r1 = naturals();
    RCP<const Set> r2 = interval(zero, one);
    RCP<const Set> r3 = finiteset({zero, one});
    RCP<const Set> r4 = reals();
    RCP<const Set> r5 = set_union({r1, r2});
    RCP<const Set> r6 = universalset();
    RCP<const Set> r7 = finiteset({zero, one, c1, x});
    RCP<const Set> r8 = finiteset({zero, c1, x});
    RCP<const Set> r9 = set_union({r1, r8});
    RCP<const Set> r10 = emptyset();
    RCP<const Set> r11 = interval(zero, rat2);
    RCP<const Set> r12 = finiteset({one, integer(2)});
    RCP<const Set> r13 = interval(zero, one, true, true);
    RCP<const Set> r14 = interval(zero, rat2, true, false);
    RCP<const Set> r15 = finiteset({one, integer(2)});
    RCP<const Set> r16 = finiteset({zero, one, c1});
    RCP<const Set> r17 = set_complement(reals(), naturals());
    RCP<const Set> r18 = set_complement(universalset(), naturals());
    RCP<const Set> r19 = set_complement(naturals(), r3);
    RCP<const Set> r20 = finiteset({zero, c1});
    RCP<const Set> r21 = set_complement(naturals(), r16);
    RCP<const Set> r22 = set_complement(naturals(), r2);
    RCP<const Set> r23 = finiteset({zero});
    RCP<const Set> r24 = set_union({r23, r1});
    RCP<const Set> r25 = finiteset({one});

    REQUIRE(is_a<Naturals>(*r1));
    REQUIRE(r1->is_subset(r4));
    REQUIRE(r1->is_proper_subset(r4));
    REQUIRE(r4->is_superset(r1));
    REQUIRE(eq(*r10, *r1->set_complement(r1)));
    REQUIRE(eq(*r17, *r1->set_complement(r4)));
    REQUIRE(eq(*r10, *r4->set_complement(r1)));
    REQUIRE(eq(*r10, *r4->set_complement(r10)));
    REQUIRE(eq(*r1, *r10->set_complement(r1)));
    REQUIRE(eq(*r10, *r6->set_complement(r1)));
    REQUIRE(eq(*r18, *r1->set_complement(r6)));
    REQUIRE(eq(*r23, *r1->set_complement(r3)));
    REQUIRE(eq(*r19, *r3->set_complement(r1)));
    REQUIRE(eq(*r20, *r1->set_complement(r16)));
    REQUIRE(eq(*r21, *r16->set_complement(r1)));
    REQUIRE(eq(*r22, *r2->set_complement(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r1)));
    REQUIRE(eq(*r5, *r1->set_union(r2)));
    REQUIRE(eq(*r5, *r2->set_union(r1)));
    REQUIRE(eq(*r24, *r1->set_union(r3)));
    REQUIRE(eq(*r24, *r3->set_union(r1)));
    REQUIRE(eq(*r4, *r1->set_union(r4)));
    REQUIRE(eq(*r4, *r4->set_union(r1)));
    REQUIRE(eq(*r6, *r1->set_union(r6)));
    REQUIRE(eq(*r6, *r6->set_union(r1)));
    REQUIRE(eq(*r9, *r1->set_union(r7)));
    REQUIRE(eq(*r9, *r7->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r10)));
    REQUIRE(eq(*r1, *r10->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r1)));
    REQUIRE(eq(*r1, *r4->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r4)));
    REQUIRE(eq(*r10, *r1->set_intersection(r10)));
    REQUIRE(eq(*r10, *r10->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r6)));
    REQUIRE(eq(*r1, *r6->set_intersection(r1)));
    REQUIRE(eq(*r25, *r1->set_intersection(r2)));
    REQUIRE(eq(*r25, *r2->set_intersection(r1)));
    REQUIRE(eq(*r12, *r1->set_intersection(r11)));
    REQUIRE(eq(*r12, *r11->set_intersection(r1)));
    REQUIRE(eq(*r10, *r1->set_intersection(r13)));
    REQUIRE(eq(*r10, *r13->set_intersection(r1)));
    REQUIRE(eq(*r15, *r1->set_intersection(r14)));
    REQUIRE(eq(*r15, *r14->set_intersection(r1)));
    REQUIRE(eq(*r25, *r1->set_intersection(r16)));
    REQUIRE(eq(*r25, *r16->set_intersection(r1)));
    REQUIRE(eq(*r25, *r1->set_intersection(r3)));
    REQUIRE(eq(*r25, *r3->set_intersection(r1)));
    REQUIRE(eq(*r1->contains(zero), *boolFalse));
    REQUIRE(eq(*r1->contains(one), *boolTrue));
    REQUIRE(eq(*r1->contains(c1), *boolFalse));
    REQUIRE(eq(*r1->contains(r1), *boolFalse));
    REQUIRE(eq(*r1->contains(rat1), *boolFalse));
    REQUIRE(eq(*r1->contains(x), *make_rcp<Contains>(x, r1)));
    REQUIRE(r1->__eq__(*r1));
    REQUIRE(!r1->__eq__(*r2));
    REQUIRE(r1->compare(*naturals()) == 0);
    REQUIRE(r1->get_args().empty());
    REQUIRE(r1->__str__() == "Naturals");
    REQUIRE(r1->__hash__() == naturals()->__hash__());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("Naturals0 : Basic", "[basic]")
{
    RCP<const Number> i1 = integer(3);
    RCP<const Number> i2 = integer(5);
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i2);
    RCP<const Symbol> x = symbol("x");
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Number> rat2 = Rational::from_two_ints(*integer(17), *integer(8));
    RCP<const Set> r1 = naturals0();
    RCP<const Set> r2 = interval(zero, one);
    RCP<const Set> r3 = finiteset({zero, one});
    RCP<const Set> r4 = reals();
    RCP<const Set> r5 = set_union({r1, r2});
    RCP<const Set> r6 = universalset();
    RCP<const Set> r7 = finiteset({zero, one, c1, x});
    RCP<const Set> r8 = finiteset({c1, x});
    RCP<const Set> r9 = set_union({r1, r8});
    RCP<const Set> r10 = emptyset();
    RCP<const Set> r11 = interval(zero, rat2);
    RCP<const Set> r12 = finiteset({zero, one, integer(2)});
    RCP<const Set> r13 = interval(zero, one, true, true);
    RCP<const Set> r14 = interval(zero, rat2, true, false);
    RCP<const Set> r15 = finiteset({one, integer(2)});
    RCP<const Set> r16 = finiteset({zero, one, c1});
    RCP<const Set> r17 = set_complement(reals(), naturals0());
    RCP<const Set> r18 = set_complement(universalset(), naturals0());
    RCP<const Set> r19 = set_complement(naturals0(), r3);
    RCP<const Set> r20 = finiteset({c1});
    RCP<const Set> r21 = set_complement(naturals0(), r16);
    RCP<const Set> r22 = set_complement(naturals0(), r2);
    RCP<const Set> r23 = finiteset({zero});
    RCP<const Set> r24 = set_union({r23, r1});

    REQUIRE(is_a<Naturals0>(*r1));
    REQUIRE(r1->is_subset(r4));
    REQUIRE(r1->is_proper_subset(r4));
    REQUIRE(r4->is_superset(r1));
    REQUIRE(eq(*r10, *r1->set_complement(r1)));
    REQUIRE(eq(*r17, *r1->set_complement(r4)));
    REQUIRE(eq(*r10, *r4->set_complement(r1)));
    REQUIRE(eq(*r1, *r10->set_complement(r1)));
    REQUIRE(eq(*r10, *r6->set_complement(r1)));
    REQUIRE(eq(*r18, *r1->set_complement(r6)));
    REQUIRE(eq(*r10, *r1->set_complement(r3)));
    REQUIRE(eq(*r19, *r3->set_complement(r1)));
    REQUIRE(eq(*r20, *r1->set_complement(r16)));
    REQUIRE(eq(*r21, *r16->set_complement(r1)));
    REQUIRE(eq(*r22, *r2->set_complement(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r1)));
    REQUIRE(eq(*r5, *r1->set_union(r2)));
    REQUIRE(eq(*r5, *r2->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r3)));
    REQUIRE(eq(*r1, *r3->set_union(r1)));
    REQUIRE(eq(*r4, *r1->set_union(r4)));
    REQUIRE(eq(*r4, *r4->set_union(r1)));
    REQUIRE(eq(*r6, *r1->set_union(r6)));
    REQUIRE(eq(*r6, *r6->set_union(r1)));
    REQUIRE(eq(*r9, *r1->set_union(r7)));
    REQUIRE(eq(*r9, *r7->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_union(r10)));
    REQUIRE(eq(*r1, *r10->set_union(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r1)));
    REQUIRE(eq(*r1, *r4->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r4)));
    REQUIRE(eq(*r10, *r1->set_intersection(r10)));
    REQUIRE(eq(*r10, *r10->set_intersection(r1)));
    REQUIRE(eq(*r1, *r1->set_intersection(r6)));
    REQUIRE(eq(*r1, *r6->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r2)));
    REQUIRE(eq(*r3, *r2->set_intersection(r1)));
    REQUIRE(eq(*r12, *r1->set_intersection(r11)));
    REQUIRE(eq(*r12, *r11->set_intersection(r1)));
    REQUIRE(eq(*r10, *r1->set_intersection(r13)));
    REQUIRE(eq(*r10, *r13->set_intersection(r1)));
    REQUIRE(eq(*r15, *r1->set_intersection(r14)));
    REQUIRE(eq(*r15, *r14->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r16)));
    REQUIRE(eq(*r3, *r16->set_intersection(r1)));
    REQUIRE(eq(*r3, *r1->set_intersection(r3)));
    REQUIRE(eq(*r3, *r3->set_intersection(r1)));
    REQUIRE(eq(*r1->contains(zero), *boolTrue));
    REQUIRE(eq(*r1->contains(one), *boolTrue));
    REQUIRE(eq(*r1->contains(c1), *boolFalse));
    REQUIRE(eq(*r1->contains(r1), *boolFalse));
    REQUIRE(eq(*r1->contains(rat1), *boolFalse));
    REQUIRE(eq(*r1->contains(x), *make_rcp<Contains>(x, r1)));
    REQUIRE(r1->__eq__(*r1));
    REQUIRE(!r1->__eq__(*r2));
    REQUIRE(r1->compare(*naturals0()) == 0);
    REQUIRE(r1->get_args().empty());
    REQUIRE(r1->__str__() == "Naturals0");
    REQUIRE(r1->__hash__() == naturals0()->__hash__());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("EmptySet : Basic", "[basic]")
{
    RCP<const Set> r1 = emptyset();
    RCP<const Set> r2 = interval(zero, one);

    REQUIRE(not is_a<Interval>(*r1));
    REQUIRE(is_a<EmptySet>(*r1));
    REQUIRE(not is_a<UniversalSet>(*r1));
    REQUIRE(r1->is_subset(r2));
    REQUIRE(r1->is_proper_subset(r2));
    REQUIRE(not r1->is_proper_superset(r2));
    REQUIRE(r1->is_superset(r1));
    REQUIRE(not r1->is_superset(r2));
    REQUIRE(eq(*r1, *r1->set_intersection(r2)));
    REQUIRE(eq(*r2, *r1->set_union(r2)));
    REQUIRE(eq(*r2, *set_union({r1, r2})));
    REQUIRE(eq(*r1->set_complement(interval(NegInf, Inf, true, true)),
               *interval(NegInf, Inf, true, true)));
    REQUIRE(eq(*r1->set_complement(r2), *r2));
    REQUIRE(r1->__str__() == "EmptySet");
    REQUIRE(r1->__hash__() == emptyset()->__hash__());
    REQUIRE(not r1->is_proper_subset(r1));
    REQUIRE(not r1->__eq__(*r2));
    REQUIRE(r1->compare(*emptyset()) == 0);
    REQUIRE(eq(*r1->contains(zero), *boolFalse));
    REQUIRE(r1->get_args().empty());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("UniversalSet : Basic", "[basic]")
{
    RCP<const Set> r1 = universalset();
    RCP<const Set> r2 = interval(zero, one);
    RCP<const Set> e = emptyset();

    REQUIRE(not is_a<Interval>(*r1));
    REQUIRE(not is_a<EmptySet>(*r1));
    REQUIRE(is_a<UniversalSet>(*r1));
    REQUIRE(not r1->is_subset(r2));
    REQUIRE(not r1->is_subset(e));
    REQUIRE(not r1->is_proper_subset(r2));
    REQUIRE(not r1->is_proper_subset(e));
    REQUIRE(r1->is_proper_superset(r2));
    REQUIRE(r1->is_proper_superset(e));
    REQUIRE(r1->is_superset(r2));
    REQUIRE(r1->is_superset(e));
    REQUIRE(r1->is_subset(r1));
    REQUIRE(not r1->is_proper_subset(r1));
    REQUIRE(r1->is_superset(r1));
    REQUIRE(not r1->is_proper_superset(r1));
    REQUIRE(eq(*r1, *r1->set_union(r2)));
    REQUIRE(eq(*r1, *r1->set_union(e)));
    REQUIRE(eq(*r1, *set_union({r1, r2})));
    REQUIRE(eq(*r1, *set_union({r1, e})));
    REQUIRE(eq(*r2, *r1->set_intersection(r2)));
    REQUIRE(eq(*e, *r1->set_intersection(e)));
    REQUIRE(eq(*r1->set_complement(interval(NegInf, Inf, true, true)), *e));
    REQUIRE(eq(*r1->set_complement(r1), *e));
    REQUIRE(eq(*r1->set_complement(e), *e));
    REQUIRE(eq(*r1->contains(zero), *boolTrue));
    REQUIRE(r1->__str__() == "UniversalSet");
    REQUIRE(r1->__hash__() == universalset()->__hash__());
    REQUIRE(not r1->__eq__(*r2));
    REQUIRE(r1->compare(*universalset()) == 0);
    REQUIRE(r1->get_args().empty());
    CHECK_THROWS_AS(r1->diff(symbol("x")), SymEngineException);
}

TEST_CASE("FiniteSet : Basic", "[basic]")
{
    RCP<const Set> r1 = finiteset({zero, one, symbol("x")});
    RCP<const Set> r2 = finiteset({zero, one, integer(2)});
    RCP<const Set> r3 = r1->set_union(r2); // {0, 1, 2, x}
    REQUIRE(eq(*r3, *set_union({r1, r2})));
    r3 = r1->set_intersection(r2);
    REQUIRE(eq(*r3->contains(one), *boolTrue));
    REQUIRE(eq(*r3->contains(zero), *boolTrue));
    REQUIRE(r3->is_subset(r2));
    REQUIRE(r3->is_proper_subset(r2));
    REQUIRE(r1->get_args().size() == 3);

    r1 = finiteset({zero, one});
    REQUIRE(r1->__str__() == "{0, 1}");
    RCP<const Set> r4 = interval(zero, one);
    r3 = r2->set_intersection(r4);
    REQUIRE(eq(*r3->contains(one), *boolTrue));
    REQUIRE(eq(*r3->contains(zero), *boolTrue));
    r3 = r2->set_complement(r1);
    REQUIRE(eq(*r3, *emptyset()));
    r3 = r1->set_complement(r2);
    REQUIRE(eq(*r3, *finiteset({integer(2)})));

    r2 = finiteset({zero, one});
    r3 = r2->set_union(r4);
    REQUIRE(eq(*r3, *set_union({r2, r4})));
    REQUIRE(r3->__str__() == "[0, 1]");
    REQUIRE(r1->is_subset(r4));
    REQUIRE(r1->is_proper_subset(r4));
    r3 = SymEngine::set_union({interval(NegInf, zero, true, true),
                               interval(zero, one, true, true),
                               interval(one, Inf, true, true)});
    REQUIRE(eq(*r2->set_complement(interval(NegInf, Inf, true, true)), *r3));

    r1 = finiteset({symbol("x")}); // {x} U [1, 2] (issue #1648)
    r2 = interval(integer(1), integer(2));
    r3 = r2->set_union(r1);
    REQUIRE(is_a<Union>(*r3));
    REQUIRE(eq(*r3->get_args()[0], *r1));
    REQUIRE(eq(*r3->get_args()[1], *r2));

    r3 = SymEngine::set_union(
        {interval(NegInf, real_double(1.0), true, true),
         interval(real_double(1.0), real_double(2.0), true, true),
         interval(real_double(2.0), Inf, true, true)});
    r2 = finiteset({real_double(2.0), real_double(1.0)});
    REQUIRE(eq(*r2->set_complement(interval(NegInf, Inf, true, true)), *r3));

    r3 = SymEngine::set_union(
        {interval(integer(-3), Rational::from_mpq(rational_class(3, 4)), true,
                  true),
         interval(Rational::from_mpq(rational_class(3, 4)),
                  Rational::from_mpq(rational_class(4, 3)), true, true),
         interval(Rational::from_mpq(rational_class(4, 3)), integer(3), true,
                  true)});
    r2 = finiteset({Rational::from_mpq(rational_class(3, 4)),
                    Rational::from_mpq(rational_class(4, 3))});
    REQUIRE(
        eq(*r2->set_complement(interval(integer(-3), integer(3), true, true)),
           *r3));

    r3 = finiteset({symbol("x"), symbol("y"), symbol("z")});
    r2 = interval(integer(5), integer(10));
    r3 = r3->set_complement(r2);
    REQUIRE(is_a<Complement>(*r3));
    auto &r5 = down_cast<const Complement &>(*r3);
    REQUIRE(eq(*r5.get_container(),
               *finiteset({symbol("x"), symbol("y"), symbol("z")})));
    REQUIRE(eq(*r5.get_universe(), *r2));

    r4 = interval(zero, zero);
    r1 = finiteset({zero});
    REQUIRE(r1->is_subset(r4));
    REQUIRE(not r1->is_proper_subset(r4));
    REQUIRE(r1->__eq__(*r4));
    REQUIRE(r4->__eq__(*r1));
    r1 = finiteset({zero, one});
    r4 = interval(zero, one, true, true); // (0, 1)
    r3 = r1->set_union(r4);
    r2 = interval(zero, one); // [0, 1]
    REQUIRE(eq(*r2, *r3));
    REQUIRE(eq(*r2, *set_union({r1, r4})));
    REQUIRE(eq(*(r4->set_complement(r1)), *r1));
    REQUIRE(eq(*(r1->set_complement(r4)), *r4));

    r1 = finiteset({zero, one, integer(2)});
    r3 = r1->set_union(r4);
    REQUIRE(eq(*r3, *set_union({r1, r4})));
    r4 = interval(zero, one, false, true); // [0, 1)
    r3 = r1->set_union(r4);
    REQUIRE(eq(*r3, *set_union({r1, r4})));

    r4 = emptyset();
    r3 = r2->set_intersection(r4);
    REQUIRE(eq(*r3, *emptyset()));
    r3 = r2->set_union(r4);
    REQUIRE(eq(*r3, *r2));
    REQUIRE(eq(*r3, *set_union({r2, r4})));
    REQUIRE(r1->is_superset(r4));
    REQUIRE(not r1->is_proper_subset(r4));
    REQUIRE(eq(*r1->set_complement(r4), *r4));

    r4 = universalset();
    r3 = r2->set_intersection(r4);
    REQUIRE(eq(*r3, *r2));
    r3 = r2->set_union(r4);
    REQUIRE(eq(*r3, *set_union({r2, r4})));
    REQUIRE(eq(*r3, *universalset()));
    REQUIRE(not r1->is_superset(r4));
    REQUIRE(r1->is_proper_subset(r4));
}

TEST_CASE("Union : Basic", "[basic]")
{
    auto check_union_str = [](std::string to_chk, set_set sets) {
        if ((size_t)std::count(to_chk.begin(), to_chk.end(), 'U')
            != sets.size() - 1)
            return false;
        for (auto &a : sets) {
            if (to_chk.find(a->__str__()) == std::string::npos)
                return false;
        }
        return true;
    };
    RCP<const Set> f1 = finiteset({zero, one, symbol("x")});
    RCP<const Set> r1 = set_union({f1, emptyset()});
    REQUIRE(r1->get_args().size() == 3);
    REQUIRE(eq(*r1, *f1));
    r1 = set_union({emptyset()});
    REQUIRE(eq(*r1, *emptyset()));
    r1 = set_union({universalset()});
    REQUIRE(eq(*r1, *universalset()));
    r1 = set_union({f1});
    REQUIRE(eq(*r1, *f1));
    r1 = set_union({f1, emptyset(), universalset()});
    REQUIRE(eq(*r1, *universalset()));
    RCP<const Set> i1 = interval(zero, integer(3));
    RCP<const Set> i2 = interval(integer(4), integer(5));
    RCP<const Set> i3 = interval(integer(3), integer(4));
    RCP<const Set> reals = interval(NegInf, Inf, true, true);
    r1 = set_union({i1, i2, i3});
    REQUIRE(eq(*r1, *interval(integer(0), integer(5))));

    i1 = interval(zero, one);
    i2 = interval(integer(3), integer(4));
    i3 = interval(integer(2), integer(3));
    RCP<const Set> r2 = set_union({i1, i2, i3});
    RCP<const Union> u = rcp_dynamic_cast<const Union>(r2);
    REQUIRE(u->get_container().size() == 2);
    REQUIRE(u->get_container().find(interval(zero, one))
            != u->get_container().end());
    REQUIRE(u->get_container().find(interval(integer(2), integer(4)))
            != u->get_container().end());
    REQUIRE(
        check_union_str(u->__str__(), {i1, interval(integer(2), integer(4))}));

    r2 = set_union({r1, r2});
    REQUIRE(eq(*r1, *r2));

    r2 = set_union({i1, finiteset({integer(2)})});
    REQUIRE(is_a<Union>(*r2));
    r1 = r2->set_complement(reals);
    REQUIRE(is_a<Union>(*r1));
    r2 = set_union({interval(NegInf, zero, true, true),
                    interval(one, integer(2), true, true),
                    interval(integer(2), Inf, true, true)});
    REQUIRE(eq(*r1, *r2));

    r2 = set_union({i1, i2, i3});
    r1 = set_union({finiteset({zero}), i2});
    u = rcp_dynamic_cast<const Union>(set_union({r1, r2}));
    REQUIRE(u->get_container().find(interval(integer(2), integer(4)))
            != u->get_container().end());
    REQUIRE(u->get_container().find(interval(zero, one))
            != u->get_container().end());
    REQUIRE(
        check_union_str(u->__str__(), {i1, interval(integer(2), integer(4))}));
    REQUIRE(eq(*u->contains(one), *boolTrue));
    REQUIRE(eq(*u->contains(integer(2)), *boolTrue));
    REQUIRE(eq(*u->contains(integer(7)), *boolFalse));
    REQUIRE(u->is_superset(r1));
    REQUIRE(u->is_superset(r2));
    REQUIRE(u->is_superset(u));
    REQUIRE(r2->is_subset(u));
    REQUIRE(r1->is_subset(u));
    REQUIRE(u->is_subset(u));
    REQUIRE(u->is_proper_superset(r1));
    REQUIRE(not u->is_proper_superset(r2));
    REQUIRE(not u->is_proper_superset(u));
    REQUIRE(not r2->is_proper_subset(u));
    REQUIRE(r1->is_proper_subset(u));
    REQUIRE(not u->is_proper_subset(u));
}

TEST_CASE("Complement : Basic", "[basic]")
{
    RCP<const Set> f1 = finiteset({zero, one, symbol("x")});
    RCP<const Set> f2 = finiteset({symbol("y")});
    RCP<const Set> i1 = interval(NegInf, Inf, true, true);
    RCP<const Set> r1, r2, r3;

    r1 = set_complement(i1, f2);
    REQUIRE(is_a<Complement>(*r1));
    auto &r4 = down_cast<const Complement &>(*r1);
    REQUIRE(eq(*r4.get_container(), *f2));
    REQUIRE(eq(*r4.get_universe(), *i1));

    REQUIRE(r1->get_args().size() == 2);
    REQUIRE(is_a<Not>(*r1->contains(one)));
    REQUIRE(eq(*r1->contains(symbol("y")), *boolFalse));

    r2 = set_complement(i1, finiteset({symbol("x")}));
    REQUIRE(r1->__hash__() != r2->__hash__());
    REQUIRE(not r1->__eq__(*r2));

    r1 = set_complement(i1, finiteset({symbol("x")}));
    r2 = set_complement(i1, f2);
    REQUIRE(r2->compare(*r1) == 1);
    REQUIRE(r1->compare(*r2) == -1);

    auto r5 = r2->set_intersection(finiteset({symbol("x")}));
    auto r6 = set_intersection({r2, finiteset({symbol("x")})});
    REQUIRE(eq(*r5, *r6));
    auto r7 = r2->set_intersection(finiteset({zero, integer(2)}));
    auto r8 = set_intersection({r2, finiteset({zero, integer(2)})});
    REQUIRE(eq(*r7, *r8));

    r2 = set_complement(i1, f1);
    REQUIRE(is_a<Complement>(*r2));
}

TEST_CASE("set_intersection : Basic", "[basic]")
{
    RCP<const Set> f1 = finiteset({zero, one, integer(2)});
    RCP<const Set> f2 = finiteset({one, integer(2), integer(3)});
    RCP<const Set> e = emptyset();
    RCP<const Set> u = universalset();
    RCP<const Set> interval1 = interval(zero, one);
    RCP<const Set> r1, r2, i1, i2, i3;

    // Trivial cases
    r1 = set_intersection({});
    REQUIRE(eq(*r1, *u));

    r1 = set_intersection({f1, f2, e});
    REQUIRE(eq(*r1, *e));

    r1 = set_intersection({u, u});
    REQUIRE(eq(*r1, *u));

    r1 = set_intersection({u, u});
    REQUIRE(eq(*r1, *u));

    r1 = set_intersection({u, f1});
    REQUIRE(eq(*r1, *f1));

    // Finitesets
    r1 = set_intersection({f1, f2});
    REQUIRE(eq(*r1, *finiteset({one, integer(2)})));

    r1 = set_intersection({f1, f2, interval1});
    REQUIRE(eq(*r1, *finiteset({one})));

    r1 = set_intersection({f2, f1, interval1});
    REQUIRE(eq(*r1, *finiteset({one})));

    r2 = finiteset({zero, integer(5)});
    r1 = set_intersection({r2, interval(integer(-10), integer(10))});
    REQUIRE(eq(*r1, *r2));

    r1 = finiteset({symbol("x")});
    r2 = finiteset({symbol("y")});
    auto r3 = set_intersection({r1, r2});
    auto r4 = r1->set_intersection(r2);
    REQUIRE(eq(*r3, *r4));

    r3 = finiteset({symbol("x"), symbol("y")});
    r4 = interval(integer(-10), integer(10));
    auto r5 = set_intersection({r3, r4});
    auto r6 = r3->set_intersection(r4);
    auto r7 = r4->set_intersection(r3);
    REQUIRE(eq(*r5, *r6));
    REQUIRE(eq(*r5, *r7));

    // One of the arg is Union
    i1 = interval(zero, one);
    i2 = interval(integer(3), integer(4));
    r2 = set_union({i1, i2});
    i3 = interval(integer(2), integer(3));
    r1 = set_intersection({r2, i3});
    REQUIRE(eq(*r1, *finiteset({integer(3)})));

    i3 = interval(one, integer(3));
    r1 = set_intersection({r2, i3});
    REQUIRE(eq(*r1, *finiteset({one, integer(3)})));

    i3 = interval(one, integer(3), true, true);
    r1 = set_intersection({r2, i3});
    REQUIRE(eq(*r1, *e));

    auto s1 = reals();
    auto s2 = finiteset({symbol("x")});
    auto s3 = set_intersection({s1, s2});
    REQUIRE(s3->__str__() == "Intersection(Reals, {x})");
    REQUIRE(!s3->__eq__(*s1));

    REQUIRE(eq(*s3->set_intersection(s1), *s3));
}

TEST_CASE("set_complement : Basic", "[basic]")
{
    RCP<const Set> f1 = finiteset({zero, one, integer(2)});
    RCP<const Set> f2 = finiteset({integer(2), integer(3), integer(4)});
    RCP<const Set> e = emptyset();
    RCP<const Set> u = universalset();
    RCP<const Set> interval1 = interval(one, integer(3));
    RCP<const Set> r1, r2, i1, i2;

    r1 = finiteset({zero, one});
    r2 = set_complement(r1, f1);
    REQUIRE(eq(*r2, *e));

    i1 = interval(zero, integer(2));
    i2 = interval(zero, one, false, true);
    r1 = set_union({i1, f2});
    r2 = set_complement(r1, interval1);
    r1 = set_union({i2, finiteset({integer(4)})});
    REQUIRE(eq(*r2, *r1));

    r1 = set_union({i1, finiteset({integer(2)})});
    r2 = set_complement(r1, f1);
    i1 = interval(zero, one, true, true);
    i2 = interval(one, integer(2), true, true);
    r1 = set_union({i1, i2});
    REQUIRE(eq(*r1, *r2));

    i1 = interval(zero, integer(2));
    r1 = set_complement(interval1, i1);
    r2 = interval(integer(2), integer(3), true, false);
    REQUIRE(eq(*r1, *r2));

    r1 = set_complement(e, i1);
    REQUIRE(eq(*r1, *e));
    r1 = set_complement(e, f1);
    REQUIRE(eq(*r1, *e));

    r1 = set_complement(f1, interval1);
    r2 = finiteset({zero});
    REQUIRE(eq(*r1, *r2));

    r1 = set_complement(interval1, f1);
    i1 = interval(integer(2), integer(3), true, false);
    i2 = interval(one, integer(2), true, true);
    r2 = set_union({i1, i2});
    REQUIRE(eq(*r1, *r2));

    r1 = set_complement(f1, f2);
    r2 = finiteset({zero, one});
    REQUIRE(eq(*r1, *r2));
}

TEST_CASE("ConditionSet : Basic", "[basic]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Set> r1, r2;

    RCP<const Set> i1 = interval(NegInf, Inf, true, true);
    RCP<const Set> i2 = interval(one, Inf, true, true);
    RCP<const Set> i3 = interval(zero, integer(4), false, true);
    RCP<const Set> f1 = finiteset({zero, one, integer(2), integer(4)});
    RCP<const Set> f2 = finiteset({zero, one, integer(-3), y});
    RCP<const Set> f3 = finiteset({y});

    RCP<const Boolean> cond1, cond2, cond3;
    cond1 = Ge(mul(x, x), integer(9));
    cond2 = Le(x, zero);
    cond3 = Gt(x, zero);

    r1 = conditionset(x, logical_and({cond1, i1->contains(x)}));
    REQUIRE(is_a<ConditionSet>(*r1));

    auto &r3 = down_cast<const ConditionSet &>(*r1);
    REQUIRE(unified_eq(r3.get_symbol(), x));
    REQUIRE(eq(*r3.get_condition(), *logical_and({cond1, i1->contains(x)})));

    r1 = conditionset(x, logical_and({cond1, f1->contains(x)}));
    REQUIRE(eq(*r1, *finiteset({integer(4)})));

    r1 = conditionset(x, logical_and({cond3, f2->contains(x)}));
    REQUIRE(is_a<Union>(*r1));
    auto &r4 = down_cast<const Union &>(*r1);
    REQUIRE(r4.get_container().size() == 2);
    REQUIRE(r4.get_container().find(finiteset({one}))
            != r4.get_container().end());
    REQUIRE(r4.get_container().find(
                conditionset(x, logical_and({cond3, f3->contains(x)})))
            != r4.get_container().end());

    r1 = conditionset(x, logical_and({cond2, i1->contains(x)}));
    REQUIRE(is_a<ConditionSet>(*r1));
    REQUIRE(eq(*r1->contains(pi),
               *logical_and({Le(pi, zero), i1->contains(pi)}))); // pi can't be
                                                                 // compared
                                                                 // with zero
                                                                 // now. remains
                                                                 // unevaluated
                                                                 // as And(pi <=
                                                                 // 0,
                                                                 // Contains(pi,
                                                                 // (-oo, oo)))
    REQUIRE(eq(*r1->contains(integer(2)), *boolFalse));
    REQUIRE(eq(*r1->contains(integer(-2)), *boolTrue));

    r1 = conditionset(x, logical_and({Eq(zero, one), i1->contains(x)}));
    REQUIRE(eq(*r1, *emptyset()));

    r1 = conditionset(x, logical_and({Gt(one, zero), i1->contains(x)}));
    REQUIRE(eq(*r1, *i1));

    cond1 = Eq(sin(x), zero);
    r1 = conditionset(x, logical_and({cond1, i3->contains(x)}));
    REQUIRE(is_a<ConditionSet>(*r1));
    // following doesn't work because we can't compare pi with number as of now.
    // REQUIRE(eq(*r1->contains(pi), *boolTrue));
    // REQUIRE(eq(*r1->contains(finiteset({pi})), *boolTrue));

    cond1 = Eq(mul(x, x), integer(9));
    r1 = conditionset(x, logical_and({cond1, i2->contains(x)}));
    REQUIRE(is_a<ConditionSet>(*r1));
    REQUIRE(eq(*r1->contains(integer(3)), *boolTrue));
    REQUIRE(eq(*r1->contains(integer(-3)), *boolFalse));
    REQUIRE(eq(*r1->contains(integer(2)), *boolFalse));
    REQUIRE(eq(*r1->contains(finiteset({integer(3)})), *boolFalse));
    REQUIRE(
        eq(*r1->contains(finiteset({integer(3), integer(-3)})), *boolFalse));

    cond1 = Le(sin(x), y);
    r1 = conditionset(x, logical_and({cond1, i1->contains(x)}));
    REQUIRE(is_a<ConditionSet>(*r1));
    REQUIRE(eq(*r1->contains(integer(5)), *Le(sin(integer(5)), y)));

    cond1 = Eq(mul(x, x), integer(9));
    r1 = conditionset(x, logical_and({cond1, i1->contains(x)}));
    REQUIRE(is_a<ConditionSet>(*r1));
    REQUIRE(eq(*r1->contains(integer(3)), *boolTrue));
    REQUIRE(eq(*r1->contains(integer(-3)), *boolTrue));
    REQUIRE(
        eq(*r1->contains(finiteset({integer(3), integer(-3)})), *boolFalse));

    r2 = r1->set_intersection(interval(integer(-10), integer(10)));
    REQUIRE(is_a<ConditionSet>(*r2));
    REQUIRE(r1->compare(*r2) == -1);
    REQUIRE(eq(*down_cast<const ConditionSet &>(*r2).get_condition(),
               *logical_and({Eq(mul(x, x), integer(9)),
                             interval(integer(-10), integer(10))->contains(x),
                             i1->contains(x)})));

    r2 = r1->set_union(i1);
    REQUIRE(is_a<Union>(*r2));

    r2 = r1->set_complement(i1);
    REQUIRE(is_a<Complement>(*r2));

    cond3 = logical_and({Eq(y, x), f1->contains(x)});
    r1 = conditionset(x, cond3);
    REQUIRE(is_a<ConditionSet>(*r1));
    REQUIRE(eq(*down_cast<const ConditionSet &>(*r1).get_condition(), *cond3));
}

TEST_CASE("ImageSet : Basic", "[basic]")
{
    RCP<const Set> r1, r2;
    RCP<const Symbol> x = symbol("x");
    RCP<const Set> i1 = interval(zero, one);

    r1 = imageset(x, mul(x, x), i1);
    REQUIRE(is_a<ImageSet>(*r1));
    auto &r3 = down_cast<const ImageSet &>(*r1);
    REQUIRE(not r3.is_canonical(sin(x), x, i1));
    REQUIRE(eq(*r3.get_symbol(), *x));
    REQUIRE(eq(*r3.get_expr(), *mul(x, x)));
    REQUIRE(eq(*r3.get_baseset(), *i1));
    CHECK_THROWS_AS(r1->contains(one), SymEngineException);

    r2 = imageset(x, mul(x, x), interval(zero, Inf));
    REQUIRE(r2->compare(*r1) == 1);

    r2 = r1->set_complement(finiteset({one}));
    REQUIRE(is_a<Complement>(*r2));
    REQUIRE(eq(*r2, *set_complement(r1, finiteset({one}))));

    r1 = r1->set_union(finiteset({one}));
    REQUIRE(is_a<Union>(*r1));
    REQUIRE(eq(*r1, *set_union({r1, finiteset({one})})));

    REQUIRE(eq(*r1->set_intersection(i1), *set_intersection({r1, i1})));

    r1 = imageset(x, one, i1);
    REQUIRE(eq(*r1, *finiteset({one})));

    CHECK_THROWS_AS(imageset(sin(x), x, i1), SymEngineException);

    r1 = imageset(x, x, i1);
    REQUIRE(eq(*r1, *i1));

    r1 = imageset(x, mul(mul(integer(2), x), pi),
                  interval(zero, Inf, false, true));
    r2 = imageset(x, mul(mul(integer(2), add(one, x)), pi),
                  interval(zero, Inf, false, true));
    auto r4 = down_cast<const Union &>(*r1->set_union(r2)).get_container();
    REQUIRE(r4.find(r1) != r4.end());
    REQUIRE(r4.find(r2) != r4.end());

    r1 = imageset(x, mul(x, x), emptyset());
    REQUIRE(eq(*r1, *emptyset()));

    auto y = symbol("y"), z = symbol("z");
    r1 = imageset(x, add({mul(x, x), mul(y, y), mul(z, z)}), i1);
    REQUIRE(is_a<ImageSet>(*r1));
    auto &r5 = down_cast<const ImageSet &>(*r1);
    REQUIRE(eq(*r5.get_symbol(), *x));
    REQUIRE(eq(*r5.get_expr(), *add({mul(x, x), mul(y, y), mul(z, z)})));
    REQUIRE(eq(*r5.get_baseset(), *i1));

    auto f1 = finiteset({one, integer(2), y});
    r1 = imageset(x, f1, i1);
    REQUIRE(is_a<ImageSet>(*r1));
    REQUIRE(eq(*down_cast<const ImageSet &>(*r1).get_baseset(), *i1));
    REQUIRE(eq(*down_cast<const ImageSet &>(*r1).get_expr(), *f1));
    REQUIRE(eq(*down_cast<const ImageSet &>(*r1).get_symbol(), *x));

    f1 = finiteset({one, integer(2)});
    r1 = imageset(x, f1, i1);
    REQUIRE(eq(*r1, *finiteset({f1})));

    auto i2 = interval(one, integer(2));
    r1 = imageset(x, i2, i1);
    REQUIRE(eq(*r1, *finiteset({i2})));

    f1 = finiteset({one, y});
    r1 = imageset(x, add(x, integer(2)), f1);
    REQUIRE(eq(*r1, *finiteset({integer(3), add(y, integer(2))})));

    r1 = imageset(x, div(x, pi), imageset(x, mul({integer(2), x, pi}), i1));
    r2 = imageset(x, mul(integer(2), x), i1);
    REQUIRE(eq(*r1, *r2));

    auto xD = dummy("x");
    REQUIRE(is_a<ImageSet>(*imageset(xD, mul(xD, xD), i1)));
}

TEST_CASE("sup : Basic", "[basic]")
{
    REQUIRE(eq(*sup(*reals()), *infty(1)));
    REQUIRE(eq(*sup(*rationals()), *infty(1)));
    REQUIRE(eq(*sup(*integers()), *infty(1)));
    REQUIRE(eq(*sup(*naturals()), *infty(1)));
    REQUIRE(eq(*sup(*naturals0()), *infty(1)));
    REQUIRE(eq(*sup(*interval(one, integer(2), true, true)), *integer(2)));
    REQUIRE(eq(*sup(*interval(one, integer(2), false, false)), *integer(2)));

    RCP<const Set> r1 = finiteset({zero, one, symbol("x")});
    RCP<const Set> r2 = finiteset({zero, one, integer(2)});
    REQUIRE(eq(*sup(*r1), *max({one, symbol("x")})));
    REQUIRE(eq(*sup(*r2), *integer(2)));

    RCP<const Set> r3 = set_union(
        {interval(integer(-1), integer(1)), finiteset({integer(-23)})});
    REQUIRE(eq(*sup(*r3), *integer(1)));

    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Set> r4 = set_union({integers(), finiteset({rat1})});
    REQUIRE(eq(*sup(*r4), *infty(1)));

    CHECK_THROWS_AS(sup(*emptyset()), SymEngineException);
    CHECK_THROWS_AS(sup(*rationals()->set_complement(reals())),
                    NotImplementedError);

    RCP<const Symbol> x = symbol("x");
    RCP<const Set> i1 = interval(zero, one);
    r1 = imageset(x, mul(x, x), i1);
    CHECK_THROWS_AS(sup(*r1), NotImplementedError);
}

TEST_CASE("inf : Basic", "[basic]")
{
    REQUIRE(eq(*inf(*reals()), *infty(-1)));
    REQUIRE(eq(*inf(*rationals()), *infty(-1)));
    REQUIRE(eq(*inf(*integers()), *infty(-1)));
    REQUIRE(eq(*inf(*naturals()), *integer(1)));
    REQUIRE(eq(*inf(*naturals0()), *integer(0)));
    REQUIRE(eq(*inf(*interval(one, integer(2), true, true)), *integer(1)));
    REQUIRE(eq(*inf(*interval(one, integer(2), false, false)), *integer(1)));

    RCP<const Set> r1 = finiteset({zero, one, symbol("x")});
    RCP<const Set> r2 = finiteset({zero, one, integer(2)});
    REQUIRE(eq(*inf(*r1), *min({zero, symbol("x")})));
    REQUIRE(eq(*inf(*r2), *integer(0)));

    RCP<const Set> r3 = set_union(
        {interval(integer(-1), integer(1)), finiteset({integer(-23)})});
    REQUIRE(eq(*inf(*r3), *integer(-23)));

    RCP<const Number> rat1 = Rational::from_two_ints(*integer(5), *integer(6));
    RCP<const Set> r4 = set_union({integers(), finiteset({rat1})});
    REQUIRE(eq(*inf(*r4), *infty(-1)));

    CHECK_THROWS_AS(inf(*emptyset()), SymEngineException);
    CHECK_THROWS_AS(inf(*rationals()->set_complement(reals())),
                    NotImplementedError);

    RCP<const Symbol> x = symbol("x");
    RCP<const Set> i1 = interval(zero, one);
    r1 = imageset(x, mul(x, x), i1);
    CHECK_THROWS_AS(inf(*r1), NotImplementedError);
}

TEST_CASE("boundary : Basic", "[basic]")
{
    auto i1 = interval(integer(1), integer(2));
    auto i2 = interval(integer(3), integer(4));
    REQUIRE(eq(*boundary(*emptyset()), *emptyset()));
    REQUIRE(eq(*boundary(*universalset()), *emptyset()));
    REQUIRE(eq(*boundary(*complexes()), *emptyset()));
    REQUIRE(eq(*boundary(*reals()), *emptyset()));
    REQUIRE(eq(*boundary(*rationals()), *reals()));
    REQUIRE(eq(*boundary(*integers()), *integers()));
    REQUIRE(eq(*boundary(*naturals()), *naturals()));
    REQUIRE(eq(*boundary(*naturals0()), *naturals0()));
    REQUIRE(eq(*boundary(*i1), *finiteset({integer(1), integer(2)})));
    REQUIRE(eq(*boundary(*finiteset({integer(1), integer(2)})),
               *finiteset({integer(1), integer(2)})));
    REQUIRE(eq(*boundary(*set_union({i1, i2})),
               *finiteset({integer(1), integer(2), integer(3), integer(4)})));

    RCP<const Symbol> x = symbol("x");
    auto r1 = imageset(x, mul(x, x), i1);
    CHECK_THROWS_AS(boundary(*r1), NotImplementedError);
    CHECK_THROWS_AS(boundary(*rationals()->set_complement(reals())),
                    NotImplementedError);
}

TEST_CASE("closure : Basic", "[basic]")
{
    auto i1 = interval(integer(1), integer(2));
    auto i2 = interval(integer(1), integer(2), true, true);
    auto i3 = interval(integer(3), integer(4));
    REQUIRE(eq(*closure(*emptyset()), *emptyset()));
    REQUIRE(eq(*closure(*universalset()), *universalset()));
    REQUIRE(eq(*closure(*complexes()), *complexes()));
    REQUIRE(eq(*closure(*reals()), *reals()));
    REQUIRE(eq(*closure(*rationals()), *reals()));
    REQUIRE(eq(*closure(*integers()), *integers()));
    REQUIRE(eq(*closure(*naturals()), *naturals()));
    REQUIRE(eq(*closure(*naturals0()), *naturals0()));
    REQUIRE(eq(*closure(*finiteset({integer(1), integer(2)})),
               *finiteset({integer(1), integer(2)})));
    REQUIRE(eq(*closure(*i1), *i1));
    REQUIRE(eq(*closure(*i2), *i1));
    REQUIRE(eq(*closure(*set_union({i1, i3})), *set_union({i1, i3})));
}

TEST_CASE("interior : Basic", "[basic]")
{
    auto i1 = interval(integer(1), integer(2));
    auto i2 = interval(integer(1), integer(2), true, true);
    auto i3 = interval(integer(3), integer(4));
    auto i4 = interval(integer(3), integer(4), true, true);
    REQUIRE(eq(*interior(*emptyset()), *emptyset()));
    REQUIRE(eq(*interior(*universalset()), *universalset()));
    REQUIRE(eq(*interior(*complexes()), *complexes()));
    REQUIRE(eq(*interior(*reals()), *reals()));
    REQUIRE(eq(*interior(*rationals()), *emptyset()));
    REQUIRE(eq(*interior(*integers()), *emptyset()));
    REQUIRE(eq(*interior(*naturals()), *emptyset()));
    REQUIRE(eq(*interior(*naturals0()), *emptyset()));
    REQUIRE(eq(*interior(*finiteset({integer(1), integer(2)})), *emptyset()));
    REQUIRE(eq(*interior(*i1), *i2));
    REQUIRE(eq(*interior(*i2), *i2));
    REQUIRE(eq(*interior(*set_union({i1, i3})), *set_union({i2, i4})));
}
