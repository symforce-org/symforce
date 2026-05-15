#include "catch.hpp"

#include <symengine/logic.h>
#include <symengine/add.h>
#include <symengine/real_double.h>
#include <symengine/symengine_exception.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::boolean;
using SymEngine::boolFalse;
using SymEngine::boolTrue;
using SymEngine::contains;
using SymEngine::Contains;
using SymEngine::DomainError;
using SymEngine::eq;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::Interval;
using SymEngine::interval;
using SymEngine::is_a;
using SymEngine::logical_and;
using SymEngine::logical_nand;
using SymEngine::logical_nor;
using SymEngine::logical_not;
using SymEngine::logical_or;
using SymEngine::logical_xnor;
using SymEngine::logical_xor;
using SymEngine::Not;
using SymEngine::one;
using SymEngine::piecewise;
using SymEngine::real_double;
using SymEngine::set_boolean;
using SymEngine::symbol;
using SymEngine::SymEngineException;
using SymEngine::unified_eq;
using SymEngine::vec_basic;
using SymEngine::vec_boolean;
using SymEngine::Xor;
using SymEngine::zero;

TEST_CASE("BooleanAtom : Basic", "[basic]")
{
    REQUIRE(boolTrue->__str__() == "True");
    REQUIRE(boolFalse->__str__() == "False");

    vec_basic v = boolTrue->get_args();
    vec_basic u;
    REQUIRE(unified_eq(v, u));

    auto x = symbol("x");
    CHECK_THROWS_AS(boolTrue->diff(x), SymEngineException);

    REQUIRE(not eq(*boolTrue, *boolFalse));
    REQUIRE(eq(*boolFalse, *boolean(false)));
}

TEST_CASE("Contains", "[logic]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(2), true, true);

    auto p = contains(integer(1), int2);
    REQUIRE(eq(*p, *boolFalse));

    p = contains(integer(2), int2);
    REQUIRE(eq(*p, *boolFalse));

    p = contains(integer(1), int1);
    REQUIRE(eq(*p, *boolTrue));

    p = contains(integer(2), int1);
    REQUIRE(eq(*p, *boolTrue));

    p = contains(real_double(1.5), int1);
    REQUIRE(eq(*p, *boolTrue));

    p = contains(integer(3), int1);
    REQUIRE(eq(*p, *boolFalse));

    p = contains(x, int1);
    REQUIRE(is_a<Contains>(*p));
    REQUIRE(p->__str__() == "Contains(x, [1, 2])");
    REQUIRE(eq(*p, *p));

    vec_basic v = p->get_args();
    vec_basic u = {x, int1};
    REQUIRE(unified_eq(v, u));

    CHECK_THROWS_AS(p->diff(x), SymEngineException);
}

TEST_CASE("Piecewise", "[logic]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto int1 = interval(integer(1), integer(2), true, false);
    auto int2 = interval(integer(2), integer(5), true, false);
    auto int3 = interval(integer(5), integer(10), true, false);
    auto int4 = interval(integer(10), integer(12), true, false);
    auto p = piecewise({{x, contains(x, int1)},
                        {y, contains(x, int2)},
                        {add(x, y), contains(x, int3)}});

    vec_basic v = p->get_args();
    vec_basic u = {x,         contains(x, int1), y, contains(x, int2),
                   add(x, y), contains(x, int3)};
    REQUIRE(unified_eq(v, u));

    std::string s = "Piecewise((x, Contains(x, (1, 2])), (y, Contains(x, (2, "
                    "5])), (x + y, Contains(x, (5, 10])))";
    REQUIRE(s == p->__str__());

    auto q = piecewise({{one, contains(x, int1)},
                        {zero, contains(x, int2)},
                        {one, contains(x, int3)}});

    REQUIRE((p->diff(x))->__hash__() == q->__hash__());
    REQUIRE(eq(*p->diff(x), *q));

    auto q2 = piecewise({
        {one, contains(x, int1)},
        {zero, contains(x, int2)},
        {one, contains(x, int3)},
        {one, contains(x, int2)},
    });

    REQUIRE(eq(*q2, *q));

    q2 = piecewise({
        {one, contains(x, int1)},
        {zero, contains(x, int2)},
        {one, boolFalse},
        {one, contains(x, int3)},
    });

    REQUIRE(eq(*q2, *q));

    p = piecewise({
        {one, contains(x, int1)},
        {zero, contains(x, int2)},
        {one, contains(x, int3)},
        {one, boolTrue},
    });

    q = piecewise({
        {one, contains(x, int1)},
        {zero, contains(x, int2)},
        {one, contains(x, int3)},
        {one, boolTrue},
        {one, contains(x, int4)},
    });

    REQUIRE(eq(*q, *p));

    q = piecewise({{one, boolTrue}});

    REQUIRE(eq(*q, *one));

    CHECK_THROWS_AS(piecewise({{one, boolFalse}}), DomainError);
}

TEST_CASE("And, Or : Basic", "[basic]")
{
    set_boolean e;
    REQUIRE(eq(*logical_and(e), *boolTrue));
    REQUIRE(eq(*logical_or(e), *boolFalse));

    REQUIRE(eq(*logical_and({boolTrue}), *boolTrue));
    REQUIRE(eq(*logical_and({boolFalse}), *boolFalse));
    REQUIRE(eq(*logical_or({boolTrue}), *boolTrue));
    REQUIRE(eq(*logical_or({boolFalse}), *boolFalse));

    REQUIRE(eq(*logical_and({boolTrue, boolFalse}), *boolFalse));
    REQUIRE(eq(*logical_or({boolTrue, boolFalse}), *boolTrue));

    auto x = symbol("x");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(5), false, false);
    auto c1 = contains(x, int1);
    auto c2 = contains(x, int2);

    auto s1 = logical_and({c1, c2});
    std::string str = s1->__str__();
    REQUIRE(str.find("And(") == 0);
    REQUIRE(str.find(c1->__str__()) != std::string::npos);
    REQUIRE(str.find(c2->__str__()) != std::string::npos);
    auto s2 = logical_and({c2, c1});
    REQUIRE(s1->__hash__() == s2->__hash__());
    REQUIRE(eq(*s1, *s2));
    vec_basic v = s2->get_args();
    vec_basic u = {c2, c1};
    REQUIRE(vec_basic_eq_perm(v, u));

    s1 = logical_or({c1, c2});
    str = s1->__str__();
    REQUIRE(str.find("Or(") == 0);
    REQUIRE(str.find(c1->__str__()) != std::string::npos);
    REQUIRE(str.find(c2->__str__()) != std::string::npos);
    s2 = logical_or({c2, c1});
    REQUIRE(s1->__hash__() == s2->__hash__());
    REQUIRE(eq(*s1, *s2));
    v = s2->get_args();
    u = {c2, c1};
    REQUIRE(vec_basic_eq_perm(v, u));

    REQUIRE(eq(*logical_and({c1}), *c1));
    REQUIRE(eq(*logical_or({c1}), *c1));

    REQUIRE(eq(*logical_and({c1, logical_not(c1)}), *boolFalse));
    REQUIRE(eq(*logical_or({c1, logical_not(c1)}), *boolTrue));

    REQUIRE(eq(*logical_and({c1, boolTrue}), *c1));
    REQUIRE(eq(*logical_and({c1, boolFalse}), *boolFalse));
    REQUIRE(eq(*logical_or({c1, boolTrue}), *boolTrue));
    REQUIRE(eq(*logical_or({c1, boolFalse}), *c1));

    REQUIRE(eq(*logical_and({c1, c1, c2}), *logical_and({c1, c2})));
    REQUIRE(eq(*logical_or({c1, c1, c2}), *logical_or({c1, c2})));

    auto y = symbol("y");
    auto c3 = contains(y, int1);
    auto c4 = contains(y, int2);
    REQUIRE(eq(*logical_and({c1, c1, c2}), *logical_and({c1, c2})));
    REQUIRE(eq(*logical_and({logical_and({c1, c2}), logical_and({c3, c4})}),
               *logical_and({c1, c2, c3, c4})));
    REQUIRE(eq(
        *logical_and(
            {logical_and({c1, logical_and({c2, logical_and({c3, c4})})}), c2}),
        *logical_and({c1, c2, c3, c4})));
    REQUIRE(eq(*logical_or({c2, c1, c2}), *logical_or({c1, c2})));
    REQUIRE(eq(*logical_or({logical_or({c1, c2}), logical_or({c3, c4})}),
               *logical_or({c1, c2, c3, c4})));
    REQUIRE(eq(
        *logical_or({c1, logical_and({c2, c3, c4}), logical_and({c2, c4}),
                     logical_and({c2, c3, c4}), c1, logical_and({c2, c4})}),
        *logical_or({c1, logical_and({c2, c3, c4}), logical_and({c2, c4})})));
}

TEST_CASE("Nand : Basic", "[basic]")
{
    set_boolean e;
    REQUIRE(eq(*logical_nand(e), *boolFalse));

    REQUIRE(eq(*logical_nand({boolTrue}), *boolFalse));
    REQUIRE(eq(*logical_nand({boolFalse}), *boolTrue));

    auto x = symbol("x");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(5), false, false);
    auto c1 = contains(x, int1);
    auto c2 = contains(x, int2);
    REQUIRE(eq(*logical_nand({boolTrue, c1}), *logical_not(c1)));
    REQUIRE(eq(*logical_nand({boolFalse, c2}), *boolTrue));
}

TEST_CASE("Nor : Basic", "[basic]")
{
    REQUIRE(eq(*logical_nor({boolTrue}), *boolFalse));
    REQUIRE(eq(*logical_nor({boolFalse}), *boolTrue));

    auto x = symbol("x");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(5), false, false);
    auto c1 = contains(x, int1);
    auto c2 = contains(x, int2);

    REQUIRE(eq(*logical_nor({boolTrue, c1}), *boolFalse));
    REQUIRE(eq(*logical_nor({boolFalse, c1}), *logical_not(c1)));
    REQUIRE(eq(*logical_nor({boolTrue, boolTrue, boolTrue}), *boolFalse));
    REQUIRE(eq(*logical_nor({boolTrue, boolTrue, c1}), *boolFalse));
    REQUIRE(eq(*logical_nor({boolTrue, boolFalse, c1}), *boolFalse));
}

TEST_CASE("Not : Basic", "[basic]")
{
    auto x = symbol("x");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(5), false, false);
    auto c1 = contains(x, int1);
    auto c2 = contains(x, int2);

    REQUIRE(eq(*logical_not(boolTrue), *boolFalse));
    REQUIRE(eq(*logical_not(boolFalse), *boolTrue));
    REQUIRE(logical_not(c1)->__str__() == "Not(Contains(x, [1, 2]))");
    REQUIRE(eq(*logical_not(logical_and({c1, c2})),
               *logical_or({logical_not(c1), logical_not(c2)})));
    REQUIRE(eq(*logical_not(logical_or({c1, c2})),
               *logical_and({logical_not(c1), logical_not(c2)})));
}

TEST_CASE("Xor : Basic", "[basic]")
{
    vec_boolean e;
    REQUIRE(eq(*logical_xor(e), *boolFalse));
    REQUIRE(eq(*logical_xor({boolTrue}), *boolTrue));
    REQUIRE(eq(*logical_xor({boolFalse}), *boolFalse));
    REQUIRE(eq(*logical_xor({boolFalse, boolFalse, boolFalse, boolTrue}),
               *boolTrue));
    REQUIRE(eq(*logical_xor({boolTrue, boolTrue}), *boolFalse));
    REQUIRE(eq(*logical_xor({boolTrue, boolTrue, boolTrue}), *boolTrue));
    REQUIRE(eq(*logical_xor({boolFalse, boolFalse}), *boolFalse));

    auto x = symbol("x");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(5), false, false);
    auto c1 = contains(x, int1);
    auto c2 = contains(x, int2);

    auto p = logical_xor({c2, c1});
    vec_basic v = p->get_args();
    vec_basic u = {c2, c1};
    REQUIRE(vec_basic_eq_perm(v, u));

    auto s1 = logical_xor({c1, c2});
    auto s2 = logical_xor({c2, c1});
    REQUIRE(s1->__hash__() == s2->__hash__());

    auto y = symbol("y");
    auto c3 = contains(y, int1);
    auto c4 = contains(y, int2);
    REQUIRE(eq(*logical_xor({c1, c1, c2}), *c2));
    REQUIRE(eq(*logical_xor({logical_xor({c1, c2}), logical_xor({c3, c4})}),
               *logical_xor({c1, c2, c3, c4})));

    REQUIRE(eq(*logical_xor({boolTrue, c1, p}), *logical_not(c2)));
    REQUIRE(eq(*logical_xor({boolTrue, logical_not(c2), p}), *c1));
    REQUIRE(eq(*logical_xor({boolTrue, c1}), *logical_not(c1)));
    REQUIRE(eq(*logical_xor({boolTrue, c1, c1, c1}), *logical_not(c1)));
    REQUIRE(eq(*logical_xor({boolTrue, c1, c2}), *logical_xnor({c1, c2})));
    REQUIRE(
        eq(*logical_xor({boolTrue, c1, logical_not(c1), c2, logical_not(c2)}),
           *boolTrue));
    REQUIRE(eq(*logical_xor({boolTrue, c1, c1}), *boolTrue));
    REQUIRE(eq(*logical_xor({c1, c1}), *boolFalse));
    REQUIRE(eq(*logical_xor({boolFalse, c2}), *(c2)));
    REQUIRE(eq(*logical_xor({logical_not(c2), c2}), *boolTrue));
    REQUIRE(is_a<Xor>(*logical_xor({c1, c2})));
}

TEST_CASE("Xnor : Basic", "[basic]")
{
    vec_boolean e;
    REQUIRE(eq(*logical_xnor(e), *boolTrue));
    REQUIRE(eq(*logical_xnor({boolTrue}), *boolFalse));
    REQUIRE(eq(*logical_xnor({boolFalse}), *boolTrue));
    REQUIRE(eq(*logical_xnor({boolFalse, boolFalse, boolFalse, boolTrue}),
               *boolFalse));
    REQUIRE(eq(*logical_xnor({boolTrue, boolTrue}), *boolTrue));
    REQUIRE(eq(*logical_xnor({boolTrue, boolTrue, boolTrue}), *boolFalse));

    auto x = symbol("x");
    auto int1 = interval(integer(1), integer(2), false, false);
    auto int2 = interval(integer(1), integer(5), false, false);
    auto c1 = contains(x, int1);
    auto c2 = contains(x, int2);

    REQUIRE(eq(*logical_xnor({boolFalse, c1}), *logical_not(c1)));
    REQUIRE(eq(*logical_xnor({c2, c2}), *boolTrue));
    REQUIRE(eq(*logical_xnor({c2, c2, c1}), *logical_not(c1)));
    REQUIRE(eq(*logical_xnor({boolTrue, boolFalse, c2}), *(c2)));
    REQUIRE(is_a<Not>(*logical_xnor({c1, c2})));
}
