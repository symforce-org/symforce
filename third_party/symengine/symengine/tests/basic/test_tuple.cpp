#include "catch.hpp"
#include <iostream>
#include <symengine/basic.h>
#include <symengine/tuple.h>
#include <symengine/integer.h>

using SymEngine::Basic;
using SymEngine::integer;
using SymEngine::RCP;
using SymEngine::SymEngineException;
using SymEngine::tuple;

TEST_CASE("Tuple", "[Tuple]")
{
    auto i1 = integer(1);
    auto i2 = integer(2);
    auto i3 = integer(3);
    RCP<const Basic> t1 = tuple({i1, i2});
    RCP<const Basic> t2 = tuple({i1, i2});
    RCP<const Basic> t3 = tuple({i1, i3});
    RCP<const Basic> t4 = tuple({i1});
    RCP<const Basic> t5 = tuple({i2, i1});

    REQUIRE(eq(*t1, *t2));
    REQUIRE(not eq(*t1, *t3));
    REQUIRE(not eq(*t1, *t4));
    REQUIRE(not eq(*t1, *t5));
    REQUIRE(not eq(*t1, *i1));
    REQUIRE(t1->compare(*t2) == 0);
    REQUIRE(t1->compare(*t4) == 1);
    REQUIRE(t4->compare(*t1) == -1);
    REQUIRE(t1->__hash__() == t2->__hash__());
    REQUIRE(eq(*tuple(t1->get_args()), *t1));
    REQUIRE(t1->__str__() == "(1, 2)");
}
