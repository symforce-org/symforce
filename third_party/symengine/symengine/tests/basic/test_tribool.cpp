#include "catch.hpp"
#include <symengine/tribool.h>

using SymEngine::tribool;
using SymEngine::tribool_from_bool;

TEST_CASE("tribool", "[basic]")
{
    REQUIRE(!is_true(tribool::indeterminate));
    REQUIRE(!is_true(tribool::trifalse));
    REQUIRE(is_true(tribool::tritrue));

    REQUIRE(!is_false(tribool::indeterminate));
    REQUIRE(is_false(tribool::trifalse));
    REQUIRE(!is_false(tribool::tritrue));

    REQUIRE(is_indeterminate(tribool::indeterminate));
    REQUIRE(!is_indeterminate(tribool::trifalse));
    REQUIRE(!is_indeterminate(tribool::tritrue));

    REQUIRE(is_true(tribool_from_bool(true)));
    REQUIRE(is_false(tribool_from_bool(false)));

    REQUIRE(is_false(and_tribool(tribool::trifalse, tribool::trifalse)));
    REQUIRE(is_false(and_tribool(tribool::trifalse, tribool::indeterminate)));
    REQUIRE(is_false(and_tribool(tribool::trifalse, tribool::tritrue)));
    REQUIRE(is_false(and_tribool(tribool::indeterminate, tribool::trifalse)));
    REQUIRE(is_false(and_tribool(tribool::tritrue, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        and_tribool(tribool::indeterminate, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        and_tribool(tribool::tritrue, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        and_tribool(tribool::indeterminate, tribool::tritrue)));
    REQUIRE(is_true(and_tribool(tribool::tritrue, tribool::tritrue)));

    REQUIRE(is_false(or_tribool(tribool::trifalse, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        or_tribool(tribool::trifalse, tribool::indeterminate)));
    REQUIRE(is_true(or_tribool(tribool::trifalse, tribool::tritrue)));
    REQUIRE(is_indeterminate(
        or_tribool(tribool::indeterminate, tribool::trifalse)));
    REQUIRE(is_true(or_tribool(tribool::tritrue, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        or_tribool(tribool::indeterminate, tribool::indeterminate)));
    REQUIRE(is_true(or_tribool(tribool::tritrue, tribool::indeterminate)));
    REQUIRE(is_true(or_tribool(tribool::indeterminate, tribool::tritrue)));
    REQUIRE(is_true(or_tribool(tribool::tritrue, tribool::tritrue)));

    REQUIRE(is_true(not_tribool(tribool::trifalse)));
    REQUIRE(is_false(not_tribool(tribool::tritrue)));
    REQUIRE(is_indeterminate(not_tribool(tribool::indeterminate)));

    REQUIRE(is_false(andwk_tribool(tribool::trifalse, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::trifalse, tribool::indeterminate)));
    REQUIRE(is_false(andwk_tribool(tribool::trifalse, tribool::tritrue)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::indeterminate, tribool::trifalse)));
    REQUIRE(is_false(andwk_tribool(tribool::tritrue, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::indeterminate, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::tritrue, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::indeterminate, tribool::tritrue)));
    REQUIRE(is_true(andwk_tribool(tribool::tritrue, tribool::tritrue)));

    REQUIRE(is_false(orwk_tribool(tribool::trifalse, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        orwk_tribool(tribool::trifalse, tribool::indeterminate)));
    REQUIRE(is_true(orwk_tribool(tribool::trifalse, tribool::tritrue)));
    REQUIRE(is_indeterminate(
        orwk_tribool(tribool::indeterminate, tribool::trifalse)));
    REQUIRE(is_true(orwk_tribool(tribool::tritrue, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        orwk_tribool(tribool::indeterminate, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        orwk_tribool(tribool::tritrue, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        orwk_tribool(tribool::indeterminate, tribool::tritrue)));
    REQUIRE(is_true(orwk_tribool(tribool::tritrue, tribool::tritrue)));
}
