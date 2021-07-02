/*
 * This file was automatically generated: DO NOT EDIT.
 *
 * Use symengine/utilities/matchpycpp/generate_tests.py to generate this file.
 *
 * Decision tree matching expressions:
 * ['x']
 *
 * Wildcards:
 * []
 */
#include "catch.hpp"
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <symengine/basic.h>
#include <symengine/pow.h>
#include <symengine/utilities/matchpycpp/common.h>
#include <symengine/utilities/matchpycpp/substitution.h>
#include <tuple>

RCP<const Basic> x = symbol("x");
RCP<const Basic> y = symbol("y");

generator<tuple<int, SubstitutionMultiset>>
match_root(const RCP<const Basic> &subject)
{
    generator<tuple<int, SubstitutionMultiset>> result;
    Deque subjects;
    subjects.push_front(subject);
    SubstitutionMultiset subst0;
    // State 2194
    if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
        RCP<const Basic> tmp1 = subjects.front();
        subjects.pop_front();
        // State 2195
        if (subjects.size() == 0) {
            // 0: x
            result.push_back(make_tuple(0, subst0));
        }
        subjects.push_front(tmp1);
    }
    return result;
}

TEST_CASE("GeneratedMatchPyTest1", "")
{
    generator<tuple<int, SubstitutionMultiset>> ret;
    SubstitutionMultiset substitution;

    // Pattern x matching x with substitution {}:
    ret = match_root(x);
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);

    // Pattern y not matching:
    ret = match_root(y);
    REQUIRE(ret.size() == 0);
}
