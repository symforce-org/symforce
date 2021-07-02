/*
 * This file was automatically generated: DO NOT EDIT.
 *
 * Use symengine/utilities/matchpycpp/generate_tests.py to generate this file.
 *
 * Decision tree matching expressions:
 * ['x**y']
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
RCP<const Basic> z = symbol("z");

generator<tuple<int, SubstitutionMultiset>>
match_root(const RCP<const Basic> &subject)
{
    generator<tuple<int, SubstitutionMultiset>> result;
    Deque subjects;
    subjects.push_front(subject);
    SubstitutionMultiset subst0;
    // State 2196
    if (subjects.size() >= 1 && is_a<Pow>(*subjects[0])) {
        RCP<const Basic> tmp1 = subjects.front();
        subjects.pop_front();
        Deque subjects2 = get_deque(tmp1);
        // State 2197
        if (subjects2.size() >= 1 && eq(*subjects2[0], *x)) {
            RCP<const Basic> tmp3 = subjects2.front();
            subjects2.pop_front();
            // State 2198
            if (subjects2.size() >= 1 && eq(*subjects2[0], *y)) {
                RCP<const Basic> tmp4 = subjects2.front();
                subjects2.pop_front();
                // State 2199
                if (subjects2.size() == 0) {
                    // State 2200
                    if (subjects.size() == 0) {
                        // 0: x**y
                        result.push_back(make_tuple(0, subst0));
                    }
                }
                subjects2.push_front(tmp4);
            }
            subjects2.push_front(tmp3);
        }
        subjects.push_front(tmp1);
    }
    return result;
}

TEST_CASE("GeneratedMatchPyTest2", "")
{
    generator<tuple<int, SubstitutionMultiset>> ret;
    SubstitutionMultiset substitution;

    // Pattern x**y matching x**y with substitution {}:
    ret = match_root(pow(x, y));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);

    // Pattern x**z not matching:
    ret = match_root(pow(x, z));
    REQUIRE(ret.size() == 0);
}
