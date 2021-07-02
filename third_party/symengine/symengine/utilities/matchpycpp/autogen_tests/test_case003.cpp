/*
 * This file was automatically generated: DO NOT EDIT.
 *
 * Use symengine/utilities/matchpycpp/generate_tests.py to generate this file.
 *
 * Decision tree matching expressions:
 * ['x**y', 'w_']
 *
 * Wildcards:
 * ['w']
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
    // State 2201
    if (subjects.size() >= 1 && is_a<Pow>(*subjects[0])) {
        RCP<const Basic> tmp1 = subjects.front();
        subjects.pop_front();
        Deque subjects2 = get_deque(tmp1);
        // State 2202
        if (subjects2.size() >= 1 && eq(*subjects2[0], *x)) {
            RCP<const Basic> tmp3 = subjects2.front();
            subjects2.pop_front();
            // State 2203
            if (subjects2.size() >= 1 && eq(*subjects2[0], *y)) {
                RCP<const Basic> tmp4 = subjects2.front();
                subjects2.pop_front();
                // State 2204
                if (subjects2.size() == 0) {
                    // State 2205
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
    if (subjects.size() >= 1) {
        RCP<const Basic> tmp5 = subjects.front();
        subjects.pop_front();
        SubstitutionMultiset subst1 = SubstitutionMultiset(subst0);
        if (!try_add_variable(subst1, "i0", tmp5)) {
            // State 2206
            if (subjects.size() == 0) {
                SubstitutionMultiset tmp_subst;
                tmp_subst["w"] = subst1["i0"];
                // 1: w_
                result.push_back(make_tuple(1, tmp_subst));
            }
        }
        subjects.push_front(tmp5);
    }
    return result;
}

TEST_CASE("GeneratedMatchPyTest3", "")
{
    generator<tuple<int, SubstitutionMultiset>> ret;
    SubstitutionMultiset substitution;

    // Pattern x**y matching x**y with substitution {}:
    ret = match_root(pow(x, y));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);

    // Pattern w_ matching x with substitution {'w': 'x'}:
    ret = match_root(x);
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 1);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("w") != substitution.end());
    REQUIRE(eq(*(*substitution.at("w").begin()), *x));

    // Pattern w_ matching x + y with substitution {'w': 'x + y'}:
    ret = match_root(add(x, y));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 1);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("w") != substitution.end());
    REQUIRE(eq(*(*substitution.at("w").begin()), *add(x, y)));
}
