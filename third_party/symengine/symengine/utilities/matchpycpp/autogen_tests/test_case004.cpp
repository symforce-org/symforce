/*
 * This file was automatically generated: DO NOT EDIT.
 *
 * Use symengine/utilities/matchpycpp/generate_tests.py to generate this file.
 *
 * Decision tree matching expressions:
 * ['x + y', 'x**2']
 *
 * Wildcards:
 * []
 */
#include "catch.hpp"
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <symengine/basic.h>
#include <symengine/pow.h>
#include <symengine/utilities/matchpycpp/bipartite.h>
#include <symengine/utilities/matchpycpp/common.h>
#include <symengine/utilities/matchpycpp/many_to_one.h>
#include <symengine/utilities/matchpycpp/substitution.h>
#include <symengine/utilities/matchpycpp/utils.h>
#include <tuple>

RCP<const Basic> x = symbol("x");
RCP<const Basic> y = symbol("y");

class CommutativeMatcher2209 : public CommutativeMatcher
{
public:
    CommutativeMatcher2209()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0, 1}, {})}};
        subjects = {};
        subjects_by_id = {};
        associative = [](const RCP<const Basic> &x, const RCP<const Basic> &y) {
            return add(x, y);
        };
        max_optional_count = 0;
        anonymous_patterns = {0, 1};

        add_subject(None);
    }

    generator<tuple<int, SubstitutionMultiset>>
    get_match_iter(const RCP<const Basic> &subject)
    {
        generator<tuple<int, SubstitutionMultiset>> result;
        Deque subjects;
        subjects.push_front(subject);
        SubstitutionMultiset subst0;
        // State 2208
        if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2210
            if (subjects.size() == 0) {
                // 0: x
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *y)) {
            RCP<const Basic> tmp2 = subjects.front();
            subjects.pop_front();
            // State 2211
            if (subjects.size() == 0) {
                // 1: y
                result.push_back(make_tuple(1, subst0));
            }
            subjects.push_front(tmp2);
        }
        return result;
    }
};

generator<tuple<int, SubstitutionMultiset>>
match_root(const RCP<const Basic> &subject)
{
    generator<tuple<int, SubstitutionMultiset>> result;
    Deque subjects;
    subjects.push_front(subject);
    SubstitutionMultiset subst0;
    // State 2207
    if (subjects.size() >= 1 && is_a<Add>(*subjects[0])) {
        RCP<const Basic> tmp1 = subjects.front();
        subjects.pop_front();
        RCP<const Basic> associative1 = tmp1;
        string associative_type1 = tmp1->__str__();
        Deque subjects2 = get_deque(tmp1);
        CommutativeMatcher2209 matcher;
        Deque tmp3 = subjects2;
        subjects2 = {};
        for (RCP<const Basic> &s : tmp3) {
            matcher.add_subject(s);
        }
        for (tuple<int, SubstitutionMultiset> &p :
             matcher.match(tmp3, subst0)) {
            int pattern_index = get<0>(p);
            SubstitutionMultiset subst1 = get<1>(p);
            if (pattern_index == 0) {
                // State 2212
                if (subjects.size() == 0) {
                    // 0: x + y
                    result.push_back(make_tuple(0, subst1));
                }
            }
        }
        subjects.push_front(tmp1);
    }
    if (subjects.size() >= 1 && is_a<Pow>(*subjects[0])) {
        RCP<const Basic> tmp4 = subjects.front();
        subjects.pop_front();
        Deque subjects5 = get_deque(tmp4);
        // State 2213
        if (subjects5.size() >= 1 && eq(*subjects5[0], *x)) {
            RCP<const Basic> tmp6 = subjects5.front();
            subjects5.pop_front();
            // State 2214
            if (subjects5.size() >= 1 && eq(*subjects5[0], *integer(2))) {
                RCP<const Basic> tmp7 = subjects5.front();
                subjects5.pop_front();
                // State 2215
                if (subjects5.size() == 0) {
                    // State 2216
                    if (subjects.size() == 0) {
                        // 1: x**2
                        result.push_back(make_tuple(1, subst0));
                    }
                }
                subjects5.push_front(tmp7);
            }
            subjects5.push_front(tmp6);
        }
        subjects.push_front(tmp4);
    }
    return result;
}

TEST_CASE("GeneratedMatchPyTest4", "")
{
    generator<tuple<int, SubstitutionMultiset>> ret;
    SubstitutionMultiset substitution;

    // Pattern x + y matching x + y with substitution {}:
    ret = match_root(add(x, y));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);

    // Pattern x**2 matching x**2 with substitution {}:
    ret = match_root(pow(x, integer(2)));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 1);
    substitution = get<1>(ret[0]);

    // Pattern x**3 not matching:
    ret = match_root(pow(x, integer(3)));
    REQUIRE(ret.size() == 0);
}
