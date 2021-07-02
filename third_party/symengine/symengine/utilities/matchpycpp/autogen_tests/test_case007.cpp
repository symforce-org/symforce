/*
 * This file was automatically generated: DO NOT EDIT.
 *
 * Use symengine/utilities/matchpycpp/generate_tests.py to generate this file.
 *
 * Decision tree matching expressions:
 * ['x + y + wo', '3*x*y*wo', '7*x*y*w']
 *
 * Wildcards:
 * ['w', 'wo']
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
RCP<const Basic> z = symbol("z");

class CommutativeMatcher2248 : public CommutativeMatcher
{
public:
    CommutativeMatcher2248()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0, 1}, {make_tuple(VariableWithCount("i1.0", 1, 1, integer(0)), SYMENGINE_ADD)})}};
        subjects = {};
        subjects_by_id = {};
        associative = [](const RCP<const Basic> &x, const RCP<const Basic> &y) {
            return add(x, y);
        };
        max_optional_count = 1;
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
        // State 2247
        if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2249
            if (subjects.size() == 0) {
                // 0: x
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *y)) {
            RCP<const Basic> tmp2 = subjects.front();
            subjects.pop_front();
            // State 2250
            if (subjects.size() == 0) {
                // 1: y
                result.push_back(make_tuple(1, subst0));
            }
            subjects.push_front(tmp2);
        }
        return result;
    }
};

class CommutativeMatcher2253 : public CommutativeMatcher
{
public:
    CommutativeMatcher2253()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0, 1, 2}, {make_tuple(VariableWithCount("i1.0", 1, 1, integer(1)), SYMENGINE_MUL)})}, {{1}, make_tuple<int, multiset<int>, PatternSet>(1, {3, 1, 2}, {make_tuple(VariableWithCount("i1.0", 1, 1, None), SYMENGINE_MUL)})}};
        subjects = {};
        subjects_by_id = {};
        associative = [](const RCP<const Basic> &x, const RCP<const Basic> &y) {
            return mul(x, y);
        };
        max_optional_count = 1;
        anonymous_patterns = {0, 1, 2, 3};

        add_subject(None);
    }

    generator<tuple<int, SubstitutionMultiset>>
    get_match_iter(const RCP<const Basic> &subject)
    {
        generator<tuple<int, SubstitutionMultiset>> result;
        Deque subjects;
        subjects.push_front(subject);
        SubstitutionMultiset subst0;
        // State 2252
        if (subjects.size() >= 1 && eq(*subjects[0], *integer(3))) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2254
            if (subjects.size() == 0) {
                // 0: 3
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
            RCP<const Basic> tmp2 = subjects.front();
            subjects.pop_front();
            // State 2255
            if (subjects.size() == 0) {
                // 1: x
                result.push_back(make_tuple(1, subst0));
            }
            subjects.push_front(tmp2);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *y)) {
            RCP<const Basic> tmp3 = subjects.front();
            subjects.pop_front();
            // State 2256
            if (subjects.size() == 0) {
                // 2: y
                result.push_back(make_tuple(2, subst0));
            }
            subjects.push_front(tmp3);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *integer(7))) {
            RCP<const Basic> tmp4 = subjects.front();
            subjects.pop_front();
            // State 2258
            if (subjects.size() == 0) {
                // 3: 7
                result.push_back(make_tuple(3, subst0));
            }
            subjects.push_front(tmp4);
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
    // State 2246
    if (subjects.size() >= 1 && is_a<Add>(*subjects[0])) {
        RCP<const Basic> tmp1 = subjects.front();
        subjects.pop_front();
        RCP<const Basic> associative1 = tmp1;
        string associative_type1 = tmp1->__str__();
        Deque subjects2 = get_deque(tmp1);
        CommutativeMatcher2248 matcher;
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
                // State 2251
                if (subjects.size() == 0) {
                    SubstitutionMultiset tmp_subst;
                    tmp_subst["wo"] = subst1["i1.0"];
                    // 0: x + y + wo
                    result.push_back(make_tuple(0, tmp_subst));
                }
            }
        }
        subjects.push_front(tmp1);
    }
    if (subjects.size() >= 1 && is_a<Mul>(*subjects[0])) {
        RCP<const Basic> tmp4 = subjects.front();
        subjects.pop_front();
        RCP<const Basic> associative1 = tmp4;
        string associative_type1 = tmp4->__str__();
        Deque subjects5 = get_deque(tmp4);
        CommutativeMatcher2253 matcher;
        Deque tmp6 = subjects5;
        subjects5 = {};
        for (RCP<const Basic> &s : tmp6) {
            matcher.add_subject(s);
        }
        for (tuple<int, SubstitutionMultiset> &p :
             matcher.match(tmp6, subst0)) {
            int pattern_index = get<0>(p);
            SubstitutionMultiset subst1 = get<1>(p);
            if (pattern_index == 0) {
                // State 2257
                if (subjects.size() == 0) {
                    SubstitutionMultiset tmp_subst;
                    tmp_subst["wo"] = subst1["i1.0"];
                    // 1: 3*x*y*wo
                    result.push_back(make_tuple(1, tmp_subst));
                }
            }
            if (pattern_index == 1) {
                // State 2259
                if (subjects.size() == 0) {
                    SubstitutionMultiset tmp_subst;
                    tmp_subst["w"] = subst1["i1.0"];
                    // 2: 7*x*y*w
                    result.push_back(make_tuple(2, tmp_subst));
                }
            }
        }
        subjects.push_front(tmp4);
    }
    return result;
}

TEST_CASE("GeneratedMatchPyTest7", "")
{
    generator<tuple<int, SubstitutionMultiset>> ret;
    SubstitutionMultiset substitution;

    // Pattern x + y + wo matching x + y with substitution {'wo': '0'}:
    ret = match_root(add(x, y));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("wo") != substitution.end());
    REQUIRE(eq(*(*substitution.at("wo").begin()), *integer(0)));

    // Pattern 3*x*y*wo matching 3*x*y with substitution {'wo': '1'}:
    ret = match_root(mul(integer(3), mul(x, y)));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 1);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("wo") != substitution.end());
    REQUIRE(eq(*(*substitution.at("wo").begin()), *integer(1)));

    // Pattern 2*x*y not matching:
    ret = match_root(mul(integer(2), mul(x, y)));
    REQUIRE(ret.size() == 0);

    // Pattern x + y + wo matching x + y + z with substitution {'wo': 'z'}:
    ret = match_root(add(x, add(y, z)));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("wo") != substitution.end());
    REQUIRE(eq(*(*substitution.at("wo").begin()), *z));

    // Pattern 7*x*y not matching:
    ret = match_root(mul(integer(7), mul(x, y)));
    REQUIRE(ret.size() == 0);
}
