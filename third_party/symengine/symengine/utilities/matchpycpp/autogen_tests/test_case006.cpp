/*
 * This file was automatically generated: DO NOT EDIT.
 *
 * Use symengine/utilities/matchpycpp/generate_tests.py to generate this file.
 *
 * Decision tree matching expressions:
 * ['x**(x + w)', 'x + y + w', 'w**(-x*w + 1)']
 *
 * Wildcards:
 * ['w']
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

class CommutativeMatcher2226 : public CommutativeMatcher
{
public:
    CommutativeMatcher2226()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0}, {make_tuple(VariableWithCount("i3.0", 1, 1, None), SYMENGINE_ADD)})}};
        subjects = {};
        subjects_by_id = {};
        associative = [](const RCP<const Basic> &x, const RCP<const Basic> &y) {
            return add(x, y);
        };
        max_optional_count = 0;
        anonymous_patterns = {0};

        add_subject(None);
    }

    generator<tuple<int, SubstitutionMultiset>>
    get_match_iter(const RCP<const Basic> &subject)
    {
        generator<tuple<int, SubstitutionMultiset>> result;
        Deque subjects;
        subjects.push_front(subject);
        SubstitutionMultiset subst0;
        // State 2225
        if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2227
            if (subjects.size() == 0) {
                // 0: x
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        return result;
    }
};

class CommutativeMatcher2240 : public CommutativeMatcher
{
public:
    CommutativeMatcher2240()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0, 1}, {make_tuple(VariableWithCount("i1", 1, 1, None), SYMENGINE_MUL)})}};
        subjects = {};
        subjects_by_id = {};
        associative = [](const RCP<const Basic> &x, const RCP<const Basic> &y) {
            return mul(x, y);
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
        // State 2239
        if (subjects.size() >= 1 && eq(*subjects[0], *integer(-1))) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2241
            if (subjects.size() == 0) {
                // 0: -1
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
            RCP<const Basic> tmp2 = subjects.front();
            subjects.pop_front();
            // State 2242
            if (subjects.size() == 0) {
                // 1: x
                result.push_back(make_tuple(1, subst0));
            }
            subjects.push_front(tmp2);
        }
        return result;
    }
};

class CommutativeMatcher2237 : public CommutativeMatcher
{
public:
    CommutativeMatcher2237()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0, 1}, {})}};
        subjects = {};
        subjects_by_id = {};
        associative = [](const RCP<const Basic> &x, const RCP<const Basic> &y) {
            return add(x, y);
        };
        max_optional_count = 0;
        anonymous_patterns = {0};

        add_subject(None);
    }

    generator<tuple<int, SubstitutionMultiset>>
    get_match_iter(const RCP<const Basic> &subject)
    {
        generator<tuple<int, SubstitutionMultiset>> result;
        Deque subjects;
        subjects.push_front(subject);
        SubstitutionMultiset subst0;
        // State 2236
        if (subjects.size() >= 1 && eq(*subjects[0], *integer(1))) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2238
            if (subjects.size() == 0) {
                // 0: 1
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        if (subjects.size() >= 1 && is_a<Mul>(*subjects[0])) {
            RCP<const Basic> tmp2 = subjects.front();
            subjects.pop_front();
            RCP<const Basic> associative1 = tmp2;
            string associative_type1 = tmp2->__str__();
            Deque subjects3 = get_deque(tmp2);
            CommutativeMatcher2240 matcher;
            Deque tmp4 = subjects3;
            subjects3 = {};
            for (RCP<const Basic> &s : tmp4) {
                matcher.add_subject(s);
            }
            for (tuple<int, SubstitutionMultiset> &p :
                 matcher.match(tmp4, subst0)) {
                int pattern_index = get<0>(p);
                SubstitutionMultiset subst1 = get<1>(p);
                if (pattern_index == 0) {
                    // State 2243
                    if (subjects.size() == 0) {
                        // 1: -x*w
                        result.push_back(make_tuple(1, subst1));
                    }
                }
            }
            subjects.push_front(tmp2);
        }
        return result;
    }
};

class CommutativeMatcher2231 : public CommutativeMatcher
{
public:
    CommutativeMatcher2231()
    {
        patterns = {{{0}, make_tuple<int, multiset<int>, PatternSet>(0, {0, 1}, {make_tuple(VariableWithCount("i1.0", 1, 1, None), SYMENGINE_ADD)})}};
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
        // State 2230
        if (subjects.size() >= 1 && eq(*subjects[0], *x)) {
            RCP<const Basic> tmp1 = subjects.front();
            subjects.pop_front();
            // State 2232
            if (subjects.size() == 0) {
                // 0: x
                result.push_back(make_tuple(0, subst0));
            }
            subjects.push_front(tmp1);
        }
        if (subjects.size() >= 1 && eq(*subjects[0], *y)) {
            RCP<const Basic> tmp2 = subjects.front();
            subjects.pop_front();
            // State 2233
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
    // State 2222
    if (subjects.size() >= 1 && is_a<Pow>(*subjects[0])) {
        RCP<const Basic> tmp1 = subjects.front();
        subjects.pop_front();
        Deque subjects2 = get_deque(tmp1);
        // State 2223
        if (subjects2.size() >= 1 && eq(*subjects2[0], *x)) {
            RCP<const Basic> tmp3 = subjects2.front();
            subjects2.pop_front();
            // State 2224
            if (subjects2.size() >= 1 && is_a<Add>(*subjects2[0])) {
                RCP<const Basic> tmp4 = subjects2.front();
                subjects2.pop_front();
                RCP<const Basic> associative1 = tmp4;
                string associative_type1 = tmp4->__str__();
                Deque subjects5 = get_deque(tmp4);
                CommutativeMatcher2226 matcher;
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
                        // State 2228
                        if (subjects2.size() == 0) {
                            // State 2229
                            if (subjects.size() == 0) {
                                SubstitutionMultiset tmp_subst;
                                tmp_subst["w"] = subst1["i3.0"];
                                // 0: x**(x + w)
                                result.push_back(make_tuple(0, tmp_subst));
                            }
                        }
                    }
                }
                subjects2.push_front(tmp4);
            }
            subjects2.push_front(tmp3);
        }
        if (subjects2.size() >= 1) {
            RCP<const Basic> tmp7 = subjects2.front();
            subjects2.pop_front();
            SubstitutionMultiset subst1 = SubstitutionMultiset(subst0);
            if (!try_add_variable(subst1, "i1", tmp7)) {
                // State 2235
                if (subjects2.size() >= 1 && is_a<Add>(*subjects2[0])) {
                    RCP<const Basic> tmp9 = subjects2.front();
                    subjects2.pop_front();
                    RCP<const Basic> associative1 = tmp9;
                    string associative_type1 = tmp9->__str__();
                    Deque subjects10 = get_deque(tmp9);
                    CommutativeMatcher2237 matcher;
                    Deque tmp11 = subjects10;
                    subjects10 = {};
                    for (RCP<const Basic> &s : tmp11) {
                        matcher.add_subject(s);
                    }
                    for (tuple<int, SubstitutionMultiset> &p :
                         matcher.match(tmp11, subst1)) {
                        int pattern_index = get<0>(p);
                        SubstitutionMultiset subst2 = get<1>(p);
                        if (pattern_index == 0) {
                            // State 2244
                            if (subjects2.size() == 0) {
                                // State 2245
                                if (subjects.size() == 0) {
                                    SubstitutionMultiset tmp_subst;
                                    tmp_subst["w"] = subst2["i1"];
                                    // 2: w**(-x*w + 1)
                                    result.push_back(make_tuple(2, tmp_subst));
                                }
                            }
                        }
                    }
                    subjects2.push_front(tmp9);
                }
            }
            subjects2.push_front(tmp7);
        }
        subjects.push_front(tmp1);
    }
    if (subjects.size() >= 1 && is_a<Add>(*subjects[0])) {
        RCP<const Basic> tmp12 = subjects.front();
        subjects.pop_front();
        RCP<const Basic> associative1 = tmp12;
        string associative_type1 = tmp12->__str__();
        Deque subjects13 = get_deque(tmp12);
        CommutativeMatcher2231 matcher;
        Deque tmp14 = subjects13;
        subjects13 = {};
        for (RCP<const Basic> &s : tmp14) {
            matcher.add_subject(s);
        }
        for (tuple<int, SubstitutionMultiset> &p :
             matcher.match(tmp14, subst0)) {
            int pattern_index = get<0>(p);
            SubstitutionMultiset subst1 = get<1>(p);
            if (pattern_index == 0) {
                // State 2234
                if (subjects.size() == 0) {
                    SubstitutionMultiset tmp_subst;
                    tmp_subst["w"] = subst1["i1.0"];
                    // 1: x + y + w
                    result.push_back(make_tuple(1, tmp_subst));
                }
            }
        }
        subjects.push_front(tmp12);
    }
    return result;
}

TEST_CASE("GeneratedMatchPyTest6", "")
{
    generator<tuple<int, SubstitutionMultiset>> ret;
    SubstitutionMultiset substitution;

    // Pattern x + y not matching:
    ret = match_root(add(x, y));
    REQUIRE(ret.size() == 0);

    // Pattern x + y + w matching x + y + z with substitution {'w': 'z'}:
    ret = match_root(add(x, add(y, z)));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 1);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("w") != substitution.end());
    REQUIRE(eq(*(*substitution.at("w").begin()), *z));

    // Pattern x**2 not matching:
    ret = match_root(pow(x, integer(2)));
    REQUIRE(ret.size() == 0);

    // Pattern x**(x + w) matching x**(x + y) with substitution {'w': 'y'}:
    ret = match_root(pow(x, add(x, y)));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("w") != substitution.end());
    REQUIRE(eq(*(*substitution.at("w").begin()), *y));

    // Pattern x**(x + w) matching x**(x + 2) with substitution {'w': '2'}:
    ret = match_root(pow(x, add(integer(2), x)));
    REQUIRE(ret.size() > 0);
    REQUIRE(get<0>(ret[0]) == 0);
    substitution = get<1>(ret[0]);
    REQUIRE(substitution.find("w") != substitution.end());
    REQUIRE(eq(*(*substitution.at("w").begin()), *integer(2)));
}
