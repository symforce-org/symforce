#ifndef SYMENGINE_UTILITIES_MATCHPYCPP_MANY_TO_ONE_H_
#define SYMENGINE_UTILITIES_MATCHPYCPP_MANY_TO_ONE_H_

#include "bipartite.h"
#include "utils.h"

#include <map>
#include <list>
#include <iterator>
#include <cstdint>

typedef int OperationMeta;

typedef BipartiteGraph<tuple<int, int>, tuple<int, int>,
                       vector<SubstitutionMultiset>>
    Subgraph;
typedef map<tuple<int, int>, tuple<int, int>> Matching;

//! Needed for `set` sorting:
struct lessVariableCount {
    //! true if memory locations `x < y`, false otherwise
    bool operator()(const tuple<VariableWithCount, OperationMeta> &x,
                    const tuple<VariableWithCount, OperationMeta> &y) const
    {
        return ((intptr_t)&x) < ((intptr_t)&y);
    }
};

typedef set<tuple<VariableWithCount, OperationMeta>, lessVariableCount>
    PatternSet;

class CommutativeMatcher
{
public:
    std::function<RCP<const Basic>(const RCP<const Basic> &,
                                   const RCP<const Basic> &)>
        associative;
    map<RCP<const Basic>, tuple<int, set<int>>, RCPBasicKeyLess> subjects;
    BipartiteGraph<int, int, vector<SubstitutionMultiset>> bipartite;
    int max_optional_count;
    // map subject_pattern_ids;
    map<int, RCP<const Basic>> subjects_by_id;
    map<set<int>, tuple<int, multiset<int>, PatternSet>> patterns;
    set<int> anonymous_patterns;

    CommutativeMatcher()
    {
    }

    virtual ~CommutativeMatcher()
    {
    }

    virtual generator<tuple<int, SubstitutionMultiset>>
    get_match_iter(const RCP<const Basic> &subject) = 0;

    int add_subject(const RCP<const Basic> &subject)
    {
        int subject_id;
        auto elem = subjects.find(subject);
        if (elem == subjects.end()) {
            subject_id = subjects.size();
            set<int> pattern_set;
            // tuple<int, set<RCP<const Basic>>> elem =
            // make_tuple(subjects.size(), set<RCP<const Basic>>());
            subjects[subject] = make_tuple(subject_id, pattern_set);
            subjects_by_id[subject_id] = subject;
            for (tuple<int, SubstitutionMultiset> &p :
                 get_match_iter(subject)) {
                int pattern_index = get<0>(p);
                SubstitutionMultiset substitution = get<1>(p);

                bipartite.setdefault(make_tuple(subject_id, pattern_index),
                                     vector<SubstitutionMultiset>());
                vector<SubstitutionMultiset> &edge_value = bipartite._edges.at(
                    make_tuple(subject_id, pattern_index));
                edge_value.push_back(SubstitutionMultiset(substitution));
                pattern_set.insert(pattern_index);
            }
        } else {
            subject_id = get<0>(elem->second);
        }
        return subject_id;
    }

    generator<tuple<int, SubstitutionMultiset>>
    match(const RCP<const Basic> &subject,
          const SubstitutionMultiset &substitution)
    {
        Deque subjects = {subject};
        return match(subjects, substitution);
    }

    generator<tuple<int, SubstitutionMultiset>>
    match(const Deque &subjects, const SubstitutionMultiset &substitution)
    {
        vector<tuple<int, SubstitutionMultiset>> result;

        multiset<int> subject_ids;
        multiset<int> pattern_ids;
        int subject_id;
        // int pattern_index;
        multiset<int> pattern_set;
        // tuple<> pattern_vars;
        set<int> subject_pattern_ids;
        if (max_optional_count > 0) {
            subject_id = get<0>(this->subjects[None]);
            subject_pattern_ids = get<1>(this->subjects[None]);
            subject_ids.insert(subject_id);
            for (int i = 0; i < max_optional_count; ++i) {
                pattern_ids.insert(subject_pattern_ids.begin(),
                                   subject_pattern_ids.end());
            }
        }
        for (const RCP<const Basic> &subject : subjects) {
            tuple<int, set<int>> p = this->subjects[subject];
            subject_id = get<0>(p);
            subject_pattern_ids = get<1>(p);
            subject_ids.insert(subject_id);
            pattern_ids.insert(subject_pattern_ids.begin(),
                               subject_pattern_ids.end());
        }
        for (const pair<const set<int>, tuple<int, multiset<int>, PatternSet>> &p :
             patterns) {
            int pattern_index = get<0>(p.second);
            multiset<int> pattern_set = get<1>(p.second);
            PatternSet pattern_vars = get<2>(p.second);
            if (!pattern_set.empty()) {
                if (!includes(pattern_set.begin(), pattern_set.end(),
                              pattern_ids.begin(), pattern_ids.end())) {
                    continue;
                }
                vector<tuple<SubstitutionMultiset, multiset<int>>>
                    bipartite_match_iter = _match_with_bipartite(
                        subject_ids, pattern_set, substitution);
                for (tuple<SubstitutionMultiset, multiset<int>> &p :
                     bipartite_match_iter) {
                    SubstitutionMultiset bipartite_substitution = get<0>(p);
                    multiset<int> matched_subjects = get<1>(p);
                    multiset<int> ids;
                    set_difference(subject_ids.begin(), subject_ids.end(),
                                   matched_subjects.begin(),
                                   matched_subjects.end(),
                                   inserter(ids, ids.begin()));
                    multiset_basic remaining;
                    for (int id : ids) {
                        if (eq(*subjects_by_id.at(id), *None)) {
                            continue;
                        }
                        remaining.insert(subjects_by_id.at(id));
                    }
                    if (!pattern_vars.empty()) {
                        vector<SubstitutionMultiset> sequence_var_iter
                            = _match_sequence_variables(remaining, pattern_vars,
                                                        bipartite_substitution);
                        for (SubstitutionMultiset &result_substitution :
                             sequence_var_iter) {
                            // YIELD:
                            result.push_back(
                                make_tuple(pattern_index, result_substitution));
                        }
                    } else if (remaining.empty()) {
                        result.push_back(
                            make_tuple(pattern_index, bipartite_substitution));
                    }
                }
            } else if (!pattern_vars.empty()) {
                multiset_basic multiset_arg(subjects.begin(), subjects.end());
                vector<SubstitutionMultiset> sequence_var_iter
                    = _match_sequence_variables(multiset_arg, pattern_vars,
                                                substitution);
                for (SubstitutionMultiset &variable_substitution :
                     sequence_var_iter) {
                    // YIELD
                    result.push_back(
                        make_tuple(pattern_index, variable_substitution));
                }
            } else if (subjects.empty()) {
                // YIELD:
                result.push_back(make_tuple(pattern_index, substitution));
            }
        }
        return result;
    }

    generator<tuple<SubstitutionMultiset, multiset<int>>>
    _match_with_bipartite(const multiset<int> &subject_ids,
                          const multiset<int> &pattern_set,
                          const SubstitutionMultiset &substitution)
    {
        vector<tuple<SubstitutionMultiset, multiset<int>>> result;

        typedef vector<SubstitutionMultiset> TEdgeValue;

        Subgraph bipartite = _build_bipartite(subject_ids, pattern_set);
        for (const Matching &matching :
             enum_maximum_matchings_iter<tuple<int, int>, tuple<int, int>,
                                         vector<SubstitutionMultiset>>(
                 bipartite)) {
            if (matching.size() < pattern_set.size()) {
                break;
            }
            if (!_is_canonical_matching(matching)) {
                continue;
            }
            /*
            auto loop = [&](list<vector<SubstitutionMultiset>> elab) {

            };
             */
            vector<TEdgeValue> iterobjs;
            for (const pair<const tuple<int, int>, tuple<int, int>> &p3 : matching) {
                iterobjs.push_back(
                    bipartite.__getitem__(make_tuple(p3.first, p3.second)));
            }
            for (vector<SubstitutionMultiset> &substs :
                 itertools_product(iterobjs)) {
                SubstitutionMultiset bipartite_substitution
                    = substitution_union(substitution, substs);

                multiset<int> matched_subjects;
                for (const pair<const tuple<int, int>, tuple<int, int>> &p3 :
                     matching) {
                    int elem = get<0>(p3.first);
                    matched_subjects.insert(elem);
                }
                // YIELD:
                result.push_back(
                    make_tuple(bipartite_substitution, matched_subjects));
            }
        }
        return result;
    }

    vector<SubstitutionMultiset>
    _match_sequence_variables(const multiset_basic &subjects,
                              const PatternSet &pattern_vars,
                              const SubstitutionMultiset &substitution)
    {
        vector<SubstitutionMultiset> result;

        vector<VariableWithCount> only_counts;
        list<string> wrapped_vars;
        for (const tuple<VariableWithCount, OperationMeta> &p : pattern_vars) {
            only_counts.push_back(get<0>(p));
            wrapped_vars.push_back(get<0>(p).name->__str__());
        }
        for (SubstitutionMultiset &variable_substitution :
             commutative_sequence_variable_partition_iter(subjects,
                                                          only_counts)) {
            for (const string &var : wrapped_vars) {
                multiset_basic operands = variable_substitution[var];

                // for (const pair<RCP<const Basic>, int> &pp :
                // count_multiset(operands)) {
                //    RCP<const Basic> operands_var = pp.first;
                //    int operands_size = pp.second;
                if (operands.size() > 1) {
                    throw runtime_error("not implemented");
                } else {
                    variable_substitution[var] = {*operands.begin()};
                }
            }
            SubstitutionMultiset result_substitution
                = substitution_union(substitution, {variable_substitution});

            // YIELD:
            result.push_back(result_substitution);
        }
        return result;
    }

    bool _is_canonical_matching(const Matching &matching)
    {
        for (const pair<const tuple<int, int>, tuple<int, int>> &pair1 : matching) {
            //.items():
            int s1 = get<0>(pair1.first);
            int n1 = get<1>(pair1.first);
            int p1 = get<0>(pair1.second);
            int m1 = get<1>(pair1.second);
            for (const pair<const tuple<int, int>, tuple<int, int>> &pair2 :
                 matching) {
                int s2 = get<0>(pair2.first);
                int n2 = get<1>(pair2.first);
                int p2 = get<0>(pair2.second);
                int m2 = get<1>(pair2.second);

                if ((anonymous_patterns.find(p1) != anonymous_patterns.end())
                    && (anonymous_patterns.find(p2)
                        != anonymous_patterns.end())) {
                    if (n1 < n2 && m1 > m2) {
                        return false;
                    } else if (s1 == s2 && n1 < n2 && m1 > m2) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    Subgraph _build_bipartite(const multiset<int> &subjects,
                              const multiset<int> &patterns)
    {
        Subgraph bipartite;
        int n = 0;
        int m = 0;
        map<int, int> p_states;
        for (const pair<const int, int> &p : count_multiset(subjects)) {
            int subject = p.first;
            int s_count = p.second;
            auto elem = this->bipartite._graph_left.find(subject);
            if (elem != this->bipartite._graph_left.end()) {
                bool any_patterns = false;
                for (int pattern : elem->second) {
                    if (patterns.find(pattern) != patterns.end()) {
                        any_patterns = true;
                        vector<SubstitutionMultiset> subst
                            = this->bipartite.__getitem__(
                                make_tuple(subject, pattern));
                        int p_count = count_multiset(patterns).at(pattern);
                        int p_start;
                        if (p_states.find(pattern) != p_states.end()) {
                            p_start = p_states[pattern];
                        } else {
                            p_start = p_states[pattern] = m;
                            m += p_count;
                        }
                        for (int i = n; i < n + s_count; i++) {
                            for (int j = p_start; j < p_start + p_count; j++) {
                                // (subject, i), (pattern, j)
                                bipartite.__setitem__(
                                    make_tuple(make_tuple(subject, i),
                                               make_tuple(pattern, j)),
                                    subst);
                            }
                        }
                    }
                }
                if (any_patterns) {
                    n += s_count;
                }
            }
        }

        return bipartite;
    }
};

#endif /* SYMENGINE_UTILITIES_MATCHPYCPP_MANY_TO_ONE_H_ */
