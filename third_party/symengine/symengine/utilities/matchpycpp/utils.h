#ifndef SYMENGINE_UTILITIES_MATCHPYCPP_UTILS_H_
#define SYMENGINE_UTILITIES_MATCHPYCPP_UTILS_H_

#include <vector>
#include <map>
#include <functional>

#include "common.h"
#include "bipartite.h"

using namespace std;

RCP<const Basic> None = symbol("None");

template <typename T, typename Comparison>
map<T, int, Comparison> count_multiset(const multiset<T, Comparison> &m)
{
    map<T, int, Comparison> result;
    for (const T &elem : m) {
        auto it = result.find(elem);
        if (it == result.end()) {
            result[elem] = 1;
        } else {
            it->second++;
        }
    }
    return result;
}

template <typename T>
vector<vector<T>> itertools_product(const vector<vector<T>> &v)
{
    vector<vector<T>> result;
    vector<vector<T>> temp;
    for (const T &e : v[0]) {
        vector<T> val;
        val.push_back(e);
        result.push_back(val);
    }

    for (unsigned int i = 1; i < v.size(); ++i) {
        temp.clear();
        const vector<T> &vi = v[i];
        for (vector<T> current : result) {
            for (const T &elem : vi) {
                vector<T> copycurr(current.begin(), current.end());
                copycurr.push_back(elem);
                temp.push_back(copycurr);
            }
        }
        result = temp;
    }
    return result;
}

// template <typename T>
class VariableWithCount
{
public:
    VariableWithCount(RCP<const Basic> name, unsigned count, unsigned minimum,
                      RCP<const Basic> defaultv)
        : name(name), count(count), minimum(minimum), defaultv(defaultv)
    {
    }
    VariableWithCount(const string &name, unsigned count, unsigned minimum,
                      RCP<const Basic> defaultv)
        : count(count), minimum(minimum), defaultv(defaultv)
    {
        this->name = symbol(name);
    }
    RCP<const Basic> name;
    unsigned count;
    unsigned minimum;
    RCP<const Basic> defaultv;
};

generator<SubstitutionMultiset>
_commutative_single_variable_partiton_iter(const multiset_basic &values,
                                           const VariableWithCount &variable)
{
    string name = variable.name->__str__();
    unsigned count = variable.count;
    unsigned minimum = variable.minimum;
    RCP<const Basic> defaultv = variable.defaultv;

    generator<SubstitutionMultiset> result;

    if (values.empty() && defaultv != None) {
        result.push_back(SubstitutionMultiset{{name, {defaultv}}});
        return result;
    }
    if (count == 1) {
        if (values.size() >= minimum) {
            if (name != "None") {
                result.push_back(SubstitutionMultiset{{name, {values}}});
            }
        }
    } else {
        multiset_basic new_values;
        for (const pair<const RCP<const Basic>, int> &p : count_multiset(values)) {
            RCP<const Basic> element = p.first;
            int element_count = p.second;
            if (element_count % count != 0) {
                return generator<SubstitutionMultiset>();
            }
            for (int i = 0; i < element_count; i++) {
                new_values.insert(element); // TODO: make this more efficient
            }
        }
        if (new_values.size() >= minimum) {
            if (name != "None") {
                result.push_back(SubstitutionMultiset{{name, {new_values}}});
            }
        }
    }
    return result;
}

function<void(SubstitutionMultiset)>
_make_variable_generator_factory(const RCP<const Basic> &value, const int total,
                                 const vector<VariableWithCount> &variables)
{
    vector<int> var_counts;
    for (const VariableWithCount &v : variables) {
        var_counts.push_back(v.count);
    }
    // vector<int> cache_key;
    // cache_key.push_back(total);
    // cache_key.insert(var_counts.begin(), var_counts.end());

    auto _factory = [&](SubstitutionMultiset subst) {
        // if cache_key in _linear_diop_solution_cache:
        //    solutions = _linear_diop_solution_cache[cache_key]
        // else:
        vector<SubstitutionMultiset> result;
        // TODO: finish
        /*
            solutions = list(solve_linear_diop(total, *var_counts))
            _linear_diop_solution_cache[cache_key] = solutions
        for (auto &solution : solutions) {
            new_subst = copy.copy(subst)
            for var, count in zip(variables, solution):
                new_subst[var.name][value] = count
            result.push_back(new_subst);
        }
        */
        return result;
    };

    return _factory;
}

generator<SubstitutionMultiset> commutative_sequence_variable_partition_iter(
    const multiset_basic &values, const vector<VariableWithCount> &variables)
{
    generator<SubstitutionMultiset> result;

    if (variables.size() == 1) {
        return _commutative_single_variable_partiton_iter(values, variables[0]);
    }

    vector<function<void(SubstitutionMultiset)>> generators;
    for (const pair<const RCP<const Basic>, int> &p : count_multiset(values)) {
        RCP<const Basic> value = p.first;
        int count = p.second;
        generators.push_back(
            _make_variable_generator_factory(value, count, variables));
    }

    map<string, multiset<int>> initial;
    for (const VariableWithCount &var : variables) {
        initial[var.name->__str__()] = multiset<int>();
    }
    // bool valid;
    // vector<map<string, multiset<int>>> result;
    // TODO:
    /*
    for (map<string, multiset<int>> &subst :
         generator_chain(initial, *generators)) {
        valid = true;
        for (VariableWithCount &var : variables) {
            if (var.defaultv != None && subst[var.name].size() == 0) {
                subst[var.name] = var.defaultv;
            } else if (subst[var.name].size() < var.minimum) {
                valid = false;
                break;
            }
        }
        if (valid) {
            if (subst.find("") != subst.end()) {
                delete subst[None];
            }
            result.push_back(subst);
        }
    }
    */
    return result;
}

#endif /* SYMENGINE_UTILITIES_MATCHPYCPP_UTILS_H_ */
