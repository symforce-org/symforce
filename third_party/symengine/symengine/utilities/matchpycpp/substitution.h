#ifndef SYMENGINE_UTILITIES_MATCHPYCPP_SUBSTITUTION_H_
#define SYMENGINE_UTILITIES_MATCHPYCPP_SUBSTITUTION_H_

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/mul.h>

#include <map>
#include <vector>

using namespace std;
using namespace SymEngine;

typedef map<string, multiset_basic> SubstitutionMultiset;

int try_add_variable(SubstitutionMultiset &subst, const string &variable_name,
                     const multiset_basic &replacement)
{
    auto elem = subst.find(variable_name);
    if (elem == subst.end()) {
        subst[variable_name] = replacement;
    } else {
        const multiset_basic &existing_value = elem->second;
        return unified_eq(existing_value, replacement);
    }
    return 0;
}

int try_add_variable(SubstitutionMultiset &subst, const string &variable_name,
                     const vector<RCP<const Basic>> &replacement)
{
    multiset_basic new_repl;
    new_repl.insert(replacement.begin(), replacement.end());
    return try_add_variable(subst, variable_name, new_repl);
}

int try_add_variable(SubstitutionMultiset &subst, const string &variable_name,
                     const RCP<const Basic> &replacement)
{
    multiset_basic new_repl = {replacement};
    return try_add_variable(subst, variable_name, new_repl);
}

SubstitutionMultiset
substitution_union(const SubstitutionMultiset &subst,
                   const vector<SubstitutionMultiset> &others)
{
    SubstitutionMultiset new_subst = subst;
    for (const SubstitutionMultiset &other : others) {
        for (const pair<const string, multiset_basic> &p : other) {
            int ret = try_add_variable(new_subst, p.first, p.second);
            assert(ret == 0);
        }
    }
    return new_subst;
}

#endif /* SYMENGINE_UTILITIES_MATCHPYCPP_SUBSTITUTION_H_ */
