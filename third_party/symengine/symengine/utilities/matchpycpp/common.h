#ifndef SYMENGINE_UTILITIES_MATCHPYCPP_COMMON_H_
#define SYMENGINE_UTILITIES_MATCHPYCPP_COMMON_H_

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/mul.h>
#include <map>
#include <string>
#include <queue>

#include <setjmp.h>

#include "substitution.h"

using namespace std;
using namespace SymEngine;

template <typename T>
using generator = vector<T>;

// Assuming TLeft = TRight, otherwise
// Node should be tuple<int, variant<TLeft, TRight>>
#define TYPES_DERIVED_FROM_TLEFT_TRIGHT                                        \
    typedef tuple<int, TLeft> Node;                                            \
    typedef vector<Node> NodeList;                                             \
    typedef set<Node> NodeSet;                                                 \
    typedef tuple<TLeft, TRight> Edge;

constexpr int LEFT = 0;
constexpr int RIGHT = 1;

typedef deque<RCP<const Basic>> Deque;

Deque get_deque(const RCP<const Basic> &expr)
{
    vec_basic v = expr->get_args();
    Deque d(v.begin(), v.end());
    return d;
}

#endif /* SYMENGINE_UTILITIES_MATCHPYCPP_COMMON_H_ */
