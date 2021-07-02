/**
 *  \file dict.h
 *  Dictionary implementation
 *
 **/

#ifndef SYMENGINE_DICT_H
#define SYMENGINE_DICT_H
#include <symengine/mp_class.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>
#include <unordered_map>
#include <set>
#include <unordered_set>

namespace SymEngine
{

class Basic;
class Number;
class Integer;
class Expression;
class Symbol;
struct RCPBasicHash;
struct RCPBasicKeyEq;
struct RCPBasicKeyLess;
struct RCPIntegerKeyLess;

bool eq(const Basic &, const Basic &);
typedef uint64_t hash_t;
typedef std::unordered_map<RCP<const Basic>, RCP<const Number>, RCPBasicHash,
                           RCPBasicKeyEq>
    umap_basic_num;
typedef std::unordered_map<short, RCP<const Basic>> umap_short_basic;
typedef std::unordered_map<int, RCP<const Basic>> umap_int_basic;
typedef std::unordered_map<RCP<const Basic>, RCP<const Basic>, RCPBasicHash,
                           RCPBasicKeyEq>
    umap_basic_basic;
typedef std::unordered_set<RCP<const Basic>, RCPBasicHash, RCPBasicKeyEq>
    uset_basic;

typedef std::vector<int> vec_int;
typedef std::vector<RCP<const Basic>> vec_basic;
typedef std::vector<RCP<const Integer>> vec_integer;
typedef std::vector<unsigned int> vec_uint;
typedef std::vector<integer_class> vec_integer_class;
typedef std::vector<RCP<const Symbol>> vec_sym;
typedef std::set<RCP<const Basic>, RCPBasicKeyLess> set_basic;
typedef std::multiset<RCP<const Basic>, RCPBasicKeyLess> multiset_basic;
typedef std::map<vec_uint, unsigned long long int> map_vec_uint;
typedef std::map<vec_uint, integer_class> map_vec_mpz;
typedef std::map<RCP<const Basic>, RCP<const Number>, RCPBasicKeyLess>
    map_basic_num;
typedef std::map<RCP<const Basic>, RCP<const Basic>, RCPBasicKeyLess>
    map_basic_basic;
typedef std::map<RCP<const Integer>, unsigned, RCPIntegerKeyLess>
    map_integer_uint;
typedef std::map<unsigned, integer_class> map_uint_mpz;
typedef std::map<unsigned, rational_class> map_uint_mpq;
typedef std::map<int, Expression> map_int_Expr;
typedef std::unordered_map<RCP<const Basic>, unsigned int, RCPBasicHash,
                           RCPBasicKeyEq>
    umap_basic_uint;
typedef std::vector<std::pair<RCP<const Basic>, RCP<const Basic>>> vec_pair;

template <typename T>
struct vec_hash {
    hash_t operator()(const T &v) const;
};

typedef std::unordered_map<vec_uint, integer_class, vec_hash<vec_uint>>
    umap_uvec_mpz;
typedef std::unordered_map<vec_int, integer_class, vec_hash<vec_int>>
    umap_vec_mpz;
typedef std::unordered_map<vec_int, Expression, vec_hash<vec_int>>
    umap_vec_expr;
//! `insert(m, first, second)` is equivalent to `m[first] = second`, just
//! faster,
//! because no default constructor is called on the `second` type.
template <typename T1, typename T2, typename T3>
inline void insert(T1 &m, const T2 &first, const T3 &second)
{
    m.insert(std::pair<T2, T3>(first, second));
}

// Takes an unordered map of type M with key type K and returns a vector of K
// ordered by C.
template <class M, typename C = std::less<typename M::key_type>>
std::vector<typename M::key_type> sorted_keys(const M &d)
{
    std::vector<typename M::key_type> v;
    v.reserve(d.size());
    for (auto &p : d) {
        v.push_back(p.first);
    }
    std::sort(v.begin(), v.end(), C());
    return v;
}

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T, typename U>
inline bool unified_eq(const std::pair<T, U> &a, const std::pair<T, U> &b)
{
    return unified_eq(a.first, b.first) and unified_eq(a.second, b.second);
}

template <typename T, typename U>
inline bool unified_eq(const std::set<T, U> &a, const std::set<T, U> &b)
{
    return ordered_eq(a, b);
}

template <typename T, typename U>
inline bool unified_eq(const std::multiset<T, U> &a,
                       const std::multiset<T, U> &b)
{
    return ordered_eq(a, b);
}

template <typename K, typename V, typename C>
inline bool unified_eq(const std::map<K, V, C> &a, const std::map<K, V, C> &b)
{
    return ordered_eq(a, b);
}

template <typename K, typename V, typename H, typename E>
inline bool unified_eq(const std::unordered_map<K, V, H, E> &a,
                       const std::unordered_map<K, V, H, E> &b)
{
    return unordered_eq(a, b);
}

template <typename T, typename U,
          typename = enable_if_t<std::is_base_of<Basic, T>::value
                                 and std::is_base_of<Basic, U>::value>>
inline bool unified_eq(const RCP<const T> &a, const RCP<const U> &b)
{
    return eq(*a, *b);
}

template <typename T,
          typename = enable_if_t<std::is_arithmetic<T>::value
                                 or std::is_same<T, integer_class>::value>>
inline bool unified_eq(const T &a, const T &b)
{
    return a == b;
}

//! eq function base
//! \return true if the two dictionaries `a` and `b` are equal. Otherwise false
template <class T>
inline bool unordered_eq(const T &a, const T &b)
{
    // This follows the same algorithm as Python's dictionary comparison
    // (a==b), which is implemented by "dict_equal" function in
    // Objects/dictobject.c.

    // Can't be equal if # of entries differ:
    if (a.size() != b.size())
        return false;
    // Loop over keys in "a":
    for (const auto &p : a) {
        // O(1) lookup of the key in "b":
        auto f = b.find(p.first);
        if (f == b.end())
            return false; // no such element in "b"
        if (not unified_eq(p.second, f->second))
            return false; // values not equal
    }
    return true;
}

template <class T>
inline bool ordered_eq(const T &A, const T &B)
{
    // Can't be equal if # of entries differ:
    if (A.size() != B.size())
        return false;
    // Loop over elements in "a" and "b":
    auto a = A.begin();
    auto b = B.begin();
    for (; a != A.end(); ++a, ++b) {
        if (not unified_eq(*a, *b))
            return false; // values not equal
    }
    return true;
}

template <typename T>
inline bool unified_eq(const std::vector<T> &a, const std::vector<T> &b)
{
    return ordered_eq(a, b);
}

//! compare functions base
//! \return -1, 0, 1 for a < b, a == b, a > b
template <typename T,
          typename = enable_if_t<std::is_arithmetic<T>::value
                                 or std::is_same<T, integer_class>::value
                                 or std::is_same<T, rational_class>::value>>
inline int unified_compare(const T &a, const T &b)
{
    if (a == b)
        return 0;
    return a < b ? -1 : 1;
}

template <typename T, typename U,
          typename = enable_if_t<std::is_base_of<Basic, T>::value
                                 and std::is_base_of<Basic, U>::value>>
inline int unified_compare(const RCP<const T> &a, const RCP<const U> &b)
{
    return a->__cmp__(*b);
}

template <class T>
inline int ordered_compare(const T &A, const T &B);

template <typename T>
inline int unified_compare(const std::vector<T> &a, const std::vector<T> &b)
{
    return ordered_compare(a, b);
}

template <typename T, typename U>
inline int unified_compare(const std::set<T, U> &a, const std::set<T, U> &b)
{
    return ordered_compare(a, b);
}

template <typename T, typename U>
inline int unified_compare(const std::multiset<T, U> &a,
                           const std::multiset<T, U> &b)
{
    return ordered_compare(a, b);
}

template <typename T, typename U>
inline int unified_compare(const std::pair<T, U> &a, const std::pair<T, U> &b)
{
    auto t = unified_compare(a.first, b.first);
    if (t == 0) {
        return unified_compare(a.second, b.second);
    } else {
        return t;
    }
}

template <typename K, typename V, typename C>
inline int unified_compare(const std::map<K, V, C> &a,
                           const std::map<K, V, C> &b)
{
    return ordered_compare(a, b);
}

template <typename K, typename V, typename H, typename E>
inline int unified_compare(const std::unordered_map<K, V, H, E> &a,
                           const std::unordered_map<K, V, H, E> &b)
{
    return unordered_compare(a, b);
}

template <class T>
inline int ordered_compare(const T &A, const T &B)
{
    // Can't be equal if # of entries differ:
    if (A.size() != B.size())
        return A.size() < B.size() ? -1 : 1;

    // Loop over elements in "a" and "b":
    auto a = A.begin();
    auto b = B.begin();
    for (; a != A.end(); ++a, ++b) {
        auto t = unified_compare(*a, *b);
        if (t != 0)
            return t; // values not equal
    }
    return 0;
}

template <class M, typename C = std::less<typename M::key_type>>
inline int unordered_compare(const M &a, const M &b)
{
    // Can't be equal if # of entries differ:
    if (a.size() != b.size())
        return a.size() < b.size() ? -1 : 1;

    std::vector<typename M::key_type> va = sorted_keys<M, C>(a);
    std::vector<typename M::key_type> vb = sorted_keys<M, C>(b);

    for (unsigned int i = 0; i < va.size() && i < vb.size(); i++) {
        bool s = C()(va[i], vb[i]);
        if (s)
            return -1;
        s = C()(vb[i], va[i]);
        if (s)
            return 1;

        int t = unified_compare(a.find(va[i])->second, b.find(vb[i])->second);
        if (t != 0)
            return t;
    }
    return 0;
}

//! misc functions
bool vec_basic_eq_perm(const vec_basic &a, const vec_basic &b);

//! print functions
std::ostream &operator<<(std::ostream &out, const SymEngine::umap_basic_num &d);
std::ostream &operator<<(std::ostream &out, const SymEngine::map_basic_num &d);
std::ostream &operator<<(std::ostream &out,
                         const SymEngine::map_basic_basic &d);
std::ostream &operator<<(std::ostream &out,
                         const SymEngine::umap_basic_basic &d);
std::ostream &operator<<(std::ostream &out, const SymEngine::vec_basic &d);
std::ostream &operator<<(std::ostream &out, const SymEngine::set_basic &d);
std::ostream &operator<<(std::ostream &out, const SymEngine::map_int_Expr &d);
std::ostream &operator<<(std::ostream &out, const SymEngine::vec_pair &d);

} // SymEngine

#endif
