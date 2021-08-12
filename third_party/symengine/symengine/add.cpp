/**
 *  @file   add.cpp
 *  @author SymEngine Developers
 *  @date   2021-02-25
 *  @brief  Definitions for arithmatic
 *
 *  Created on: 2012-07-11
 *
 *  This file contains the basic binary operations defined for symbolic enties.
 *   In particular the @ref Add class for representing addition is
 *   @b defined here, along with the `add` and `substract` functions.
 */

#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/complex.h>

namespace SymEngine
{

/**
 *  @class Add
 *  @details Internally this is implemented in as a numeric coefficient `coef_`
 *    and a dictionary `dict_` of key-value pairs. Consider the following
 *    example:
 *
 *       Add(coef_, {{key1, value1}, {key2, value2}, ... })
 *
 *  This represents the following expression,
 *
 *       coef_ + key1*value1 + key2*value2 + ...
 *
 *  `coef_` and the values of the dictionary may be numeric coefficients like
 *    Integer, RealDouble, Complex while their corresponding  `key`s can be any
 *    symbolic expression except numeric coefficients and `Mul` objects with
 *    coefficient != 1.
 *
 *  For example, the following are valid representations
 *
 *       Add(1, {{x, 2}, {y, 5}})
 *       Add(0, {{x, 1}, {y, 4}, {z, 3}})
 *
 *  The following representations **are invalid**  (their valid equivalent is
 *   shown next to each of them)
 *
 *        Add(1, {{x, 1}, {2*y, 3})   -> Add(1, {{x, 1}, {y, 6}})
 *        Add(0, {{x, 2}})             -> Mul(2, {{x, 1}})
 *        Add(1, {{x, 2}, {4, 6}})    -> Add(25, {{x, 2}})
 *
 *  A visual aid (from the [SymEngine
 *   Wiki](https://github.com/symengine/symengine/wiki/OSS-World-Submission))
 *   for understanding this class in the broader context of the data structure
 *   for mathematical expressions is:
 *  @image html symEngineTree.png "Sample Expression Data Structure"
 *
 *  @see `Basic` for an explanation of how the intialization works in conjuction
 *    to the constructors of the `Basic` class and the guarantees in Release.
 **/

/**
 *  @details Constructs Add from a dictionary by copying the contents of
 *    the dictionary.
 */
Add::Add(const RCP<const Number> &coef, add_operands_map &&dict)
    : coef_{coef}, dict_{std::move(dict)}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(coef, dict_))
}

/**
 * @details This uses `Basic.hash()` to give a cached version of the hash.
 */
hash_t Add::__hash__() const
{
    hash_t seed = SYMENGINE_ADD, temp;
    hash_combine<Basic>(seed, *coef_);
    for (const auto &p : dict_) {
        temp = p.first->hash();
        hash_combine<Basic>(temp, *(p.second));
        seed ^= temp;
    }
    return seed;
}

/**
 * @details This older implementation compares the elements of the coefficients
 *  and expressions for two objects.
 */
bool Add::__eq__(const Basic &o) const
{
    if (is_a<Add>(o) and eq(*coef_, *(down_cast<const Add &>(o).coef_))
        and unified_eq(dict_, down_cast<const Add &>(o).dict_))
        return true;

    return false;
}

/**
 *  @details This function takes a `Basic` object, checks if it is an `Add`
 *   object, and subsequently compares exhaustively:
 *    - The number of elements.
 *    - The coefficients.
 *    - Each element of the dictionary.
 *
 *  @note Since the `map_basic_num` representation is not cached by `Add` after
 *   being computed, this is slow.
 * */
int Add::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Add>(o))
    const Add &s = down_cast<const Add &>(o);
    // # of elements
    if (dict_.size() != s.dict_.size())
        return (dict_.size() < s.dict_.size()) ? -1 : 1;

    // coef
    int cmp = coef_->__cmp__(*s.coef_);
    if (cmp != 0)
        return cmp;

    // Compare dictionaries (slow):
    //!< @todo cache `adict` and `bdict`
    map_basic_num adict(dict_.begin(), dict_.end());
    map_basic_num bdict(s.dict_.begin(), s.dict_.end());
    return unified_compare(adict, bdict);
}

/**
 *  @details Quick implementation which depends only on the size of @a d.
 *
 *  The speed benefits also arise from the fact that when using the
 *   `SymEngine::RCP` in production mode (which is not thread safe) then when a
 *   single `Mul` object is encountered, instead of copying its `dict_`, it is
 *   reused instead.
 *
 *  That is, when `WITH_SYMENGINE_THREAD_SAFE` is not defined and
 *   `WITH_SYMENGINE_RCP` is defined, we can "steal" its dictionary by explictly
 *   casting away the const'ness. Since the `refcount_` is 1, nothing else is
 *   using the `Mul`.
 */
RCP<const Basic> Add::from_dict(const RCP<const Number> &coef,
                                add_operands_map &&d)
{
    if (d.size() == 0) {
        return coef;
    } else if (d.size() == 1 and coef->is_zero()) {
        auto p = d.begin();
        if (is_a<Integer>(*(p->second))) {
            if (down_cast<const Integer &>(*(p->second)).is_zero()) {
                return p->second; // Symbol
            }
            if (down_cast<const Integer &>(*(p->second)).is_one()) {
                return p->first; // Integer
            }
            if (is_a<Mul>(*(p->first))) {
#if !defined(WITH_SYMENGINE_THREAD_SAFE) && defined(WITH_SYMENGINE_RCP)
                if (down_cast<const Mul &>(*(p->first)).use_count() == 1) {
                    // We can steal the dictionary:
                    // Cast away const'ness, so that we can move 'dict_', since
                    // 'p->first' will be destroyed when 'd' is at the end of
                    // this function, so we "steal" its dict_ to avoid an
                    // unnecessary copy. We know the refcount_ is one, so
                    // nobody else is using the Mul except us.
                    const map_basic_basic &d2
                        = down_cast<const Mul &>(*(p->first)).get_dict();
                    map_basic_basic &d3 = const_cast<map_basic_basic &>(d2);
                    return Mul::from_dict(p->second, std::move(d3));
                } else {
#else
                {
#endif
                    // We need to copy the dictionary:
                    map_basic_basic d2
                        = down_cast<const Mul &>(*(p->first)).get_dict();
                    return Mul::from_dict(
                        p->second,
                        std::move(d2)); // Can return a Pow object here
                }
            }
            map_basic_basic m;
            if (is_a<Pow>(*(p->first))) {
                insert(m, down_cast<const Pow &>(*(p->first)).get_base(),
                       down_cast<const Pow &>(*(p->first)).get_exp());
            } else {
                insert(m, p->first, one);
            }
            return make_rcp<const Mul>(p->second,
                                       std::move(m)); // Returns a Mul from here
        }
        map_basic_basic m;
        if (is_a_Number(*p->second)) {
            if (is_a<Mul>(*(p->first))) {
#if !defined(WITH_SYMENGINE_THREAD_SAFE) && defined(WITH_SYMENGINE_RCP)
                if (down_cast<const Mul &>(*(p->first)).use_count() == 1) {
                    // We can steal the dictionary:
                    // Cast away const'ness, so that we can move 'dict_', since
                    // 'p->first' will be destroyed when 'd' is at the end of
                    // this function, so we "steal" its dict_ to avoid an
                    // unnecessary copy. We know the refcount_ is one, so
                    // nobody else is using the Mul except us.
                    const map_basic_basic &d2
                        = down_cast<const Mul &>(*(p->first)).get_dict();
                    map_basic_basic &d3 = const_cast<map_basic_basic &>(d2);
                    return Mul::from_dict(p->second, std::move(d3));
                } else {
#else
                {
#endif
                    // We need to copy the dictionary:
                    map_basic_basic d2
                        = down_cast<const Mul &>(*(p->first)).get_dict();
                    return Mul::from_dict(p->second,
                                          std::move(d2)); // May return a Pow
                }
            }
            if (is_a<Pow>(*p->first)) {
                insert(m, down_cast<const Pow &>(*(p->first)).get_base(),
                       down_cast<const Pow &>(*(p->first)).get_exp());
            } else {
                insert(m, p->first, one);
            }
            return make_rcp<const Mul>(p->second, std::move(m));
        } else {
            insert(m, p->first, one);
            insert(m, p->second, one);
            return make_rcp<const Mul>(one, std::move(m));
        }
    } else {
        return make_rcp<const Add>(coef, std::move(d)); // returns an Add
    }
}

/**
 *  @details Adds `(coeff*t)` to the dict @a d inplace.
 *  @warning We assume that `t` has no numerical coefficients, and `coef` has
 *   only numerical coefficients.
 */
void Add::dict_add_term(add_operands_map &d, const RCP<const Number> &coef,
                        const RCP<const Basic> &t)
{
    auto it = d.find(t);
    if (it == d.end()) {
        // Not found, add it in if it is nonzero:
        if (not(coef->is_zero()))
            insert(d, t, coef);
    } else {
        iaddnum(outArg(it->second), coef);
        if (it->second->is_zero())
            d.erase(it);
    }
}

/**
 *  @details This implements the following logic:
 *   - If both `c` and `term` are numbers, then the term `(c* term)` is
 *      added to the existing `coeff`.
 *   - If `term` is not a number then the pair (`c, term`) is used to update
 *      the existing  dict `d` (as a pair `c, term`).
 *   - In case `term` is `Add` and `c=1`, expands the `Add` into the `coeff`
 *      and `d`.
 */
void Add::coef_dict_add_term(const Ptr<RCP<const Number>> &coef,
                             add_operands_map &d, const RCP<const Number> &c,
                             const RCP<const Basic> &term)
{
    if (is_a_Number(*term)) {
        iaddnum(coef, mulnum(c, rcp_static_cast<const Number>(term)));
    } else if (is_a<Add>(*term)) {
        if (c->is_one()) {
            for (const auto &q : (down_cast<const Add &>(*term)).dict_)
                Add::dict_add_term(d, q.second, q.first);
            iaddnum(coef, down_cast<const Add &>(*term).coef_);
        } else {
            Add::dict_add_term(d, c, term);
        }
    } else {
        RCP<const Number> coef2;
        RCP<const Basic> t;
        Add::as_coef_term(term, outArg(coef2), outArg(t));
        Add::dict_add_term(d, mulnum(c, coef2), t);
    }
}

/**
 * @details This implementation first converts @a a to a `Mul` and then performs
 *  addition.
 */
void Add::as_two_terms(const Ptr<RCP<const Basic>> &a,
                       const Ptr<RCP<const Basic>> &b) const
{
    auto p = dict_.begin();
    *a = mul(p->first, p->second);
    add_operands_map d = dict_;
    d.erase(p->first);
    *b = Add::from_dict(coef_, std::move(d));
}

/**
 *  @details This function converts the its representation as per the following
 *   logic:
 *    - If `self` is a `Mul` return the coefficient and the remaining term.
 *    - If `self` is not `Mul` or `Add` the coefficient is set one and the term
 *       is unchanged.
 *    - If `self` is a `Number` the term is set one and the coefficient is
 *       unchanged.
 */
void Add::as_coef_term(const RCP<const Basic> &self,
                       const Ptr<RCP<const Number>> &coef,
                       const Ptr<RCP<const Basic>> &term)
{
    if (is_a<Mul>(*self)) {
        if (neq(*(down_cast<const Mul &>(*self).get_coef()), *one)) {
            *coef = (down_cast<const Mul &>(*self)).get_coef();
            // We need to copy our 'dict_' here, as 'term' has to have its own.
            map_basic_basic d2 = (down_cast<const Mul &>(*self)).get_dict();
            *term = Mul::from_dict(one, std::move(d2));
        } else {
            *coef = one;
            *term = self;
        }
    } else if (is_a_Number(*self)) {
        *coef = rcp_static_cast<const Number>(self);
        *term = one;
    } else {
        SYMENGINE_ASSERT(not is_a<Add>(*self));
        *coef = one;
        *term = self;
    }
}

/**
 *  @details This function ensures that each term in *dict* is in canonical
 *   form. The implementation in the form of a exclusion list (defaults to
 *   true).
 *
 *  @note **Canonical form** requires the existance of both `coef` and
 *   `dict`, so `null` coefficients and purely numerical (empty dictionaries)
 *   are also not considered to be in canonical form. Also, the ordering is
 *   important, it must be `(coeff, dict)` and **not** `(dict, coeff)`.
 *
 *  Some **non-cannonical** forms are:
 *   - @f$0 + x@f$.
 *   - @f$0 + 2x@f$.
 *   - @f$ 2 \times 3 @f$.
 *   - @f$ x \times 0 @f$.
 *   - @f$ 1 \times x @f$ has the wrong order.
 *   - @f$ 3x \times 2 @f$ is actually just @f$6x@f$.
 */
bool Add::is_canonical(const RCP<const Number> &coef,
                       const add_operands_map &dict) const
{
    if (coef == null)
        return false;
    if (dict.size() == 0)
        return false;
    if (dict.size() == 1) {
        // e.g. 0 + x, 0 + 2x
        if (coef->is_zero())
            return false;
    }
    // Check that each term in 'dict' is in canonical form
    for (const auto &p : dict) {
        if (p.first == null)
            return false;
        if (p.second == null)
            return false;
        // e.g. 2*3
        if (is_a_Number(*p.first))
            return false;
        // e.g. 1*x (={1:x}), this should rather be just x (={x:1})
        if (is_a<Integer>(*p.first)
            and down_cast<const Integer &>(*p.first).is_one())
            return false;
        // e.g. x*0
        if (is_a_Number(*p.second)
            and down_cast<const Number &>(*p.second).is_zero())
            return false;

        // e.g. {3x: 2}, this should rather be just {x: 6}
        if (is_a<Mul>(*p.first)
            and not(down_cast<const Mul &>(*p.first).get_coef()->is_one()))
            return false;
    }
    return true;
}

/**
 * @details For an `Add` of the form:
 *
 *     Add(coef_, {{key1, value1}, {key2, value2}, ... })
 *  If coef_ is non-zero it returns:
 *
 *      {coef_, key1*value1, key2*value2, ... }
 *  otherwise it returns:
 *
 *      {key1*value1, key2*value2, ... }
 */
vec_basic Add::get_args() const
{
    vec_basic args;
    if (not coef_->is_zero()) {
        args.reserve(dict_.size() + 1);
        args.push_back(coef_);
    } else {
        args.reserve(dict_.size());
    }
    for (const auto &p : dict_) {
        if (eq(*p.second, *one)) {
            args.push_back(p.first);
        } else {
            args.push_back(Add::from_dict(zero, {{p.first, p.second}}));
        }
    }
    return args;
}

/**
 * @details This implementation is slower than the methods of `Add`, however it
 *  is conceptually simpler and also safer, as it is more general and can
 *  perform canonicalization.
 *
 * Note that:
 * > x + y will return an `Add`.
 * > x + x will return `Mul (2*x)`.
 */
RCP<const Basic> add(const RCP<const Basic> &a, const RCP<const Basic> &b)
{
    SymEngine::add_operands_map d;
    RCP<const Number> coef;
    RCP<const Basic> t;
    if (is_a<Add>(*a) and is_a<Add>(*b)) {
        coef = (down_cast<const Add &>(*a)).get_coef();
        d = (down_cast<const Add &>(*a)).get_dict();
        for (const auto &p : (down_cast<const Add &>(*b)).get_dict())
            Add::dict_add_term(d, p.second, p.first);
        iaddnum(outArg(coef), down_cast<const Add &>(*b).get_coef());
    } else if (is_a<Add>(*a)) {
        coef = (down_cast<const Add &>(*a)).get_coef();
        d = (down_cast<const Add &>(*a)).get_dict();
        if (is_a_Number(*b)) {
            if (not down_cast<const Number &>(*b).is_zero()) {
                iaddnum(outArg(coef), rcp_static_cast<const Number>(b));
            }
        } else {
            RCP<const Number> coef2;
            Add::as_coef_term(b, outArg(coef2), outArg(t));
            Add::dict_add_term(d, coef2, t);
        }
    } else if (is_a<Add>(*b)) {
        coef = (down_cast<const Add &>(*b)).get_coef();
        d = (down_cast<const Add &>(*b)).get_dict();
        if (is_a_Number(*a)) {
            if (not down_cast<const Number &>(*a).is_zero()) {
                iaddnum(outArg(coef), rcp_static_cast<const Number>(a));
            }
        } else {
            RCP<const Number> coef2;
            Add::as_coef_term(a, outArg(coef2), outArg(t));
            Add::dict_add_term(d, coef2, t);
        }
    } else {
        Add::as_coef_term(a, outArg(coef), outArg(t));
        Add::dict_add_term(d, coef, t);
        Add::as_coef_term(b, outArg(coef), outArg(t));
        Add::dict_add_term(d, coef, t);
        auto it = d.find(one);
        if (it == d.end()) {
            coef = zero;
        } else {
            coef = it->second;
            d.erase(it);
        }
        return Add::from_dict(coef, std::move(d));
    }
    return Add::from_dict(coef, std::move(d));
}

/**
 * @details This should be faster for `n` elements compared to performing `n-1`
 *  additions.
 */
RCP<const Basic> add(const vec_basic &a)
{
    SymEngine::add_operands_map d;
    RCP<const Number> coef = zero;
    for (const auto &i : a) {
        Add::coef_dict_add_term(outArg(coef), d, one, i);
    }
    return Add::from_dict(coef, std::move(d));
}

/**
 * @details This essentially implements the addition of `a` and `-b`. Note that
 *  since this calls `add()` it performs canonicalization if required.
 */
RCP<const Basic> sub(const RCP<const Basic> &a, const RCP<const Basic> &b)
{
    return add(a, mul(minus_one, b));
}

} // namespace SymEngine
