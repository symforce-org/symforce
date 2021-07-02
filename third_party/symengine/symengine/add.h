/**
 *  @file   add.h
 *  @author SymEngine Developers
 *  @date   2021-02-25
 *  @brief  Classes and functions relating to the binary operation of addition
 *
 *  Created on: 2012-07-11
 *
 *  This file contains the basic binary operations defined for symbolic enties.
 *   In particular the @ref Add class for representing addition is
 *   @b declared here, along with the `add` and `substract` functions.
 */

#ifndef SYMENGINE_ADD_H
#define SYMENGINE_ADD_H

#include <symengine/basic.h>

namespace SymEngine
{

/**
 *   @class Add
 *   @brief The base class for representing addition in symbolic expressions.
 **/
class Add : public Basic
{
private:
    RCP<const Number> coef_; //!< The numeric coefficient of the expression
                             //!< (e.g. `2` in `2+x+y`).
    umap_basic_num dict_;    //!< The expression without its coefficient as a
                             //!< dictionary (e.g. `x+y` in `2+x+y`).

public:
    IMPLEMENT_TYPEID(SYMENGINE_ADD)

    /**
     *  @brief Default constructor.
     *  @pre Input must be in cannonical form.
     *  @param coef numeric coefficient.
     *  @param dict dictionary of the expression without the coefficient.
     */
    Add(const RCP<const Number> &coef, umap_basic_num &&dict);

    /**
     *  @brief Generates the hash representation.
     *  @see Basic for an implementation to get the cached version.
     *  @return 64-bit integer value for the hash.
     */
    virtual hash_t __hash__() const;

    /**
     *  @brief Test equality.
     *  @deprecated Use eq(const Basic &a, const Basic &b) instead.
     *  @param o a constant reference to object to test against.
     *  @return True if `this` is equal to `o`.
     */
    virtual bool __eq__(const Basic &o) const;

    /**
     *  @brief Compares `Add` objects.
     *  @param o object to test against.
     *  @see `unified_compare()` for the actual implementation.
     *  @return 1 if `this` is equal to `o` otherwise (-1).
     */
    virtual int compare(const Basic &o) const;

    /**
     *  @brief Create an appropriate instance from dictionary quickly.
     *  @pre The dictionary must be in canonical form.
     *  @see `Mul` for how `Pow` gets returned.
     *  @see `Basic` for the guarantees and expectations.
     *  @param coef the numeric coefficient.
     *  @param d the dictionary of the expression without the coefficient.
     *  @return `coef` if the dictionary is empty (size 0).
     *  @return `Mul` if the dictionary has one element which is a `Mul`.
     *  @return `Integer` if the dictionary has one element which is a
     *   `Integer`.
     *  @return `Symbol` if the dictionary has one element which is a `Symbol`.
     *  @return `Pow` if the dictionary has one element which is a `Pow`.
     *  @return `Add` if the size of the dictionary is greater than 1.
     */
    static RCP<const Basic> from_dict(const RCP<const Number> &coef,
                                      umap_basic_num &&d);
    /**
     *  @brief Adds a new term to the expression.
     *  @pre The coefficient of the new term is non-zero.
     *  @param d dictionary to be updated.
     *  @param coef new coefficient.
     *  @param t new term.
     *  @return Void.
     */
    static void dict_add_term(umap_basic_num &d, const RCP<const Number> &coef,
                              const RCP<const Basic> &t);
    /**
     *  @brief Updates the numerical coefficient and the dictionary.
     *  @param coef the numerical coefficient.
     *  @param d  the dictionary containing the expression.
     *  @param c  the numerical coefficient to be added.
     *  @param term the new term.
     *  @return Void.
     */
    static void coef_dict_add_term(const Ptr<RCP<const Number>> &coef,
                                   umap_basic_num &d,
                                   const RCP<const Number> &c,
                                   const RCP<const Basic> &term);

    /**
     *  @brief Converts the Add into a sum of two Basic objects.
     *  @param a first basic object.
     *  @param b second basic object.
     *  @return Void.
     */
    void as_two_terms(const Ptr<RCP<const Basic>> &a,
                      const Ptr<RCP<const Basic>> &b) const;

    /**
     *  @brief Converts a Basic `self` into the form of `coefficient * term`.
     *  @param coef numerical coefficient.
     *  @param term the term.
     *  @return Void.
     */
    static void as_coef_term(const RCP<const Basic> &self,
                             const Ptr<RCP<const Number>> &coef,
                             const Ptr<RCP<const Basic>> &term);
    /**
     *  @brief Checks if a given dictionary and coeffient is in cannonical form.
     *  @param coef numerical coefficient.
     *  @param dict dictionary of remaining expression terms.
     *  @return `true` if canonical.
     */
    bool is_canonical(const RCP<const Number> &coef,
                      const umap_basic_num &dict) const;

    /**
     * @brief Returns the arguments of the Add.
     * @return list of arguments.
     */
    virtual vec_basic get_args() const;

    //!< @return const reference to the coefficient of the `Add`.
    inline const RCP<const Number> &get_coef() const
    {
        return coef_;
    }

    //!< @return const reference to the dictionary of the `Add`
    inline const umap_basic_num &get_dict() const
    {
        return dict_;
    }
};

/**
 *  @brief Adds two objects (safely).
 *  @param a is a `Basic` object.
 *  @param b is a `Basic` object.
 *  @returns `a+b` in its most aproriate form.
 *
 *  @relatesalso Add
 */
RCP<const Basic> add(const RCP<const Basic> &a, const RCP<const Basic> &b);

/**
 *  @brief Sums the elements of a vector.
 *  @param a is a vector.
 *  @returns sum of elements of the input vector `a`.
 *
 *  @relatesalso Add.
 */
RCP<const Basic> add(const vec_basic &a);

/**
 *  @brief Substracts `b` from `a`.
 *  @param a is the minuend.
 *  @param b is the subtrahend.
 *  @returns the difference `a-b`.
 *
 *  @relatesalso Add.
 */
RCP<const Basic> sub(const RCP<const Basic> &a, const RCP<const Basic> &b);

} // namespace SymEngine

#endif
