/**
 * \file mul.h
 * Header containing definition of Mul and related functions mul, div, neg
 *
 **/
#ifndef SYMENGINE_MUL_H
#define SYMENGINE_MUL_H

#include <symengine/basic.h>

namespace SymEngine
{

/*! \class Mul
   Mul class keeps a product of symbolic expressions. Internal representation
   of an Mul is a numeric coefficient `coef_` and a dictionary `dict_` of
   key-value pairs.

        Mul(coef_, {{key1, value1}, {key2, value2}, ... })

   This represents the following expression,

        coef_ * key1^value1 * key2^value2 * ...

   `coef_` is an objecct of type Number, i.e. a numeric coefficient like
   Integer,
   RealDouble, Complex.

   For example, the following are valid representations

        Mul(2, {{x, 2}, {y, 5}})
        Mul(3, {{x, 1}, {y, 4}, {z, 3}})

   Following are invalid representations. (valid equivalent is shown next to
   them)

    * When key is a numeric and value is an integers,

       Mul(2, {{3, 2}, {x, 2}})     -> Mul(18, {{x, 2}})
       Mul(2, {{I, 3}, {x, 2}})     -> Mul(-2*I, {{x, 2}})

    * When key is an integer and value is a Rational not in the range (0, 1)

       Mul(2, {{3, 3/2}, {x, 2}})   -> Mul(6, {{3, 1/2}, {x, 2}})
       Mul(2, {{3, -1/2}, {x, 2}})  -> Mul(2/3, {{3, 1/2}, {x, 2}})

    * When the value is zero

       Mul(3, {{x, 0}, {y, 2}})     -> Mul(3, {{y, 2}})

    * When key and value are numeric and one of them is inexact

       Mul(2, {{3, 0.5}, {x, 2}})   -> Mul(3.464..., {x, 2}})

    * When `coef_` is one and the dictionary is of size 1

       Mul(1, {{x, 2}})             -> Pow(x, 2)

    * When `coef_` is zero

       Mul(0, {{x, 2}})             -> Integer(0)
       Mul(0.0, {{x, 2}})           -> RealDouble(0.0)

    * When key is 1

       Mul(2, {{1, x}, {x, 2}})     -> Mul(2, {{x, 2}})

    * When value is zero

       Mul(2, {{1, x}, {x, 2}})     -> Mul(2, {{x, 2}})

*/
class Mul : public Basic
{
private:
    RCP<const Number> coef_; //! The coefficient (e.g. `2` in `2*x*y`)
    map_basic_basic
        dict_; //! the dictionary of the rest (e.g. `x*y` in `2*x*y`)

public:
    IMPLEMENT_TYPEID(SYMENGINE_MUL)
    //! Constructs Mul from a dictionary by copying the contents of the
    //! dictionary:
    Mul(const RCP<const Number> &coef, map_basic_basic &&dict);
    //! \return size of the hash
    hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;

    // Performs canonicalization first:
    //! Create a Mul from a dict
    static RCP<const Basic> from_dict(const RCP<const Number> &coef,
                                      map_basic_basic &&d);
    //! Add terms to dict
    static void dict_add_term(map_basic_basic &d, const RCP<const Basic> &exp,
                              const RCP<const Basic> &t);
    static void dict_add_term_new(const Ptr<RCP<const Number>> &coef,
                                  map_basic_basic &d,
                                  const RCP<const Basic> &exp,
                                  const RCP<const Basic> &t);
    //! Convert to a base and exponent form
    static void as_base_exp(const RCP<const Basic> &self,
                            const Ptr<RCP<const Basic>> &exp,
                            const Ptr<RCP<const Basic>> &base);
    //! Rewrite as 2 terms
    /*!
        Example: if this=3*x**2*y**2*z**2`, then `a=x**2` and `b=3*y**2*z**2`
    * */
    void as_two_terms(const Ptr<RCP<const Basic>> &a,
                      const Ptr<RCP<const Basic>> &b) const;
    //! Power all terms with the exponent `exp`
    void power_num(const Ptr<RCP<const Number>> &coef, map_basic_basic &d,
                   const RCP<const Number> &exp) const;

    //! \return true if both `coef` and `dict` are in canonical form
    bool is_canonical(const RCP<const Number> &coef,
                      const map_basic_basic &dict) const;

    vec_basic get_args() const override;

    inline const RCP<const Number> &get_coef() const
    {
        return coef_;
    }
    inline const map_basic_basic &get_dict() const
    {
        return dict_;
    }
};
//! Multiplication
RCP<const Basic> mul(const RCP<const Basic> &a, const RCP<const Basic> &b);
RCP<const Basic> mul(const vec_basic &a);
//! Division
RCP<const Basic> div(const RCP<const Basic> &a, const RCP<const Basic> &b);
//! Negation
RCP<const Basic> neg(const RCP<const Basic> &a);

} // namespace SymEngine

#endif
