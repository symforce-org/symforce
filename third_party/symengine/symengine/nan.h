/**
 *  \file nan.h
 *
 **/

#ifndef SYMENGINE_NAN_H
#define SYMENGINE_NAN_H

#include <symengine/basic.h>
#include <symengine/number.h>

namespace SymEngine
{

/**
 * This serves as a place holder for numeric values that are indeterminate.
 *  Most operations on NaN, produce another NaN.
 **/
class NaN : public Number
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_NOT_A_NUMBER)
    //! Constructs NaN
    NaN();

    //! \return size of the hash
    hash_t __hash__() const;

    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const;
    int compare(const Basic &o) const;

    //! \return `true` if `0`
    inline bool is_zero() const
    {
        return false;
    }
    //! \return `true` if `1`
    inline bool is_one() const
    {
        return false;
    }
    //! \return `true` if `-1`
    inline bool is_minus_one() const
    {
        return false;
    }

    inline bool is_positive() const
    {
        return false;
    }

    inline bool is_negative() const
    {
        return false;
    }

    inline bool is_complex() const
    {
        return false;
    }
    //! \return the conjugate if the class is complex
    virtual RCP<const Basic> conjugate() const;
    inline bool is_exact() const
    {
        return false;
    }
    virtual Evaluate &get_eval() const;

    RCP<const Number> add(const Number &other) const;
    RCP<const Number> mul(const Number &other) const;
    RCP<const Number> div(const Number &other) const;
    RCP<const Number> pow(const Number &other) const;
    RCP<const Number> rpow(const Number &other) const;
};

} // SymEngine
#endif
