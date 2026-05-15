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
    hash_t __hash__() const override;

    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;

    //! \return `true` if `0`
    inline bool is_zero() const override
    {
        return false;
    }
    //! \return `true` if `1`
    inline bool is_one() const override
    {
        return false;
    }
    //! \return `true` if `-1`
    inline bool is_minus_one() const override
    {
        return false;
    }

    inline bool is_positive() const override
    {
        return false;
    }

    inline bool is_negative() const override
    {
        return false;
    }

    inline bool is_complex() const override
    {
        return false;
    }
    //! \return the conjugate if the class is complex
    RCP<const Basic> conjugate() const override;
    inline bool is_exact() const override
    {
        return false;
    }
    Evaluate &get_eval() const override;

    RCP<const Number> add(const Number &other) const override;
    RCP<const Number> mul(const Number &other) const override;
    RCP<const Number> div(const Number &other) const override;
    RCP<const Number> pow(const Number &other) const override;
    RCP<const Number> rpow(const Number &other) const override;
};

} // namespace SymEngine
#endif
