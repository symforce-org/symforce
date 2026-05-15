/**
 *  \file number.h
 *  Basic Number class.
 *
 **/

#ifndef SYMENGINE_NUMBER_H
#define SYMENGINE_NUMBER_H

#include <symengine/basic.h>
#include <symengine/assumptions.h>

namespace SymEngine
{

class Evaluate;

class Number : public Basic
{
public:
    //! \return true if `0`
    virtual bool is_zero() const = 0;
    //! \return true if `1`
    virtual bool is_one() const = 0;
    //! \return true if `-1`
    virtual bool is_minus_one() const = 0;
    //! \return true if negative
    virtual bool is_negative() const = 0;
    //! \return true if positive
    virtual bool is_positive() const = 0;
    //! \return true if a complex number
    virtual bool is_complex() const = 0;
    //! \return the conjugate if the class is complex
    virtual RCP<const Basic> conjugate() const;
    //! return true if the number is an exact representation
    //  false if the number is an approximation
    virtual bool is_exact() const
    {
        return true;
    };
    //! \return true if the number is equal to 0 and not an approximation
    inline bool is_exact_zero() const
    {
        return is_exact() and is_zero();
    };
    //! Get `Evaluate` singleton to evaluate numerically
    virtual Evaluate &get_eval() const
    {
        throw NotImplementedError("Not Implemented.");
    };

    //! Addition
    virtual RCP<const Number> add(const Number &other) const = 0;
    //! Subtraction
    virtual RCP<const Number> sub(const Number &other) const;
    virtual RCP<const Number> rsub(const Number &other) const;
    //! Multiplication
    virtual RCP<const Number> mul(const Number &other) const = 0;
    //! Division
    virtual RCP<const Number> div(const Number &other) const;
    virtual RCP<const Number> rdiv(const Number &other) const;
    //! Power
    virtual RCP<const Number> pow(const Number &other) const = 0;
    virtual RCP<const Number> rpow(const Number &other) const = 0;

    vec_basic get_args() const override
    {
        return {};
    }

    virtual bool is_perfect_power(bool is_expected = false) const
    {
        throw NotImplementedError("Not Implemented.");
    };
    virtual bool nth_root(const Ptr<RCP<const Number>> &, unsigned long n) const
    {
        throw NotImplementedError("Not Implemented.");
    };
};
//! Add `self` and `other`
inline RCP<const Number> addnum(const RCP<const Number> &self,
                                const RCP<const Number> &other)
{
    return self->add(*other);
}
//! Subtract `self` and `other`
inline RCP<const Number> subnum(const RCP<const Number> &self,
                                const RCP<const Number> &other)
{
    return self->sub(*other);
}
//! Multiply `self` and `other`
inline RCP<const Number> mulnum(const RCP<const Number> &self,
                                const RCP<const Number> &other)
{
    return self->mul(*other);
}
//! Divide `self` and `other`
inline RCP<const Number> divnum(const RCP<const Number> &self,
                                const RCP<const Number> &other)
{
    return self->div(*other);
}
//! Raise `self` to power `other`
inline RCP<const Number> pownum(const RCP<const Number> &self,
                                const RCP<const Number> &other)
{
    return self->pow(*other);
}

inline void iaddnum(const Ptr<RCP<const Number>> &self,
                    const RCP<const Number> &other)
{
    *self = addnum(*self, other);
}

inline void imulnum(const Ptr<RCP<const Number>> &self,
                    const RCP<const Number> &other)
{
    *self = mulnum(*self, other);
}

inline void idivnum(const Ptr<RCP<const Number>> &self,
                    const RCP<const Number> &other)
{
    *self = divnum(*self, other);
}

//! \return true if 'b' is a Number or any of its subclasses
inline bool is_a_Number(const Basic &b)
{
    // `NumberWraper` is the last subclass of Number in TypeID
    // An enum should be before `SYMENIGNE_NUMBER_WRAPPER` iff it is a
    // subclass of Number
    return b.get_type_code() <= SYMENGINE_NUMBER_WRAPPER;
}

//! \return true if 'b' is a Number and is zero
inline bool is_number_and_zero(const Basic &b)
{
    return is_a_Number(b) and down_cast<const Number &>(b).is_zero();
}

/**
 * @brief Check if a number is zero
 * @param b Basic
 * @param assumptions Assumptions
 * @returns tribool
 *
 * Check if b is zero. If b is not numeric an exception will be thrown.
 */
tribool is_zero(const Basic &b, const Assumptions *assumptions = nullptr);
/**
 * @brief Check if a number is non-zero
 * @param b Basic
 * @param assumptions Assumptions
 * @returns tribool
 *
 * Check if b is non-zero. If b is not numeric an exception will be thrown.
 */
tribool is_nonzero(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_positive(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_nonpositive(const Basic &b,
                       const Assumptions *assumptions = nullptr);
tribool is_negative(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_nonnegative(const Basic &b,
                       const Assumptions *assumptions = nullptr);
tribool is_integer(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_real(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_complex(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_rational(const Basic &b);
tribool is_irrational(const Basic &b);
tribool is_finite(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_infinite(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_even(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_odd(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_algebraic(const Basic &b, const Assumptions *assumptions = nullptr);
tribool is_transcendental(const Basic &b,
                          const Assumptions *assumptions = nullptr);

class NumberWrapper : public Number
{
public:
    NumberWrapper(){SYMENGINE_ASSIGN_TYPEID()}

    IMPLEMENT_TYPEID(SYMENGINE_NUMBER_WRAPPER)

        virtual std::string __str__() const
    {
        throw NotImplementedError("Not Implemented.");
    };
    virtual RCP<const Number> eval(long bits) const
    {
        throw NotImplementedError("Not Implemented.");
    };
};

//! A class that will evaluate functions numerically.
class Evaluate
{
public:
    virtual RCP<const Basic> sin(const Basic &) const = 0;
    virtual RCP<const Basic> cos(const Basic &) const = 0;
    virtual RCP<const Basic> tan(const Basic &) const = 0;
    virtual RCP<const Basic> cot(const Basic &) const = 0;
    virtual RCP<const Basic> sec(const Basic &) const = 0;
    virtual RCP<const Basic> csc(const Basic &) const = 0;
    virtual RCP<const Basic> asin(const Basic &) const = 0;
    virtual RCP<const Basic> acos(const Basic &) const = 0;
    virtual RCP<const Basic> atan(const Basic &) const = 0;
    virtual RCP<const Basic> acot(const Basic &) const = 0;
    virtual RCP<const Basic> asec(const Basic &) const = 0;
    virtual RCP<const Basic> acsc(const Basic &) const = 0;
    virtual RCP<const Basic> sinh(const Basic &) const = 0;
    virtual RCP<const Basic> csch(const Basic &) const = 0;
    virtual RCP<const Basic> cosh(const Basic &) const = 0;
    virtual RCP<const Basic> sech(const Basic &) const = 0;
    virtual RCP<const Basic> tanh(const Basic &) const = 0;
    virtual RCP<const Basic> coth(const Basic &) const = 0;
    virtual RCP<const Basic> asinh(const Basic &) const = 0;
    virtual RCP<const Basic> acsch(const Basic &) const = 0;
    virtual RCP<const Basic> acosh(const Basic &) const = 0;
    virtual RCP<const Basic> atanh(const Basic &) const = 0;
    virtual RCP<const Basic> acoth(const Basic &) const = 0;
    virtual RCP<const Basic> asech(const Basic &) const = 0;
    virtual RCP<const Basic> log(const Basic &) const = 0;
    virtual RCP<const Basic> gamma(const Basic &) const = 0;
    virtual RCP<const Basic> abs(const Basic &) const = 0;
    virtual RCP<const Basic> exp(const Basic &) const = 0;
    virtual RCP<const Basic> floor(const Basic &) const = 0;
    virtual RCP<const Basic> ceiling(const Basic &) const = 0;
    virtual RCP<const Basic> truncate(const Basic &) const = 0;
    virtual RCP<const Basic> erf(const Basic &) const = 0;
    virtual RCP<const Basic> erfc(const Basic &) const = 0;
};

} // namespace SymEngine

#endif
