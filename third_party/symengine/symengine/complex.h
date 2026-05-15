/**
 *  \file complex.h
 *  Class for Complex built on top of Number class
 *
 **/
#ifndef SYMENGINE_COMPLEX_H
#define SYMENGINE_COMPLEX_H

#include <symengine/rational.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{
//! ComplexBase Class for deriving all complex classes
class ComplexBase : public Number
{
public:
    virtual RCP<const Number> real_part() const = 0;
    virtual RCP<const Number> imaginary_part() const = 0;
    bool is_re_zero() const;
};

//! \return true if 'b' is any of the subclasses of ComplexBase
inline bool is_a_Complex(const Basic &b)
{
    return (b.get_type_code() == SYMENGINE_COMPLEX
            || b.get_type_code() == SYMENGINE_COMPLEX_MPC
            || b.get_type_code() == SYMENGINE_COMPLEX_DOUBLE);
}

//! Complex Class
class Complex : public ComplexBase
{
public:
    //! `real_` : Real part of the complex Number
    //! `imaginary_` : Imaginary part of the complex Number
    // Complex Number is of the form `real + i(imaginary)`
    rational_class real_;
    rational_class imaginary_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_COMPLEX)
    //! Constructor of Complex class
    Complex(rational_class real, rational_class imaginary);
    /*! Creates an instance of Complex if imaginary part is non-zero
     * \param `re` must already be in rational_class canonical form
     * \param `im` must already be in rational_class canonical form
     * \return Complex or Rational depending on imaginary part.
     * */
    static RCP<const Number> from_mpq(const rational_class re,
                                      const rational_class im);
    //! \return true if canonical
    bool is_canonical(const rational_class &real,
                      const rational_class &imaginary) const;
    //! \return size of the hash
    hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    //! Get the real part of the complex number
    RCP<const Number> real_part() const override;
    //! Get the imaginary part of the complex number
    RCP<const Number> imaginary_part() const override;
    //! Get the conjugate of the complex number
    RCP<const Basic> conjugate() const override;
    //! \returns `false`
    // False is returned because complex cannot be compared with zero
    inline bool is_positive() const override
    {
        return false;
    }
    //! \returns `false`
    // False is returned because complex cannot be compared with zero
    inline bool is_negative() const override
    {
        return false;
    }
    //! \returns `true`
    inline bool is_complex() const override
    {
        return true;
    }

    /*! Constructs Complex from re, im. If im is 0
     * it will return a Rational instead.
     * */
    static RCP<const Number> from_two_rats(const Rational &re,
                                           const Rational &im);

    /*! Constructs Complex from re, im. If im is 0
     * it will return a Rational instead.
     * */
    static RCP<const Number> from_two_nums(const Number &re, const Number &im);

    //! \return `false` since `imaginary_` cannot be zero
    bool is_zero() const override
    {
        return false;
    }
    //! \return `false` since `imaginary_` cannot be zero
    bool is_one() const override
    {
        return false;
    }
    //! \return `false` since `imaginary_` cannot be zero
    bool is_minus_one() const override
    {
        return false;
    }

    /*! Add Complex
     * \param other of type Complex
     * */
    inline RCP<const Number> addcomp(const Complex &other) const
    {
        return from_mpq(this->real_ + other.real_,
                        this->imaginary_ + other.imaginary_);
    }
    /*! Add Complex
     * \param other of type Rational
     * */
    inline RCP<const Number> addcomp(const Rational &other) const
    {
        return from_mpq(this->real_ + other.as_rational_class(),
                        this->imaginary_);
    }
    /*! Add Complex
     * \param other of type Integer
     * */
    inline RCP<const Number> addcomp(const Integer &other) const
    {
        return from_mpq(this->real_ + other.as_integer_class(),
                        this->imaginary_);
    }

    /*! Subtract Complex
     * \param other of type Complex
     * */
    inline RCP<const Number> subcomp(const Complex &other) const
    {
        return from_mpq(this->real_ - other.real_,
                        this->imaginary_ - other.imaginary_);
    }
    /*! Subtract Complex
     * \param other of type Rational
     * */
    inline RCP<const Number> subcomp(const Rational &other) const
    {
        return from_mpq(this->real_ - other.as_rational_class(),
                        this->imaginary_);
    }
    /*! Subtract Complex
     * \param other of type Integer
     * */
    inline RCP<const Number> subcomp(const Integer &other) const
    {
        return from_mpq(this->real_ - other.as_integer_class(),
                        this->imaginary_);
    }
    /*! Subtract Complex from other
     * \param other of type Complex
     * */
    inline RCP<const Number> rsubcomp(const Complex &other) const
    {
        return from_mpq(other.real_ - this->real_,
                        other.imaginary_ - this->imaginary_);
    }
    /*! Subtract Complex from other
     * \param other of type Rational
     * */
    inline RCP<const Number> rsubcomp(const Rational &other) const
    {
        return from_mpq(other.as_rational_class() - this->real_,
                        -this->imaginary_);
    }
    /*! Subtract Complex from other
     * \param other of type Integer
     * */
    inline RCP<const Number> rsubcomp(const Integer &other) const
    {
        return from_mpq(other.as_integer_class() - this->real_,
                        -this->imaginary_);
    }

    /*! Multiply Complex
     * \param other of type Complex
     * */
    inline RCP<const Number> mulcomp(const Complex &other) const
    {
        return from_mpq(
            this->real_ * other.real_ - this->imaginary_ * other.imaginary_,
            this->real_ * other.imaginary_ + this->imaginary_ * other.real_);
    }
    /*! Multiply Complex
     * \param other of type Rational
     * */
    inline RCP<const Number> mulcomp(const Rational &other) const
    {
        return from_mpq(this->real_ * other.as_rational_class(),
                        this->imaginary_ * other.as_rational_class());
    }
    /*! Multiply Complex
     * \param other of type Integer
     * */
    inline RCP<const Number> mulcomp(const Integer &other) const
    {
        return from_mpq(this->real_ * other.as_integer_class(),
                        this->imaginary_ * other.as_integer_class());
    }

    /*! Divide Complex
     * \param other of type Complex
     * */
    inline RCP<const Number> divcomp(const Complex &other) const
    {
        rational_class modulus_sq_other
            = other.real_ * other.real_ + other.imaginary_ * other.imaginary_;

        if (get_num(modulus_sq_other) == 0) {
            rational_class modulus_sq_this
                = this->real_ * this->real_
                  + this->imaginary_ * this->imaginary_;
            if (get_num(modulus_sq_this) == 0) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq((this->real_ * other.real_
                             + this->imaginary_ * other.imaginary_)
                                / modulus_sq_other,
                            (-this->real_ * other.imaginary_
                             + this->imaginary_ * other.real_)
                                / modulus_sq_other);
        }
    }
    /*! Divide Complex
     * \param other of type Rational
     * */
    inline RCP<const Number> divcomp(const Rational &other) const
    {
        if (other.is_zero()) {
            rational_class modulus_sq_this
                = this->real_ * this->real_
                  + this->imaginary_ * this->imaginary_;

            if (get_num(modulus_sq_this) == 0) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq(this->real_ / other.as_rational_class(),
                            this->imaginary_ / other.as_rational_class());
        }
    }
    /*! Divide Complex
     * \param other of type Integer
     * */
    inline RCP<const Number> divcomp(const Integer &other) const
    {
        if (other.is_zero()) {
            rational_class modulus_sq_this
                = this->real_ * this->real_
                  + this->imaginary_ * this->imaginary_;

            if (get_num(modulus_sq_this) == 0) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq(this->real_ / other.as_integer_class(),
                            this->imaginary_ / other.as_integer_class());
        }
    }
    /*! Divide other by the Complex
     * \param other of type Integer
     * */
    inline RCP<const Number> rdivcomp(const Integer &other) const
    {
        rational_class modulus_sq_this
            = this->real_ * this->real_ + this->imaginary_ * this->imaginary_;

        if (get_num(modulus_sq_this) == 0) {
            if (other.is_zero()) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq((this->real_ * other.as_integer_class())
                                / modulus_sq_this,
                            (this->imaginary_ * (-other.as_integer_class()))
                                / modulus_sq_this);
        }
    }
    /*! Pow Complex
     * \param other of type Integer
     * */
    RCP<const Number> powcomp(const Integer &other) const;

    //! Converts the param `other` appropriately and then calls `addcomp`
    RCP<const Number> add(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return addcomp(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return addcomp(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return addcomp(down_cast<const Complex &>(other));
        } else {
            return other.add(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `subcomp`
    RCP<const Number> sub(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return subcomp(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return subcomp(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return subcomp(down_cast<const Complex &>(other));
        } else {
            return other.rsub(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `rsubcomp`
    RCP<const Number> rsub(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rsubcomp(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rsubcomp(down_cast<const Integer &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    };
    //! Converts the param `other` appropriately and then calls `mulcomp`
    RCP<const Number> mul(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return mulcomp(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return mulcomp(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return mulcomp(down_cast<const Complex &>(other));
        } else {
            return other.mul(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `divcomp`
    RCP<const Number> div(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return divcomp(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return divcomp(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return divcomp(down_cast<const Complex &>(other));
        } else {
            return other.rdiv(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `rdivcomp`
    RCP<const Number> rdiv(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return rdivcomp(down_cast<const Integer &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    };
    //! Converts the param `other` appropriately and then calls `powcomp`
    RCP<const Number> pow(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return powcomp(down_cast<const Integer &>(other));
        } else {
            return other.rpow(*this);
        }
    };

    RCP<const Number> rpow(const Number &other) const override
    {
        throw NotImplementedError("Not Implemented");
    };
};

} // namespace SymEngine

#endif
