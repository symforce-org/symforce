/**
 *  \file RealDouble.h
 *  Class for RealDouble built on top of Number class
 *
 **/
#ifndef SYMENGINE_REAL_DOUBLE_H
#define SYMENGINE_REAL_DOUBLE_H

#include <symengine/complex.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{

RCP<const Number> number(std::complex<double> x);
RCP<const Number> number(double x);

//! RealDouble Class to hold double values
class RealDouble : public Number
{
public:
    double i;

public:
    IMPLEMENT_TYPEID(SYMENGINE_REAL_DOUBLE)
    //! Constructor of RealDouble class
    explicit RealDouble(double i);
    //! \return size of the hash
    hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    //! \return `true` if positive
    inline bool is_positive() const override
    {
        return i > 0;
    }
    //! \return `true` if negative
    inline bool is_negative() const override
    {
        return i < 0;
    }
    //! \return self as a double
    inline double as_double() const
    {
        return i;
    }
    //! \return `true` if this number is an exact number
    inline bool is_exact() const override
    {
        return false;
    }
    //! Get `Evaluate` singleton to evaluate numerically
    Evaluate &get_eval() const override;

    //! \return `true` when equals to 0
    bool is_zero() const override
    {
        return this->i == 0.0;
    }
    //! \return `false`
    // A double is not exactly equal to `1`
    bool is_one() const override
    {
        return false;
    }
    //! \return `false`
    // A double is not exactly equal to `-1`
    bool is_minus_one() const override
    {
        return false;
    }
    //! \returns `false`
    // False is returned because a RealDouble cannot have a imaginary part
    bool is_complex() const override
    {
        return false;
    }

    /*! Add RealDoubles
     * \param other of type Integer
     * */
    RCP<const Number> addreal(const Integer &other) const
    {
        return make_rcp<const RealDouble>(i
                                          + mp_get_d(other.as_integer_class()));
    }

    /*! Add RealDoubles
     * \param other of type Rational
     * */
    RCP<const Number> addreal(const Rational &other) const
    {
        return make_rcp<const RealDouble>(
            i + mp_get_d(other.as_rational_class()));
    }

    /*! Add RealDoubles
     * \param other of type Complex
     * */
    RCP<const Number> addreal(const Complex &other) const
    {
        return number(i
                      + std::complex<double>(mp_get_d(other.real_),
                                             mp_get_d(other.imaginary_)));
    }

    /*! Add RealDoubles
     * \param other of type RealDouble
     * */
    RCP<const Number> addreal(const RealDouble &other) const
    {
        return make_rcp<const RealDouble>(i + other.i);
    }

    //! Converts the param `other` appropriately and then calls `addreal`
    RCP<const Number> add(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return addreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return addreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return addreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return addreal(down_cast<const RealDouble &>(other));
        } else {
            return other.add(*this);
        }
    }

    /*! Subtract RealDoubles
     * \param other of type Integer
     * */
    RCP<const Number> subreal(const Integer &other) const
    {
        return make_rcp<const RealDouble>(i
                                          - mp_get_d(other.as_integer_class()));
    }

    /*! Subtract RealDoubles
     * \param other of type Rational
     * */
    RCP<const Number> subreal(const Rational &other) const
    {
        return make_rcp<const RealDouble>(
            i - mp_get_d(other.as_rational_class()));
    }

    /*! Subtract RealDoubles
     * \param other of type Complex
     * */
    RCP<const Number> subreal(const Complex &other) const
    {
        return number(i
                      - std::complex<double>(mp_get_d(other.real_),
                                             mp_get_d(other.imaginary_)));
    }

    /*! Subtract RealDoubles
     * \param other of type RealDouble
     * */
    RCP<const Number> subreal(const RealDouble &other) const
    {
        return make_rcp<const RealDouble>(i - other.i);
    }

    //! Converts the param `other` appropriately and then calls `subreal`
    RCP<const Number> sub(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return subreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return subreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return subreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return subreal(down_cast<const RealDouble &>(other));
        } else {
            return other.rsub(*this);
        }
    }

    /*! Subtract RealDoubles
     * \param other of type Integer
     * */
    RCP<const Number> rsubreal(const Integer &other) const
    {
        return make_rcp<const RealDouble>(mp_get_d(other.as_integer_class())
                                          - i);
    }

    /*! Subtract RealDoubles
     * \param other of type Rational
     * */
    RCP<const Number> rsubreal(const Rational &other) const
    {
        return make_rcp<const RealDouble>(mp_get_d(other.as_rational_class())
                                          - i);
    }

    /*! Subtract RealDoubles
     * \param other of type Complex
     * */
    RCP<const Number> rsubreal(const Complex &other) const
    {
        return number(-i
                      + std::complex<double>(mp_get_d(other.real_),
                                             mp_get_d(other.imaginary_)));
    }

    //! Converts the param `other` appropriately and then calls `subreal`
    RCP<const Number> rsub(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rsubreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rsubreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return rsubreal(down_cast<const Complex &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }

    /*! Multiply RealDoubles
     * \param other of type Integer
     * */
    RCP<const Number> mulreal(const Integer &other) const
    {
        if (other.is_zero()) {
            return zero;
        }
        return make_rcp<const RealDouble>(i
                                          * mp_get_d(other.as_integer_class()));
    }

    /*! Multiply RealDoubles
     * \param other of type Rational
     * */
    RCP<const Number> mulreal(const Rational &other) const
    {
        return make_rcp<const RealDouble>(
            i * mp_get_d(other.as_rational_class()));
    }

    /*! Multiply RealDoubles
     * \param other of type Complex
     * */
    RCP<const Number> mulreal(const Complex &other) const
    {
        return number(i
                      * std::complex<double>(mp_get_d(other.real_),
                                             mp_get_d(other.imaginary_)));
    }

    /*! Multiply RealDoubles
     * \param other of type RealDouble
     * */
    RCP<const Number> mulreal(const RealDouble &other) const
    {
        return make_rcp<const RealDouble>(i * other.i);
    }

    //! Converts the param `other` appropriately and then calls `mulreal`
    RCP<const Number> mul(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return mulreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return mulreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return mulreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return mulreal(down_cast<const RealDouble &>(other));
        } else {
            return other.mul(*this);
        }
    }

    /*! Divide RealDoubles
     * \param other of type Integer
     * */
    RCP<const Number> divreal(const Integer &other) const
    {
        return make_rcp<const RealDouble>(i
                                          / mp_get_d(other.as_integer_class()));
    }

    /*! Divide RealDoubles
     * \param other of type Rational
     * */
    RCP<const Number> divreal(const Rational &other) const
    {
        return make_rcp<const RealDouble>(
            i / mp_get_d(other.as_rational_class()));
    }

    /*! Divide RealDoubles
     * \param other of type Complex
     * */
    RCP<const Number> divreal(const Complex &other) const
    {
        return number(i
                      / std::complex<double>(mp_get_d(other.real_),
                                             mp_get_d(other.imaginary_)));
    }

    /*! Divide RealDoubles
     * \param other of type RealDouble
     * */
    RCP<const Number> divreal(const RealDouble &other) const
    {
        return make_rcp<const RealDouble>(i / other.i);
    }

    //! Converts the param `other` appropriately and then calls `divreal`
    RCP<const Number> div(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return divreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return divreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return divreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return divreal(down_cast<const RealDouble &>(other));
        } else {
            return other.rdiv(*this);
        }
    }

    /*! Divide RealDoubles
     * \param other of type Integer
     * */
    RCP<const Number> rdivreal(const Integer &other) const
    {
        return make_rcp<const RealDouble>(mp_get_d(other.as_integer_class())
                                          / i);
    }

    /*! Divide RealDoubles
     * \param other of type Rational
     * */
    RCP<const Number> rdivreal(const Rational &other) const
    {
        return make_rcp<const RealDouble>(mp_get_d(other.as_rational_class())
                                          / i);
    }

    /*! Divide RealDoubles
     * \param other of type Complex
     * */
    RCP<const Number> rdivreal(const Complex &other) const
    {
        return number(std::complex<double>(mp_get_d(other.real_),
                                           mp_get_d(other.imaginary_))
                      / i);
    }

    //! Converts the param `other` appropriately and then calls `divreal`
    RCP<const Number> rdiv(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rdivreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rdivreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return rdivreal(down_cast<const Complex &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }

    /*! Raise RealDouble to power `other`
     * \param other of type Integer
     * */
    RCP<const Number> powreal(const Integer &other) const
    {
        return make_rcp<const RealDouble>(
            std::pow(i, mp_get_d(other.as_integer_class())));
    }

    /*! Raise RealDouble to power `other`
     * \param other of type Rational
     * */
    RCP<const Number> powreal(const Rational &other) const
    {
        if (i < 0) {
            return number(std::pow(std::complex<double>(i),
                                   mp_get_d(other.as_rational_class())));
        }
        return make_rcp<const RealDouble>(
            std::pow(i, mp_get_d(other.as_rational_class())));
    }

    /*! Raise RealDouble to power `other`
     * \param other of type Complex
     * */
    RCP<const Number> powreal(const Complex &other) const
    {
        return number(
            std::pow(i, std::complex<double>(mp_get_d(other.real_),
                                             mp_get_d(other.imaginary_))));
    }

    /*! Raise RealDouble to power `other`
     * \param other of type RealDouble
     * */
    RCP<const Number> powreal(const RealDouble &other) const
    {
        if (i < 0) {
            return number(std::pow(std::complex<double>(i), other.i));
        }
        return make_rcp<const RealDouble>(std::pow(i, other.i));
    }

    //! Converts the param `other` appropriately and then calls `powreal`
    RCP<const Number> pow(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return powreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return powreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return powreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return powreal(down_cast<const RealDouble &>(other));
        } else {
            return other.rpow(*this);
        }
    }

    /*! Raise `other` to power RealDouble
     * \param other of type Integer
     * */
    RCP<const Number> rpowreal(const Integer &other) const
    {
        if (other.is_negative()) {
            return number(std::pow(mp_get_d(other.as_integer_class()),
                                   std::complex<double>(i)));
        }
        return make_rcp<const RealDouble>(
            std::pow(mp_get_d(other.as_integer_class()), i));
    }

    /*! Raise `other` to power RealDouble
     * \param other of type Rational
     * */
    RCP<const Number> rpowreal(const Rational &other) const
    {
        if (other.is_negative()) {
            return number(std::pow(std::complex<double>(i),
                                   mp_get_d(other.as_rational_class())));
        }
        return make_rcp<const RealDouble>(
            std::pow(mp_get_d(other.as_rational_class()), i));
    }

    /*! Raise `other` to power RealDouble
     * \param other of type Complex
     * */
    RCP<const Number> rpowreal(const Complex &other) const
    {
        return number(std::pow(std::complex<double>(mp_get_d(other.real_),
                                                    mp_get_d(other.imaginary_)),
                               i));
    }

    //! Converts the param `other` appropriately and then calls `powreal`
    RCP<const Number> rpow(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rpowreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rpowreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return rpowreal(down_cast<const Complex &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }
};

RCP<const RealDouble> real_double(double x);

} // namespace SymEngine

#endif
