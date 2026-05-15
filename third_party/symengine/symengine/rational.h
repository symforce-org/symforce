/**
 *  \file rational.h
 *  Class for Rationals built on top of Number class
 *
 **/
#ifndef SYMENGINE_RATIONAL_H
#define SYMENGINE_RATIONAL_H

#include <symengine/constants.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{
//! Rational Class
class Rational : public Number
{
private:
    //! `i` : object of `rational_class`
    rational_class i;

public:
    IMPLEMENT_TYPEID(SYMENGINE_RATIONAL)
    //! Constructor of Rational class
    Rational(rational_class &&_i) : i(std::move(_i))
    {
        SYMENGINE_ASSIGN_TYPEID()
    }
    /*! \param `i` must already be in rational_class canonical form
     *   \return Integer or Rational depending on denumerator.
     * */
    static RCP<const Number> from_mpq(const rational_class &i);
    static RCP<const Number> from_mpq(rational_class &&i);
    //! \return size of the hash
    hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    //! \return true if canonical
    bool is_canonical(const rational_class &i) const;

    /*! Constructs Rational as n/d, where n, d can be any Integers. If n/d is an
     *   Integer, it will return an Integer instead.
     * */
    static RCP<const Number> from_two_ints(const Integer &n, const Integer &d);
    static RCP<const Number> from_two_ints(const long n, const long d);
    //! Convert to `rational_class`.
    inline const rational_class &as_rational_class() const
    {
        return this->i;
    }
    //! \return `true` if `0`
    bool is_zero() const override
    {
        return this->i == 0;
    }
    //! \return `true` if `1`
    bool is_one() const override
    {
        return this->i == 1;
    }
    //! \return `true` if `-1`
    bool is_minus_one() const override
    {
        return this->i == -1;
    }
    //! \return `true` if denominator is `1`
    inline bool is_int() const
    {
        return this->i == 1;
    }
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
    //! \returns `false`
    // False is returned because a rational cannot have an imaginary part
    inline bool is_complex() const override
    {
        return false;
    }

    //! \return negative of `this`
    inline RCP<const Rational> neg() const
    {
        return make_rcp<const Rational>(-i);
    }

    bool is_perfect_power(bool is_expected = false) const override;
    // \return true if there is a exact nth root of self.
    bool nth_root(const Ptr<RCP<const Number>> &,
                  unsigned long n) const override;

    /*! Add Rationals
     * \param other of type Rational
     * */
    inline RCP<const Number> addrat(const Rational &other) const
    {
        return from_mpq(this->i + other.i);
    }
    /*! Add Rationals
     * \param other of type Integer
     * */
    inline RCP<const Number> addrat(const Integer &other) const
    {
        return from_mpq(this->i + other.as_integer_class());
    }
    /*! Subtract Rationals
     * \param other of type Rational
     * */
    inline RCP<const Number> subrat(const Rational &other) const
    {
        return from_mpq(this->i - other.i);
    }
    /*! Subtract Rationals
     * \param other of type Integer
     * */
    inline RCP<const Number> subrat(const Integer &other) const
    {
        return from_mpq(this->i - other.as_integer_class());
    }
    inline RCP<const Number> rsubrat(const Integer &other) const
    {
        return from_mpq(other.as_integer_class() - this->i);
    }
    /*! Multiply Rationals
     * \param other of type Rational
     * */
    inline RCP<const Number> mulrat(const Rational &other) const
    {
        return from_mpq(this->i * other.i);
    }
    /*! Multiply Rationals
     * \param other of type Integer
     * */
    inline RCP<const Number> mulrat(const Integer &other) const
    {
        return from_mpq(this->i * other.as_integer_class());
    }
    /*! Divide Rationals
     * \param other of type Rational
     * */
    inline RCP<const Number> divrat(const Rational &other) const
    {
        if (other.i == 0) {
            if (this->i == 0) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq(this->i / other.i);
        }
    }
    /*! Divide Rationals
     * \param other of type Integer
     * */
    inline RCP<const Number> divrat(const Integer &other) const
    {
        if (other.as_integer_class() == 0) {
            if (this->i == 0) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq(this->i / other.as_integer_class());
        }
    }
    inline RCP<const Number> rdivrat(const Integer &other) const
    {
        if (this->i == 0) {
            if (other.is_zero()) {
                return Nan;
            } else {
                return ComplexInf;
            }
        } else {
            return from_mpq(other.as_integer_class() / this->i);
        }
    }
    /*! Raise Rationals to power `other`
     * \param other power to be raised
     * */
    inline RCP<const Number> powrat(const Integer &other) const
    {
        bool neg = other.is_negative();
        integer_class exp_ = other.as_integer_class();
        if (neg)
            exp_ = -exp_;
        if (not mp_fits_ulong_p(exp_))
            throw SymEngineException("powrat: 'exp' does not fit ulong.");
        unsigned long exp = mp_get_ui(exp_);

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP
        // boost::multiprecision::cpp_rational doesn't provide
        // non-const references to num and den
        integer_class num;
        integer_class den;
        mp_pow_ui(num, SymEngine::get_num(i), exp);
        mp_pow_ui(den, SymEngine::get_den(i), exp);
        rational_class val(num, den);
#else
        rational_class val;
        mp_pow_ui(SymEngine::get_num(val), SymEngine::get_num(i), exp);
        mp_pow_ui(SymEngine::get_den(val), SymEngine::get_den(i), exp);
#endif

        // Since 'this' is in canonical form, so is this**other, so we simply
        // pass val into the constructor directly without canonicalizing:
        if (not neg) {
            return Rational::from_mpq(std::move(val));
        } else {
            return Rational::from_mpq(1 / val);
        }
    }
    /*! Raise *this to power `other`
     * \param other exponent
     * */
    RCP<const Basic> powrat(const Rational &other) const;
    /*!Reverse powrat
     * Raise 'other' to power *this
     * \param other base
     * */
    RCP<const Basic> rpowrat(const Integer &other) const;

    //! Converts the param `other` appropriately and then calls `addrat`
    RCP<const Number> add(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return addrat(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return addrat(down_cast<const Integer &>(other));
        } else {
            return other.add(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `subrat`
    RCP<const Number> sub(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return subrat(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return subrat(down_cast<const Integer &>(other));
        } else {
            return other.rsub(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `rsubrat`
    RCP<const Number> rsub(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return rsubrat(down_cast<const Integer &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    };
    //! Converts the param `other` appropriately and then calls `mulrat`
    RCP<const Number> mul(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return mulrat(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return mulrat(down_cast<const Integer &>(other));
        } else {
            return other.mul(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `divrat`
    RCP<const Number> div(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return divrat(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return divrat(down_cast<const Integer &>(other));
        } else {
            return other.rdiv(*this);
        }
    };
    //! Converts the param `other` appropriately and then calls `rdivrat`
    RCP<const Number> rdiv(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return rdivrat(down_cast<const Integer &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    };
    //! Converts the param `other` appropriately and then calls `powrat`
    RCP<const Number> pow(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return powrat(down_cast<const Integer &>(other));
        } else {
            return other.rpow(*this);
        }
    };

    RCP<const Number> rpow(const Number &other) const override
    {
        throw NotImplementedError("Not Implemented");
    };

    RCP<const Integer> get_num() const
    {
        return integer(SymEngine::get_num(i));
    }

    RCP<const Integer> get_den() const
    {
        return integer(SymEngine::get_den(i));
    }
};

//! returns the `num` and `den` of rational `rat` as `RCP<const Integer>`
void get_num_den(const Rational &rat, const Ptr<RCP<const Integer>> &num,
                 const Ptr<RCP<const Integer>> &den);

//! convenience creator from two longs
inline RCP<const Number> rational(long n, long d)
{
    return Rational::from_two_ints(n, d);
}
} // namespace SymEngine

#endif
