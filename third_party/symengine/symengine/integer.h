/**
 *  \file integer.h
 *  Class to implement Integers
 *
 **/

#ifndef SYMENGINE_INTEGER_H
#define SYMENGINE_INTEGER_H

#include <symengine/number.h>
#include <symengine/symengine_exception.h>
#include <symengine/symengine_casts.h>

namespace SymEngine
{

//! Integer Class
class Integer : public Number
{
private:
    //! `i` : object of `integer_class`
    integer_class i;

public:
    IMPLEMENT_TYPEID(SYMENGINE_INTEGER)
    //! Constructor of Integer using `integer_class`
    // explicit Integer(integer_class i);
    Integer(const integer_class &_i)
        : i(_i){SYMENGINE_ASSIGN_TYPEID()} Integer(integer_class && _i)
        : i(std::move(_i)){SYMENGINE_ASSIGN_TYPEID()}
          //! \return size of the hash
          hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;

    //! Convert to `int`, raise an exception if it does not fit
    signed long int as_int() const;
    //! Convert to `uint`, raise an exception if it does not fit
    unsigned long int as_uint() const;
    //! Convert to `integer_class`.
    inline const integer_class &as_integer_class() const
    {
        return this->i;
    }
    //! \return `true` if `0`
    inline bool is_zero() const override
    {
        return this->i == 0u;
    }
    //! \return `true` if `1`
    inline bool is_one() const override
    {
        return this->i == 1u;
    }
    //! \return `true` if `-1`
    inline bool is_minus_one() const override
    {
        return this->i == -1;
    }
    //! \return `true` if positive
    inline bool is_positive() const override
    {
        return this->i > 0u;
    }
    //! \return `true` if negative
    inline bool is_negative() const override
    {
        return this->i < 0u;
    }
    //! \returns `false`
    // False is returned because a pure integer cannot have an imaginary part
    inline bool is_complex() const override
    {
        return false;
    }

    /* These are very fast methods for add/sub/mul/div/pow on Integers only */
    //! Fast Integer Addition
    inline RCP<const Integer> addint(const Integer &other) const
    {
        return make_rcp<const Integer>(this->i + other.i);
    }
    //! Fast Integer Subtraction
    inline RCP<const Integer> subint(const Integer &other) const
    {
        return make_rcp<const Integer>(this->i - other.i);
    }
    //! Fast Integer Multiplication
    inline RCP<const Integer> mulint(const Integer &other) const
    {
        return make_rcp<const Integer>(this->i * other.i);
    }
    //!  Integer Division
    RCP<const Number> divint(const Integer &other) const;
    //! Fast Negative Power Evaluation
    RCP<const Number> pow_negint(const Integer &other) const;
    //! Fast Power Evaluation
    inline RCP<const Number> powint(const Integer &other) const
    {
        if (not(mp_fits_ulong_p(other.i))) {
            if (other.i > 0u)
                throw SymEngineException(
                    "powint: 'exp' does not fit unsigned long.");
            else
                return pow_negint(other);
        }
        integer_class tmp;
        mp_pow_ui(tmp, i, mp_get_ui(other.i));
        return make_rcp<const Integer>(std::move(tmp));
    }
    //! \return negative of self.
    inline RCP<const Integer> neg() const
    {
        return make_rcp<const Integer>(-i);
    }

    /* These are general methods, overriden from the Number class, that need to
     * check types to decide what operation to do, and so are a bit slower. */
    //! Slower Addition
    RCP<const Number> add(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return addint(down_cast<const Integer &>(other));
        } else {
            return other.add(*this);
        }
    };
    //! Slower Subtraction
    RCP<const Number> sub(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return subint(down_cast<const Integer &>(other));
        } else {
            return other.rsub(*this);
        }
    };

    RCP<const Number> rsub(const Number &other) const override
    {
        throw NotImplementedError("Not Implemented");
    };

    //! Slower Multiplication
    RCP<const Number> mul(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return mulint(down_cast<const Integer &>(other));
        } else {
            return other.mul(*this);
        }
    };
    //! Slower Division
    RCP<const Number> div(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return divint(down_cast<const Integer &>(other));
        } else {
            return other.rdiv(*this);
        }
    };

    RCP<const Number> rdiv(const Number &other) const override;

    //! Slower power evaluation
    RCP<const Number> pow(const Number &other) const override
    {
        if (is_a<Integer>(other)) {
            return powint(down_cast<const Integer &>(other));
        } else {
            return other.rpow(*this);
        }
    };

    RCP<const Number> rpow(const Number &other) const override
    {
        throw NotImplementedError("Not Implemented");
    };
};

//! less operator (<) for Integers
struct RCPIntegerKeyLess {
    //! \return true according to `<` operator
    bool operator()(const RCP<const Integer> &a,
                    const RCP<const Integer> &b) const
    {
        return a->as_integer_class() < b->as_integer_class();
    }
};
//! \return RCP<const Integer> from integral values
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value,
                               RCP<const Integer>>::type
integer(T i)
{
    return make_rcp<const Integer>(integer_class(i));
}

//! \return RCP<const Integer> from integer_class
inline RCP<const Integer> integer(integer_class i)
{
    return make_rcp<const Integer>(std::move(i));
}

//! Integer Square root
RCP<const Integer> isqrt(const Integer &n);
//! Integer nth root
int i_nth_root(const Ptr<RCP<const Integer>> &r, const Integer &a,
               unsigned long int n);
//! Perfect Square
bool perfect_square(const Integer &n);
//! Perfect Square
bool perfect_power(const Integer &n);
//! Integer Absolute value
RCP<const Integer> iabs(const Integer &n);

} // namespace SymEngine

#endif
