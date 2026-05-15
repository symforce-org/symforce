/**
 *  \file RealMPFR.h
 *  Class for RealMPFR built on top of Number class
 *
 **/
#ifndef SYMENGINE_REAL_MPFR_H
#define SYMENGINE_REAL_MPFR_H

#include <symengine/complex_double.h>
#include <symengine/symengine_exception.h>

#ifdef HAVE_SYMENGINE_MPFR
#include <mpfr.h>

namespace SymEngine
{

class mpfr_class
{
private:
    mpfr_t mp;

public:
    mpfr_ptr get_mpfr_t()
    {
        return mp;
    }
    mpfr_srcptr get_mpfr_t() const
    {
        return mp;
    }
    explicit mpfr_class(mpfr_srcptr m)
    {
        mpfr_init2(mp, mpfr_get_prec(m));
        mpfr_set(mp, m, MPFR_RNDN);
    }
    explicit mpfr_class(mpfr_prec_t prec = 53)
    {
        mpfr_init2(mp, prec);
    }
    mpfr_class(std::string s, mpfr_prec_t prec = 53, unsigned base = 10)
    {
        mpfr_init2(mp, prec);
        mpfr_set_str(mp, s.c_str(), base, MPFR_RNDN);
    }
    mpfr_class(const mpfr_class &other)
    {
        mpfr_init2(mp, mpfr_get_prec(other.get_mpfr_t()));
        mpfr_set(mp, other.get_mpfr_t(), MPFR_RNDN);
    }
    mpfr_class(mpfr_class &&other)
    {
        mp->_mpfr_d = nullptr;
        mpfr_swap(mp, other.get_mpfr_t());
    }
    mpfr_class &operator=(const mpfr_class &other)
    {
        mpfr_set_prec(mp, mpfr_get_prec(other.get_mpfr_t()));
        mpfr_set(mp, other.get_mpfr_t(), MPFR_RNDN);
        return *this;
    }
    mpfr_class &operator=(mpfr_class &&other)
    {
        mpfr_swap(mp, other.get_mpfr_t());
        return *this;
    }
    ~mpfr_class()
    {
        if (mp->_mpfr_d != nullptr) {
            mpfr_clear(mp);
        }
    }
    mpfr_prec_t get_prec() const
    {
        return mpfr_get_prec(mp);
    }
};

void hash_combine_impl(hash_t &, mpfr_srcptr);

RCP<const Number> number(mpfr_ptr x);

//! RealMPFR Class to hold mpfr_t values
class RealMPFR : public Number
{
public:
    mpfr_class i;

public:
    IMPLEMENT_TYPEID(SYMENGINE_REAL_MPFR)
    //! Constructor of RealMPFR class
    RealMPFR(mpfr_class i);
    inline const mpfr_class &as_mpfr() const
    {
        return i;
    }
    inline mpfr_prec_t get_prec() const
    {
        return mpfr_get_prec(i.get_mpfr_t());
    }
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
        return mpfr_cmp_si(i.get_mpfr_t(), 0) > 0;
    }
    //! \return `true` if negative
    inline bool is_negative() const override
    {
        return mpfr_cmp_si(i.get_mpfr_t(), 0) < 0;
    }
    //! \return `true` if this number is an exact number
    inline bool is_exact() const override
    {
        return false;
    }
    //! Get `Evaluate` singleton to evaluate numerically
    Evaluate &get_eval() const override;

    //! \return if equal to `0`
    bool is_zero() const override
    {
        return mpfr_cmp_si(i.get_mpfr_t(), 0) == 0;
    }
    //! \return `false`
    // A mpfr_t is not exactly equal to `1`
    bool is_one() const override
    {
        return false;
    }
    //! \return `false`
    // A mpfr_t is not exactly equal to `-1`
    bool is_minus_one() const override
    {
        return false;
    }
    //! \returns `false`
    // False is returned because an 'mpfr' cannot have an imaginary part
    bool is_complex() const override
    {
        return false;
    }

    /*! Add RealMPFRs
     * \param other of type Integer
     * */
    RCP<const Number> addreal(const Integer &other) const;
    RCP<const Number> addreal(const Rational &other) const;
    RCP<const Number> addreal(const Complex &other) const;
    RCP<const Number> addreal(const RealDouble &other) const;
    RCP<const Number> addreal(const ComplexDouble &other) const;
    RCP<const Number> addreal(const RealMPFR &other) const;

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
        } else if (is_a<ComplexDouble>(other)) {
            return addreal(down_cast<const ComplexDouble &>(other));
        } else if (is_a<RealMPFR>(other)) {
            return addreal(down_cast<const RealMPFR &>(other));
        } else {
            return other.add(*this);
        }
    }

    RCP<const Number> subreal(const Integer &other) const;
    RCP<const Number> subreal(const Rational &other) const;
    RCP<const Number> subreal(const Complex &other) const;
    RCP<const Number> subreal(const RealDouble &other) const;
    RCP<const Number> subreal(const ComplexDouble &other) const;
    RCP<const Number> subreal(const RealMPFR &other) const;

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
        } else if (is_a<ComplexDouble>(other)) {
            return subreal(down_cast<const ComplexDouble &>(other));
        } else if (is_a<RealMPFR>(other)) {
            return subreal(down_cast<const RealMPFR &>(other));
        } else {
            return other.rsub(*this);
        }
    }

    RCP<const Number> rsubreal(const Integer &other) const;
    RCP<const Number> rsubreal(const Rational &other) const;
    RCP<const Number> rsubreal(const Complex &other) const;
    RCP<const Number> rsubreal(const RealDouble &other) const;
    RCP<const Number> rsubreal(const ComplexDouble &other) const;

    //! Converts the param `other` appropriately and then calls `subreal`
    RCP<const Number> rsub(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rsubreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rsubreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return rsubreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return rsubreal(down_cast<const RealDouble &>(other));
        } else if (is_a<ComplexDouble>(other)) {
            return rsubreal(down_cast<const ComplexDouble &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }

    RCP<const Number> mulreal(const Integer &other) const;
    RCP<const Number> mulreal(const Rational &other) const;
    RCP<const Number> mulreal(const Complex &other) const;
    RCP<const Number> mulreal(const RealDouble &other) const;
    RCP<const Number> mulreal(const ComplexDouble &other) const;
    RCP<const Number> mulreal(const RealMPFR &other) const;

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
        } else if (is_a<ComplexDouble>(other)) {
            return mulreal(down_cast<const ComplexDouble &>(other));
        } else if (is_a<RealMPFR>(other)) {
            return mulreal(down_cast<const RealMPFR &>(other));
        } else {
            return other.mul(*this);
        }
    }

    RCP<const Number> divreal(const Integer &other) const;
    RCP<const Number> divreal(const Rational &other) const;
    RCP<const Number> divreal(const Complex &other) const;
    RCP<const Number> divreal(const RealDouble &other) const;
    RCP<const Number> divreal(const ComplexDouble &other) const;
    RCP<const Number> divreal(const RealMPFR &other) const;

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
        } else if (is_a<ComplexDouble>(other)) {
            return divreal(down_cast<const ComplexDouble &>(other));
        } else if (is_a<RealMPFR>(other)) {
            return divreal(down_cast<const RealMPFR &>(other));
        } else {
            return other.rdiv(*this);
        }
    }

    RCP<const Number> rdivreal(const Integer &other) const;
    RCP<const Number> rdivreal(const Rational &other) const;
    RCP<const Number> rdivreal(const Complex &other) const;
    RCP<const Number> rdivreal(const RealDouble &other) const;
    RCP<const Number> rdivreal(const ComplexDouble &other) const;

    //! Converts the param `other` appropriately and then calls `divreal`
    RCP<const Number> rdiv(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rdivreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rdivreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return rdivreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return rdivreal(down_cast<const RealDouble &>(other));
        } else if (is_a<ComplexDouble>(other)) {
            return rdivreal(down_cast<const ComplexDouble &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }

    RCP<const Number> powreal(const Integer &other) const;
    RCP<const Number> powreal(const Rational &other) const;
    RCP<const Number> powreal(const Complex &other) const;
    RCP<const Number> powreal(const RealDouble &other) const;
    RCP<const Number> powreal(const ComplexDouble &other) const;
    RCP<const Number> powreal(const RealMPFR &other) const;

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
        } else if (is_a<ComplexDouble>(other)) {
            return powreal(down_cast<const ComplexDouble &>(other));
        } else if (is_a<RealMPFR>(other)) {
            return powreal(down_cast<const RealMPFR &>(other));
        } else {
            return other.rpow(*this);
        }
    }

    RCP<const Number> rpowreal(const Integer &other) const;
    RCP<const Number> rpowreal(const Rational &other) const;
    RCP<const Number> rpowreal(const Complex &other) const;
    RCP<const Number> rpowreal(const RealDouble &other) const;
    RCP<const Number> rpowreal(const ComplexDouble &other) const;

    //! Converts the param `other` appropriately and then calls `powreal`
    RCP<const Number> rpow(const Number &other) const override
    {
        if (is_a<Rational>(other)) {
            return rpowreal(down_cast<const Rational &>(other));
        } else if (is_a<Integer>(other)) {
            return rpowreal(down_cast<const Integer &>(other));
        } else if (is_a<Complex>(other)) {
            return rpowreal(down_cast<const Complex &>(other));
        } else if (is_a<RealDouble>(other)) {
            return rpowreal(down_cast<const RealDouble &>(other));
        } else if (is_a<ComplexDouble>(other)) {
            return rpowreal(down_cast<const ComplexDouble &>(other));
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }
};

inline RCP<const RealMPFR> real_mpfr(mpfr_class x)
{
    return rcp(new RealMPFR(std::move(x)));
}
} // namespace SymEngine
#else

namespace SymEngine
{
class RealMPFR : public Number
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_REAL_MPFR)
};
} // namespace SymEngine

#endif // HAVE_SYMENGINE_MPFR
#endif // SymEngine
