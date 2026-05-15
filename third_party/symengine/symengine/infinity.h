/**
 *  \file infinity.h
 *
 **/

#ifndef SYMENGINE_INFINITY_H
#define SYMENGINE_INFINITY_H

#include <symengine/basic.h>
#include <symengine/number.h>
#include <symengine/integer.h>
#include <symengine/mul.h>

namespace SymEngine
{

/** This class holds "infinity"
 *  It includes a direction (like -infinity).
 **/
class Infty : public Number
{
    RCP<const Number> _direction;

public:
    IMPLEMENT_TYPEID(SYMENGINE_INFTY)
    //! Constructs Infty using the sign of `_direction`
    Infty(const RCP<const Number> &direction);
    //! Copy Constructor
    Infty(const Infty &inf);
    static RCP<const Infty> from_direction(const RCP<const Number> &direction);
    //! Constructs Infty using sign of `val`
    static RCP<const Infty> from_int(const int val);

    //! \return true if canonical
    bool is_canonical(const RCP<const Number> &num) const;
    //! \return size of the hash
    hash_t __hash__() const override;

    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    // Implement these
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;

    vec_basic get_args() const override
    {
        return {_direction};
    }

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

    //! \return `true` if this number is an exact number
    inline bool is_exact() const override
    {
        return false;
    }
    // //! Get `Evaluate` singleton to evaluate numerically
    Evaluate &get_eval() const override;

    inline RCP<const Number> get_direction() const
    {
        return _direction;
    }

    bool is_unsigned_infinity() const;
    bool is_positive_infinity() const;
    bool is_negative_infinity() const;

    inline bool is_positive() const override
    {
        return is_positive_infinity();
    }

    inline bool is_negative() const override
    {
        return is_negative_infinity();
    }

    inline bool is_complex() const override
    {
        return is_unsigned_infinity();
    }
    //! \return the conjugate if the class is complex
    RCP<const Basic> conjugate() const override;

    // Think about it again
    RCP<const Number> add(const Number &other) const override;
    RCP<const Number> mul(const Number &other) const override;
    RCP<const Number> div(const Number &other) const override;
    RCP<const Number> pow(const Number &other) const override;
    RCP<const Number> rpow(const Number &other) const override;
};

inline RCP<const Infty> infty(int n = 1)
{
    return make_rcp<Infty>(integer(n));
}

RCP<const Infty> infty(const RCP<const Number> &direction);

} // namespace SymEngine
#endif
