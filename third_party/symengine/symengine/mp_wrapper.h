#ifndef SYMENGINE_MP_WRAPPER_H
#define SYMENGINE_MP_WRAPPER_H

#include <symengine/symengine_rcp.h>
#include <gmp.h>

#define SYMENGINE_UI(f) f##_ui
#define SYMENGINE_SI(f) f##_si

#define SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(op, func, val, rev_op)      \
    template <typename T,                                                      \
              typename std::enable_if<std::is_integral<T>::value               \
                                          && std::is_unsigned<T>::value,       \
                                      int>::type                               \
              = 0>                                                             \
    inline friend bool operator op(const mpz_wrapper &a, const T b)            \
    {                                                                          \
        return SYMENGINE_UI(func)(a.get_mpz_t(), b) op val;                    \
    }                                                                          \
    template <typename T,                                                      \
              typename std::enable_if<std::is_integral<T>::value, int>::type   \
              = 0>                                                             \
    inline friend bool operator op(const T a, const mpz_wrapper &b)            \
    {                                                                          \
        return b rev_op a;                                                     \
    }                                                                          \
    template <                                                                 \
        typename T,                                                            \
        typename std::enable_if<                                               \
            std::is_integral<T>::value && std::is_signed<T>::value, int>::type \
        = 0>                                                                   \
    inline friend bool operator op(const mpz_wrapper &a, const T b)            \
    {                                                                          \
        return SYMENGINE_SI(func)(a.get_mpz_t(), b) op val;                    \
    }                                                                          \
    inline friend bool operator op(const mpz_wrapper &a, const mpz_wrapper &b) \
    {                                                                          \
        return func(a.get_mpz_t(), b.get_mpz_t()) op val;                      \
    }

#define SYMENGINE_MPZ_WRAPPER_IMPLEMENT_IN_PLACE(op, func)                     \
    inline mpz_wrapper operator op(const mpz_wrapper &a)                       \
    {                                                                          \
        func(get_mpz_t(), get_mpz_t(), a.get_mpz_t());                         \
        return *this;                                                          \
    }                                                                          \
    template <typename T,                                                      \
              typename std::enable_if<std::is_integral<T>::value               \
                                          && std::is_unsigned<T>::value,       \
                                      int>::type                               \
              = 0>                                                             \
    inline mpz_wrapper operator op(const T a)                                  \
    {                                                                          \
        SYMENGINE_UI(func)(get_mpz_t(), get_mpz_t(), a);                       \
        return *this;                                                          \
    }

#define SYMENGINE_MPZ_WRAPPER_IMPLEMENT_NON_COMMUTATIVE(op, func, op_eq)       \
    template <typename T,                                                      \
              typename std::enable_if<std::is_integral<T>::value               \
                                          && std::is_unsigned<T>::value,       \
                                      int>::type                               \
              = 0>                                                             \
    inline friend mpz_wrapper operator op(const mpz_wrapper &a, const T b)     \
    {                                                                          \
        mpz_wrapper res;                                                       \
        SYMENGINE_UI(func)(res.get_mpz_t(), a.get_mpz_t(), b);                 \
        return res;                                                            \
    }                                                                          \
    inline friend mpz_wrapper operator op(const mpz_wrapper &a,                \
                                          const mpz_wrapper &b)                \
    {                                                                          \
        mpz_wrapper res;                                                       \
        func(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());                   \
        return res;                                                            \
    }                                                                          \
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_IN_PLACE(op_eq, func)

#define SYMENGINE_MPZ_WRAPPER_IMPLEMENT_COMMUTATIVE(op, func, op_eq)           \
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_NON_COMMUTATIVE(op, func, op_eq)           \
    template <typename T,                                                      \
              typename std::enable_if<std::is_integral<T>::value, int>::type   \
              = 0>                                                             \
    inline friend mpz_wrapper operator op(const T a, mpz_wrapper &b)           \
    {                                                                          \
        return b op a;                                                         \
    }

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT

#include <symengine/flint_wrapper.h>

#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP

namespace SymEngine
{

class mpz_wrapper
{
private:
    mpz_t mp;

public:
    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_unsigned<T>::value, int>::type
        = 0>
    mpz_wrapper(const T i)
    {
        mpz_init_set_ui(mp, i);
    }
    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_signed<T>::value, int>::type
        = 0>
    mpz_wrapper(const T i)
    {
        mpz_init_set_si(mp, i);
    }
    inline mpz_wrapper()
    {
        mpz_init(mp);
    }
    inline mpz_wrapper(const mpz_t m)
    {
        mpz_init_set(mp, m);
    }
    inline mpz_wrapper(const std::string &s, unsigned base = 10)
    {
        mpz_init_set_str(mp, s.c_str(), base);
    }
    inline mpz_wrapper(const mpz_wrapper &other)
    {
        mpz_init_set(mp, other.get_mpz_t());
    }
    inline mpz_wrapper(mpz_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        mp->_mp_d = nullptr;
        mpz_swap(mp, other.get_mpz_t());
    }
    inline mpz_wrapper &operator=(const mpz_wrapper &other)
    {
        if (mp->_mp_d == nullptr) {
            mpz_init_set(mp, other.get_mpz_t());
        } else {
            mpz_set(mp, other.get_mpz_t());
        }
        return *this;
    }
    inline mpz_wrapper &operator=(mpz_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        mpz_swap(mp, other.get_mpz_t());
        return *this;
    }
    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_unsigned<T>::value, int>::type
        = 0>
    inline mpz_wrapper &operator=(T other)
    {
        if (mp->_mp_d == nullptr) {
            mpz_init_set_ui(mp, other);
        } else {
            mpz_set_ui(mp, other);
        }
        return *this;
    }
    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_signed<T>::value, int>::type
        = 0>
    inline mpz_wrapper &operator=(T other)
    {
        if (mp->_mp_d == nullptr) {
            mpz_init_set_si(mp, other);
        } else {
            mpz_set_si(mp, other);
        }
        return *this;
    }
    inline ~mpz_wrapper() SYMENGINE_NOEXCEPT
    {
        if (mp->_mp_d != nullptr) {
            mpz_clear(mp);
        }
    }
    inline mpz_ptr get_mpz_t()
    {
        return mp;
    }
    inline mpz_srcptr get_mpz_t() const
    {
        return mp;
    }

    //! + operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_COMMUTATIVE(+, mpz_add, +=)
    //! * operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_COMMUTATIVE(*, mpz_mul, *=)
    //! - operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_NON_COMMUTATIVE(-, mpz_sub, -=)

    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_unsigned<T>::value, int>::type
        = 0>
    inline friend mpz_wrapper operator-(const T b, const mpz_wrapper &a)
    {
        mpz_wrapper res;
        mpz_ui_sub(res.get_mpz_t(), b, a.get_mpz_t());
        return res;
    }
    //! / operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_NON_COMMUTATIVE(/, mpz_tdiv_q, /=)
    //! % operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_NON_COMMUTATIVE(%, mpz_tdiv_r, %=)

    inline mpz_wrapper operator-() const
    {
        mpz_wrapper res;
        mpz_neg(res.get_mpz_t(), mp);
        return res;
    }

    inline mpz_wrapper operator++()
    {
        mpz_add_ui(mp, mp, 1);
        return *this;
    }
    inline mpz_wrapper operator++(int)
    {
        mpz_wrapper orig = *this;
        ++(*this);
        return orig;
    }
    inline mpz_wrapper operator--()
    {
        mpz_sub_ui(mp, mp, 1);
        return *this;
    }
    inline mpz_wrapper operator--(int)
    {
        mpz_wrapper orig = *this;
        --(*this);
        return orig;
    }

    //! < operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(<, mpz_cmp, 0, >)
    //! <= operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(<=, mpz_cmp, 0, >=)
    //! > operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(>, mpz_cmp, 0, <)
    //! >= operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(>=, mpz_cmp, 0, <=)
    //! == operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(==, mpz_cmp, 0, ==)
    //! != operator
    SYMENGINE_MPZ_WRAPPER_IMPLEMENT_RELATIONAL(!=, mpz_cmp, 0, !=)

    inline mpz_wrapper operator<<=(unsigned long u)
    {
        mpz_mul_2exp(mp, mp, u);
        return *this;
    }
    inline mpz_wrapper operator<<(unsigned long u) const
    {
        mpz_wrapper res;
        mpz_mul_2exp(res.get_mpz_t(), mp, u);
        return res;
    }
    inline mpz_wrapper operator>>=(unsigned long u)
    {
        mpz_tdiv_q_2exp(mp, mp, u);
        return *this;
    }
    inline mpz_wrapper operator>>(unsigned long u) const
    {
        mpz_wrapper res;
        mpz_tdiv_q_2exp(res.get_mpz_t(), mp, u);
        return res;
    }
    inline unsigned long get_ui() const
    {
        return mpz_get_ui(mp);
    }
    inline signed long get_si() const
    {
        return mpz_get_si(mp);
    }
    inline double long get_d() const
    {
        return mpz_get_d(mp);
    }
    inline int fits_ulong_p() const
    {
        return mpz_fits_ulong_p(mp);
    }
    inline int fits_slong_p() const
    {
        return mpz_fits_slong_p(mp);
    }
};

class mpq_wrapper
{
private:
    mpq_t mp;

public:
    mpq_ptr get_mpq_t()
    {
        return mp;
    }
    mpq_srcptr get_mpq_t() const
    {
        return mp;
    }
    mpq_wrapper()
    {
        mpq_init(mp);
    }
    mpq_wrapper(const mpz_t m)
    {
        mpq_init(mp);
        mpz_set(mpq_numref(mp), m);
    }
    mpq_wrapper(const mpq_t m)
    {
        mpq_init(mp);
        mpq_set(mp, m);
    }
    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_unsigned<T>::value, int>::type
        = 0>
    mpq_wrapper(const T i)
    {
        mpq_init(mp);
        mpz_set_ui(mpq_numref(mp), i);
    }
    template <
        typename T,
        typename std::enable_if<
            std::is_integral<T>::value && std::is_signed<T>::value, int>::type
        = 0>
    mpq_wrapper(const T i)
    {
        mpq_init(mp);
        mpz_set_si(mpq_numref(mp), i);
    }
    mpq_wrapper(const mpz_wrapper &n, const mpz_wrapper &d = 1)
    {
        mpq_init(mp);
        mpz_set(mpq_numref(mp), n.get_mpz_t());
        mpz_set(mpq_denref(mp), d.get_mpz_t());
        mpq_canonicalize(mp);
    }
    mpq_wrapper(const mpq_wrapper &other)
    {
        mpq_init(mp);
        mpq_set(mp, other.get_mpq_t());
    }
    mpq_wrapper(mpq_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        mpq_init(mp);
        mpq_swap(mp, other.get_mpq_t());
    }
    mpq_wrapper &operator=(const mpq_wrapper &other)
    {
        mpq_set(mp, other.get_mpq_t());
        return *this;
    }
    mpq_wrapper &operator=(mpq_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        mpq_swap(mp, other.get_mpq_t());
        return *this;
    }
    ~mpq_wrapper() SYMENGINE_NOEXCEPT
    {
        mpq_clear(mp);
    }
    const mpz_wrapper &get_den() const
    {
        return reinterpret_cast<const mpz_wrapper &>(*mpq_denref(mp));
    }
    const mpz_wrapper &get_num() const
    {
        return reinterpret_cast<const mpz_wrapper &>(*mpq_numref(mp));
    }
    mpz_wrapper &get_den()
    {
        return reinterpret_cast<mpz_wrapper &>(*mpq_denref(mp));
    }
    mpz_wrapper &get_num()
    {
        return reinterpret_cast<mpz_wrapper &>(*mpq_numref(mp));
    }
    friend mpq_wrapper operator+(const mpq_wrapper &a, const mpq_wrapper &b)
    {
        mpq_wrapper res;
        mpq_add(res.get_mpq_t(), a.get_mpq_t(), b.get_mpq_t());
        return res;
    }
    mpq_wrapper operator+=(const mpq_wrapper &a)
    {
        mpq_add(mp, mp, a.get_mpq_t());
        return *this;
    }
    friend mpq_wrapper operator-(const mpq_wrapper &a, const mpq_wrapper &b)
    {
        mpq_wrapper res;
        mpq_sub(res.get_mpq_t(), a.get_mpq_t(), b.get_mpq_t());
        return res;
    }
    mpq_wrapper operator-=(const mpq_wrapper &a)
    {
        mpq_sub(mp, mp, a.get_mpq_t());
        return *this;
    }
    mpq_wrapper operator-() const
    {
        mpq_wrapper res;
        mpq_neg(res.get_mpq_t(), mp);
        return res;
    }
    friend mpq_wrapper operator*(const mpq_wrapper &a, const mpq_wrapper &b)
    {
        mpq_wrapper res;
        mpq_mul(res.get_mpq_t(), a.get_mpq_t(), b.get_mpq_t());
        return res;
    }
    mpq_wrapper operator*=(const mpq_wrapper &a)
    {
        mpq_mul(mp, mp, a.get_mpq_t());
        return *this;
    }
    friend mpq_wrapper operator/(const mpq_wrapper &a, const mpq_wrapper &b)
    {
        mpq_wrapper res;
        mpq_div(res.get_mpq_t(), a.get_mpq_t(), b.get_mpq_t());
        return res;
    }
    mpq_wrapper operator/=(const mpq_wrapper &a)
    {
        mpq_div(mp, mp, a.get_mpq_t());
        return *this;
    }
    bool operator==(const mpq_wrapper &other) const
    {
        return mpq_cmp(mp, other.get_mpq_t()) == 0;
    }
    bool operator!=(const mpq_wrapper &other) const
    {
        return not(*this == other);
    }
    bool operator<(const mpq_wrapper &other) const
    {
        return mpq_cmp(mp, other.get_mpq_t()) < 0;
    }
    bool operator<=(const mpq_wrapper &other) const
    {
        return mpq_cmp(mp, other.get_mpq_t()) <= 0;
    }
    bool operator>(const mpq_wrapper &other) const
    {
        return mpq_cmp(mp, other.get_mpq_t()) > 0;
    }
    bool operator>=(const mpq_wrapper &other) const
    {
        return mpq_cmp(mp, other.get_mpq_t()) >= 0;
    }
    double get_d() const
    {
        return mpq_get_d(mp);
    }
    void canonicalize()
    {
        mpq_canonicalize(mp);
    }
};

} // namespace SymEngine

#endif

namespace SymEngine
{

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT
std::ostream &operator<<(std::ostream &os, const SymEngine::fmpq_wrapper &f);
std::ostream &operator<<(std::ostream &os, const SymEngine::fmpz_wrapper &f);
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP
std::ostream &operator<<(std::ostream &os, const SymEngine::mpq_wrapper &f);
std::ostream &operator<<(std::ostream &os, const SymEngine::mpz_wrapper &f);
#endif

} // namespace SymEngine

#endif // SYMENGINE_MP_WRAPPER_H
