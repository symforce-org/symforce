#ifndef SYMENGINE_FLINT_WRAPPER_H
#define SYMENGINE_FLINT_WRAPPER_H

#include <symengine/symengine_rcp.h>
#include <gmp.h>

#include <flint/fmpz.h>
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>

namespace SymEngine
{

class fmpz_wrapper
{
private:
    fmpz_t mp;

public:
    template <typename T,
              typename std::enable_if<std::is_integral<T>::value
                                          && std::is_unsigned<T>::value,
                                      int>::type
              = 0>
    inline fmpz_wrapper(const T i)
    {
        fmpz_init(mp);
        fmpz_set_ui(mp, i);
    }
    template <typename T,
              typename std::enable_if<std::is_integral<T>::value
                                          && std::is_signed<T>::value,
                                      int>::type
              = 0>
    inline fmpz_wrapper(const T i)
    {
        fmpz_init(mp);
        fmpz_set_si(mp, i);
    }
    inline fmpz_wrapper()
    {
        fmpz_init(mp);
    }
    inline fmpz_wrapper(const mpz_t m)
    {
        fmpz_init(mp);
        fmpz_set_mpz(mp, m);
    }
    inline fmpz_wrapper(const fmpz_t m)
    {
        fmpz_init(mp);
        fmpz_set(mp, m);
    }
    inline fmpz_wrapper(const std::string &s, unsigned base = 10)
    {
        fmpz_init(mp);
        fmpz_set_str(mp, s.c_str(), base);
    }
    inline fmpz_wrapper(const fmpz_wrapper &other)
    {
        fmpz_init(mp);
        fmpz_set(mp, other.get_fmpz_t());
    }
    inline fmpz_wrapper(fmpz_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        fmpz_init(mp);
        fmpz_swap(mp, other.get_fmpz_t());
    }
    inline fmpz_wrapper &operator=(const fmpz_wrapper &other)
    {
        fmpz_set(mp, other.get_fmpz_t());
        return *this;
    }
    inline fmpz_wrapper &operator=(fmpz_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        fmpz_swap(mp, other.get_fmpz_t());
        return *this;
    }
    inline ~fmpz_wrapper() SYMENGINE_NOEXCEPT
    {
        fmpz_clear(mp);
    }
    inline fmpz *get_fmpz_t()
    {
        return mp;
    }
    inline const fmpz *get_fmpz_t() const
    {
        return mp;
    }
    inline friend fmpz_wrapper operator+(const fmpz_wrapper &a,
                                         const fmpz_wrapper &b)
    {
        fmpz_wrapper res;
        fmpz_add(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
        return res;
    }
    inline fmpz_wrapper operator+=(const fmpz_wrapper &a)
    {
        fmpz_add(mp, mp, a.get_fmpz_t());
        return *this;
    }
    inline friend fmpz_wrapper operator-(const fmpz_wrapper &a,
                                         const fmpz_wrapper &b)
    {
        fmpz_wrapper res;
        fmpz_sub(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
        return res;
    }
    inline fmpz_wrapper operator-=(const fmpz_wrapper &a)
    {
        fmpz_sub(mp, mp, a.get_fmpz_t());
        return *this;
    }
    inline fmpz_wrapper operator-() const
    {
        fmpz_wrapper res;
        fmpz_neg(res.get_fmpz_t(), mp);
        return res;
    }
    inline friend fmpz_wrapper operator*(const fmpz_wrapper &a,
                                         const fmpz_wrapper &b)
    {
        fmpz_wrapper res;
        fmpz_mul(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
        return res;
    }
    inline fmpz_wrapper operator*=(const fmpz_wrapper &a)
    {
        fmpz_mul(mp, mp, a.get_fmpz_t());
        return *this;
    }
    inline friend fmpz_wrapper operator/(const fmpz_wrapper &a,
                                         const fmpz_wrapper &b)
    {
        fmpz_wrapper res;
        fmpz_tdiv_q(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
        return res;
    }
    inline fmpz_wrapper operator/=(const fmpz_wrapper &a)
    {
        fmpz_tdiv_q(mp, mp, a.get_fmpz_t());
        return *this;
    }
    inline friend fmpz_wrapper operator%(const fmpz_wrapper &a,
                                         const fmpz_wrapper &b)
    {
        fmpz_wrapper res, tmp;
        fmpz_tdiv_qr(tmp.get_fmpz_t(), res.get_fmpz_t(), a.get_fmpz_t(),
                     b.get_fmpz_t());
        return res;
    }
    inline fmpz_wrapper operator%=(const fmpz_wrapper &a)
    {
        fmpz_wrapper tmp;
        fmpz_tdiv_qr(tmp.get_fmpz_t(), mp, mp, a.get_fmpz_t());
        return *this;
    }
    inline fmpz_wrapper operator++()
    {
        fmpz_add_ui(mp, mp, 1);
        return *this;
    }
    inline fmpz_wrapper operator++(int)
    {
        fmpz_wrapper orig = *this;
        ++(*this);
        return orig;
    }
    inline fmpz_wrapper operator--()
    {
        fmpz_sub_ui(mp, mp, 1);
        return *this;
    }
    inline fmpz_wrapper operator--(int)
    {
        fmpz_wrapper orig = *this;
        --(*this);
        return orig;
    }
    inline friend bool operator==(const fmpz_wrapper &a, const fmpz_wrapper &b)
    {
        return fmpz_equal(a.get_fmpz_t(), b.get_fmpz_t()) == 1;
    }
    inline friend bool operator!=(const fmpz_wrapper &a, const fmpz_wrapper &b)
    {
        return fmpz_equal(a.get_fmpz_t(), b.get_fmpz_t()) != 1;
    }
    inline friend bool operator<(const fmpz_wrapper &a, const fmpz_wrapper &b)
    {
        return fmpz_cmp(a.get_fmpz_t(), b.get_fmpz_t()) < 0;
    }
    inline friend bool operator<=(const fmpz_wrapper &a, const fmpz_wrapper &b)
    {
        return fmpz_cmp(a.get_fmpz_t(), b.get_fmpz_t()) <= 0;
    }
    inline friend bool operator>(const fmpz_wrapper &a, const fmpz_wrapper &b)
    {
        return fmpz_cmp(a.get_fmpz_t(), b.get_fmpz_t()) > 0;
    }
    inline friend bool operator>=(const fmpz_wrapper &a, const fmpz_wrapper &b)
    {
        return fmpz_cmp(a.get_fmpz_t(), b.get_fmpz_t()) >= 0;
    }
    inline fmpz_wrapper operator<<=(unsigned long u)
    {
        fmpz_mul_2exp(mp, mp, u);
        return *this;
    }
    inline fmpz_wrapper operator<<(unsigned long u) const
    {
        fmpz_wrapper res;
        fmpz_mul_2exp(res.get_fmpz_t(), mp, u);
        return res;
    }
    inline fmpz_wrapper operator>>=(unsigned long u)
    {
        fmpz_tdiv_q_2exp(mp, mp, u);
        return *this;
    }
    inline fmpz_wrapper operator>>(unsigned long u) const
    {
        fmpz_wrapper res;
        fmpz_tdiv_q_2exp(res.get_fmpz_t(), mp, u);
        return res;
    }
    inline fmpz_wrapper root(unsigned int n) const
    {
        fmpz_wrapper res;
        fmpz_root(res.get_fmpz_t(), mp, n);
        return res;
    }
};

class mpz_view_flint
{

public:
    mpz_view_flint(const fmpz_wrapper &i)
    {
        if (!COEFF_IS_MPZ(*i.get_fmpz_t())) {
            mpz_init_set_si(m, *i.get_fmpz_t());
        } else {
            ptr = COEFF_TO_PTR(*i.get_fmpz_t());
        }
    }
    operator mpz_srcptr() const
    {
        if (ptr == nullptr)
            return m;
        return ptr;
    }
    ~mpz_view_flint()
    {
        if (ptr == nullptr)
            mpz_clear(m);
    }

private:
    mpz_srcptr ptr = nullptr;
    mpz_t m;
};

class fmpq_wrapper
{
private:
    fmpq_t mp;

public:
    fmpq *get_fmpq_t()
    {
        return mp;
    }
    const fmpq *get_fmpq_t() const
    {
        return mp;
    }
    fmpq_wrapper()
    {
        fmpq_init(mp);
    }
    fmpq_wrapper(const mpz_t m)
    {
        fmpq_init(mp);
        fmpz_set_mpz(fmpq_numref(mp), m);
    }
    fmpq_wrapper(const fmpz_t m)
    {
        fmpq_init(mp);
        fmpz_set(fmpq_numref(mp), m);
    }
    fmpq_wrapper(const mpq_t m)
    {
        fmpq_init(mp);
        fmpq_set_mpq(mp, m);
    }
    fmpq_wrapper(const fmpq_t m)
    {
        fmpq_init(mp);
        fmpq_set(mp, m);
    }
    template <typename T,
              typename std::enable_if<std::is_integral<T>::value
                                          && std::is_unsigned<T>::value,
                                      int>::type
              = 0>
    fmpq_wrapper(const T i)
    {
        fmpq_init(mp);
        fmpz_set_ui(fmpq_numref(mp), i);
    }
    template <typename T,
              typename std::enable_if<std::is_integral<T>::value
                                          && std::is_signed<T>::value,
                                      int>::type
              = 0>
    fmpq_wrapper(const T i)
    {
        fmpq_init(mp);
        fmpz_set_si(fmpq_numref(mp), i);
    }
    fmpq_wrapper(const fmpz_wrapper &n, const fmpz_wrapper &d = 1)
    {
        fmpq_init(mp);
        fmpz_set(fmpq_numref(mp), n.get_fmpz_t());
        fmpz_set(fmpq_denref(mp), d.get_fmpz_t());
        fmpq_canonicalise(mp);
    }
    fmpq_wrapper(const fmpq_wrapper &other)
    {
        fmpq_init(mp);
        fmpq_set(mp, other.get_fmpq_t());
    }
    fmpq_wrapper(fmpq_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        fmpq_init(mp);
        fmpq_swap(mp, other.get_fmpq_t());
    }
    fmpq_wrapper &operator=(const fmpq_wrapper &other)
    {
        fmpq_set(mp, other.get_fmpq_t());
        return *this;
    }
    fmpq_wrapper &operator=(fmpq_wrapper &&other) SYMENGINE_NOEXCEPT
    {
        fmpq_swap(mp, other.get_fmpq_t());
        return *this;
    }
    ~fmpq_wrapper() SYMENGINE_NOEXCEPT
    {
        fmpq_clear(mp);
    }
    void canonicalise()
    {
        fmpq_canonicalise(mp);
    }
    const fmpz_wrapper &get_den() const
    {
        return reinterpret_cast<const fmpz_wrapper &>(*fmpq_denref(mp));
    }
    const fmpz_wrapper &get_num() const
    {
        return reinterpret_cast<const fmpz_wrapper &>(*fmpq_numref(mp));
    }
    fmpz_wrapper &get_den()
    {
        return reinterpret_cast<fmpz_wrapper &>(*fmpq_denref(mp));
    }
    fmpz_wrapper &get_num()
    {
        return reinterpret_cast<fmpz_wrapper &>(*fmpq_numref(mp));
    }
    friend fmpq_wrapper operator+(const fmpq_wrapper &a, const fmpq_wrapper &b)
    {
        fmpq_wrapper res;
        fmpq_add(res.get_fmpq_t(), a.get_fmpq_t(), b.get_fmpq_t());
        return res;
    }
    fmpq_wrapper operator+=(const fmpq_wrapper &a)
    {
        fmpq_add(mp, mp, a.get_fmpq_t());
        return *this;
    }
    friend fmpq_wrapper operator-(const fmpq_wrapper &a, const fmpq_wrapper &b)
    {
        fmpq_wrapper res;
        fmpq_sub(res.get_fmpq_t(), a.get_fmpq_t(), b.get_fmpq_t());
        return res;
    }
    fmpq_wrapper operator-=(const fmpq_wrapper &a)
    {
        fmpq_sub(mp, mp, a.get_fmpq_t());
        return *this;
    }
    fmpq_wrapper operator-() const
    {
        fmpq_wrapper res;
        fmpq_neg(res.get_fmpq_t(), mp);
        return res;
    }
    friend fmpq_wrapper operator*(const fmpq_wrapper &a, const fmpq_wrapper &b)
    {
        fmpq_wrapper res;
        fmpq_mul(res.get_fmpq_t(), a.get_fmpq_t(), b.get_fmpq_t());
        return res;
    }
    fmpq_wrapper operator*=(const fmpq_wrapper &a)
    {
        fmpq_mul(mp, mp, a.get_fmpq_t());
        return *this;
    }
    friend fmpq_wrapper operator/(const fmpq_wrapper &a, const fmpq_wrapper &b)
    {
        fmpq_wrapper res;
        fmpq_div(res.get_fmpq_t(), a.get_fmpq_t(), b.get_fmpq_t());
        return res;
    }
    fmpq_wrapper operator/=(const fmpq_wrapper &a)
    {
        fmpq_div(mp, mp, a.get_fmpq_t());
        return *this;
    }
    bool operator==(const fmpq_wrapper &other) const
    {
        return fmpq_equal(mp, other.get_fmpq_t());
    }
    bool operator!=(const fmpq_wrapper &other) const
    {
        return not(*this == other);
    }
    bool operator<(const fmpq_wrapper &other) const
    {
        return fmpq_cmp(mp, other.get_fmpq_t()) < 0;
    }
    bool operator<=(const fmpq_wrapper &other) const
    {
        return fmpq_cmp(mp, other.get_fmpq_t()) <= 0;
    }
    bool operator>(const fmpq_wrapper &other) const
    {
        return fmpq_cmp(mp, other.get_fmpq_t()) > 0;
    }
    bool operator>=(const fmpq_wrapper &other) const
    {
        return fmpq_cmp(mp, other.get_fmpq_t()) >= 0;
    }
    bool is_zero() const
    {
        return fmpq_is_zero(mp);
    }
    bool is_one() const
    {
        return fmpq_is_one(mp);
    }
};

class mpq_view_flint
{

public:
    mpq_view_flint(const fmpq_wrapper &i)
    {
        mpq_init(m);
        fmpq_get_mpq(m, i.get_fmpq_t());
    }
    operator mpq_srcptr() const
    {
        return m;
    }
    ~mpq_view_flint()
    {
        mpq_clear(m);
    }

private:
    mpq_t m;
};

class fmpz_poly_factor_wrapper
{
private:
    fmpz_poly_factor_t fac;

public:
    typedef fmpz_wrapper internal_coef_type;

    fmpz_poly_factor_wrapper()
    {
        fmpz_poly_factor_init(fac);
    }
    fmpz_poly_factor_wrapper(const fmpz_poly_factor_wrapper &other)
    {
        fmpz_poly_factor_init(fac);
        fmpz_poly_factor_set(fac, other.get_fmpz_poly_factor_t());
    }
    fmpz_poly_factor_wrapper &operator=(const fmpz_poly_factor_wrapper &other)
    {
        fmpz_poly_factor_set(fac, other.get_fmpz_poly_factor_t());
        return *this;
    }
    ~fmpz_poly_factor_wrapper()
    {
        fmpz_poly_factor_clear(fac);
    }

    const fmpz_poly_factor_t &get_fmpz_poly_factor_t() const
    {
        return fac;
    }
    fmpz_poly_factor_t &get_fmpz_poly_factor_t()
    {
        return fac;
    }
};

class fmpz_poly_wrapper
{
private:
    fmpz_poly_t poly;

public:
    typedef fmpz_wrapper internal_coef_type;

    fmpz_poly_wrapper()
    {
        fmpz_poly_init(poly);
    }
    fmpz_poly_wrapper(int i)
    {
        fmpz_poly_init(poly);
        fmpz_poly_set_si(poly, i);
    }
    fmpz_poly_wrapper(const char *cp)
    {
        fmpz_poly_init(poly);
        fmpz_poly_set_str(poly, cp);
    }
    fmpz_poly_wrapper(const fmpz_wrapper &z)
    {
        fmpz_poly_init(poly);
        fmpz_poly_set_fmpz(poly, z.get_fmpz_t());
    }
    fmpz_poly_wrapper(const fmpz_poly_wrapper &other)
    {
        fmpz_poly_init(poly);
        fmpz_poly_set(poly, *other.get_fmpz_poly_t());
    }
    fmpz_poly_wrapper(fmpz_poly_wrapper &&other)
    {
        fmpz_poly_init(poly);
        fmpz_poly_swap(poly, *other.get_fmpz_poly_t());
    }
    fmpz_poly_wrapper &operator=(const fmpz_poly_wrapper &other)
    {
        fmpz_poly_set(poly, *other.get_fmpz_poly_t());
        return *this;
    }
    fmpz_poly_wrapper &operator=(fmpz_poly_wrapper &&other)
    {
        fmpz_poly_swap(poly, *other.get_fmpz_poly_t());
        return *this;
    }
    void swap_fmpz_poly_t(fmpz_poly_struct &other)
    {
        fmpz_poly_swap(poly, &other);
    }
    ~fmpz_poly_wrapper()
    {
        fmpz_poly_clear(poly);
    }
    const fmpz_poly_t *get_fmpz_poly_t() const
    {
        return &poly;
    }
    fmpz_poly_t *get_fmpz_poly_t()
    {
        return &poly;
    }
    bool operator==(const fmpz_poly_wrapper &other) const
    {
        return fmpz_poly_equal(poly, *other.get_fmpz_poly_t()) == 1;
    }
    long degree() const
    {
        return fmpz_poly_degree(poly);
    }
    long length() const
    {
        return fmpz_poly_length(poly);
    }
    std::string to_string() const
    {
        return fmpz_poly_get_str(poly);
    }
    fmpz_wrapper get_coeff(unsigned int n) const
    {
        fmpz_wrapper z;
        fmpz_poly_get_coeff_fmpz(z.get_fmpz_t(), poly, n);
        return z;
    }
    void set_coeff(unsigned int n, const fmpz_wrapper &z)
    {
        fmpz_poly_set_coeff_fmpz(poly, n, z.get_fmpz_t());
    }
    fmpz_wrapper eval(const fmpz_wrapper &z) const
    {
        fmpz_wrapper r;
        fmpz_poly_evaluate_fmpz(r.get_fmpz_t(), poly, z.get_fmpz_t());
        return r;
    }
    void eval_vec(fmpz *ovec, fmpz *ivec, unsigned int n) const
    {
        fmpz_poly_evaluate_fmpz_vec(ovec, *get_fmpz_poly_t(), ivec, n);
    }
    fmpz_poly_wrapper operator-() const
    {
        fmpz_poly_wrapper r;
        fmpz_poly_neg(*r.get_fmpz_poly_t(), *get_fmpz_poly_t());
        return r;
    }
    void operator+=(const fmpz_poly_wrapper &other)
    {
        fmpz_poly_add(*get_fmpz_poly_t(), *get_fmpz_poly_t(),
                      *other.get_fmpz_poly_t());
    }
    void operator-=(const fmpz_poly_wrapper &other)
    {
        fmpz_poly_sub(*get_fmpz_poly_t(), *get_fmpz_poly_t(),
                      *other.get_fmpz_poly_t());
    }
    void operator*=(const fmpz_poly_wrapper &other)
    {
        fmpz_poly_mul(*get_fmpz_poly_t(), *get_fmpz_poly_t(),
                      *other.get_fmpz_poly_t());
    }

    friend fmpz_poly_wrapper operator*(const fmpz_poly_wrapper &a,
                                       const fmpz_poly_wrapper &b)
    {
        fmpz_poly_wrapper res;
        fmpz_poly_mul(*res.get_fmpz_poly_t(), *a.get_fmpz_poly_t(),
                      *b.get_fmpz_poly_t());
        return res;
    }

    fmpz_poly_wrapper gcd(const fmpz_poly_wrapper &other) const
    {
        fmpz_poly_wrapper r;
        fmpz_poly_gcd(*r.get_fmpz_poly_t(), poly, *other.get_fmpz_poly_t());
        return r;
    }
    fmpz_poly_wrapper lcm(const fmpz_poly_wrapper &other) const
    {
        fmpz_poly_wrapper r;
        fmpz_poly_lcm(*r.get_fmpz_poly_t(), poly, *other.get_fmpz_poly_t());
        return r;
    }
    fmpz_poly_wrapper pow(unsigned int n) const
    {
        fmpz_poly_wrapper r;
        fmpz_poly_pow(*r.get_fmpz_poly_t(), poly, n);
        return r;
    }
    void divrem(fmpz_poly_wrapper &q, fmpz_poly_wrapper &r,
                const fmpz_poly_wrapper &b) const
    {
        return fmpz_poly_divrem(*q.get_fmpz_poly_t(), *r.get_fmpz_poly_t(),
                                *get_fmpz_poly_t(), *b.get_fmpz_poly_t());
    }
    fmpz_poly_factor_wrapper factors() const
    {
        fmpz_poly_factor_wrapper r;
#if __FLINT_RELEASE > 20502
        fmpz_poly_factor(r.get_fmpz_poly_factor_t(), poly);
#else
        throw std::runtime_error(
            "FLINT's Version must be higher than 2.5.2 to obtain factors");
#endif
        return r;
    }
    fmpz_poly_wrapper derivative() const
    {
        fmpz_poly_wrapper r;
        fmpz_poly_derivative(*r.get_fmpz_poly_t(), poly);
        return r;
    }
};

class fmpq_poly_wrapper
{
private:
    fmpq_poly_t poly;

public:
    typedef fmpq_wrapper internal_coef_type;

    fmpq_poly_wrapper()
    {
        fmpq_poly_init(poly);
    }
    fmpq_poly_wrapper(int i)
    {
        fmpq_poly_init(poly);
        fmpq_poly_set_si(poly, i);
    }
    fmpq_poly_wrapper(const char *cp)
    {
        fmpq_poly_init(poly);
        fmpq_poly_set_str(poly, cp);
    }
    fmpq_poly_wrapper(const mpz_t z)
    {
        fmpq_poly_init(poly);
        fmpq_poly_set_mpz(poly, z);
    }
    fmpq_poly_wrapper(const mpq_t q)
    {
        fmpq_poly_init(poly);
        fmpq_poly_set_mpq(poly, q);
    }
    fmpq_poly_wrapper(const fmpq_wrapper &q)
    {
        fmpq_poly_init(poly);
        fmpq_poly_set_fmpq(poly, q.get_fmpq_t());
    }
    fmpq_poly_wrapper(const fmpq_poly_wrapper &other)
    {
        fmpq_poly_init(poly);
        fmpq_poly_set(poly, *other.get_fmpq_poly_t());
    }
    fmpq_poly_wrapper(fmpq_poly_wrapper &&other)
    {
        fmpq_poly_init(poly);
        fmpq_poly_swap(poly, *other.get_fmpq_poly_t());
    }
    fmpq_poly_wrapper &operator=(const fmpq_poly_wrapper &other)
    {
        fmpq_poly_set(poly, *other.get_fmpq_poly_t());
        return *this;
    }
    fmpq_poly_wrapper &operator=(fmpq_poly_wrapper &&other)
    {
        fmpq_poly_swap(poly, *other.get_fmpq_poly_t());
        return *this;
    }
    ~fmpq_poly_wrapper()
    {
        fmpq_poly_clear(poly);
    }

    const fmpq_poly_t *get_fmpq_poly_t() const
    {
        return &poly;
    }
    fmpq_poly_t *get_fmpq_poly_t()
    {
        return &poly;
    }
    std::string to_string() const
    {
        return fmpq_poly_get_str(poly);
    }
    long degree() const
    {
        return fmpq_poly_degree(poly);
    }
    long length() const
    {
        return fmpq_poly_length(poly);
    }
    void set_coeff(unsigned int n, const fmpq_wrapper &z)
    {
        fmpq_poly_set_coeff_fmpq(poly, n, z.get_fmpq_t());
    }
    fmpq_wrapper eval(const fmpq_wrapper &z) const
    {
        fmpq_wrapper r;
        fmpq_poly_evaluate_fmpq(r.get_fmpq_t(), poly, z.get_fmpq_t());
        return r;
    }
    fmpq_wrapper get_coeff(unsigned int deg) const
    {
        fmpq_wrapper q;
        fmpq_poly_get_coeff_fmpq(q.get_fmpq_t(), poly, deg);
        return q;
    }

    fmpq_poly_wrapper mullow(const fmpq_poly_wrapper &o,
                             unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_mullow(*r.get_fmpq_poly_t(), poly, *o.get_fmpq_poly_t(),
                         prec);
        return r;
    }
    fmpq_poly_wrapper pow(unsigned int n) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_pow(*r.get_fmpq_poly_t(), poly, n);
        return r;
    }
    fmpq_poly_wrapper derivative() const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_derivative(*r.get_fmpq_poly_t(), poly);
        return r;
    }
    fmpq_poly_wrapper integral() const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_integral(*r.get_fmpq_poly_t(), poly);
        return r;
    }

    fmpq_poly_wrapper inv_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_inv_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper revert_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_revert_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper log_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_log_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper exp_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_exp_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper sin_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_sin_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper cos_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_cos_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper tan_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_tan_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper asin_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_asin_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper atan_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_atan_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper sinh_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_sinh_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper cosh_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_cosh_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper tanh_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_tanh_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper asinh_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_asinh_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper atanh_series(unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_atanh_series(*r.get_fmpq_poly_t(), poly, prec);
        return r;
    }
    fmpq_poly_wrapper subs(const fmpq_poly_wrapper &o, unsigned int prec) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_compose_series(*r.get_fmpq_poly_t(), poly,
                                 *o.get_fmpq_poly_t(), prec);
        return r;
    }
    void set_zero()
    {
        fmpq_poly_zero(poly);
    }
    void set_one()
    {
        fmpq_poly_one(poly);
    }

    bool operator==(const fmpq_poly_wrapper &o) const
    {
        return fmpq_poly_equal(poly, *o.get_fmpq_poly_t()) != 0;
    }
    bool operator<(const fmpq_poly_wrapper &o) const
    {
        return fmpq_poly_cmp(poly, *o.get_fmpq_poly_t()) == -1;
    }

    friend fmpq_poly_wrapper operator+(const fmpq_poly_wrapper &a,
                                       const fmpq_poly_wrapper &o)
    {
        fmpq_poly_wrapper r;
        fmpq_poly_add(*r.get_fmpq_poly_t(), *a.get_fmpq_poly_t(),
                      *o.get_fmpq_poly_t());
        return r;
    }
    friend fmpq_poly_wrapper operator-(const fmpq_poly_wrapper &a,
                                       const fmpq_poly_wrapper &o)
    {
        fmpq_poly_wrapper r;
        fmpq_poly_sub(*r.get_fmpq_poly_t(), *a.get_fmpq_poly_t(),
                      *o.get_fmpq_poly_t());
        return r;
    }
    fmpq_poly_wrapper operator-() const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_neg(*r.get_fmpq_poly_t(), *get_fmpq_poly_t());
        return r;
    }
    friend fmpq_poly_wrapper operator*(const fmpq_poly_wrapper &a,
                                       const fmpq_wrapper &q)
    {
        fmpq_poly_wrapper r;
        fmpq_poly_scalar_mul_fmpq(*r.get_fmpq_poly_t(), *a.get_fmpq_poly_t(),
                                  q.get_fmpq_t());
        return r;
    }
    friend fmpq_poly_wrapper operator*(const fmpq_poly_wrapper &a,
                                       const fmpq_poly_wrapper &o)
    {
        fmpq_poly_wrapper r;
        fmpq_poly_mul(*r.get_fmpq_poly_t(), *a.get_fmpq_poly_t(),
                      *o.get_fmpq_poly_t());
        return r;
    }
    friend fmpq_poly_wrapper operator/(const fmpq_poly_wrapper &a,
                                       const fmpq_wrapper &q)
    {
        fmpq_poly_wrapper r;
        fmpq_poly_scalar_div_fmpq(*r.get_fmpq_poly_t(), *a.get_fmpq_poly_t(),
                                  q.get_fmpq_t());
        return r;
    }
    void divrem(fmpq_poly_wrapper &q, fmpq_poly_wrapper &r,
                const fmpq_poly_wrapper &b) const
    {
        return fmpq_poly_divrem(*q.get_fmpq_poly_t(), *r.get_fmpq_poly_t(),
                                *get_fmpq_poly_t(), *b.get_fmpq_poly_t());
    }

    fmpq_poly_wrapper gcd(const fmpq_poly_wrapper &other) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_gcd(*r.get_fmpq_poly_t(), poly, *other.get_fmpq_poly_t());
        return r;
    }
    fmpq_poly_wrapper lcm(const fmpq_poly_wrapper &other) const
    {
        fmpq_poly_wrapper r;
        fmpq_poly_lcm(*r.get_fmpq_poly_t(), poly, *other.get_fmpq_poly_t());
        return r;
    }
    void operator+=(const fmpq_poly_wrapper &o)
    {
        fmpq_poly_add(poly, poly, *o.get_fmpq_poly_t());
    }
    void operator-=(const fmpq_poly_wrapper &o)
    {
        fmpq_poly_sub(poly, poly, *o.get_fmpq_poly_t());
    }
    void operator*=(const fmpq_poly_wrapper &o)
    {
        fmpq_poly_mul(poly, poly, *o.get_fmpq_poly_t());
    }
};

} // SymEngine

#endif // SYMENGINE_FLINT_WRAPPER_H
