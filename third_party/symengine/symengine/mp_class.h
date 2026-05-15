#ifndef SYMENGINE_INTEGER_CLASS_H
#define SYMENGINE_INTEGER_CLASS_H

#include <symengine/symengine_config.h>
#include <symengine/symengine_casts.h>
#include <cstring>
#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
#include <symengine/mp_wrapper.h>
#endif

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <symengine/symengine_rcp.h>
#include <ios>
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_PIRANHA
#include <piranha/mp_integer.hpp>
#include <piranha/mp_rational.hpp>
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_GMPXX
#define __GMPXX_USE_CXX11 1
#include <gmpxx.h>
#endif

namespace SymEngine
{
/*
 * integer_class and rational_class are the storage classes for
 * SymEngine::Integer
 * and SymEngine::Rational respectively. There are 4 choices for the
 * integer_class.
 * mpz_class from libgmpxx, piranha::integer from piranha, mpz_wrapper which
 * wraps
 * mpz_t from libgmp and fmpz_wrapper which wraps fmpz_t from libflint. This
 * choice
 * is made at compile time with SYMENGINE_INTEGER_CLASS.
 *
 * Each of these classes has to have all the arithmetic operators overloaded
 * with
 * operands of these classes and C++ integer types except long long. Also, shift
 * operators, move operators, string, integer and mpz_t constructors are also
 * required.
 *
 * To add a new type, several non-member functions need to be defined for the
 * new
 * type. See mpz_wrapper implementation for all the non-member functions that
 * need
 * to be defined.
 */

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP
typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>,
                                      boost::multiprecision::et_off>
    integer_class;
typedef boost::multiprecision::number<
    boost::multiprecision::cpp_rational_backend, boost::multiprecision::et_off>
    rational_class;
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_PIRANHA
typedef piranha::integer integer_class;
typedef piranha::rational rational_class;
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT
typedef fmpz_wrapper integer_class;
typedef fmpq_wrapper rational_class;
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP
typedef mpz_wrapper integer_class;
typedef mpq_wrapper rational_class;
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_GMPXX
typedef mpz_class integer_class;
typedef mpq_class rational_class;
#endif

// needs to be in a separate namespace to import the literals.
// eg: using namespace SymEngine::literals;
inline namespace literals
{
//! Literal for creating multiple precision integers
inline integer_class operator"" _z(const char *str)
{
    return integer_class(str);
}

inline rational_class operator"" _q(const char *str)
{
    return rational_class(integer_class(str));
}
} // namespace literals

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_GMPXX                                 \
    || SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP
// Helper functions for mpz
inline integer_class mp_abs(const integer_class &i)
{
    integer_class res;
    mpz_abs(res.get_mpz_t(), i.get_mpz_t());
    return res;
}

inline int mp_sign(const integer_class &i)
{
    return mpz_sgn(i.get_mpz_t());
}

inline integer_class mp_sqrt(const integer_class &i)
{
    integer_class res;
    mpz_sqrt(res.get_mpz_t(), i.get_mpz_t());
    return res;
}

inline double mp_get_d(const integer_class &i)
{
    return static_cast<double>(i.get_d());
}

inline void mp_set_d(integer_class &i, double a)
{
    mpz_set_d(i.get_mpz_t(), a);
}

inline void mp_set_str(integer_class &i, const std::string &a)
{
    mpz_set_str(i.get_mpz_t(), a.c_str(), 0);
}

inline std::string mp_get_hex_str(const integer_class &i)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = mpz_get_str(NULL, 16, i.get_mpz_t());
    std::string r = std::string(c);
    freefunc(c, strlen(c) + 1);
    return r;
}

inline void mp_demote(integer_class &i) {}

inline bool mp_fits_ulong_p(const integer_class &i)
{
    return i.fits_ulong_p() != 0;
}

inline bool mp_fits_slong_p(const integer_class &i)
{
    return i.fits_slong_p() != 0;
}

inline unsigned long mp_get_ui(const integer_class &i)
{
    return i.get_ui();
}

inline long mp_get_si(const integer_class &i)
{
    return i.get_si();
}

inline mpz_srcptr get_mpz_t(const integer_class &i)
{
    return i.get_mpz_t();
}

inline mpz_ptr get_mpz_t(integer_class &i)
{
    return i.get_mpz_t();
}

inline void mp_pow_ui(integer_class &res, const integer_class &i,
                      unsigned long n)
{
    mpz_pow_ui(res.get_mpz_t(), i.get_mpz_t(), n);
}

inline void mp_pow_ui(rational_class &res, const rational_class &i,
                      unsigned long n)
{
    mpz_pow_ui(res.get_num().get_mpz_t(), i.get_num().get_mpz_t(), n);
    mpz_pow_ui(res.get_den().get_mpz_t(), i.get_den().get_mpz_t(), n);
}

inline void mp_powm(integer_class &res, const integer_class &a,
                    const integer_class &b, const integer_class &m)
{
    mpz_powm(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t(), m.get_mpz_t());
}

inline bool mp_invert(integer_class &res, const integer_class &a,
                      const integer_class &m)
{
    return mpz_invert(res.get_mpz_t(), a.get_mpz_t(), m.get_mpz_t()) != 0;
}

inline void mp_gcd(integer_class &res, const integer_class &a,
                   const integer_class &b)
{
    mpz_gcd(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_gcdext(integer_class &res, integer_class &r, integer_class &s,
                      const integer_class &a, const integer_class &b)
{
    mpz_gcdext(res.get_mpz_t(), r.get_mpz_t(), s.get_mpz_t(), a.get_mpz_t(),
               b.get_mpz_t());
}

inline void mp_and(integer_class &res, const integer_class &a,
                   const integer_class &b)
{
    mpz_and(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_fdiv_r(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    mpz_fdiv_r(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_fdiv_q(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    mpz_fdiv_q(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_cdiv_q(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    mpz_cdiv_q(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_tdiv_q(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    mpz_tdiv_q(res.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_fdiv_qr(integer_class &q, integer_class &r,
                       const integer_class &a, const integer_class &b)
{
    mpz_fdiv_qr(q.get_mpz_t(), r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_divexact(integer_class &q, const integer_class &a,
                        const integer_class &b)
{
    mpz_divexact(q.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_lcm(integer_class &q, const integer_class &a,
                   const integer_class &b)
{
    mpz_lcm(q.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_tdiv_qr(integer_class &q, integer_class &r,
                       const integer_class &a, const integer_class &b)
{
    mpz_tdiv_qr(q.get_mpz_t(), r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

inline void mp_addmul(integer_class &r, const integer_class &a,
                      const integer_class &b)
{
    mpz_addmul(r.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
}

// Helper functions for mpq
inline const integer_class &get_den(const rational_class &i)
{
    return i.get_den();
}

inline const integer_class &get_num(const rational_class &i)
{
    return i.get_num();
}

inline integer_class &get_den(rational_class &i)
{
    return i.get_den();
}

inline integer_class &get_num(rational_class &i)
{
    return i.get_num();
}

inline mpq_srcptr get_mpq_t(const rational_class &i)
{
    return i.get_mpq_t();
}

inline void canonicalize(rational_class &i)
{
    i.canonicalize();
}

inline double mp_get_d(const rational_class &i)
{
    return i.get_d();
}

inline int mp_sign(const rational_class &i)
{
    return mpq_sgn(i.get_mpq_t());
}

inline rational_class mp_abs(const rational_class &i)
{
    rational_class res;
    mpq_abs(res.get_mpq_t(), i.get_mpq_t());
    return res;
}

inline integer_class mp_primorial(unsigned long n)
{
    integer_class res;
    mpz_primorial_ui(res.get_mpz_t(), n);
    return res;
}

#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_PIRANHA
// Helper functions for piranha::integer
inline piranha::integer mp_abs(const piranha::integer &i)
{
    return i.abs();
}

inline piranha::integer mp_sqrt(const piranha::integer &i)
{
    return i.sqrt();
}

inline void mp_demote(piranha::integer &i) {}

inline mpz_ptr get_mpz_t(piranha::integer &i)
{
    return i._get_mpz_ptr();
}

inline auto get_mpz_t(const piranha::integer &i) -> decltype(i.get_mpz_view())
{
    return i.get_mpz_view();
}

inline void mp_pow_ui(piranha::integer &res, const piranha::integer &i,
                      unsigned long n)
{
    res = i.pow(n);
}

inline void mp_pow_ui(piranha::rational &res, const piranha::rational &i,
                      unsigned long n)
{
    res = i.pow(n);
}

inline void mp_powm(piranha::integer &res, const piranha::integer &a,
                    const piranha::integer &b, const piranha::integer &m)
{
    auto _res = get_mpz_t(res);
    mpz_powm(_res, get_mpz_t(a), get_mpz_t(b), get_mpz_t(m));
}

inline bool mp_invert(piranha::integer &res, const piranha::integer &a,
                      const piranha::integer &m)
{
    auto _res = get_mpz_t(res);
    return mpz_invert(_res, get_mpz_t(a), get_mpz_t(m)) != 0;
}

inline void mp_gcd(piranha::integer &g, const piranha::integer &a,
                   const piranha::integer &b)
{
    piranha::integer::gcd(g, a, b);
}

inline void mp_gcdext(piranha::integer &g, piranha::integer &r,
                      piranha::integer &s, const piranha::integer &a,
                      const piranha::integer &b)
{
    auto _g = get_mpz_t(g);
    auto _r = get_mpz_t(r);
    auto _s = get_mpz_t(s);
    mpz_gcdext(_g, _r, _s, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_and(piranha::integer &res, const piranha::integer &a,
                   const piranha::integer &b)
{
    auto _res = get_mpz_t(res);
    mpz_and(_res, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_fdiv_r(piranha::integer &res, const piranha::integer &a,
                      const piranha::integer &b)
{
    auto _res = get_mpz_t(res);
    mpz_fdiv_r(_res, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_fdiv_q(piranha::integer &res, const piranha::integer &a,
                      const piranha::integer &b)
{
    auto _res = get_mpz_t(res);
    mpz_fdiv_q(_res, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_cdiv_q(piranha::integer &res, const piranha::integer &a,
                      const piranha::integer &b)
{
    auto _res = get_mpz_t(res);
    mpz_cdiv_q(_res, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_tdiv_q(piranha::integer &res, const piranha::integer &a,
                      const piranha::integer &b)
{
    auto _res = get_mpz_t(res);
    mpz_tdiv_q(_res, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_fdiv_qr(piranha::integer &q, piranha::integer &r,
                       const piranha::integer &a, const piranha::integer &b)
{
    auto _q = get_mpz_t(q);
    mpz_fdiv_qr(_q, get_mpz_t(r), get_mpz_t(a), get_mpz_t(b));
}

inline void mp_divexact(piranha::integer &q, const piranha::integer &a,
                        const piranha::integer &b)
{
    piranha::integer::_divexact(q, a, b);
}

inline void mp_lcm(piranha::integer &q, const piranha::integer &a,
                   const piranha::integer &b)
{
    auto _q = get_mpz_t(q);
    mpz_lcm(_q, get_mpz_t(a), get_mpz_t(b));
}

inline void mp_tdiv_qr(piranha::integer &q, piranha::integer &r,
                       const piranha::integer &a, const piranha::integer &b)
{
    piranha::integer::divrem(q, r, a, b);
}

inline int mp_sign(const piranha::integer &i)
{
    return i.sign();
}

inline long mp_get_si(const piranha::integer &i)
{
    return mpz_get_si(i.get_mpz_view());
}

inline unsigned long mp_get_ui(const piranha::integer &i)
{
    return mpz_get_ui(i.get_mpz_view());
}

inline double mp_get_d(const piranha::integer &i)
{
    return mpz_get_d(i.get_mpz_view());
}

inline void mp_set_d(piranha::integer &i, double a)
{
    i = a;
}

inline void mp_set_str(integer_class &i, const std::string &a)
{
    mpz_t m;
    mpz_init(m);
    mpz_set_str(m, a.c_str(), 0);
    i = integer_class(m);
}

inline std::string mp_get_hex_str(const integer_class &i)
{

    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = mpz_get_str(NULL, 16, i.get_mpz_view());
    std::string r = std::string(c);
    freefunc(c, strlen(c) + 1);
    return r;
}

inline bool mp_fits_ulong_p(const piranha::integer &i)
{
    return mpz_fits_ulong_p(i.get_mpz_view()) != 0;
}

inline bool mp_fits_slong_p(const piranha::integer &i)
{
    return mpz_fits_slong_p(i.get_mpz_view()) != 0;
}

inline void mp_addmul(integer_class &r, const integer_class &a,
                      const integer_class &b)
{
    piranha::math::multiply_accumulate(r, a, b);
}

inline integer_class mp_primorial(unsigned long n)
{
    integer_class res;
    mpz_primorial_ui(get_mpz_t(res), n);
    return res;
}

// Helper functions for piranha::rational

inline piranha::rational mp_abs(const piranha::rational &i)
{
    return i.abs();
}

inline const piranha::integer &get_den(const piranha::rational &i)
{
    return i.den();
}

inline const piranha::integer &get_num(const piranha::rational &i)
{
    return i.num();
}

inline piranha::integer &get_den(piranha::rational &i)
{
    return i._den();
}

inline piranha::integer &get_num(piranha::rational &i)
{
    return i._num();
}

inline void canonicalize(piranha::rational &i)
{
    i.canonicalise();
}

inline double mp_get_d(const piranha::rational &i)
{
    return mpq_get_d(i.get_mpq_view().get());
}

inline auto get_mpq_t(const piranha::rational &i) -> decltype(i.get_mpq_view())
{
    return i.get_mpq_view();
}

inline int mp_sign(const piranha::rational &i)
{
    return i.num().sign();
}
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT

// helper functions for fmpz

inline mpz_view_flint get_mpz_t(const fmpz_wrapper &i)
{
    return mpz_view_flint(i);
}

inline mpz_ptr get_mpz_t(fmpz_wrapper &i)
{
    return _fmpz_promote_val(i.get_fmpz_t());
}

inline void mp_demote(fmpz_wrapper &i)
{
    _fmpz_demote_val(i.get_fmpz_t());
}

inline int mp_sign(const fmpz_wrapper &i)
{
    return fmpz_sgn(i.get_fmpz_t());
}

inline long mp_get_si(const fmpz_wrapper &i)
{
    return fmpz_get_si(i.get_fmpz_t());
}

inline unsigned long mp_get_ui(const fmpz_wrapper &i)
{
    return fmpz_get_ui(i.get_fmpz_t());
}

inline bool mp_fits_slong_p(const fmpz_wrapper &i)
{
    return fmpz_fits_si(i.get_fmpz_t());
}

inline bool mp_fits_ulong_p(const fmpz_wrapper &i)
{
    return fmpz_sgn(i.get_fmpz_t()) >= 0 && fmpz_abs_fits_ui(i.get_fmpz_t());
}

inline double mp_get_d(const fmpz_wrapper &i)
{
    return fmpz_get_d(i.get_fmpz_t());
}

inline void mp_set_d(fmpz_wrapper &i, double a)
{
    return fmpz_set_d(i.get_fmpz_t(), a);
}

inline void mp_set_str(fmpz_wrapper &i, const std::string &a)
{
    fmpz_set_str(i.get_fmpz_t(), a.c_str(), 0);
}

inline std::string mp_get_hex_str(const fmpz_wrapper &i)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = fmpz_get_str(NULL, 16, i.get_fmpz_t());
    std::string r = std::string(c);
    freefunc(c, strlen(c) + 1);
    return r;
}

inline fmpz_wrapper mp_abs(const fmpz_wrapper &i)
{
    fmpz_wrapper res;
    fmpz_abs(res.get_fmpz_t(), i.get_fmpz_t());
    return res;
}

inline fmpz_wrapper mp_sqrt(const fmpz_wrapper &i)
{
    fmpz_wrapper res;
    fmpz_sqrt(res.get_fmpz_t(), i.get_fmpz_t());
    return res;
}

inline void mp_pow_ui(fmpz_wrapper &res, const fmpz_wrapper &i, unsigned long n)
{
    fmpz_pow_ui(res.get_fmpz_t(), i.get_fmpz_t(), n);
}

inline void mp_pow_ui(fmpq_wrapper &res, const fmpq_wrapper &i, unsigned long n)
{
    fmpq_pow_si(res.get_fmpq_t(), i.get_fmpq_t(), n);
}

inline void mp_powm(fmpz_wrapper &res, const fmpz_wrapper &a,
                    const fmpz_wrapper &b, const fmpz_wrapper &m)
{
    if (b >= 0) {
        fmpz_powm(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t(),
                  m.get_fmpz_t());
    } else {
        fmpz_neg(res.get_fmpz_t(), b.get_fmpz_t());
        fmpz_powm(res.get_fmpz_t(), a.get_fmpz_t(), res.get_fmpz_t(),
                  m.get_fmpz_t());
        fmpz_invmod(res.get_fmpz_t(), res.get_fmpz_t(), m.get_fmpz_t());
    }
}

inline bool mp_invert(fmpz_wrapper &res, const fmpz_wrapper &a,
                      const fmpz_wrapper &m)
{
    return fmpz_invmod(res.get_fmpz_t(), a.get_fmpz_t(), m.get_fmpz_t()) != 0;
}

inline void mp_gcd(fmpz_wrapper &res, const fmpz_wrapper &a,
                   const fmpz_wrapper &b)
{
    fmpz_gcd(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_gcdext(fmpz_wrapper &g, fmpz_wrapper &r, fmpz_wrapper &s,
                      const fmpz_wrapper &a, const fmpz_wrapper &b)
{
    fmpz_xgcd(g.get_fmpz_t(), r.get_fmpz_t(), s.get_fmpz_t(), a.get_fmpz_t(),
              b.get_fmpz_t());
}

inline void mp_and(fmpz_wrapper &res, const fmpz_wrapper &a,
                   const fmpz_wrapper &b)
{
    fmpz_and(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_fdiv_r(fmpz_wrapper &res, const fmpz_wrapper &a,
                      const fmpz_wrapper &b)
{
    fmpz_fdiv_r(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_fdiv_q(fmpz_wrapper &res, const fmpz_wrapper &a,
                      const fmpz_wrapper &b)
{
    fmpz_fdiv_q(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_cdiv_q(fmpz_wrapper &res, const fmpz_wrapper &a,
                      const fmpz_wrapper &b)
{
    fmpz_cdiv_q(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_tdiv_q(fmpz_wrapper &res, const fmpz_wrapper &a,
                      const fmpz_wrapper &b)
{
    fmpz_tdiv_q(res.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_fdiv_qr(fmpz_wrapper &q, fmpz_wrapper &r, const fmpz_wrapper &a,
                       const fmpz_wrapper &b)
{
    fmpz_fdiv_qr(q.get_fmpz_t(), r.get_fmpz_t(), a.get_fmpz_t(),
                 b.get_fmpz_t());
}

inline void mp_divexact(fmpz_wrapper &q, const fmpz_wrapper &a,
                        const fmpz_wrapper &b)
{
    fmpz_divexact(q.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_lcm(fmpz_wrapper &q, const fmpz_wrapper &a,
                   const fmpz_wrapper &b)
{
    fmpz_lcm(q.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline void mp_tdiv_qr(fmpz_wrapper &q, fmpz_wrapper &r, const fmpz_wrapper &a,
                       const fmpz_wrapper &b)
{
    fmpz_tdiv_qr(q.get_fmpz_t(), r.get_fmpz_t(), a.get_fmpz_t(),
                 b.get_fmpz_t());
}

inline void mp_addmul(fmpz_wrapper &r, const fmpz_wrapper &a,
                      const fmpz_wrapper &b)
{
    fmpz_addmul(r.get_fmpz_t(), a.get_fmpz_t(), b.get_fmpz_t());
}

inline integer_class mp_primorial(unsigned long n)
{
    fmpz_wrapper res;
    fmpz_primorial(res.get_fmpz_t(), n);
    return res;
}

// helper functions for fmpq

inline const fmpz_wrapper &get_den(const fmpq_wrapper &i)
{
    return i.get_den();
}

inline const fmpz_wrapper &get_num(const fmpq_wrapper &i)
{
    return i.get_num();
}

inline fmpz_wrapper &get_den(fmpq_wrapper &i)
{
    return i.get_den();
}

inline fmpz_wrapper &get_num(fmpq_wrapper &i)
{
    return i.get_num();
}

inline mpq_view_flint get_mpq_t(const fmpq_wrapper &i)
{
    return mpq_view_flint(i);
}

inline void canonicalize(fmpq_wrapper &i)
{
    fmpq_canonicalise(i.get_fmpq_t());
}

inline double mp_get_d(const fmpq_wrapper &i)
{
    return mp_get_d(i.get_num()) / mp_get_d(i.get_den());
}

inline int mp_sign(const fmpq_wrapper &i)
{
    return fmpq_sgn(i.get_fmpq_t());
}

inline fmpq_wrapper mp_abs(const fmpq_wrapper &i)
{
    fmpq_wrapper res;
    fmpq_abs(res.get_fmpq_t(), i.get_fmpq_t());
    return res;
}

#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP

inline integer_class mp_abs(const integer_class &i)
{
    // boost::multiprecision::abs(i) returns
    // an expression template, not a cpp_int
    // but it's ok: cpp_int is constructible from an expression template
    return boost::multiprecision::abs(i);
}

inline int mp_cmpabs(const integer_class &a, const integer_class &b)
{
    if (mp_abs(a) > mp_abs(b)) {
        return 1;
    }
    if (mp_abs(a) == mp_abs(b)) {
        return 0;
    }
    return -1;
}

inline int mp_sign(const integer_class &i)
{
    return boost::math::sign(i);
}

inline double mp_get_d(const integer_class &i)
{
    return i.convert_to<double>();
}

inline void mp_set_d(integer_class &i, double a)
{
    i.assign(a);
}

inline void mp_set_str(integer_class &i, const std::string &a)
{
    i = integer_class(a.c_str());
}

inline std::string mp_get_hex_str(const integer_class &i)
{
    if (mp_sign(i) >= 0) {
        return i.str(0, std::ios_base::hex);
    } else {
        return std::string("-") + (-i).str(0, std::ios_base::hex);
    }
}

inline unsigned long mp_get_ui(const integer_class &i)
{
    return mp_abs(i).convert_to<unsigned long>();
}

inline long mp_get_si(const integer_class &i)
{
    return i.convert_to<long>();
}

inline bool mp_fits_ulong_p(const integer_class &i)
{
    return (i >= 0) && (i <= ULONG_MAX);
}

inline bool mp_fits_slong_p(const integer_class &i)
{
    return (i >= LONG_MIN) && (i <= LONG_MAX);
}

// bitwise and
inline void mp_and(integer_class &res, const integer_class &a,
                   const integer_class &b)
{
    res = boost::multiprecision::operator&(a, b);
}

inline void mp_pow_ui(integer_class &res, const integer_class &i,
                      unsigned long n)
{
    res = boost::multiprecision::pow(i, numeric_cast<unsigned>(n));
}

inline void mp_gcd(integer_class &res, const integer_class &a,
                   const integer_class &b)
{
    res = boost::multiprecision::gcd(a, b);
}

void mp_fdiv_qr(integer_class &q, integer_class &r, const integer_class &a,
                const integer_class &b);

void mp_cdiv_qr(integer_class &q, integer_class &r, const integer_class &a,
                const integer_class &b);

inline void mp_fdiv_r(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    // TODO: benchmark this speed
    integer_class quo;
    mp_fdiv_qr(quo, res, a, b);
}

inline void mp_fdiv_q(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    // TODO:  benchmark this speed
    integer_class rem;
    mp_fdiv_qr(res, rem, a, b);
}

inline void mp_cdiv_q(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    // TODO:  benchmark this speed
    integer_class rem;
    mp_cdiv_qr(res, rem, a, b);
}

inline void mp_tdiv_qr(integer_class &q, integer_class &r,
                       const integer_class &a, const integer_class &b)
{
    boost::multiprecision::divide_qr(a, b, q, r);
}

inline void mp_tdiv_q(integer_class &res, const integer_class &a,
                      const integer_class &b)
{
    integer_class rem;
    mp_tdiv_qr(res, rem, a, b);
}

inline void mp_divexact(integer_class &q, const integer_class &a,
                        const integer_class &b)
{
    // TODO: make faster
    q = a / b;
}

inline void mp_lcm(integer_class &q, const integer_class &a,
                   const integer_class &b)
{
    q = boost::multiprecision::lcm(a, b);
}

inline void mp_addmul(integer_class &r, const integer_class &a,
                      const integer_class &b)
{
    // boost::multiprecision::default_ops::eval_multiply_add(r,a,b);
    // //segfaults.
    r += a * b;
}

// Helper functions for cpp_rational
inline const integer_class get_den(const rational_class &i)
{
    return boost::multiprecision::denominator(i);
}

inline const integer_class get_num(const rational_class &i)
{
    return boost::multiprecision::numerator(i);
}

inline void canonicalize(rational_class &i)
{
    // do nothing; boost::multiprecision::cpp_int
    // is always stored in canonical form
    // numerator and denominator share no common factors
    // denominator is positive.
}

inline double mp_get_d(const rational_class &i)
{
    return i.convert_to<double>();
}

inline int mp_sign(const rational_class &i)
{
    return i.sign();
}

inline rational_class mp_abs(const rational_class &i)
{
    return boost::multiprecision::abs(i);
}

inline bool mp_divisible_p(const integer_class &a, const integer_class &b)
{
    if (b == 0) {
        return (a == 0);
    }
    return a % b == 0;
}

void mp_pow_ui(rational_class &res, const rational_class &i, unsigned long n);

void mp_powm(integer_class &res, const integer_class &a, const integer_class &b,
             const integer_class &m);

/*  Extended Euclidean algorithm in Z
 *  inargs:  integers a, b
 *  outargs:  gcd, the greatest common divisor of a and b
 *            s, t such that sa + tb = gcd
 */
void mp_gcdext(integer_class &gcd, integer_class &s, integer_class &t,
               const integer_class &a, const integer_class &b);

bool mp_invert(integer_class &res, const integer_class &a,
               const integer_class &m);

bool mp_root(integer_class &res, const integer_class &i, unsigned long n);

integer_class mp_sqrt(const integer_class &i);

void mp_rootrem(integer_class &a, integer_class &b, const integer_class &i,
                unsigned long n);

void mp_sqrtrem(integer_class &a, integer_class &b, const integer_class &i);

int mp_probab_prime_p(const integer_class &i, unsigned retries);

void mp_nextprime(integer_class &res, const integer_class &i);

unsigned long mp_scan1(const integer_class &i);

void mp_fib_ui(integer_class &res, unsigned long n);

void mp_fib2_ui(integer_class &a, integer_class &b, unsigned long n);

void mp_lucnum_ui(integer_class &res, unsigned long n);

void mp_lucnum2_ui(integer_class &a, integer_class &b, unsigned long n);

void mp_fac_ui(integer_class &res, unsigned long n);

void mp_bin_ui(integer_class &res, const integer_class &n, unsigned long r);

bool mp_perfect_power_p(const integer_class &i);

bool mp_perfect_square_p(const integer_class &i);

int mp_legendre(const integer_class &a, const integer_class &n);

int mp_jacobi(const integer_class &a, const integer_class &n);

int mp_kronecker(const integer_class &a, const integer_class &n);

integer_class mp_primorial(unsigned long n);

class mp_randstate
{
public:
    // returns a uniformly distributed random integer between 0 and a-1,
    // inclusive
    void urandomint(integer_class &res, const integer_class &a)
    {
        boost::random::uniform_int_distribution<integer_class> ui(0, a);
        res = ui(_twister);
    }

    void seed(const uint32_t &i)
    {
        _twister.seed(i);
    }

private:
    boost::random::mt19937 _twister;
};

#endif // SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP

// The implementation of each of the following
// requires only get_mpz_t(integer_class), mp_demote
// and functions from GMP.  They don't depend on
// any members of the backend class mpz, fmpz, or piranha::integer.
// These functions will all need to be implemented separately
// for a GMP-free build (with boost::multiprecision, for instance)
#if SYMENGINE_INTEGER_CLASS == SYMENGINE_PIRANHA                               \
    || SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT                              \
    || SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP                                \
    || SYMENGINE_INTEGER_CLASS == SYMENGINE_GMPXX

inline bool mp_root(integer_class &res, const integer_class &i, unsigned long n)
{
    auto _res = get_mpz_t(res);
    int ret = mpz_root(_res, get_mpz_t(i), n);
    mp_demote(res);
    return ret != 0;
}

inline void mp_nextprime(integer_class &res, const integer_class &i)
{
    auto _res = get_mpz_t(res);
    mpz_nextprime(_res, get_mpz_t(i));
    mp_demote(res);
}

inline void mp_sqrtrem(integer_class &a, integer_class &b,
                       const integer_class &i)
{
    auto _a = get_mpz_t(a);
    auto _b = get_mpz_t(b);
    mpz_sqrtrem(_a, _b, get_mpz_t(i));
    mp_demote(a);
    mp_demote(b);
}

inline void mp_rootrem(integer_class &a, integer_class &b,
                       const integer_class &i, unsigned long n)
{
    auto _a = get_mpz_t(a);
    auto _b = get_mpz_t(b);
    mpz_rootrem(_a, _b, get_mpz_t(i), n);
    mp_demote(a);
    mp_demote(b);
}

inline unsigned long mp_scan1(const integer_class &i)
{
    return mpz_scan1(get_mpz_t(i), 0);
}

inline void mp_fib_ui(integer_class &res, unsigned long n)
{
    mpz_fib_ui(get_mpz_t(res), n);
    mp_demote(res);
}

inline void mp_fib2_ui(integer_class &a, integer_class &b, unsigned long n)
{
    mpz_fib2_ui(get_mpz_t(a), get_mpz_t(b), n);
    mp_demote(a);
    mp_demote(b);
}

inline void mp_lucnum_ui(integer_class &res, unsigned long n)
{
    mpz_lucnum_ui(get_mpz_t(res), n);
    mp_demote(res);
}

inline void mp_lucnum2_ui(integer_class &a, integer_class &b, unsigned long n)
{
    mpz_lucnum2_ui(get_mpz_t(a), get_mpz_t(b), n);
    mp_demote(a);
    mp_demote(b);
}

inline void mp_bin_ui(integer_class &res, const integer_class &n,
                      unsigned long r)
{
    auto _res = get_mpz_t(res);
    mpz_bin_ui(_res, get_mpz_t(n), r);
    mp_demote(res);
}

inline void mp_fac_ui(integer_class &res, unsigned long n)
{
    mpz_fac_ui(get_mpz_t(res), n);
    mp_demote(res);
}

inline int mp_legendre(const integer_class &a, const integer_class &n)
{
    return mpz_legendre(get_mpz_t(a), get_mpz_t(n));
}

inline int mp_kronecker(const integer_class &a, const integer_class &n)
{
    return mpz_kronecker(get_mpz_t(a), get_mpz_t(n));
}

inline int mp_jacobi(const integer_class &a, const integer_class &n)
{
    return mpz_jacobi(get_mpz_t(a), get_mpz_t(n));
}

inline bool mp_perfect_power_p(const integer_class &i)
{
    return mpz_perfect_power_p(get_mpz_t(i)) != 0;
}

inline bool mp_perfect_square_p(const integer_class &i)
{
    return mpz_perfect_square_p(get_mpz_t(i)) != 0;
}

inline int mp_probab_prime_p(const integer_class &i, unsigned retries)
{
    return mpz_probab_prime_p(get_mpz_t(i), retries);
}

inline bool mp_divisible_p(const integer_class &a, const integer_class &b)
{
    return mpz_divisible_p(get_mpz_t(a), get_mpz_t(b)) != 0;
}

inline void mp_urandomm(integer_class &a, gmp_randstate_t &t,
                        const integer_class &b)
{
    auto _a = get_mpz_t(a);
    mpz_urandomm(_a, t, get_mpz_t(b));
    mp_demote(a);
}

inline auto get_mp_t(const integer_class &x) -> decltype(get_mpz_t(x))
{
    return get_mpz_t(x);
}

inline auto get_mp_t(const rational_class &x) -> decltype(get_mpq_t(x))
{
    return get_mpq_t(x);
}

inline int mp_cmpabs(const integer_class &a, const integer_class &b)
{
    return mpz_cmpabs(get_mpz_t(a), get_mpz_t(b));
}

class mp_randstate
{
public:
    // returns a uniformly distributed random integer between 0 and a-1,
    // inclusive
    void urandomint(integer_class &res, const integer_class &a)
    {
        mpz_urandomm(get_mpz_t(res), _state, get_mpz_t(a));
        mp_demote(res);
    }

    void seed(const integer_class &i)
    {
        gmp_randseed(_state, get_mpz_t(i));
    }

    mp_randstate()
    {
        gmp_randinit_default(_state);
        gmp_randseed_ui(_state, std::rand());
    }

    ~mp_randstate()
    {
        gmp_randclear(_state);
    }

private:
    gmp_randstate_t _state;
};

#endif // SYMENGINE_INTEGER_CLASS == Piranha or Flint or GMP or GMPXX

} // namespace SymEngine

#if !defined(HAVE_SYMENGINE_GMP) && defined(HAVE_SYMENGINE_BOOST)              \
    && BOOST_VERSION < 105900
namespace boost
{
namespace detail
{
template <>
struct make_unsigned_imp<SymEngine::integer_class> {
    typedef SymEngine::integer_class type;
};
} // namespace detail
} // namespace boost
#endif

#endif // SYMENGINE_INTEGER_CLASS_H
