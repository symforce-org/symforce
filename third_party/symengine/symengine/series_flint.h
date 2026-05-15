#ifndef SYMENGINE_SERIES_FLINT_H
#define SYMENGINE_SERIES_FLINT_H

#include <symengine/series.h>
#include <symengine/expression.h>
#include <symengine/symengine_exception.h>

#ifdef HAVE_SYMENGINE_FLINT
#include <symengine/flint_wrapper.h>

namespace SymEngine
{

using fqp_t = fmpq_poly_wrapper;
// Univariate Rational Coefficient Power SeriesBase using Flint
class URatPSeriesFlint
    : public SeriesBase<fqp_t, fmpq_wrapper, URatPSeriesFlint>
{
public:
    URatPSeriesFlint(const fqp_t p, const std::string varname,
                     const unsigned degree);
    IMPLEMENT_TYPEID(SYMENGINE_URATPSERIESFLINT)
    int compare(const Basic &o) const override;
    hash_t __hash__() const override;
    RCP<const Basic> as_basic() const override;
    umap_int_basic as_dict() const override;
    RCP<const Basic> get_coeff(int) const override;

    static RCP<const URatPSeriesFlint>
    series(const RCP<const Basic> &t, const std::string &x, unsigned int prec);
    static fqp_t var(const std::string &s);
    static fqp_t convert(const integer_class &x);
    static fqp_t convert(const rational_class &x);
    static fqp_t convert(const Rational &x);
    static fqp_t convert(const Integer &x);
    static fqp_t convert(const Basic &x);
    static inline fqp_t mul(const fqp_t &s, const fqp_t &r, unsigned prec)
    {
        return s.mullow(r, prec);
    }
    static fqp_t pow(const fqp_t &s, int n, unsigned prec);
    static unsigned ldegree(const fqp_t &s);
    static inline fmpq_wrapper find_cf(const fqp_t &s, const fqp_t &var,
                                       unsigned deg)
    {
        return s.get_coeff(deg);
    }
    static fmpq_wrapper root(fmpq_wrapper &c, unsigned n);
    static fqp_t diff(const fqp_t &s, const fqp_t &var);
    static fqp_t integrate(const fqp_t &s, const fqp_t &var);
    static fqp_t subs(const fqp_t &s, const fqp_t &var, const fqp_t &r,
                      unsigned prec);

    static inline fqp_t series_invert(const fqp_t &s, const fqp_t &var,
                                      unsigned int prec)
    {
        SYMENGINE_ASSERT(not s.get_coeff(0).is_zero());
        return s.inv_series(prec);
    }
    static inline fqp_t series_reverse(const fqp_t &s, const fqp_t &var,
                                       unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero()
                         and not s.get_coeff(1).is_zero());
        return s.revert_series(prec);
    }
    static inline fqp_t series_log(const fqp_t &s, const fqp_t &var,
                                   unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_one());
        return s.log_series(prec);
    }
    static inline fqp_t series_exp(const fqp_t &s, const fqp_t &var,
                                   unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.exp_series(prec);
    }
    static inline fqp_t series_sin(const fqp_t &s, const fqp_t &var,
                                   unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.sin_series(prec);
    }

    static inline fqp_t series_cos(const fqp_t &s, const fqp_t &var,
                                   unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.cos_series(prec);
    }

    static inline fqp_t series_tan(const fqp_t &s, const fqp_t &var,
                                   unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.tan_series(prec);
    }
    static inline fqp_t series_atan(const fqp_t &s, const fqp_t &var,
                                    unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.atan_series(prec);
    }
    static inline fqp_t series_atanh(const fqp_t &s, const fqp_t &var,
                                     unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.atanh_series(prec);
    }
    static inline fqp_t series_asin(const fqp_t &s, const fqp_t &var,
                                    unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.asin_series(prec);
    }
    static inline fqp_t series_asinh(const fqp_t &s, const fqp_t &var,
                                     unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.asinh_series(prec);
    }
    static inline fqp_t series_acos(const fqp_t &s, const fqp_t &var,
                                    unsigned int prec)
    {
        throw NotImplementedError("acos() not implemented");
    }
    static inline fqp_t series_sinh(const fqp_t &s, const fqp_t &var,
                                    unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.sinh_series(prec);
    }
    static inline fqp_t series_cosh(const fqp_t &s, const fqp_t &var,
                                    unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.cosh_series(prec);
    }
    static inline fqp_t series_tanh(const fqp_t &s, const fqp_t &var,
                                    unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());
        return s.tanh_series(prec);
    }
    static inline fqp_t series_lambertw(const fqp_t &s, const fqp_t &var,
                                        unsigned int prec)
    {
        SYMENGINE_ASSERT(s.get_coeff(0).is_zero());

        fqp_t p1;
        p1.set_zero();

        auto steps = step_list(prec);
        for (const auto step : steps) {
            const fqp_t e(series_exp(p1, var, step));
            const fqp_t p2(mul(e, p1, step) - s);
            const fqp_t p3(
                series_invert(mul(e, fqp_t(p1 + fqp_t(1)), step), var, step));
            p1 -= mul(p2, p3, step);
        }
        return p1;
    }

    static inline fqp_t series_nthroot(const fqp_t &s, int n, const fqp_t &var,
                                       unsigned int prec)
    {
        fqp_t one;
        one.set_one();
        if (n == 0)
            return one;
        if (n == 1)
            return s;
        if (n == -1)
            return series_invert(s, var, prec);

        const unsigned ldeg = ldegree(s);
        if (ldeg % n != 0) {
            throw NotImplementedError("Puiseux series not implemented.");
        }
        fqp_t ss = s;
        if (ldeg != 0) {
            ss = s * pow(var, -ldeg, prec);
        }
        fmpq_wrapper ct = find_cf(ss, var, 0);
        bool do_inv = false;
        if (n < 0) {
            n = -n;
            do_inv = true;
        }

        fmpq_wrapper ctroot = root(ct, n);
        fqp_t res_p = one, sn = fqp_t(ss / ct);
        auto steps = step_list(prec);
        for (const auto step : steps) {
            fqp_t t = mul(pow(res_p, n + 1, step), sn, step);
            res_p += (res_p - t) / n;
        }
        if (ldeg != 0) {
            res_p *= pow(var, ldeg / n, prec);
        }
        if (do_inv)
            return fqp_t(res_p * ctroot);
        else
            return fqp_t(series_invert(res_p, var, prec) * ctroot);
    }
};
} // namespace SymEngine

#endif // HAVE_SYMENGINE_FLINT

#endif // SYMENGINE_SERIES_FLINT_H
