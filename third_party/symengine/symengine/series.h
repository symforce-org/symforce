/**
 *  \file series.h
 *  Class for univariate series.
 *
 **/
#ifndef SYMENGINE_SERIES_H
#define SYMENGINE_SERIES_H

#include <list>

#include <symengine/integer.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{

class SeriesCoeffInterface : public Number
{
public:
    virtual RCP<const Basic> as_basic() const = 0;
    virtual umap_int_basic as_dict() const = 0;
    virtual RCP<const Basic> get_coeff(int) const = 0;
    virtual unsigned get_degree() const = 0;
    virtual const std::string &get_var() const = 0;
};

template <typename Poly, typename Coeff, typename Series>
class SeriesBase : public SeriesCoeffInterface
{
protected:
    const Poly p_;
    const std::string var_;
    const unsigned degree_;

public:
    inline SeriesBase(Poly p, std::string var, unsigned degree)
        : p_(std::move(p)), var_(var), degree_(degree)
    {
    }
    inline unsigned get_degree() const override
    {
        return degree_;
    }

    inline const std::string &get_var() const override
    {
        return var_;
    }

    inline const Poly &get_poly() const
    {
        return p_;
    }

    inline bool is_zero() const override
    {
        return false;
    }

    inline bool is_one() const override
    {
        return false;
    }

    inline bool is_minus_one() const override
    {
        return false;
    }

    inline bool is_negative() const override
    {
        return false;
    }

    inline bool is_positive() const override
    {
        return false;
    }

    inline bool is_complex() const override
    {
        return false;
    }

    inline bool __eq__(const Basic &o) const override
    {
        return (is_a<Series>(o) and var_ == down_cast<const Series &>(o).var_
                and p_ == down_cast<const Series &>(o).p_
                and degree_ == down_cast<const Series &>(o).degree_);
    }

    RCP<const Number> add(const Number &other) const override
    {
        if (is_a<Series>(other)) {
            const Series &o = down_cast<const Series &>(other);
            auto deg = std::min(degree_, o.degree_);
            if (var_ != o.var_) {
                throw NotImplementedError(
                    "Multivariate Series not implemented");
            }
            return make_rcp<Series>(Poly(p_ + o.p_), var_, deg);
        } else if (other.get_type_code() < Series::type_code_id) {
            Poly p = Series::series(other.rcp_from_this(), var_, degree_)->p_;
            return make_rcp<Series>(Poly(p_ + p), var_, degree_);
        } else {
            return other.add(*this);
        }
    }

    RCP<const Number> mul(const Number &other) const override
    {
        if (is_a<Series>(other)) {
            const Series &o = down_cast<const Series &>(other);
            auto deg = std::min(degree_, o.degree_);
            if (var_ != o.var_) {
                throw NotImplementedError(
                    "Multivariate Series not implemented");
            }
            return make_rcp<Series>(Series::mul(p_, o.p_, deg), var_, deg);
        } else if (other.get_type_code() < Series::type_code_id) {
            Poly p = Series::series(other.rcp_from_this(), var_, degree_)->p_;
            return make_rcp<Series>(Series::mul(p_, p, degree_), var_, degree_);
        } else {
            return other.mul(*this);
        }
    }

    RCP<const Number> pow(const Number &other) const override
    {
        auto deg = degree_;
        Poly p;
        if (is_a<Series>(other)) {
            const Series &o = down_cast<const Series &>(other);
            deg = std::min(deg, o.degree_);
            if (var_ != o.var_) {
                throw NotImplementedError(
                    "Multivariate Series not implemented");
            }
            p = o.p_;
        } else if (is_a<Integer>(other)) {
            if (other.is_negative()) {
                p = Series::pow(
                    p_,
                    (numeric_cast<int>(
                        down_cast<const Integer &>(other).neg()->as_int())),
                    deg);
                p = Series::series_invert(p, Series::var(var_), deg);
                return make_rcp<Series>(p, var_, deg);
            }
            p = Series::pow(
                p_,
                numeric_cast<int>((down_cast<const Integer &>(other).as_int())),
                deg);
            return make_rcp<Series>(p, var_, deg);
        } else if (other.get_type_code() < Series::type_code_id) {
            p = Series::series(other.rcp_from_this(), var_, degree_)->p_;
        } else {
            return other.rpow(*this);
        }
        p = Series::series_exp(
            Poly(p * Series::series_log(p_, Series::var(var_), deg)),
            Series::var(var_), deg);
        return make_rcp<Series>(p, var_, deg);
    }

    RCP<const Number> rpow(const Number &other) const override
    {
        if (other.get_type_code() < Series::type_code_id) {
            Poly p = Series::series(other.rcp_from_this(), var_, degree_)->p_;
            p = Series::series_exp(
                Poly(p_ * Series::series_log(p, Series::var(var_), degree_)),
                Series::var(var_), degree_);
            return make_rcp<Series>(p, var_, degree_);
        } else {
            throw SymEngineException("Unknown type");
        }
    }

    static inline const std::list<unsigned int> &step_list(unsigned int prec)
    {
        static std::list<unsigned int> steps;
        if (not steps.empty()) {
            if (*(steps.rbegin()) == prec)
                return steps;
            else
                steps.clear();
        }

        unsigned int tprec = prec;
        while (tprec > 4) {
            tprec = 2 + tprec / 2;
            steps.push_front(tprec);
        }
        steps.push_front(2);
        steps.push_back(prec);
        return steps;
    }

    static inline Poly series_invert(const Poly &s, const Poly &var,
                                     unsigned int prec)
    {
        if (s == 0)
            throw DivisionByZeroError(
                "Series::series_invert: Division By Zero");
        if (s == 1)
            return Poly(1);
        const int ldeg = Series::ldegree(s);
        const Coeff co = Series::find_cf(s, var, ldeg);
        Poly p(1 / co), ss = s;
        if (ldeg != 0) {
            ss = s * Series::pow(var, -ldeg, prec);
        }
        auto steps = step_list(prec);
        for (const auto step : steps) {
            p = Series::mul(2 - Series::mul(p, ss, step), p, step);
        }
        if (ldeg != 0) {
            return p * Series::pow(var, -ldeg, prec);
        } else {
            return p;
        }
    }

    static inline Poly series_reverse(const Poly &s, const Poly &var,
                                      unsigned int prec)
    {
        const Coeff co = Series::find_cf(s, var, 0);
        if (co != 0)
            throw SymEngineException("reversion of series with constant term");
        const Coeff a = Series::find_cf(s, var, 1);
        if (a == 0)
            throw SymEngineException(
                "reversion of series with zero term of degree one");
        Poly r(var);
        r /= a;
        for (unsigned int i = 2; i < prec; i++) {
            Poly sp = Series::subs(s, var, r, i + 1);
            r -= Series::pow(var, i, i + 1) * Series::find_cf(sp, var, i) / a;
        }
        return r;
    }

    static inline Poly series_nthroot(const Poly &s, int n, const Poly &var,
                                      unsigned int prec)
    {
        if (n == 0)
            return Poly(1);
        if (n == 1)
            return s;
        if (n == -1)
            return Series::series_invert(s, var, prec);

        const int ldeg = Series::ldegree(s);
        if (ldeg % n != 0) {
            throw NotImplementedError("Puiseux series not implemented.");
        }
        Poly ss = s;
        if (ldeg != 0) {
            ss = s * Series::pow(var, -ldeg, prec);
        }
        Coeff ct = Series::find_cf(ss, var, 0);
        bool do_inv = false;
        if (n < 0) {
            n = -n;
            do_inv = true;
        }

        Coeff ctroot = Series::root(ct, n);
        Poly res_p(1), sn = ss / ct;
        auto steps = step_list(prec);
        for (const auto step : steps) {
            Poly t = Series::mul(Series::pow(res_p, n + 1, step), sn, step);
            res_p += (res_p - t) / n;
        }
        if (ldeg != 0) {
            res_p *= Series::pow(var, ldeg / n, prec);
        }
        if (do_inv)
            return res_p / ctroot;
        else
            return Series::series_invert(res_p, var, prec) * ctroot;
    }

    static inline Poly series_atan(const Poly &s, const Poly &var,
                                   unsigned int prec)
    {
        Poly res_p(0);
        if (s == 0)
            return res_p;

        if (s == var) {
            //! fast atan(x)
            int sign = 1;
            Poly monom(var), vsquare(var * var);
            for (unsigned int i = 1; i < prec; i += 2, sign *= -1) {
                res_p += monom * (Coeff(sign) / Coeff(i));
                monom *= vsquare;
            }
            return res_p;
        }
        const Coeff c(Series::find_cf(s, var, 0));
        const Poly p(Series::pow(s, 2, prec - 1) + 1);
        res_p = Series::mul(Series::diff(s, var),
                            Series::series_invert(p, var, prec - 1), prec - 1);

        if (c == 0) {
            // atan(s) = integrate(diff(s)*(1+s**2))
            return Series::integrate(res_p, var);
        } else {
            return Series::integrate(res_p, var) + Series::atan(c);
        }
    }

    static inline Poly series_tan(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        Poly res_p(0), ss = s;
        const Coeff c(Series::find_cf(s, var, 0));
        if (c != 0) {
            ss = s - c;
        }

        // IDEA: use this to get tan(x) coefficients:
        //    # n -> [0, a(1), a(2), ..., a(n)] for n > 0.
        //    def A000182_list(n):
        //    ....T = [0 for i in range(1, n+2)]
        //    ....T[1] = 1
        //    ....for k in range(2, n+1):
        //    ........T[k] = (k-1)*T[k-1]
        //    ....for k in range(2, n+1):
        //    ........for j in range(k, n+1):
        //    ............T[j] = (j-k)*T[j-1]+(j-k+2)*T[j]
        //    ....return T
        //  Ref.: https://oeis.org/A000182

        auto steps = step_list(prec);
        for (const auto step : steps) {
            Poly t = Series::pow(res_p, 2, step) + 1;
            res_p += Series::mul(ss - Series::series_atan(res_p, var, step), t,
                                 step);
        }

        if (c == 0) {
            return res_p;
        } else {
            return Series::mul(
                res_p + Series::tan(c),
                Series::series_invert(1 + res_p * (-Series::tan(c)), var, prec),
                prec);
        }
    }

    static inline Poly series_cot(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        return Series::series_invert(Series::series_tan(s, var, prec), var,
                                     prec);
    }

    static inline Poly _series_sin(const Poly &s, unsigned int prec)
    {
        Poly res_p(0), monom(s);
        Poly ssquare = Series::mul(s, s, prec);
        Coeff prod(1);
        for (unsigned int i = 0; i < prec / 2; i++) {
            const int j = 2 * i + 1;
            if (i != 0)
                prod /= 1 - j;
            prod /= j;
            res_p += Series::mul(monom, Poly(prod), prec);
            monom = Series::mul(monom, ssquare, prec);
        }
        return res_p;
    }

    static inline Poly series_sin(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        if (c != 0) {
            const Poly t = s - c;
            return Series::cos(c) * _series_sin(t, prec)
                   + Series::sin(c) * _series_cos(t, prec);
        } else {
            return _series_sin(s, prec);
        }

        //        if (c == 0) {
        //            // return 2*t/(1+t**2)
        //            const Poly t(Series::series_tan(s / 2, var, prec));     //
        //            t = tan(s/2);
        //            const Poly t2(Series::pow(t, 2, prec));
        //            return Series::series_invert(t2 + 1, var, prec) * t * 2;
        //        } else {
        //            const Poly t(Series::series_tan((s - c) / 2, var, prec));
        //            // t = tan(s/2);
        //            const Poly t2(Series::pow(t, 2, prec));
        //            // return sin(c)*cos(s) + cos(c)*sin(s)
        //            return (-Series::sin(c)) * (t2 - 1) *
        //            Series::series_invert(t2 + 1, var, prec)
        //                + (Series::cos(c) * 2) * t * Series::series_invert(t2
        //                + 1, var, prec);
        //        }
    }

    static inline Poly series_csc(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        return Series::series_invert(Series::series_sin(s, var, prec), var,
                                     prec);
    }

    static inline Poly series_asin(const Poly &s, const Poly &var,
                                   unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));

        // asin(s) = integrate(sqrt(1/(1-s**2))*diff(s))
        const Poly t(1 - Series::pow(s, 2, prec - 1));
        const Poly res_p(Series::integrate(
            Series::diff(s, var) * Series::series_nthroot(t, -2, var, prec - 1),
            var));

        if (c != 0) {
            return res_p + Series::asin(c);
        } else {
            return res_p;
        }
    }

    static inline Poly series_acos(const Poly &s, const Poly &var,
                                   unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        return Series::acos(c) - series_asin(s - c, var, prec);
    }

    static inline Poly _series_cos(const Poly &s, unsigned int prec)
    {
        Poly res_p(1);
        Poly ssquare = Series::mul(s, s, prec);
        Poly monom(ssquare);
        Coeff prod(1);
        for (unsigned int i = 1; i <= prec / 2; i++) {
            const int j = 2 * i;
            if (i != 0)
                prod /= 1 - j;
            prod /= j;
            res_p += Series::mul(monom, Poly(prod), prec);
            monom = Series::mul(monom, ssquare, prec);
        }
        return res_p;
    }

    static inline Poly series_cos(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        if (c != 0) {
            const Poly t = s - c;
            return Series::cos(c) * _series_cos(t, prec)
                   - Series::sin(c) * _series_sin(t, prec);
        } else {
            return _series_cos(s, prec);
        }
        //
        //        if (c == 0) {
        //            // return (1-t**2)/(1+t**2)
        //            const Poly t(Series::series_tan(s / 2, var, prec));     //
        //            t = tan(s/2);
        //            const Poly t2(Series::pow(t, 2, prec));
        //            return Series::series_invert(t2 + 1, var, prec) * ((t2 -
        //            1) * -1);
        //        } else {
        //            const Poly t(Series::series_tan((s - c)/ 2, var, prec));
        //            // t = tan(s/2);
        //            const Poly t2(Series::pow(t, 2, prec));
        //            // return cos(c)*cos(s) - sin(c)*sin(s)
        //            return (-Series::cos(c)) * (t2 - 1) *
        //            Series::series_invert(t2 + 1, var, prec)
        //                - Series::sin(c) * 2 * t * Series::series_invert(t2 +
        //                1, var, prec);
        //        }
    }

    static inline Poly series_sec(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        return Series::series_invert(Series::series_cos(s, var, prec), var,
                                     prec);
    }

    static inline Poly series_log(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        Poly res_p(0);
        if (s == 1)
            return res_p;
        if (s == var + 1) {
            //! fast log(1+x)
            Poly monom(var);
            for (unsigned int i = 1; i < prec; i++) {
                res_p += monom * Coeff(((i % 2) == 0) ? -1 : 1) / Coeff(i);
                monom *= var;
            }
            return res_p;
        }

        const Coeff c(Series::find_cf(s, var, 0));
        res_p = Series::mul(Series::diff(s, var),
                            Series::series_invert(s, var, prec), prec - 1);
        res_p = Series::integrate(res_p, var);

        if (c != 1) {
            res_p += Series::log(c);
        }
        return res_p;
    }

    static inline Poly series_exp(const Poly &s, const Poly &var,
                                  unsigned int prec)
    {
        Poly res_p(1);
        if (s == 0)
            return res_p;

        if (s == var) {
            //! fast exp(x)
            Coeff coef(1);
            Poly monom(var);
            for (unsigned int i = 1; i < prec; i++) {
                coef /= i;
                res_p += monom * coef;
                monom *= var;
            }
            return res_p;
        }

        const Coeff c(Series::find_cf(s, var, 0));
        Poly t = s + 1;
        if (c != 0) {
            // exp(s) = exp(c)*exp(s-c)
            t = s - c + 1;
        }
        auto steps = step_list(prec);
        for (const auto step : steps) {
            res_p = Series::mul(res_p, t - Series::series_log(res_p, var, step),
                                step);
        }
        if (c != 0) {
            return res_p * Series::exp(c);
        } else {
            return res_p;
        }
    }

    static inline Poly series_lambertw(const Poly &s, const Poly &var,
                                       unsigned int prec)
    {
        if (Series::find_cf(s, var, 0) != 0)
            throw NotImplementedError("lambertw(const) not Implemented");

        Poly p1(0);

        auto steps = step_list(prec);
        for (const auto step : steps) {
            const Poly e(Series::series_exp(p1, var, step));
            const Poly p2(Series::mul(e, p1, step) - s);
            const Poly p3(Series::series_invert(Series::mul(e, (p1 + 1), step),
                                                var, step));
            p1 -= Series::mul(p2, p3, step);
        }
        return p1;
    }

    static inline Poly series_sinh(const Poly &s, const Poly &var,
                                   unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        const Poly p1(Series::series_exp(s - c, var, prec));
        const Poly p2(Series::series_invert(p1, var, prec));

        if (c == 0) {
            return (p1 - p2) / 2;
        } else {
            return Series::cosh(c) * (p1 - p2) / 2
                   + Series::sinh(c) * (p1 + p2) / 2;
        }
    }

    static inline Poly series_cosh(const Poly &s, const Poly &var,
                                   unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        const Poly p1(Series::series_exp(s - c, var, prec));
        const Poly p2(Series::series_invert(p1, var, prec));

        if (c == 0) {
            return (p1 + p2) / 2;
        } else {
            return Series::cosh(c) * (p1 + p2) / 2
                   + Series::sinh(c) * (p1 - p2) / 2;
        }
    }

    static inline Poly series_atanh(const Poly &s, const Poly &var,
                                    unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        const Poly p(1 - Series::pow(s, 2, prec - 1));
        const Poly res_p(Series::mul(Series::diff(s, var),
                                     Series::series_invert(p, var, prec - 1),
                                     prec - 1));

        if (c == 0) {
            return Series::integrate(res_p, var);
        } else {
            return Series::integrate(res_p, var) + Series::atanh(c);
        }
    }

    static inline Poly series_asinh(const Poly &s, const Poly &var,
                                    unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));

        const Poly p(Series::series_nthroot(Series::pow(s, 2, prec - 1) + 1, 2,
                                            var, prec - 1));
        const Poly res_p(Series::diff(s, var)
                         * Series::series_invert(p, var, prec - 1));

        if (c == 0) {
            return Series::integrate(res_p, var);
        } else {
            return Series::integrate(res_p, var) + Series::asinh(c);
        }
    }

    static inline Poly series_tanh(const Poly &s, const Poly &var,
                                   unsigned int prec)
    {
        const Coeff c(Series::find_cf(s, var, 0));
        Poly res_p(s);
        if (c != 0) {
            res_p -= c;
        }
        Poly s_(res_p);
        auto steps = step_list(prec);
        for (const auto step : steps) {
            const Poly p(s_ - Series::series_atanh(res_p, var, step));
            res_p += Series::mul(-p, Series::pow(res_p, 2, step) - 1, step);
        }
        if (c != 0) {
            return (res_p + Series::tanh(c))
                   * Series::series_invert(1 + Series::tanh(c) * res_p, var,
                                           prec);
        } else {
            return res_p;
        }
    }

    static inline Coeff sin(const Coeff &c)
    {
        throw NotImplementedError("sin(const) not implemented");
    }
    static inline Coeff cos(const Coeff &c)
    {
        throw NotImplementedError("cos(const) not implemented");
    }
    static inline Coeff tan(const Coeff &c)
    {
        throw NotImplementedError("tan(const) not implemented");
    }
    static inline Coeff asin(const Coeff &c)
    {
        throw NotImplementedError("asin(const) not implemented");
    }
    static inline Coeff acos(const Coeff &c)
    {
        throw NotImplementedError("acos(const) not implemented");
    }
    static inline Coeff atan(const Coeff &c)
    {
        throw NotImplementedError("atan(const) not implemented");
    }
    static inline Coeff sinh(const Coeff &c)
    {
        throw NotImplementedError("sinh(const) not implemented");
    }
    static inline Coeff cosh(const Coeff &c)
    {
        throw NotImplementedError("cosh(const) not implemented");
    }
    static inline Coeff tanh(const Coeff &c)
    {
        throw NotImplementedError("tanh(const) not implemented");
    }
    static inline Coeff asinh(const Coeff &c)
    {
        throw NotImplementedError("asinh(const) not implemented");
    }
    static inline Coeff atanh(const Coeff &c)
    {
        throw NotImplementedError("atanh(const) not implemented");
    }
    static inline Coeff exp(const Coeff &c)
    {
        throw NotImplementedError("exp(const) not implemented");
    }
    static inline Coeff log(const Coeff &c)
    {
        throw NotImplementedError("log(const) not implemented");
    }
};

RCP<const SeriesCoeffInterface> series(const RCP<const Basic> &ex,
                                       const RCP<const Symbol> &var,
                                       unsigned int prec);

RCP<const SeriesCoeffInterface> series_invfunc(const RCP<const Basic> &ex,
                                               const RCP<const Symbol> &var,
                                               unsigned int prec);

} // namespace SymEngine
#endif
