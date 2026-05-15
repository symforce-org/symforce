#ifndef SYMENGINE_SERIES_VISITOR_H
#define SYMENGINE_SERIES_VISITOR_H

#include <symengine/visitor.h>

namespace SymEngine
{

template <typename Poly, typename Coeff, typename Series>
class SeriesVisitor : public BaseVisitor<SeriesVisitor<Poly, Coeff, Series>>
{
private:
    Poly p;
    const Poly var;
    const std::string varname;
    const unsigned prec;

public:
    inline SeriesVisitor(const Poly &var_, const std::string &varname_,
                         const unsigned prec_)
        : var(var_), varname(varname_), prec(prec_)
    {
    }
    RCP<const Series> series(const RCP<const Basic> &x)
    {
        return make_rcp<Series>(apply(x), varname, prec);
    }

    Poly apply(const RCP<const Basic> &x)
    {
        x->accept(*this);
        Poly temp(std::move(p));
        return temp;
    }

    void bvisit(const Add &x)
    {
        Poly temp(apply(x.get_coef()));
        for (const auto &term : x.get_dict()) {
            temp += apply(term.first) * apply(term.second);
        }
        p = temp;
    }
    void bvisit(const Mul &x)
    {
        Poly temp(apply(x.get_coef()));
        for (const auto &term : x.get_dict()) {
            temp = Series::mul(temp, apply(pow(term.first, term.second)), prec);
        }
        p = temp;
    }
    void bvisit(const Pow &x)
    {
        const RCP<const Basic> &base = x.get_base(), exp = x.get_exp();
        if (is_a<Integer>(*exp)) {
            const Integer &ii = (down_cast<const Integer &>(*exp));
            if (not mp_fits_slong_p(ii.as_integer_class()))
                throw SymEngineException("series power exponent size");
            const int sh = numeric_cast<int>(mp_get_si(ii.as_integer_class()));
            base->accept(*this);
            if (sh == 1) {
                return;
            } else if (sh > 0) {
                p = Series::pow(p, sh, prec);
            } else if (sh == -1) {
                p = Series::series_invert(p, var, prec);
            } else {
                // Invert and then exponentiate to give the correct behavior
                // when expanding 1/x**(prec), which should return x**(-prec),
                // not 0.
                p = Series::pow(Series::series_invert(p, var, prec), -sh, prec);
            }

        } else if (is_a<Rational>(*exp)) {
            const Rational &rat = (down_cast<const Rational &>(*exp));
            const integer_class &expnumz = get_num(rat.as_rational_class());
            const integer_class &expdenz = get_den(rat.as_rational_class());
            if (not mp_fits_slong_p(expnumz) or not mp_fits_slong_p(expdenz))
                throw SymEngineException("series rational power exponent size");
            const int num = numeric_cast<int>(mp_get_si(expnumz));
            const int den = numeric_cast<int>(mp_get_si(expdenz));
            base->accept(*this);
            const Poly proot(
                Series::series_nthroot(apply(base), den, var, prec));
            if (num == 1) {
                p = proot;
            } else if (num > 0) {
                p = Series::pow(proot, num, prec);
            } else if (num == -1) {
                p = Series::series_invert(proot, var, prec);
            } else {
                p = Series::series_invert(Series::pow(proot, -num, prec), var,
                                          prec);
            }
        } else if (eq(*E, *base)) {
            p = Series::series_exp(apply(exp), var, prec);
        } else {
            p = Series::series_exp(
                Poly(apply(exp) * Series::series_log(apply(base), var, prec)),
                var, prec);
        }
    }

    void bvisit(const Function &x)
    {
        RCP<const Basic> d = x.rcp_from_this();
        RCP<const Symbol> s = symbol(varname);

        map_basic_basic m({{s, zero}});
        RCP<const Basic> const_term = d->subs(m);
        if (const_term == d) {
            p = Series::convert(*d);
            return;
        }
        Poly res_p(apply(expand(const_term)));
        Coeff prod, t;
        prod = 1;

        for (unsigned int i = 1; i < prec; i++) {
            // Workaround for flint
            t = i;
            prod /= t;
            d = d->diff(s);
            res_p += Series::pow(var, i, prec)
                     * (prod * apply(expand(d->subs(m))));
        }
        p = res_p;
    }

    void bvisit(const Gamma &x)
    {
        RCP<const Symbol> s = symbol(varname);
        RCP<const Basic> arg = x.get_args()[0];
        if (eq(*arg->subs({{s, zero}}), *zero)) {
            RCP<const Basic> g = gamma(add(arg, one));
            if (is_a<Gamma>(*g)) {
                bvisit(down_cast<const Function &>(*g));
                p *= Series::pow(var, -1, prec);
            } else {
                g->accept(*this);
            }
        } else {
            bvisit(implicit_cast<const Function &>(x));
        }
    }

    void bvisit(const Series &x)
    {
        if (x.get_var() != varname) {
            throw NotImplementedError("Multivariate Series not implemented");
        }
        if (x.get_degree() < prec) {
            throw SymEngineException("Series with lesser prec found");
        }
        p = x.get_poly();
    }
    void bvisit(const Integer &x)
    {
        p = Series::convert(x);
    }
    void bvisit(const Rational &x)
    {
        p = Series::convert(x);
    }
    void bvisit(const Complex &x)
    {
        p = Series::convert(x);
    }
    void bvisit(const RealDouble &x)
    {
        p = Series::convert(x);
    }
    void bvisit(const ComplexDouble &x)
    {
        p = Series::convert(x);
    }
#ifdef HAVE_SYMENGINE_MPFR
    void bvisit(const RealMPFR &x)
    {
        p = Series::convert(x);
    }
#endif
#ifdef HAVE_SYMENGINE_MPC
    void bvisit(const ComplexMPC &x)
    {
        p = Series::convert(x);
    }
#endif
    void bvisit(const Sin &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_sin(p, var, prec);
    }
    void bvisit(const Cos &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_cos(p, var, prec);
    }
    void bvisit(const Tan &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_tan(p, var, prec);
    }
    void bvisit(const Cot &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_cot(p, var, prec);
    }
    void bvisit(const Csc &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_csc(p, var, prec);
    }
    void bvisit(const Sec &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_sec(p, var, prec);
    }
    void bvisit(const Log &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_log(p, var, prec);
    }
    void bvisit(const ASin &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_asin(p, var, prec);
    }
    void bvisit(const ACos &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_acos(p, var, prec);
    }
    void bvisit(const ATan &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_atan(p, var, prec);
    }
    void bvisit(const Sinh &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_sinh(p, var, prec);
    }
    void bvisit(const Cosh &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_cosh(p, var, prec);
    }
    void bvisit(const Tanh &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_tanh(p, var, prec);
    }
    void bvisit(const ASinh &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_asinh(p, var, prec);
    }
    void bvisit(const ATanh &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_atanh(p, var, prec);
    }
    void bvisit(const LambertW &x)
    {
        x.get_arg()->accept(*this);
        p = Series::series_lambertw(p, var, prec);
    }
    void bvisit(const Symbol &x)
    {
        if (x.get_name() == varname) {
            p = Series::var(x.get_name());
        } else {
            p = Series::convert(x);
        }
    }
    void bvisit(const Constant &x)
    {
        p = Series::convert(x);
    }
    void bvisit(const Basic &x)
    {
        if (!has_symbol(x, *symbol(varname))) {
            p = Series::convert(x);
        } else {
            throw NotImplementedError("Not Implemented");
        }
    }
};

class NeedsSymbolicExpansionVisitor
    : public BaseVisitor<NeedsSymbolicExpansionVisitor, StopVisitor>
{
protected:
    RCP<const Symbol> x_;
    bool needs_;

public:
    template <typename T,
              typename
              = enable_if_t<std::is_base_of<TrigBase, T>::value
                            or std::is_base_of<HyperbolicBase, T>::value>>
    void bvisit(const T &f)
    {
        auto arg = f.get_arg();
        map_basic_basic subsx0{{x_, integer(0)}};
        if (arg->subs(subsx0)->__neq__(*integer(0))) {
            needs_ = true;
            stop_ = true;
        }
    }

    void bvisit(const Pow &pow)
    {
        auto base = pow.get_base();
        auto exp = pow.get_exp();
        map_basic_basic subsx0{{x_, integer(0)}};
        // exp(const) or x^-1
        if ((base->__eq__(*E) and exp->subs(subsx0)->__neq__(*integer(0)))
            or (is_a_Number(*exp)
                and down_cast<const Number &>(*exp).is_negative()
                and base->subs(subsx0)->__eq__(*integer(0)))) {
            needs_ = true;
            stop_ = true;
        }
    }

    void bvisit(const Log &f)
    {
        auto arg = f.get_arg();
        map_basic_basic subsx0{{x_, integer(0)}};
        if (arg->subs(subsx0)->__eq__(*integer(0))) {
            needs_ = true;
            stop_ = true;
        }
    }

    void bvisit(const LambertW &x)
    {
        needs_ = true;
        stop_ = true;
    }

    void bvisit(const Basic &x) {}

    bool apply(const Basic &b, const RCP<const Symbol> &x)
    {
        x_ = x;
        needs_ = false;
        stop_ = false;
        postorder_traversal_stop(b, *this);
        return needs_;
    }
};

} // namespace SymEngine
#endif // SYMENGINE_SERIES_VISITOR_H
