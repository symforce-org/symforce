#include <symengine/visitor.h>
#include <symengine/eval_mpfr.h>
#include <symengine/symengine_exception.h>

#ifdef HAVE_SYMENGINE_MPFR

namespace SymEngine
{

class EvalMPFRVisitor : public BaseVisitor<EvalMPFRVisitor>
{
protected:
    mpfr_rnd_t rnd_;
    mpfr_ptr result_;

public:
    EvalMPFRVisitor(mpfr_rnd_t rnd) : rnd_{rnd} {}

    void apply(mpfr_ptr result, const Basic &b)
    {
        mpfr_ptr tmp = result_;
        result_ = result;
        b.accept(*this);
        result_ = tmp;
    }

    void bvisit(const Integer &x)
    {
        mpfr_set_z(result_, get_mpz_t(x.as_integer_class()), rnd_);
    }

    void bvisit(const Rational &x)
    {
        mpfr_set_q(result_, get_mpq_t(x.as_rational_class()), rnd_);
    }

    void bvisit(const RealDouble &x)
    {
        mpfr_set_d(result_, x.i, rnd_);
    }

    void bvisit(const RealMPFR &x)
    {
        mpfr_set(result_, x.i.get_mpfr_t(), rnd_);
    }

    void bvisit(const Add &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;
        for (; p != d.end(); p++) {
            apply(t.get_mpfr_t(), *(*p));
            mpfr_add(result_, result_, t.get_mpfr_t(), rnd_);
        }
    }

    void bvisit(const Mul &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;
        for (; p != d.end(); p++) {
            apply(t.get_mpfr_t(), *(*p));
            mpfr_mul(result_, result_, t.get_mpfr_t(), rnd_);
        }
    }

    void bvisit(const Pow &x)
    {
        if (eq(*x.get_base(), *E)) {
            apply(result_, *(x.get_exp()));
            mpfr_exp(result_, result_, rnd_);
        } else {
            mpfr_class b(mpfr_get_prec(result_));
            apply(b.get_mpfr_t(), *(x.get_base()));
            apply(result_, *(x.get_exp()));
            mpfr_pow(result_, b.get_mpfr_t(), result_, rnd_);
        }
    }

    void bvisit(const Equality &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(t.get_mpfr_t(), *(x.get_arg1()));
        apply(result_, *(x.get_arg2()));
        if (mpfr_equal_p(t.get_mpfr_t(), result_)) {
            mpfr_set_ui(result_, 1, rnd_);
        } else {
            mpfr_set_ui(result_, 0, rnd_);
        }
    }

    void bvisit(const Unequality &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(t.get_mpfr_t(), *(x.get_arg1()));
        apply(result_, *(x.get_arg2()));
        if (mpfr_lessgreater_p(t.get_mpfr_t(), result_)) {
            mpfr_set_ui(result_, 1, rnd_);
        } else {
            mpfr_set_ui(result_, 0, rnd_);
        }
    }

    void bvisit(const LessThan &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(t.get_mpfr_t(), *(x.get_arg1()));
        apply(result_, *(x.get_arg2()));
        if (mpfr_lessequal_p(t.get_mpfr_t(), result_)) {
            mpfr_set_ui(result_, 1, rnd_);
        } else {
            mpfr_set_ui(result_, 0, rnd_);
        }
    }

    void bvisit(const StrictLessThan &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(t.get_mpfr_t(), *(x.get_arg1()));
        apply(result_, *(x.get_arg2()));
        if (mpfr_less_p(t.get_mpfr_t(), result_)) {
            mpfr_set_ui(result_, 1, rnd_);
        } else {
            mpfr_set_ui(result_, 0, rnd_);
        }
    }

    void bvisit(const Sin &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_sin(result_, result_, rnd_);
    }

    void bvisit(const Cos &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_cos(result_, result_, rnd_);
    }

    void bvisit(const Tan &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_tan(result_, result_, rnd_);
    }

    void bvisit(const Log &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_log(result_, result_, rnd_);
    }

    void bvisit(const Cot &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_cot(result_, result_, rnd_);
    }

    void bvisit(const Csc &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_csc(result_, result_, rnd_);
    }

    void bvisit(const Sec &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_sec(result_, result_, rnd_);
    }

    void bvisit(const ASin &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_asin(result_, result_, rnd_);
    }

    void bvisit(const ACos &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_acos(result_, result_, rnd_);
    }

    void bvisit(const ASec &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_ui_div(result_, 1, result_, rnd_);
        mpfr_asin(result_, result_, rnd_);
    }

    void bvisit(const ACsc &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_ui_div(result_, 1, result_, rnd_);
        mpfr_acos(result_, result_, rnd_);
    }

    void bvisit(const ATan &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_atan(result_, result_, rnd_);
    }

    void bvisit(const ACot &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_ui_div(result_, 1, result_, rnd_);
        mpfr_atan(result_, result_, rnd_);
    }

    void bvisit(const ATan2 &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(t.get_mpfr_t(), *(x.get_num()));
        apply(result_, *(x.get_den()));
        mpfr_atan2(result_, t.get_mpfr_t(), result_, rnd_);
    }

    void bvisit(const Sinh &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_sinh(result_, result_, rnd_);
    }

    void bvisit(const Csch &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_csch(result_, result_, rnd_);
    }

    void bvisit(const Cosh &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_cosh(result_, result_, rnd_);
    }

    void bvisit(const Sech &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_sech(result_, result_, rnd_);
    }

    void bvisit(const Tanh &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_tanh(result_, result_, rnd_);
    }

    void bvisit(const Coth &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_coth(result_, result_, rnd_);
    }

    void bvisit(const ASinh &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_asinh(result_, result_, rnd_);
    }

    void bvisit(const ACsch &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_ui_div(result_, 1, result_, rnd_);
        mpfr_asinh(result_, result_, rnd_);
    };

    void bvisit(const ACosh &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_acosh(result_, result_, rnd_);
    }

    void bvisit(const ATanh &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_atanh(result_, result_, rnd_);
    }

    void bvisit(const ACoth &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_ui_div(result_, 1, result_, rnd_);
        mpfr_atanh(result_, result_, rnd_);
    }

    void bvisit(const ASech &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_ui_div(result_, 1, result_, rnd_);
        mpfr_acosh(result_, result_, rnd_);
    };

    void bvisit(const Gamma &x)
    {
        apply(result_, *(x.get_args()[0]));
        mpfr_gamma(result_, result_, rnd_);
    };
#if MPFR_VERSION_MAJOR > 3
    void bvisit(const UpperGamma &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(result_, *(x.get_args()[1]));
        apply(t.get_mpfr_t(), *(x.get_args()[0]));
        mpfr_gamma_inc(result_, t.get_mpfr_t(), result_, rnd_);
    };

    void bvisit(const LowerGamma &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        apply(result_, *(x.get_args()[1]));
        apply(t.get_mpfr_t(), *(x.get_args()[0]));
        mpfr_gamma_inc(result_, t.get_mpfr_t(), result_, rnd_);
        mpfr_gamma(t.get_mpfr_t(), t.get_mpfr_t(), rnd_);
        mpfr_sub(result_, t.get_mpfr_t(), result_, rnd_);
    };
#endif
    void bvisit(const LogGamma &x)
    {
        apply(result_, *(x.get_args()[0]));
        mpfr_lngamma(result_, result_, rnd_);
    }

    void bvisit(const Beta &x)
    {
        apply(result_, *(x.rewrite_as_gamma()));
    };

    void bvisit(const Constant &x)
    {
        if (x.__eq__(*pi)) {
            mpfr_const_pi(result_, rnd_);
        } else if (x.__eq__(*E)) {
            mpfr_t one_;
            mpfr_init2(one_, mpfr_get_prec(result_));
            mpfr_set_ui(one_, 1, rnd_);
            mpfr_exp(result_, one_, rnd_);
            mpfr_clear(one_);
        } else if (x.__eq__(*EulerGamma)) {
            mpfr_const_euler(result_, rnd_);
        } else if (x.__eq__(*Catalan)) {
            mpfr_const_catalan(result_, rnd_);
        } else if (x.__eq__(*GoldenRatio)) {
            mpfr_sqrt_ui(result_, 5, rnd_);
            mpfr_add_ui(result_, result_, 1, rnd_);
            mpfr_div_ui(result_, result_, 2, rnd_);
        } else {
            throw NotImplementedError("Constant " + x.get_name()
                                      + " is not implemented.");
        }
    }

    void bvisit(const Abs &x)
    {
        apply(result_, *(x.get_arg()));
        mpfr_abs(result_, result_, rnd_);
    };

    void bvisit(const NumberWrapper &x)
    {
        x.eval(mpfr_get_prec(result_))->accept(*this);
    }

    void bvisit(const FunctionWrapper &x)
    {
        x.eval(mpfr_get_prec(result_))->accept(*this);
    }
    void bvisit(const Erf &x)
    {
        apply(result_, *(x.get_args()[0]));
        mpfr_erf(result_, result_, rnd_);
    }

    void bvisit(const Erfc &x)
    {
        apply(result_, *(x.get_args()[0]));
        mpfr_erfc(result_, result_, rnd_);
    }

    void bvisit(const Max &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;
        for (; p != d.end(); p++) {
            apply(t.get_mpfr_t(), *(*p));
            mpfr_max(result_, result_, t.get_mpfr_t(), rnd_);
        }
    }

    void bvisit(const Min &x)
    {
        mpfr_class t(mpfr_get_prec(result_));
        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;
        for (; p != d.end(); p++) {
            apply(t.get_mpfr_t(), *(*p));
            mpfr_min(result_, result_, t.get_mpfr_t(), rnd_);
        }
    }

    void bvisit(const UnevaluatedExpr &x)
    {
        apply(result_, *x.get_arg());
    }

    // Classes not implemented are
    // Subs, Dirichlet_eta, Zeta
    // LeviCivita, KroneckerDelta, LambertW
    // Derivative, Complex, ComplexDouble, ComplexMPC
    void bvisit(const Basic &)
    {
        throw NotImplementedError("Not Implemented");
    };
};

void eval_mpfr(mpfr_ptr result, const Basic &b, mpfr_rnd_t rnd)
{
    EvalMPFRVisitor v(rnd);
    v.apply(result, b);
}

} // namespace SymEngine

#endif // HAVE_SYMENGINE_MPFR
