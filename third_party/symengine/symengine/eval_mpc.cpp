#include <symengine/visitor.h>
#include <symengine/eval_mpc.h>
#include <symengine/symengine_exception.h>

#ifdef HAVE_SYMENGINE_MPC

namespace SymEngine
{

class EvalMPCVisitor : public BaseVisitor<EvalMPCVisitor>
{
protected:
    mpfr_rnd_t rnd_;
    mpc_ptr result_;

public:
    EvalMPCVisitor(mpfr_rnd_t rnd) : rnd_{rnd} {}

    void apply(mpc_ptr result, const Basic &b)
    {
        mpc_ptr tmp = result_;
        result_ = result;
        b.accept(*this);
        result_ = tmp;
    }

    void bvisit(const Integer &x)
    {
        mpc_set_z(result_, get_mpz_t(x.as_integer_class()), rnd_);
    }

    void bvisit(const Rational &x)
    {
        mpc_set_q(result_, get_mpq_t(x.as_rational_class()), rnd_);
    }

    void bvisit(const RealDouble &x)
    {
        mpc_set_d(result_, x.i, rnd_);
    }

    void bvisit(const Complex &x)
    {
        mpc_set_q_q(result_, get_mpq_t(x.real_), get_mpq_t(x.imaginary_), rnd_);
    }

    void bvisit(const ComplexDouble &x)
    {
        mpc_set_d_d(result_, x.i.real(), x.i.imag(), rnd_);
    }

    void bvisit(const RealMPFR &x)
    {
        mpc_set_fr(result_, x.i.get_mpfr_t(), rnd_);
    }

    void bvisit(const ComplexMPC &x)
    {
        mpc_set(result_, x.as_mpc().get_mpc_t(), rnd_);
    }

    void bvisit(const Add &x)
    {
        mpc_t t;
        mpc_init2(t, mpc_get_prec(result_));

        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;
        for (; p != d.end(); p++) {
            apply(t, *(*p));
            mpc_add(result_, result_, t, rnd_);
        }
        mpc_clear(t);
    }

    void bvisit(const Mul &x)
    {
        mpc_t t;
        mpc_init2(t, mpc_get_prec(result_));

        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;
        for (; p != d.end(); p++) {
            apply(t, *(*p));
            mpc_mul(result_, result_, t, rnd_);
        }
        mpc_clear(t);
    }

    void bvisit(const Pow &x)
    {
        if (eq(*x.get_base(), *E)) {
            apply(result_, *(x.get_exp()));
            mpc_exp(result_, result_, rnd_);
        } else {
            mpc_t t;
            mpc_init2(t, mpc_get_prec(result_));

            apply(t, *(x.get_base()));
            apply(result_, *(x.get_exp()));
            mpc_pow(result_, t, result_, rnd_);

            mpc_clear(t);
        }
    }

    void bvisit(const Sin &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_sin(result_, result_, rnd_);
    }

    void bvisit(const Cos &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_cos(result_, result_, rnd_);
    }

    void bvisit(const Tan &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_tan(result_, result_, rnd_);
    }

    void bvisit(const Log &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_log(result_, result_, rnd_);
    }

    void bvisit(const Cot &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_tan(result_, result_, rnd_);
        mpc_ui_div(result_, 1, result_, rnd_);
    }

    void bvisit(const Csc &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_sin(result_, result_, rnd_);
        mpc_ui_div(result_, 1, result_, rnd_);
    }

    void bvisit(const Sec &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_cos(result_, result_, rnd_);
        mpc_ui_div(result_, 1, result_, rnd_);
    }

    void bvisit(const ASin &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_asin(result_, result_, rnd_);
    }

    void bvisit(const ACos &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_acos(result_, result_, rnd_);
    }

    void bvisit(const ASec &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_ui_div(result_, 1, result_, rnd_);
        mpc_acos(result_, result_, rnd_);
    }

    void bvisit(const ACsc &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_ui_div(result_, 1, result_, rnd_);
        mpc_asin(result_, result_, rnd_);
    }

    void bvisit(const ATan &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_atan(result_, result_, rnd_);
    }

    void bvisit(const ACot &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_ui_div(result_, 1, result_, rnd_);
        mpc_atan(result_, result_, rnd_);
    }

    void bvisit(const Sinh &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_sinh(result_, result_, rnd_);
    }

    void bvisit(const Csch &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_sinh(result_, result_, rnd_);
        mpc_ui_div(result_, 1, result_, rnd_);
    }

    void bvisit(const Cosh &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_cosh(result_, result_, rnd_);
    }

    void bvisit(const Sech &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_cosh(result_, result_, rnd_);
        mpc_ui_div(result_, 1, result_, rnd_);
    }

    void bvisit(const Tanh &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_tanh(result_, result_, rnd_);
    }

    void bvisit(const Coth &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_tanh(result_, result_, rnd_);
        mpc_ui_div(result_, 1, result_, rnd_);
    }

    void bvisit(const ASinh &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_asinh(result_, result_, rnd_);
    }

    void bvisit(const ACsch &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_ui_div(result_, 1, result_, rnd_);
        mpc_asinh(result_, result_, rnd_);
    }

    void bvisit(const ACosh &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_acosh(result_, result_, rnd_);
    }

    void bvisit(const ATanh &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_atanh(result_, result_, rnd_);
    }

    void bvisit(const ACoth &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_ui_div(result_, 1, result_, rnd_);
        mpc_atanh(result_, result_, rnd_);
    }

    void bvisit(const ASech &x)
    {
        apply(result_, *(x.get_arg()));
        mpc_ui_div(result_, 1, result_, rnd_);
        mpc_acosh(result_, result_, rnd_);
    };

    void bvisit(const Constant &x)
    {
        if (x.__eq__(*pi)) {
            mpfr_t t;
            mpfr_init2(t, mpc_get_prec(result_));
            mpfr_const_pi(t, rnd_);
            mpc_set_fr(result_, t, rnd_);
            mpfr_clear(t);
        } else if (x.__eq__(*E)) {
            mpfr_t t;
            mpfr_init2(t, mpc_get_prec(result_));
            mpfr_set_ui(t, 1, rnd_);
            mpfr_exp(t, t, rnd_);
            mpc_set_fr(result_, t, rnd_);
            mpfr_clear(t);
        } else if (x.__eq__(*EulerGamma)) {
            mpfr_t t;
            mpfr_init2(t, mpc_get_prec(result_));
            mpfr_const_euler(t, rnd_);
            mpc_set_fr(result_, t, rnd_);
            mpfr_clear(t);
        } else if (x.__eq__(*Catalan)) {
            mpfr_t t;
            mpfr_init2(t, mpc_get_prec(result_));
            mpfr_const_catalan(t, rnd_);
            mpc_set_fr(result_, t, rnd_);
            mpfr_clear(t);
        } else if (x.__eq__(*GoldenRatio)) {
            mpfr_t t;
            mpfr_init2(t, mpc_get_prec(result_));
            mpfr_sqrt_ui(t, 5, rnd_);
            mpfr_add_ui(t, t, 1, rnd_);
            mpfr_div_ui(t, t, 2, rnd_);
            mpc_set_fr(result_, t, rnd_);
            mpfr_clear(t);
        } else {
            throw NotImplementedError("Constant " + x.get_name()
                                      + " is not implemented.");
        }
    }

    void bvisit(const Gamma &x)
    {
        throw NotImplementedError("Not implemented");
    }

    void bvisit(const Abs &x)
    {
        mpfr_t t;
        mpfr_init2(t, mpc_get_prec(result_));
        apply(result_, *(x.get_arg()));
        mpc_abs(t, result_, rnd_);
        mpc_set_fr(result_, t, rnd_);
        mpfr_clear(t);
    };

    void bvisit(const NumberWrapper &x)
    {
        x.eval(mpc_get_prec(result_))->accept(*this);
    }

    void bvisit(const FunctionWrapper &x)
    {
        x.eval(mpc_get_prec(result_))->accept(*this);
    }

    void bvisit(const UnevaluatedExpr &x)
    {
        apply(result_, *x.get_arg());
    }

    // Classes not implemented are
    // Subs, UpperGamma, LowerGamma, Dirichlet_eta, Zeta
    // LeviCivita, KroneckerDelta, FunctionSymbol, LambertW
    // Derivative, ATan2, Gamma
    void bvisit(const Basic &)
    {
        throw NotImplementedError("Not Implemented");
    };
};

void eval_mpc(mpc_ptr result, const Basic &b, mpfr_rnd_t rnd)
{
    EvalMPCVisitor v(rnd);
    v.apply(result, b);
}

} // namespace SymEngine

#endif // HAVE_SYMENGINE_MPFR
