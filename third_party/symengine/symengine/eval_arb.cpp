#include <symengine/visitor.h>
#include <symengine/eval_arb.h>
#include <symengine/symengine_exception.h>

#ifdef HAVE_SYMENGINE_ARB

namespace SymEngine
{

class EvalArbVisitor : public BaseVisitor<EvalArbVisitor>
{
protected:
    long prec_;
    arb_ptr result_;

public:
    EvalArbVisitor(long precision) : prec_{precision} {}

    void apply(arb_ptr result, const Basic &b)
    {
        arb_ptr tmp = result_;
        result_ = result;
        b.accept(*this);
        result_ = tmp;
    }

    void bvisit(const Integer &x)
    {
        fmpz_t z_;
        fmpz_init(z_);
        fmpz_set_mpz(z_, get_mpz_t(x.as_integer_class()));
        arb_set_fmpz(result_, z_);
        fmpz_clear(z_);
    }

    void bvisit(const Rational &x)
    {
        fmpq_t q_;
        fmpq_init(q_);
        fmpq_set_mpq(q_, get_mpq_t(x.as_rational_class()));
        arb_set_fmpq(result_, q_, prec_);
        fmpq_clear(q_);
    }

    void bvisit(const RealDouble &x)
    {
        arf_t f_;
        arf_init(f_);
        arf_set_d(f_, x.i);
        arb_set_arf(result_, f_);
        arf_clear(f_);
    }

    void bvisit(const Add &x)
    {
        arb_t t;
        arb_init(t);

        auto d = x.get_args();
        for (auto p = d.begin(); p != d.end(); p++) {

            if (p == d.begin()) {
                apply(result_, *(*p));
            } else {
                apply(t, *(*p));
                arb_add(result_, result_, t, prec_);
            }
        }

        arb_clear(t);
    }

    void bvisit(const Mul &x)
    {
        arb_t t;
        arb_init(t);

        auto d = x.get_args();
        for (auto p = d.begin(); p != d.end(); p++) {

            if (p == d.begin()) {
                apply(result_, *(*p));
            } else {
                apply(t, *(*p));
                arb_mul(result_, result_, t, prec_);
            }
        }

        arb_clear(t);
    }

    void bvisit(const Pow &x)
    {
        if (eq(*x.get_base(), *E)) {
            apply(result_, *(x.get_exp()));
            arb_exp(result_, result_, prec_);
        } else {
            arb_t b;
            arb_init(b);

            apply(b, *(x.get_base()));
            apply(result_, *(x.get_exp()));
            arb_pow(result_, b, result_, prec_);

            arb_clear(b);
        }
    }

    void bvisit(const Sin &x)
    {
        apply(result_, *(x.get_arg()));
        arb_sin(result_, result_, prec_);
    }

    void bvisit(const Cos &x)
    {
        apply(result_, *(x.get_arg()));
        arb_cos(result_, result_, prec_);
    }

    void bvisit(const Tan &x)
    {
        apply(result_, *(x.get_arg()));
        arb_tan(result_, result_, prec_);
    }

    void bvisit(const Symbol &)
    {
        throw SymEngineException("Symbol cannot be evaluated as an arb type.");
    }

    void bvisit(const UIntPoly &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Complex &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const ComplexDouble &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const RealMPFR &)
    {
        throw NotImplementedError("Not Implemented");
    }
#ifdef HAVE_SYMENGINE_MPC
    void bvisit(const ComplexMPC &)
    {
        throw NotImplementedError("Not Implemented");
    }
#endif
    void bvisit(const Log &x)
    {
        apply(result_, *(x.get_arg()));
        arb_log(result_, result_, prec_);
    }

    void bvisit(const Derivative &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Cot &x)
    {
        apply(result_, *(x.get_arg()));
        arb_cot(result_, result_, prec_);
    }

    void bvisit(const Csc &x)
    {
        apply(result_, *(x.get_arg()));
        arb_sin(result_, result_, prec_);
        arb_inv(result_, result_, prec_);
    }

    void bvisit(const Sec &x)
    {
        apply(result_, *(x.get_arg()));
        arb_cos(result_, result_, prec_);
        arb_inv(result_, result_, prec_);
    }

    void bvisit(const ASin &x)
    {
        apply(result_, *(x.get_arg()));
        arb_asin(result_, result_, prec_);
    }

    void bvisit(const ACos &x)
    {
        apply(result_, *(x.get_arg()));
        arb_acos(result_, result_, prec_);
    }

    void bvisit(const ASec &x)
    {
        apply(result_, *(x.get_arg()));
        arb_inv(result_, result_, prec_);
        arb_acos(result_, result_, prec_);
    }

    void bvisit(const ACsc &x)
    {
        apply(result_, *(x.get_arg()));
        arb_inv(result_, result_, prec_);
        arb_asin(result_, result_, prec_);
    }

    void bvisit(const ATan &x)
    {
        apply(result_, *(x.get_arg()));
        arb_atan(result_, result_, prec_);
    }

    void bvisit(const ACot &x)
    {
        apply(result_, *(x.get_arg()));
        arb_inv(result_, result_, prec_);
        arb_atan(result_, result_, prec_);
    }

    void bvisit(const ATan2 &x)
    {
        arb_t t;
        arb_init(t);

        apply(t, *(x.get_num()));
        apply(result_, *(x.get_den()));
        arb_atan2(result_, t, result_, prec_);

        arb_clear(t);
    }

    void bvisit(const LambertW &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const FunctionWrapper &x)
    {
        x.eval(prec_)->accept(*this);
    }

    void bvisit(const Sinh &x)
    {
        apply(result_, *(x.get_arg()));
        arb_sinh(result_, result_, prec_);
    }

    void bvisit(const Csch &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Cosh &x)
    {
        apply(result_, *(x.get_arg()));
        arb_cosh(result_, result_, prec_);
    }

    void bvisit(const Sech &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Tanh &x)
    {
        apply(result_, *(x.get_arg()));
        arb_tanh(result_, result_, prec_);
    }

    void bvisit(const Coth &x)
    {
        apply(result_, *(x.get_arg()));
        arb_coth(result_, result_, prec_);
    }

    void bvisit(const Max &x)
    {
        arb_t t;
        arb_init(t);

        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;

        for (; p != d.end(); p++) {

            apply(t, *(*p));
            if (arb_gt(t, result_))
                arb_set(result_, t);
        }

        arb_clear(t);
    }

    void bvisit(const Min &x)
    {
        arb_t t;
        arb_init(t);

        auto d = x.get_args();
        auto p = d.begin();
        apply(result_, *(*p));
        p++;

        for (; p != d.end(); p++) {

            apply(t, *(*p));
            if (arb_lt(t, result_))
                arb_set(result_, t);
        }

        arb_clear(t);
    }

    void bvisit(const ACsch &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const ASinh &x)
    {
        apply(result_, *(x.get_arg()));
        arb_asinh(result_, result_, prec_);
    }

    void bvisit(const ACosh &x)
    {
        apply(result_, *(x.get_arg()));
        arb_acosh(result_, result_, prec_);
    }

    void bvisit(const ATanh &x)
    {
        apply(result_, *(x.get_arg()));
        arb_atanh(result_, result_, prec_);
    }

    void bvisit(const ACoth &x)
    {
        apply(result_, *(x.get_arg()));
        arb_inv(result_, result_, prec_);
        arb_atanh(result_, result_, prec_);
    }

    void bvisit(const ASech &x)
    {
        apply(result_, *(x.get_arg()));
        arb_inv(result_, result_, prec_);
        arb_acosh(result_, result_, prec_);
    }

    void bvisit(const KroneckerDelta &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const LeviCivita &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Zeta &x)
    {
        arb_t t_;
        arb_init(t_);

        apply(t_, *(x.get_arg1()));
        apply(result_, *(x.get_arg2()));
        arb_hurwitz_zeta(result_, t_, result_, prec_);

        arb_clear(t_);
    }

    void bvisit(const Dirichlet_eta &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Gamma &x)
    {
        apply(result_, *(x.get_args())[0]);
        arb_gamma(result_, result_, prec_);
    }

    void bvisit(const LogGamma &x)
    {
        apply(result_, *(x.get_args())[0]);
        arb_lgamma(result_, result_, prec_);
    }

    void bvisit(const LowerGamma &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const UpperGamma &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const Constant &x)
    {
        if (x.__eq__(*pi)) {
            arb_const_pi(result_, prec_);
        } else if (x.__eq__(*E)) {
            arb_const_e(result_, prec_);
        } else if (x.__eq__(*EulerGamma)) {
            arb_const_euler(result_, prec_);
        } else if (x.__eq__(*Catalan)) {
            arb_const_catalan(result_, prec_);
        } else if (x.__eq__(*GoldenRatio)) {
            arb_sqrt_ui(result_, 5, prec_);
            arb_add_ui(result_, result_, 1, prec_);
            arb_div_ui(result_, result_, 2, prec_);
        } else {
            throw NotImplementedError("Constant " + x.get_name()
                                      + " is not implemented.");
        }
    }

    void bvisit(const Abs &x)
    {
        apply(result_, *(x.get_arg()));
        arb_abs(result_, result_);
    }

    void bvisit(const Basic &)
    {
        throw NotImplementedError("Not Implemented");
    }

    void bvisit(const NumberWrapper &x)
    {
        x.eval(prec_)->accept(*this);
    }

    void bvisit(const UnevaluatedExpr &x)
    {
        apply(result_, *x.get_arg());
    }
};

void eval_arb(arb_t result, const Basic &b, long precision)
{
    EvalArbVisitor v(precision);
    v.apply(result, b);
}

} // namespace SymEngine

#endif // HAVE_SYMENGINE_ARB
