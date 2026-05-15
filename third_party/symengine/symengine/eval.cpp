#include <exception>
#include <symengine/eval.h>
#include <symengine/eval_double.h>
#include <symengine/real_double.h>
#include <symengine/complex_double.h>
#include <symengine/symengine_rcp.h>
#include <symengine/visitor.h>
#ifdef HAVE_SYMENGINE_MPFR
#include <mpfr.h>
#include <symengine/real_mpfr.h>
#include <symengine/eval_mpfr.h>
#endif // HAVE_SYMENGINE_MPFR

#ifdef HAVE_SYMENGINE_MPC
#include <mpc.h>
#include <symengine/complex_mpc.h>
#include <symengine/eval_mpc.h>
#endif // HAVE_SYMENGINE_MPC

namespace SymEngine
{

RCP<const Number> evalf_numeric(const Basic &b, unsigned long bits, bool real)
{
    if (bits <= 53 && real) { // double
        double d = eval_double(b);
        return real_double(d);
    } else if (bits <= 53 && !real) { // complex double
        std::complex<double> d = eval_complex_double(b);
        return complex_double(d);
    } else if (bits > 53 && real) { // mpfr
#ifdef HAVE_SYMENGINE_MPFR
        mpfr_class mc = mpfr_class(bits);
        mpfr_ptr result = mc.get_mpfr_t();
        eval_mpfr(result, b, MPFR_RNDN);
        return make_rcp<RealMPFR>(std::move(mc));
#else
        throw std::invalid_argument(
            "For multiple bit precision, MPFR is needed");
#endif       // HAVE_SYMENGINE_MPFR
    } else { // mpc
#ifdef HAVE_SYMENGINE_MPC
        mpc_class mc = mpc_class(bits);
        mpc_ptr result = mc.get_mpc_t();
        eval_mpc(result, b, MPFR_RNDN);
        return make_rcp<ComplexMPC>(std::move(mc));
#else
        throw std::invalid_argument(
            "For multiple bit precision, MPC is needed");
#endif // HAVE_SYMENGINE_MPC
    }
}

class EvalVisitor : public BaseVisitor<EvalVisitor, TransformVisitor>
{
protected:
    unsigned long bits;

public:
    EvalVisitor(unsigned long bits) : bits(bits) {}
    using TransformVisitor::apply;
    using TransformVisitor::bvisit;
    void bvisit(const Number &x)
    {
        result_ = evalf_numeric(x, bits, true);
    }
    void bvisit(const ComplexBase &x)
    {
        result_ = evalf_numeric(x, bits, false);
    }
    void bvisit(const Constant &x)
    {
        result_ = evalf_numeric(x, bits, true);
    }
    void bvisit(const NumberWrapper &x)
    {
        result_ = x.eval(bits)->rcp_from_this();
    }
    void bvisit(const FunctionWrapper &x)
    {
        result_ = x.eval(bits)->rcp_from_this();
    }
};

RCP<const Basic> evalf(const Basic &b, unsigned long bits, EvalfDomain domain)
{
#ifndef HAVE_SYMENGINE_MPFR
    if (bits > 53) {
        throw std::invalid_argument(
            "For multiple bit precision, MPFR is needed");
    }
#endif
#ifndef HAVE_SYMENGINE_MPC
    if (bits > 53 and domain == EvalfDomain::Complex) {
        throw std::invalid_argument(
            "For multiple bit precision, MPC is needed");
    }
#endif
    if (domain == EvalfDomain::Real) {
        return evalf_numeric(b, bits, true);
    } else if (domain == EvalfDomain::Complex) {
        return evalf_numeric(b, bits, false);
    }

    EvalVisitor v(bits);
    return v.apply(b.rcp_from_this());
}

} // namespace SymEngine
