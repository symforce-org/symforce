/**
 *  \file derivative.h
 *  Includes differentation functions
 *
 **/

#ifndef SYMENGINE_DERIVATIVE_H
#define SYMENGINE_DERIVATIVE_H

#include <symengine/basic.h>
#include <symengine/visitor.h>

namespace SymEngine
{

//! Differentiation w.r.t symbols
RCP<const Basic> diff(const RCP<const Basic> &arg, const RCP<const Symbol> &x,
                      bool cache = true);

//! SymPy style differentiation w.r.t non-symbols and symbols
RCP<const Basic> sdiff(const RCP<const Basic> &arg, const RCP<const Basic> &x,
                       bool cache = true);

class DiffVisitor : public BaseVisitor<DiffVisitor>
{
protected:
    const RCP<const Symbol> x;
    RCP<const Basic> result_;
    umap_basic_basic visited;
    bool cache;

public:
    DiffVisitor(const RCP<const Symbol> &x, bool cache = true)
        : x(x), cache(cache)
    {
    }
// Uncomment the following define in order to debug the methods:
#define debug_methods
#ifndef debug_methods
    void bvisit(const Basic &self);
#else
    // Here we do not have a 'Basic' fallback, but rather must implement all
    // virtual methods explicitly (if we miss one, the code will not compile).
    // This is useful to check that we have implemented all methods that we
    // wanted.
    void bvisit(const UnivariateSeries &self);
    void bvisit(const Max &self);
    void bvisit(const Min &self);
#endif
    void bvisit(const Number &self);
    void bvisit(const Constant &self);
    void bvisit(const Symbol &self);
    void bvisit(const Log &self);
    void bvisit(const Abs &self);
    void bvisit(const ASech &self);
    void bvisit(const ACoth &self);
    void bvisit(const ATanh &self);
    void bvisit(const ACosh &self);
    void bvisit(const ACsch &self);
    void bvisit(const ASinh &self);
    void bvisit(const Coth &self);
    void bvisit(const Tanh &self);
    void bvisit(const Sech &self);
    void bvisit(const Cosh &self);
    void bvisit(const Csch &self);
    void bvisit(const Sinh &self);
    void bvisit(const Subs &self);
    void bvisit(const Derivative &self);
    void bvisit(const OneArgFunction &self);
    void bvisit(const MultiArgFunction &self);
    void bvisit(const TwoArgFunction &self);
    void bvisit(const PolyGamma &self);
    void bvisit(const UpperGamma &self);
    void bvisit(const LowerGamma &self);
    void bvisit(const Zeta &self);
    void bvisit(const LambertW &self);
    void bvisit(const Add &self);
    void bvisit(const Mul &self);
    void bvisit(const Pow &self);
    void bvisit(const Sin &self);
    void bvisit(const Cos &self);
    void bvisit(const Tan &self);
    void bvisit(const Cot &self);
    void bvisit(const Csc &self);
    void bvisit(const Sec &self);
    void bvisit(const ASin &self);
    void bvisit(const ACos &self);
    void bvisit(const ASec &self);
    void bvisit(const ACsc &self);
    void bvisit(const ATan &self);
    void bvisit(const ACot &self);
    void bvisit(const ATan2 &self);
    void bvisit(const Erf &self);
    void bvisit(const Erfc &self);
    void bvisit(const Gamma &self);
    void bvisit(const LogGamma &self);
    void bvisit(const UnevaluatedExpr &self);
    void bvisit(const UIntPoly &self);
    void bvisit(const URatPoly &self);
#ifdef HAVE_SYMENGINE_PIRANHA
    void bvisit(const UIntPolyPiranha &self);
    void bvisit(const URatPolyPiranha &self);
#endif
#ifdef HAVE_SYMENGINE_FLINT
    void bvisit(const UIntPolyFlint &self);
    void bvisit(const URatPolyFlint &self);
#endif
    void bvisit(const UExprPoly &self);
    void bvisit(const MIntPoly &self);
    void bvisit(const MExprPoly &self);
    void bvisit(const FunctionWrapper &self);
    void bvisit(const Beta &self);
    void bvisit(const Set &self);
    void bvisit(const Boolean &self);
    void bvisit(const GaloisField &self);
    void bvisit(const Piecewise &self);
    const RCP<const Basic> &apply(const Basic &b);
    const RCP<const Basic> &apply(const RCP<const Basic> &b);
};

} // namespace SymEngine

#endif // SYMENGINE_DERIVATIVE_H
