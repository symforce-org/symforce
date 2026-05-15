#ifndef SYMENGINE_MATHML_H
#define SYMENGINE_MATHML_H

#include <symengine/visitor.h>
#include <symengine/printers/strprinter.h>

namespace SymEngine
{
class MathMLPrinter : public BaseVisitor<MathMLPrinter, StrPrinter>
{
protected:
    std::ostringstream s;

public:
    static const std::vector<std::string> names_;
    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Integer &x);
    void bvisit(const Rational &x);
    void bvisit(const ComplexBase &x);
    void bvisit(const Interval &x);
    void bvisit(const Piecewise &x);
    void bvisit(const EmptySet &x);
    void bvisit(const Complexes &x);
    void bvisit(const Reals &x);
    void bvisit(const Rationals &x);
    void bvisit(const Integers &x);
    void bvisit(const FiniteSet &x);
    void bvisit(const ConditionSet &x);
    void bvisit(const Contains &x);
    void bvisit(const BooleanAtom &x);
    void bvisit(const And &x);
    void bvisit(const Or &x);
    void bvisit(const Xor &x);
    void bvisit(const Not &x);
    void bvisit(const Union &x);
    void bvisit(const Complement &x);
    void bvisit(const ImageSet &x);
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const Constant &x);
    void bvisit(const Function &x);
    void bvisit(const FunctionSymbol &x);
    void bvisit(const Derivative &x);
    void bvisit(const UnevaluatedExpr &x);
    // void bvisit(const Subs &x);
    void bvisit(const RealDouble &x);
    void bvisit(const Equality &x);
    void bvisit(const Unequality &x);
    void bvisit(const LessThan &x);
    void bvisit(const StrictLessThan &x);
#ifdef HAVE_SYMENGINE_MPFR
    void bvisit(const RealMPFR &x);
#endif
    // void bvisit(const NumberWrapper &x);
    std::string apply(const Basic &b);
};
} // namespace SymEngine

#endif // SYMENGINE_MATHML_H
