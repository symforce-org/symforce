#ifndef LATEX_H
#define LATEX_H

#include <symengine/printers/strprinter.h>

namespace SymEngine
{

class LatexPrinter : public BaseVisitor<LatexPrinter, StrPrinter>
{
public:
    using StrPrinter::bvisit;

    void bvisit(const Symbol &x);
    void bvisit(const Rational &x);
    void bvisit(const Complex &x);
    void bvisit(const ComplexBase &x);
    void bvisit(const ComplexDouble &x);
#ifdef HAVE_SYMENGINE_MPC
    void bvisit(const ComplexMPC &x);
#endif
    void bvisit(const Interval &x);
    void bvisit(const Piecewise &x);
    void bvisit(const EmptySet &x);
    void bvisit(const Complexes &x);
    void bvisit(const Reals &x);
    void bvisit(const Rationals &x);
    void bvisit(const Integers &x);
    void bvisit(const Naturals &x);
    void bvisit(const Naturals0 &x);
    void bvisit(const FiniteSet &x);
    void bvisit(const ConditionSet &x);
    void bvisit(const Contains &x);
    void bvisit(const BooleanAtom &x);
    void bvisit(const And &x);
    void bvisit(const Or &x);
    void bvisit(const Xor &x);
    void bvisit(const Not &x);
    void bvisit(const Union &x);
    void bvisit(const Intersection &x);
    void bvisit(const Complement &x);
    void bvisit(const ImageSet &x);
    void bvisit(const Infty &x);
    void bvisit(const NaN &x);
    void bvisit(const Constant &x);
    void bvisit(const Function &x);
    void bvisit(const Abs &x);
    void bvisit(const Floor &x);
    void bvisit(const Ceiling &x);
    void bvisit(const Derivative &x);
    void bvisit(const Subs &x);
    void bvisit(const Equality &x);
    void bvisit(const Unequality &x);
    void bvisit(const LessThan &x);
    void bvisit(const StrictLessThan &x);
    void bvisit(const Tuple &x);

private:
    static const std::vector<std::string> names_;

protected:
    void print_with_args(const Basic &x, const std::string &join,
                         std::ostringstream &s);
    std::string parenthesize(const std::string &expr) override;
    void _print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                    const RCP<const Basic> &b) override;
    bool split_mul_coef() override;
    std::string print_mul() override;
    std::string print_div(const std::string &num, const std::string &den,
                          bool paren) override;
};
} // namespace SymEngine

#endif // LATEX_H
