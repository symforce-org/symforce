#ifndef SYMENGINE_UNICODE_H
#define SYMENGINE_UNICODE_H

#include <symengine/printers/strprinter.h>
#include <symengine/printers/stringbox.h>

namespace SymEngine
{

class UnicodePrinter : public BaseVisitor<UnicodePrinter>
{
private:
    static const std::vector<std::string> names_;
    static const std::vector<size_t> lengths_;

protected:
    StringBox box_;
    StringBox print_mul();
    void _print_pow(const RCP<const Basic> &a, const RCP<const Basic> &b);
    std::string get_imag_symbol();
    StringBox parenthesizeLT(const RCP<const Basic> &x,
                             PrecedenceEnum precedenceEnum);
    StringBox parenthesizeLE(const RCP<const Basic> &x,
                             PrecedenceEnum precedenceEnum);

public:
    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Integer &x);
    void bvisit(const Rational &x);
    void bvisit(const Complex &x);
    void bvisit(const Interval &x);
    void bvisit(const Complexes &x);
    void bvisit(const Reals &x);
    void bvisit(const Rationals &x);
    void bvisit(const Integers &x);
    void bvisit(const Naturals &x);
    void bvisit(const Naturals0 &x);
    void bvisit(const EmptySet &x);
    void bvisit(const UniversalSet &x);
    void bvisit(const Piecewise &x);
    void bvisit(const FiniteSet &x);
    void bvisit(const ConditionSet &x);
    void bvisit(const Union &x);
    void bvisit(const Intersection &x);
    void bvisit(const Complement &x);
    void bvisit(const ImageSet &x);
    void bvisit(const Contains &x);

    void bvisit(const BooleanAtom &x);
    void bvisit(const And &x);
    void bvisit(const Or &x);
    void bvisit(const Xor &x);
    void bvisit(const Not &x);

    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const Infty &x);
    void bvisit(const NaN &x);
    void bvisit(const Constant &x);
    void bvisit(const Abs &x);
    void bvisit(const Floor &x);
    void bvisit(const Ceiling &x);
    void bvisit(const Function &x);

    void bvisit(const FunctionSymbol &x);
    void bvisit(const RealDouble &x);
    void bvisit(const ComplexDouble &x);
    void bvisit(const Equality &x);
    void bvisit(const Unequality &x);
    void bvisit(const LessThan &x);
    void bvisit(const StrictLessThan &x);

    void bvisit(const Tuple &x);

    StringBox apply(const RCP<const Basic> &b);
    StringBox apply(const vec_basic &v);
    StringBox apply(const Basic &b);
};

} // namespace SymEngine

#endif // SYMENGINE_UNICODE_H
