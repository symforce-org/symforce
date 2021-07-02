#ifndef SYMENGINE_STR_PRINTER_H
#define SYMENGINE_STR_PRINTER_H

#include <symengine/visitor.h>
#include <symengine/printers.h>

namespace SymEngine
{

std::string print_double(double d);
std::vector<std::string> init_str_printer_names();

enum class PrecedenceEnum { Relational, Add, Mul, Pow, Atom };

class Precedence : public BaseVisitor<Precedence>
{
private:
    PrecedenceEnum precedence;

public:
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Relational &x);
    void bvisit(const Pow &x);
    template <typename Poly>
    void bvisit_upoly(const Poly &x)
    {
        if (x.end() == ++x.begin()) {
            auto it = x.begin();
            precedence = PrecedenceEnum::Atom;
            if (it->second == 1) {
                if (it->first == 0 or it->first == 1) {
                    precedence = PrecedenceEnum::Atom;
                } else {
                    precedence = PrecedenceEnum::Pow;
                }
            } else {
                if (it->first == 0) {
                    Expression(it->second).get_basic()->accept(*this);
                } else {
                    precedence = PrecedenceEnum::Mul;
                }
            }
        } else if (x.begin() == x.end()) {
            precedence = PrecedenceEnum::Atom;
        } else {
            precedence = PrecedenceEnum::Add;
        }
    }

    template <typename Container, typename Poly>
    void bvisit(const UPolyBase<Container, Poly> &x)
    {
        bvisit_upoly(down_cast<const Poly &>(x));
    }

    void bvisit(const GaloisField &x);

    template <typename Container, typename Poly>
    void bvisit(const MSymEnginePoly<Container, Poly> &x)
    {
        if (0 == x.get_poly().dict_.size()) {
            precedence = PrecedenceEnum::Atom;
        } else if (1 == x.get_poly().dict_.size()) {
            auto iter = x.get_poly().dict_.begin();
            precedence = PrecedenceEnum::Atom;
            bool first = true; // true if there are no nonzero exponents, false
                               // otherwise
            for (unsigned int exp : iter->first) {
                if (exp > 0) {
                    if (first && exp > 1)
                        precedence = PrecedenceEnum::Pow;
                    if (!first)
                        precedence = PrecedenceEnum::Mul;
                    first = false;
                }
            }
            if (!first) {
                if (iter->second != 1)
                    precedence = PrecedenceEnum::Mul;
            }
        } else {
            precedence = PrecedenceEnum::Add;
        }
    }

    void bvisit(const Rational &x);
    void bvisit(const Complex &x);
    void bvisit(const Integer &x);
    void bvisit(const RealDouble &x);
#ifdef HAVE_SYMENGINE_PIRANHA
    void bvisit(const URatPSeriesPiranha &x);
    void bvisit(const UPSeriesPiranha &x);
#endif
    void bvisit(const ComplexDouble &x);
#ifdef HAVE_SYMENGINE_MPFR
    void bvisit(const RealMPFR &x);
#endif
#ifdef HAVE_SYMENGINE_MPC
    void bvisit(const ComplexMPC &x);
#endif
    void bvisit(const Basic &x);
    PrecedenceEnum getPrecedence(const RCP<const Basic> &x);
};

std::vector<std::string> init_str_printer_names();

class StrPrinter : public BaseVisitor<StrPrinter>
{
private:
    static const std::vector<std::string> names_;

protected:
    std::string str_;
    virtual std::string print_mul();
    virtual bool split_mul_coef();
    virtual void _print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                            const RCP<const Basic> &b);
    virtual std::string print_div(const std::string &num,
                                  const std::string &den, bool paren);
    virtual std::string get_imag_symbol();
    virtual std::string parenthesize(const std::string &expr);
    std::string parenthesizeLT(const RCP<const Basic> &x,
                               PrecedenceEnum precedenceEnum);
    std::string parenthesizeLE(const RCP<const Basic> &x,
                               PrecedenceEnum precedenceEnum);

public:
    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Integer &x);
    void bvisit(const Rational &x);
    void bvisit(const Complex &x);
    void bvisit(const Interval &x);
    void bvisit(const Reals &x);
    void bvisit(const Rationals &x);
    void bvisit(const Integers &x);
    void bvisit(const Piecewise &x);
    void bvisit(const EmptySet &x);
    void bvisit(const FiniteSet &x);
    void bvisit(const UniversalSet &x);
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
    void bvisit(const UIntPoly &x);
    void bvisit(const MIntPoly &x);
    void bvisit(const URatPoly &x);
#ifdef HAVE_SYMENGINE_FLINT
    void bvisit(const UIntPolyFlint &x);
    void bvisit(const URatPolyFlint &x);
#endif
    void bvisit(const UExprPoly &x);
    void bvisit(const MExprPoly &x);
    void bvisit(const GaloisField &x);
    void bvisit(const Infty &x);
    void bvisit(const NaN &x);
    void bvisit(const UnivariateSeries &x);
#ifdef HAVE_SYMENGINE_PIRANHA
    void bvisit(const URatPSeriesPiranha &x);
    void bvisit(const UPSeriesPiranha &x);
    void bvisit(const UIntPolyPiranha &x);
    void bvisit(const URatPolyPiranha &x);
#endif
    void bvisit(const Constant &x);
    void bvisit(const Function &x);
    void bvisit(const FunctionSymbol &x);
    void bvisit(const Derivative &x);
    void bvisit(const Subs &x);
    void bvisit(const RealDouble &x);
    void bvisit(const ComplexDouble &x);
    void bvisit(const Equality &x);
    void bvisit(const Unequality &x);
    void bvisit(const LessThan &x);
    void bvisit(const StrictLessThan &x);
#ifdef HAVE_SYMENGINE_MPFR
    void bvisit(const RealMPFR &x);
#endif
#ifdef HAVE_SYMENGINE_MPC
    void bvisit(const ComplexMPC &x);
#endif
    void bvisit(const NumberWrapper &x);

    std::string apply(const RCP<const Basic> &b);
    std::string apply(const vec_basic &v);
    std::string apply(const Basic &b);
};

class JuliaStrPrinter : public BaseVisitor<JuliaStrPrinter, StrPrinter>
{
public:
    using StrPrinter::bvisit;
    virtual void _print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                            const RCP<const Basic> &b);
    virtual std::string get_imag_symbol();
    void bvisit(const Constant &x);
    void bvisit(const NaN &x);
    void bvisit(const Infty &x);
};
}

#endif // SYMENGINE_STR_PRINTER_H
