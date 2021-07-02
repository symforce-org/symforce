#include <symengine/printers/codegen.h>
#include <symengine/printers.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{

void CodePrinter::bvisit(const Basic &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const Complex &x)
{
    throw NotImplementedError("Not implemented");
}
void CodePrinter::bvisit(const Interval &x)
{
    std::string var = str_;
    std::ostringstream s;
    bool is_inf = eq(*x.get_start(), *NegInf);
    if (not is_inf) {
        s << var;
        if (x.get_left_open()) {
            s << " > ";
        } else {
            s << " >= ";
        }
        s << apply(x.get_start());
    }
    if (neq(*x.get_end(), *Inf)) {
        if (not is_inf) {
            s << " && ";
        }
        s << var;
        if (x.get_right_open()) {
            s << " < ";
        } else {
            s << " <= ";
        }
        s << apply(x.get_end());
    }
    str_ = s.str();
}
void CodePrinter::bvisit(const Contains &x)
{
    x.get_expr()->accept(*this);
    x.get_set()->accept(*this);
}
void CodePrinter::bvisit(const Piecewise &x)
{
    std::ostringstream s;
    auto vec = x.get_vec();
    for (size_t i = 0;; ++i) {
        if (i == vec.size() - 1) {
            if (neq(*vec[i].second, *boolTrue)) {
                throw SymEngineException(
                    "Code generation requires a (Expr, True) at the end");
            }
            s << "(\n   " << apply(vec[i].first) << "\n";
            break;
        } else {
            s << "((";
            s << apply(vec[i].second);
            s << ") ? (\n   ";
            s << apply(vec[i].first);
            s << "\n)\n: ";
        }
    }
    for (size_t i = 0; i < vec.size(); i++) {
        s << ")";
    }
    str_ = s.str();
}
void CodePrinter::bvisit(const Rational &x)
{
    std::ostringstream o;
    double n = mp_get_d(get_num(x.as_rational_class()));
    double d = mp_get_d(get_den(x.as_rational_class()));
    o << print_double(n) << "/" << print_double(d);
    str_ = o.str();
}
void CodePrinter::bvisit(const Reals &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const Rationals &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const Integers &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const EmptySet &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const FiniteSet &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const UniversalSet &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const Abs &x)
{
    std::ostringstream s;
    s << "fabs(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void CodePrinter::bvisit(const Ceiling &x)
{
    std::ostringstream s;
    s << "ceil(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void CodePrinter::bvisit(const Truncate &x)
{
    std::ostringstream s;
    s << "trunc(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void CodePrinter::bvisit(const Max &x)
{
    std::ostringstream s;
    const auto &args = x.get_args();
    switch (args.size()) {
        case 0:
        case 1:
            throw SymEngineException("Impossible");
        case 2:
            s << "fmax(" << apply(args[0]) << ", " << apply(args[1]) << ")";
            break;
        default: {
            vec_basic inner_args(args.begin() + 1, args.end());
            auto inner = max(inner_args);
            s << "fmax(" << apply(args[0]) << ", " << apply(inner) << ")";
            break;
        }
    }
    str_ = s.str();
}
void CodePrinter::bvisit(const Min &x)
{
    std::ostringstream s;
    const auto &args = x.get_args();
    switch (args.size()) {
        case 0:
        case 1:
            throw SymEngineException("Impossible");
        case 2:
            s << "fmin(" << apply(args[0]) << ", " << apply(args[1]) << ")";
            break;
        default: {
            vec_basic inner_args(args.begin() + 1, args.end());
            auto inner = min(inner_args);
            s << "fmin(" << apply(args[0]) << ", " << apply(inner) << ")";
            break;
        }
    }
    str_ = s.str();
}
void CodePrinter::bvisit(const Constant &x)
{
    if (eq(x, *E)) {
        str_ = "exp(1)";
    } else if (eq(x, *pi)) {
        str_ = "acos(-1)";
    } else {
        str_ = x.get_name();
    }
}
void CodePrinter::bvisit(const NaN &x)
{
    std::ostringstream s;
    s << "NAN";
    str_ = s.str();
}
void CodePrinter::bvisit(const Equality &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " == " << apply(x.get_arg2());
    str_ = s.str();
}
void CodePrinter::bvisit(const Unequality &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " != " << apply(x.get_arg2());
    str_ = s.str();
}
void CodePrinter::bvisit(const LessThan &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " <= " << apply(x.get_arg2());
    str_ = s.str();
}
void CodePrinter::bvisit(const StrictLessThan &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " < " << apply(x.get_arg2());
    str_ = s.str();
}
void CodePrinter::bvisit(const UnivariateSeries &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const Derivative &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const Subs &x)
{
    throw SymEngineException("Not supported");
}
void CodePrinter::bvisit(const GaloisField &x)
{
    throw SymEngineException("Not supported");
}

void C89CodePrinter::bvisit(const Infty &x)
{
    std::ostringstream s;
    if (x.is_negative_infinity())
        s << "-HUGE_VAL";
    else if (x.is_positive_infinity())
        s << "HUGE_VAL";
    else
        throw SymEngineException("Not supported");
    str_ = s.str();
}
void C89CodePrinter::_print_pow(std::ostringstream &o,
                                const RCP<const Basic> &a,
                                const RCP<const Basic> &b)
{
    if (eq(*a, *E)) {
        o << "exp(" << apply(b) << ")";
    } else if (eq(*b, *rational(1, 2))) {
        o << "sqrt(" << apply(a) << ")";
    } else {
        o << "pow(" << apply(a) << ", " << apply(b) << ")";
    }
}

void C99CodePrinter::bvisit(const Infty &x)
{
    std::ostringstream s;
    if (x.is_negative_infinity())
        s << "-INFINITY";
    else if (x.is_positive_infinity())
        s << "INFINITY";
    else
        throw SymEngineException("Not supported");
    str_ = s.str();
}
void C99CodePrinter::_print_pow(std::ostringstream &o,
                                const RCP<const Basic> &a,
                                const RCP<const Basic> &b)
{
    if (eq(*a, *E)) {
        o << "exp(" << apply(b) << ")";
    } else if (eq(*b, *rational(1, 2))) {
        o << "sqrt(" << apply(a) << ")";
    } else if (eq(*b, *rational(1, 3))) {
        o << "cbrt(" << apply(a) << ")";
    } else {
        o << "pow(" << apply(a) << ", " << apply(b) << ")";
    }
}
void C99CodePrinter::bvisit(const Gamma &x)
{
    std::ostringstream s;
    s << "tgamma(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void C99CodePrinter::bvisit(const LogGamma &x)
{
    std::ostringstream s;
    s << "lgamma(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}

void JSCodePrinter::bvisit(const Constant &x)
{
    if (eq(x, *E)) {
        str_ = "Math.E";
    } else if (eq(x, *pi)) {
        str_ = "Math.PI";
    } else {
        str_ = x.get_name();
    }
}
void JSCodePrinter::_print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                               const RCP<const Basic> &b)
{
    if (eq(*a, *E)) {
        o << "Math.exp(" << apply(b) << ")";
    } else if (eq(*b, *rational(1, 2))) {
        o << "Math.sqrt(" << apply(a) << ")";
    } else if (eq(*b, *rational(1, 3))) {
        o << "Math.cbrt(" << apply(a) << ")";
    } else {
        o << "Math.pow(" << apply(a) << ", " << apply(b) << ")";
    }
}
void JSCodePrinter::bvisit(const Abs &x)
{
    std::ostringstream s;
    s << "Math.abs(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void JSCodePrinter::bvisit(const Sin &x)
{
    std::ostringstream s;
    s << "Math.sin(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void JSCodePrinter::bvisit(const Cos &x)
{
    std::ostringstream s;
    s << "Math.cos(" << apply(x.get_arg()) << ")";
    str_ = s.str();
}
void JSCodePrinter::bvisit(const Max &x)
{
    const auto &args = x.get_args();
    std::ostringstream s;
    s << "Math.max(";
    for (size_t i = 0; i < args.size(); ++i) {
        s << apply(args[i]);
        s << ((i == args.size() - 1) ? ")" : ", ");
    }
    str_ = s.str();
}
void JSCodePrinter::bvisit(const Min &x)
{
    const auto &args = x.get_args();
    std::ostringstream s;
    s << "Math.min(";
    for (size_t i = 0; i < args.size(); ++i) {
        s << apply(args[i]);
        s << ((i == args.size() - 1) ? ")" : ", ");
    }
    str_ = s.str();
}

std::string ccode(const Basic &x)
{
    C99CodePrinter c;
    return c.apply(x);
}

std::string jscode(const Basic &x)
{
    JSCodePrinter p;
    return p.apply(x);
}

std::string inline c89code(const Basic &x)
{
    C89CodePrinter p;
    return p.apply(x);
}

std::string inline c99code(const Basic &x)
{
    C99CodePrinter p;
    return p.apply(x);
}

} // namespace SymEngine
