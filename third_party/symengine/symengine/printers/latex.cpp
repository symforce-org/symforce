#include <symengine/printers/latex.h>

namespace SymEngine
{

std::string latex(const Basic &x)
{
    LatexPrinter p;
    return p.apply(x);
}

void print_rational_class(const rational_class &r, std::ostringstream &s)
{
    if (get_den(r) == 1) {
        s << get_num(r);
    } else {
        s << "\\frac{" << get_num(r) << "}{" << get_den(r) << "}";
    }
}

void LatexPrinter::bvisit(const Symbol &x)
{
    std::string name = x.get_name();

    if (name.find('\\') != std::string::npos
        or name.find('{') != std::string::npos) {
        str_ = name;
        return;
    }
    if (name[0] == '_') {
        name = name.substr(1, name.size());
    }
    std::vector<std::string> greeks
        = {"alpha",  "beta",  "gamma", "Gamma", "delta",   "Delta",   "epsilon",
           "zeta",   "eta",   "theta", "Theta", "iota",    "kappa",   "lambda",
           "Lambda", "mu",    "nu",    "xi",    "omicron", "pi",      "Pi",
           "rho",    "sigma", "Sigma", "tau",   "upsilon", "Upsilon", "phi",
           "Phi",    "chi",   "psi",   "Psi",   "omega",   "Omega"};

    for (auto &letter : greeks) {
        if (name == letter) {
            str_ = "\\" + name;
            return;
        }
        if (name.size() > letter.size() and name.find(letter + "_") == 0) {
            str_ = "\\" + name;
            return;
        }
    }
    str_ = name;
    return;
}

void LatexPrinter::bvisit(const Rational &x)
{
    const auto &rational = x.as_rational_class();
    std::ostringstream s;
    print_rational_class(rational, s);
    str_ = s.str();
}

void LatexPrinter::bvisit(const Complex &x)
{
    std::ostringstream s;
    if (x.real_ != 0) {
        print_rational_class(x.real_, s);
        // Since Complex is in canonical form, imaginary_ is not 0.
        if (mp_sign(x.imaginary_) == 1) {
            s << " + ";
        } else {
            s << " - ";
        }
        // If imaginary_ is not 1 or -1, print the absolute value
        if (x.imaginary_ != mp_sign(x.imaginary_)) {
            print_rational_class(mp_abs(x.imaginary_), s);
            s << "j";
        } else {
            s << "j";
        }
    } else {
        if (x.imaginary_ != mp_sign(x.imaginary_)) {
            print_rational_class(x.imaginary_, s);
            s << "j";
        } else {
            if (mp_sign(x.imaginary_) == 1) {
                s << "j";
            } else {
                s << "-j";
            }
        }
    }
    str_ = s.str();
}

void LatexPrinter::bvisit(const ComplexBase &x)
{
    RCP<const Number> imag = x.imaginary_part();
    if (imag->is_negative()) {
        std::string str = apply(imag);
        str = str.substr(1, str.length() - 1);
        str_ = apply(x.real_part()) + " - " + str + "j";
    } else {
        str_ = apply(x.real_part()) + " + " + apply(imag) + "j";
    }
}
void LatexPrinter::bvisit(const ComplexDouble &x)
{
    bvisit(static_cast<const ComplexBase &>(x));
}

#ifdef HAVE_SYMENGINE_MPC
void LatexPrinter::bvisit(const ComplexMPC &x)
{
    bvisit(static_cast<const ComplexBase &>(x));
}
#endif

void LatexPrinter::bvisit(const Infty &x)
{
    if (x.is_negative_infinity()) {
        str_ = "-\\infty";
    } else if (x.is_positive_infinity()) {
        str_ = "\\infty";
    } else {
        str_ = "\\tilde{\\infty}";
    }
}

void LatexPrinter::bvisit(const NaN &x)
{
    str_ = "\\mathrm{NaN}";
}

void LatexPrinter::bvisit(const Constant &x)
{
    if (eq(x, *pi)) {
        str_ = "\\pi";
    } else if (eq(x, *E)) {
        str_ = "e";
    } else if (eq(x, *EulerGamma)) {
        str_ = "\\gamma";
    } else if (eq(x, *Catalan)) {
        str_ = "G";
    } else if (eq(x, *GoldenRatio)) {
        str_ = "\\phi";
    } else {
        throw NotImplementedError("Constant " + x.get_name()
                                  + " is not implemented.");
    }
}

void LatexPrinter::bvisit(const Derivative &x)
{
    const auto &symbols = x.get_symbols();
    std::ostringstream s;
    if (symbols.size() == 1) {
        if (free_symbols(*x.get_arg()).size() == 1) {
            s << "\\frac{d}{d " << apply(*symbols.begin());
        } else {
            s << "\\frac{\\partial}{\\partial " << apply(*symbols.begin());
        }
    } else {
        s << "\\frac{\\partial^" << symbols.size() << "}{";
        unsigned count = 1;
        auto it = symbols.begin();
        RCP<const Basic> prev = *it;
        ++it;
        for (; it != symbols.end(); ++it) {
            if (neq(*prev, **it)) {
                if (count == 1) {
                    s << "\\partial " << apply(*prev) << " ";
                } else {
                    s << "\\partial " << apply(*prev) << "^" << count << " ";
                }
                count = 1;
            } else {
                count++;
            }
            prev = *it;
        }
        if (count == 1) {
            s << "\\partial " << apply(*prev) << " ";
        } else {
            s << "\\partial " << apply(*prev) << "^" << count << " ";
        }
    }
    s << "} " << apply(x.get_arg());
    str_ = s.str();
}

void LatexPrinter::bvisit(const Subs &x)
{
    std::ostringstream o;
    o << "\\left. " << apply(x.get_arg()) << "\\right|_{\\substack{";
    for (auto p = x.get_dict().begin(); p != x.get_dict().end(); p++) {
        if (p != x.get_dict().begin()) {
            o << " \\\\ ";
        }
        o << apply(p->first) << "=" << apply(p->second);
    }
    o << "}}";
    str_ = o.str();
}

void LatexPrinter::bvisit(const Equality &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " = " << apply(x.get_arg2());
    str_ = s.str();
}

void LatexPrinter::bvisit(const Unequality &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " \\neq " << apply(x.get_arg2());
    str_ = s.str();
}

void LatexPrinter::bvisit(const LessThan &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " \\leq " << apply(x.get_arg2());
    str_ = s.str();
}

void LatexPrinter::bvisit(const StrictLessThan &x)
{
    std::ostringstream s;
    s << apply(x.get_arg1()) << " < " << apply(x.get_arg2());
    str_ = s.str();
}

void LatexPrinter::bvisit(const Interval &x)
{
    std::ostringstream s;
    if (x.get_left_open())
        s << "\\left(";
    else
        s << "\\left[";
    s << *x.get_start() << ", " << *x.get_end();
    if (x.get_right_open())
        s << "\\right)";
    else
        s << "\\right]";
    str_ = s.str();
}

void LatexPrinter::bvisit(const BooleanAtom &x)
{
    if (x.get_val()) {
        str_ = "\\mathrm{True}";
    } else {
        str_ = "\\mathrm{False}";
    }
}

void LatexPrinter::bvisit(const And &x)
{
    std::ostringstream s;
    auto container = x.get_container();
    if (is_a<Or>(**container.begin()) or is_a<Xor>(**container.begin())) {
        s << parenthesize(apply(*container.begin()));
    } else {
        s << apply(*container.begin());
    }

    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " \\wedge ";
        if (is_a<Or>(**it) or is_a<Xor>(**it)) {
            s << parenthesize(apply(*it));
        } else {
            s << apply(*it);
        }
    }
    str_ = s.str();
}

void LatexPrinter::bvisit(const Or &x)
{
    std::ostringstream s;
    auto container = x.get_container();
    if (is_a<And>(**container.begin()) or is_a<Xor>(**container.begin())) {
        s << parenthesize(apply(*container.begin()));
    } else {
        s << apply(*container.begin());
    }

    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " \\vee ";
        if (is_a<And>(**it) or is_a<Xor>(**it)) {
            s << parenthesize(apply(*it));
        } else {
            s << apply(*it);
        }
    }
    str_ = s.str();
}

void LatexPrinter::bvisit(const Xor &x)
{
    std::ostringstream s;
    auto container = x.get_container();
    if (is_a<Or>(**container.begin()) or is_a<And>(**container.begin())) {
        s << parenthesize(apply(*container.begin()));
    } else {
        s << apply(*container.begin());
    }

    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " \\veebar ";
        if (is_a<Or>(**it) or is_a<And>(**it)) {
            s << parenthesize(apply(*it));
        } else {
            s << apply(*it);
        }
    }
    str_ = s.str();
}

void LatexPrinter::print_with_args(const Basic &x, const std::string &join,
                                   std::ostringstream &s)
{
    vec_basic v = x.get_args();
    s << apply(*v.begin());

    for (auto it = ++(v.begin()); it != v.end(); ++it) {
        s << " " << join << " " << apply(*it);
    }
}

void LatexPrinter::bvisit(const Not &x)
{
    str_ = "\\neg " + apply(*x.get_arg());
}

void LatexPrinter::bvisit(const Union &x)
{
    std::ostringstream s;
    print_with_args(x, "\\cup", s);
    str_ = s.str();
}

void LatexPrinter::bvisit(const Complement &x)
{
    std::ostringstream s;
    s << apply(x.get_universe()) << " \\setminus " << apply(x.get_container());
    str_ = s.str();
}

void LatexPrinter::bvisit(const ImageSet &x)
{
    std::ostringstream s;
    s << "\\left\\{" << apply(*x.get_expr()) << "\\; |\\; ";
    s << apply(*x.get_symbol());
    s << " \\in " << apply(*x.get_baseset()) << "\\right\\}";
    str_ = s.str();
}

void LatexPrinter::bvisit(const ConditionSet &x)
{
    std::ostringstream s;
    s << "\\left\\{" << apply(*x.get_symbol()) << "\\; |\\; ";
    s << apply(x.get_condition()) << "\\right\\}";
    str_ = s.str();
}

void LatexPrinter::bvisit(const EmptySet &x)
{
    str_ = "\\emptyset";
}

void LatexPrinter::bvisit(const Reals &x)
{
    str_ = "\\mathbf{R}";
}

void LatexPrinter::bvisit(const Rationals &x)
{
    str_ = "\\mathbf{Q}";
}

void LatexPrinter::bvisit(const Integers &x)
{
    str_ = "\\mathbf{Z}";
}

void LatexPrinter::bvisit(const FiniteSet &x)
{
    std::ostringstream s;
    s << "\\left{";
    print_with_args(x, ",", s);
    s << "\\right}";
    str_ = s.str();
}

void LatexPrinter::bvisit(const Contains &x)
{
    std::ostringstream s;
    s << apply(x.get_expr()) << " \\in " << apply(x.get_set());
    str_ = s.str();
}

std::string LatexPrinter::print_mul()
{
    return " ";
}

bool LatexPrinter::split_mul_coef()
{
    return true;
}

std::vector<std::string> init_latex_printer_names()
{
    std::vector<std::string> names = init_str_printer_names();

    for (unsigned i = 0; i < names.size(); i++) {
        if (names[i] != "") {
            names[i] = "\\operatorname{" + names[i] + "}";
        }
    }
    names[SYMENGINE_SIN] = "\\sin";
    names[SYMENGINE_COS] = "\\cos";
    names[SYMENGINE_TAN] = "\\tan";
    names[SYMENGINE_COT] = "\\cot";
    names[SYMENGINE_CSC] = "\\csc";
    names[SYMENGINE_SEC] = "\\sec";
    names[SYMENGINE_ATAN2] = "\\operatorname{atan_2}";
    names[SYMENGINE_SINH] = "\\sinh";
    names[SYMENGINE_COSH] = "\\cosh";
    names[SYMENGINE_TANH] = "\\tanh";
    names[SYMENGINE_COTH] = "\\coth";
    names[SYMENGINE_LOG] = "\\log";
    names[SYMENGINE_ZETA] = "\\zeta";
    names[SYMENGINE_LAMBERTW] = "\\operatorname{W}";
    names[SYMENGINE_DIRICHLET_ETA] = "\\eta";
    names[SYMENGINE_KRONECKERDELTA] = "\\delta_";
    names[SYMENGINE_LEVICIVITA] = "\\varepsilon_";
    names[SYMENGINE_LOWERGAMMA] = "\\gamma";
    names[SYMENGINE_UPPERGAMMA] = "\\Gamma";
    names[SYMENGINE_BETA] = "\\operatorname{B}";
    names[SYMENGINE_LOG] = "\\log";
    names[SYMENGINE_GAMMA] = "\\Gamma";
    names[SYMENGINE_TRUNCATE] = "\\operatorname{truncate}";
    return names;
}

const std::vector<std::string> LatexPrinter::names_
    = init_latex_printer_names();

void LatexPrinter::bvisit(const Function &x)
{
    std::ostringstream o;
    o << names_[x.get_type_code()] << "{";
    vec_basic vec = x.get_args();
    o << parenthesize(apply(vec)) << "}";
    str_ = o.str();
}

void LatexPrinter::bvisit(const Floor &x)
{
    std::ostringstream o;
    o << "\\lfloor{" << apply(x.get_arg()) << "}\\rfloor";
    str_ = o.str();
}

void LatexPrinter::bvisit(const Ceiling &x)
{
    std::ostringstream o;
    o << "\\lceil{" << apply(x.get_arg()) << "}\\rceil";
    str_ = o.str();
}

void LatexPrinter::bvisit(const Abs &x)
{
    std::ostringstream o;
    o << "\\left|" << apply(x.get_arg()) << "}\\right|";
    str_ = o.str();
}

std::string LatexPrinter::parenthesize(const std::string &expr)
{
    return "\\left(" + expr + "\\right)";
}

void LatexPrinter::_print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                              const RCP<const Basic> &b)
{
    if (eq(*a, *E)) {
        o << "e^{" << apply(b) << "}";
    } else if (eq(*b, *rational(1, 2))) {
        o << "\\sqrt{" << apply(a) << "}";
    } else if (is_a<Rational>(*b)
               and eq(*static_cast<const Rational &>(*b).get_num(), *one)) {
        o << "\\sqrt[" << apply(static_cast<const Rational &>(*b).get_den())
          << "]{" << apply(a) << "}";
    } else {
        o << parenthesizeLE(a, PrecedenceEnum::Pow);
        Precedence prec;
        auto b_str = apply(b);
        if (b_str.size() > 1) {
            o << "^{" << b_str << "}";
        } else {
            o << "^" << b_str;
        }
    }
}

std::string LatexPrinter::print_div(const std::string &num,
                                    const std::string &den, bool paren)
{
    return "\\frac{" + num + "}{" + den + "}";
}

void LatexPrinter::bvisit(const Piecewise &x)
{
    std::ostringstream s;
    s << "\\begin{cases} ";
    const auto &vec = x.get_vec();
    auto it = vec.begin();
    auto it_last = --vec.end();
    while (it != vec.end()) {
        s << apply(it->first);
        if (it == it_last) {
            if (eq(*it->second, *boolTrue)) {
                s << " & \\text{otherwise} \\end{cases}";
            } else {
                s << " & \\text{for}\\: ";
                s << apply(it->second);
                s << " \\end{cases}";
            }
        } else {
            s << " & \\text{for}\\: ";
            s << apply(it->second);
            s << "\\\\";
        }
        it++;
    }
    str_ = s.str();
}
}
