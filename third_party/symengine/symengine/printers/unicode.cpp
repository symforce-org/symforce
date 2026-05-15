#include <limits>
#include <symengine/printers/strprinter.h>
#include <symengine/printers/unicode.h>

// Macro to let string literals be unicode const char in all C++ standards
// Otherwise u8"" would be char8_t in C++20
#define U8(x) reinterpret_cast<const char *>(u8##x)

namespace SymEngine
{
//! Less operator `(<)` using cmp:
struct PrinterBasicCmp {
    //! true if `x < y`, false otherwise
    bool operator()(const RCP<const Basic> &x, const RCP<const Basic> &y) const
    {
        if (x->__eq__(*y))
            return false;
        return x->__cmp__(*y) == -1;
    }
};

void UnicodePrinter::bvisit(const Basic &x)
{
    std::ostringstream s;
    s << U8("<") << typeName<Basic>(x) << U8(" instance at ")
      << (const void *)this << U8(">");
    StringBox box(s.str());
    box_ = box;
}

void UnicodePrinter::bvisit(const Symbol &x)
{
    box_ = StringBox(x.get_name());
}

void UnicodePrinter::bvisit(const Infty &x)
{
    if (x.is_negative_infinity())
        box_ = StringBox(U8("-\u221E"), 2);
    else if (x.is_positive_infinity())
        box_ = StringBox(U8("\u221E"), 1);
    else
        box_ = StringBox(U8("\U0001D467\u221E"), 2);
}

void UnicodePrinter::bvisit(const NaN &x)
{
    box_ = StringBox(U8("NaN"));
}

void UnicodePrinter::bvisit(const Integer &x)
{
    std::ostringstream s;
    s << x.as_integer_class();
    box_ = StringBox(s.str());
}

void UnicodePrinter::bvisit(const Rational &x)
{
    std::ostringstream num;
    num << (*x.get_num()).as_integer_class();
    StringBox rat(num.str());
    std::ostringstream denom;
    denom << (*x.get_den()).as_integer_class();
    StringBox denbox(denom.str());
    rat.add_below_unicode_line(denbox);
    box_ = rat;
}

void UnicodePrinter::bvisit(const Complex &x)
{
    std::ostringstream s;
    bool mul = false;
    if (x.real_ != 0) {
        s << x.real_;
        // Since Complex is in canonical form, imaginary_ is not 0.
        if (mp_sign(x.imaginary_) == 1) {
            s << U8(" + ");
        } else {
            s << U8(" - ");
        }
        // If imaginary_ is not 1 or -1, print the absolute value
        if (x.imaginary_ != mp_sign(x.imaginary_)) {
            s << mp_abs(x.imaginary_);
            s << U8("\u22C5") << get_imag_symbol();
            mul = true;
        } else {
            s << get_imag_symbol();
        }
    } else {
        if (x.imaginary_ != mp_sign(x.imaginary_)) {
            s << x.imaginary_;
            s << U8("\u22C5") << get_imag_symbol();
            mul = true;
        } else {
            if (mp_sign(x.imaginary_) == 1) {
                s << get_imag_symbol();
            } else {
                s << U8("-") << get_imag_symbol();
            }
        }
    }
    std::string str = s.str();
    std::size_t width = str.length() - 3;
    if (mul)
        width--;
    StringBox box(str, width);
    box_ = box;
}

void UnicodePrinter::bvisit(const RealDouble &x)
{
    box_ = StringBox(print_double(x.i));
}

void UnicodePrinter::bvisit(const ComplexDouble &x)
{
    std::string str = print_double(x.i.real());
    if (x.i.imag() < 0) {
        str += U8(" - ") + print_double(-x.i.imag());
    } else {
        str += U8(" + ") + print_double(x.i.imag());
    }
    auto len = str.length();
    str += U8("\u22C5") + get_imag_symbol();
    box_ = StringBox(str, len + 2);
}

void UnicodePrinter::bvisit(const Equality &x)
{
    StringBox box = apply(x.get_arg1());
    StringBox eq(" = ");
    box.add_right(eq);
    StringBox rhs = apply(x.get_arg2());
    box.add_right(rhs);
    box_ = box;
}

void UnicodePrinter::bvisit(const Unequality &x)
{
    StringBox box = apply(x.get_arg1());
    StringBox eq(U8(" \u2260 "), 3);
    box.add_right(eq);
    StringBox rhs = apply(x.get_arg2());
    box.add_right(rhs);
    box_ = box;
}

void UnicodePrinter::bvisit(const LessThan &x)
{
    StringBox box = apply(x.get_arg1());
    StringBox eq(U8(" \u2264 "), 3);
    box.add_right(eq);
    StringBox rhs = apply(x.get_arg2());
    box.add_right(rhs);
    box_ = box;
}

void UnicodePrinter::bvisit(const StrictLessThan &x)
{
    StringBox box = apply(x.get_arg1());
    StringBox eq(" < ", 3);
    box.add_right(eq);
    StringBox rhs = apply(x.get_arg2());
    box.add_right(rhs);
    box_ = box;
}

void UnicodePrinter::bvisit(const Interval &x)
{
    StringBox box = apply(x.get_start());
    StringBox comma = StringBox(", ");
    box.add_right(comma);
    StringBox end = StringBox(apply(x.get_end()));
    box.add_right(end);
    if (x.get_left_open()) {
        box.add_left_parens();
    } else {
        box.add_left_sqbracket();
    }
    if (x.get_right_open())
        box.add_right_parens();
    else
        box.add_right_sqbracket();
    box_ = box;
}

void UnicodePrinter::bvisit(const BooleanAtom &x)
{
    if (x.get_val()) {
        box_ = StringBox("true");
    } else {
        box_ = StringBox("false");
    }
}

void UnicodePrinter::bvisit(const And &x)
{
    auto container = x.get_container();
    StringBox box = apply(*container.begin());
    StringBox op(U8(" \u2227 "), 3);
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        box.add_right(op);
        StringBox next = apply(*it);
        box.add_right(next);
    }
    box_ = box;
}

void UnicodePrinter::bvisit(const Or &x)
{
    auto container = x.get_container();
    StringBox box = apply(*container.begin());
    StringBox op(U8(" \u2228 "), 3);
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        box.add_right(op);
        StringBox next = apply(*it);
        box.add_right(next);
    }
    box_ = box;
}

void UnicodePrinter::bvisit(const Xor &x)
{
    auto container = x.get_container();
    StringBox box = apply(*container.begin());
    StringBox op(U8(" \u22BB "), 3);
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        box.add_right(op);
        StringBox next = apply(*it);
        box.add_right(next);
    }
    box_ = box;
}

void UnicodePrinter::bvisit(const Not &x)
{
    StringBox box(U8("\u00AC"), 1);
    StringBox expr = apply(*x.get_arg());
    expr.enclose_parens();
    box.add_right(expr);
    box_ = box;
}

void UnicodePrinter::bvisit(const Contains &x)
{
    StringBox s = apply(x.get_expr());
    StringBox op(U8(" \u220A "), 3);
    s.add_right(op);
    auto right = apply(x.get_set());
    s.add_right(right);
    box_ = s;
}

void UnicodePrinter::bvisit(const Piecewise &x)
{
    StringBox box;

    auto vec = x.get_vec();
    auto it = vec.begin();
    while (true) {
        StringBox piece = apply((*it).first);
        StringBox mid(" if ");
        piece.add_right(mid);
        StringBox second = apply((*it).second);
        piece.add_right(second);
        box.add_below(piece);
        ++it;
        if (it == vec.end()) {
            break;
        }
    }
    box.add_left_curly();
    box_ = box;
}

void UnicodePrinter::bvisit(const Complexes &x)
{
    box_ = StringBox(U8("\u2102"), 1);
}

void UnicodePrinter::bvisit(const Reals &x)
{
    box_ = StringBox(U8("\u211D"), 1);
}

void UnicodePrinter::bvisit(const Rationals &x)
{
    box_ = StringBox(U8("\u211A"), 1);
}

void UnicodePrinter::bvisit(const Integers &x)
{
    box_ = StringBox(U8("\u2124"), 1);
}

void UnicodePrinter::bvisit(const Naturals &x)
{
    box_ = StringBox(U8("\u2115"), 1);
}

void UnicodePrinter::bvisit(const Naturals0 &x)
{
    box_ = StringBox(U8("\u2115\u2080"), 2);
}

void UnicodePrinter::bvisit(const EmptySet &x)
{
    box_ = StringBox(U8("\u2205"), 1);
}

void UnicodePrinter::bvisit(const UniversalSet &x)
{
    box_ = StringBox(U8("\U0001D54C"), 1);
}

void UnicodePrinter::bvisit(const Union &x)
{
    auto container = x.get_container();
    StringBox box = apply(*container.begin());
    StringBox op(U8(" \u222A "), 3);
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        box.add_right(op);
        StringBox next = apply(*it);
        box.add_right(next);
    }
    box_ = box;
}

void UnicodePrinter::bvisit(const Intersection &x)
{
    auto container = x.get_container();
    StringBox box = apply(*container.begin());
    StringBox op(U8(" \u2229 "), 3);
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        box.add_right(op);
        StringBox next = apply(*it);
        box.add_right(next);
    }
    box_ = box;
}

void UnicodePrinter::bvisit(const Complement &x)
{
    StringBox box = apply(*x.get_universe());
    StringBox op(U8(" \\ "));
    box.add_right(op);
    StringBox rhs = apply(*x.get_container());
    box.add_right(rhs);
    box_ = box;
}

void UnicodePrinter::bvisit(const ImageSet &x)
{
    StringBox box = apply(*x.get_expr());
    StringBox bar(" | ");
    box.add_right(bar);
    StringBox symbol = apply(*x.get_symbol());
    box.add_right(symbol);
    StringBox in(U8(" \u220A "), 3);
    box.add_right(in);
    StringBox base = apply(*x.get_baseset());
    box.add_right(base);
    box.enclose_curlies();
    box_ = box;
}

void UnicodePrinter::bvisit(const FiniteSet &x)
{
    StringBox box;
    StringBox comma(", ");
    bool first = true;
    for (const auto &elem : x.get_container()) {
        if (not first) {
            box.add_right(comma);
        } else {
            first = false;
        }
        StringBox arg = apply(elem);
        box.add_right(arg);
    }
    box.enclose_curlies();
    box_ = box;
}

void UnicodePrinter::bvisit(const ConditionSet &x)
{
    StringBox box = apply(*x.get_symbol());
    StringBox bar(" | ");
    box.add_right(bar);
    StringBox cond = apply(*x.get_condition());
    box.add_right(cond);
    box.enclose_curlies();
    box_ = box;
}

void UnicodePrinter::bvisit(const Add &x)
{
    StringBox box;
    bool first = true;
    std::map<RCP<const Basic>, RCP<const Number>, PrinterBasicCmp> dict(
        x.get_dict().begin(), x.get_dict().end());

    if (neq(*(x.get_coef()), *zero)) {
        box = apply(x.get_coef());
        first = false;
    }
    bool minus = false;
    for (const auto &p : dict) {
        StringBox t;
        if (eq(*(p.second), *one)) {
            t = parenthesizeLT(p.first, PrecedenceEnum::Add);
        } else if (eq(*(p.second), *minus_one)) {
            minus = true;
            t = parenthesizeLT(p.first, PrecedenceEnum::Mul);
        } else {
            if (down_cast<const Number &>(*p.second).is_negative()) {
                minus = true;
            }
            // FIXME: Double minus here
            t = parenthesizeLT(p.second, PrecedenceEnum::Mul);
            auto op = print_mul();
            t.add_right(op);
            auto rhs = parenthesizeLT(p.first, PrecedenceEnum::Mul);
            t.add_right(rhs);
        }

        if (not first) {
            if (minus) {
                StringBox op(" - ");
                box.add_right(op);
                box.add_right(t);
                minus = false;
            } else {
                StringBox op(" + ");
                box.add_right(op);
                box.add_right(t);
            }
        } else {
            box.add_right(t);
            first = false;
        }
    }
    box_ = box;
}

void UnicodePrinter::_print_pow(const RCP<const Basic> &a,
                                const RCP<const Basic> &b)
{
    if (eq(*b, *rational(1, 2))) {
        StringBox box = apply(a);
        box.enclose_sqrt();
        box_ = box;
    } else {
        StringBox base = parenthesizeLE(a, PrecedenceEnum::Pow);
        StringBox exp = parenthesizeLE(b, PrecedenceEnum::Pow);
        base.add_power(exp);
        box_ = base;
    }
}

void UnicodePrinter::bvisit(const Mul &x)
{
    StringBox box1, box2;
    bool num = false;
    unsigned den = 0;
    StringBox mulbox = print_mul();

    bool first_box1 = true;
    bool first_box2 = true;

    if (eq(*(x.get_coef()), *minus_one)) {
        box1 = StringBox("-");
    } else if (neq(*(x.get_coef()), *one)) {
        RCP<const Basic> numer, denom;
        as_numer_denom(x.get_coef(), outArg(numer), outArg(denom));
        if (neq(*numer, *one)) {
            num = true;
            box1 = parenthesizeLT(numer, PrecedenceEnum::Mul);
            first_box1 = false;
        }
        if (neq(*denom, *one)) {
            den++;
            box2 = parenthesizeLT(denom, PrecedenceEnum::Mul);
            first_box2 = false;
        }
    }

    for (const auto &p : x.get_dict()) {
        if ((is_a<Integer>(*p.second) or is_a<Rational>(*p.second))
            and down_cast<const Number &>(*p.second).is_negative()) {
            if (not first_box2) {
                box2.add_right(mulbox);
            } else {
                first_box2 = false;
            }
            if (eq(*(p.second), *minus_one)) {
                auto expr = parenthesizeLT(p.first, PrecedenceEnum::Mul);
                box2.add_right(expr);
            } else {
                _print_pow(p.first, neg(p.second));
                box2.add_right(box_);
            }
            den++;
        } else {
            if (not first_box1) {
                box1.add_right(mulbox);
            } else {
                first_box1 = false;
            }
            if (eq(*(p.second), *one)) {
                auto expr = parenthesizeLT(p.first, PrecedenceEnum::Mul);
                box1.add_right(expr);
            } else {
                _print_pow(p.first, p.second);
                box1.add_right(box_);
            }
            num = true;
        }
    }

    if (not num) {
        auto onebox = StringBox("1");
        box1.add_right(onebox);
        box1.add_right(mulbox);
    }

    if (den != 0) {
        if (den > 1) {
            box2.enclose_parens();
        }
        box1.add_below_unicode_line(box2);
    }
    box_ = box1;
}

void UnicodePrinter::bvisit(const Pow &x)
{
    _print_pow(x.get_base(), x.get_exp());
}

void UnicodePrinter::bvisit(const Constant &x)
{
    // NOTE: Using italics for constants which is very common in mathematics
    // typesetting. (It goes against the ISO typesetting-standard though.)
    if (eq(x, *pi)) {
        box_ = StringBox(U8("\U0001D70B"), 1);
    } else if (eq(x, *E)) {
        box_ = StringBox(U8("\U0001D452"), 1);
    } else if (eq(x, *EulerGamma)) {
        box_ = StringBox(U8("\U0001D6FE"), 1);
    } else if (eq(x, *Catalan)) {
        box_ = StringBox(U8("\U0001D43A"), 1);
    } else if (eq(x, *GoldenRatio)) {
        box_ = StringBox(U8("\U0001D719"), 1);
    }
}

StringBox UnicodePrinter::apply(const vec_basic &d)
{
    StringBox box("");
    StringBox comma(", ");
    for (auto p = d.begin(); p != d.end(); p++) {
        if (p != d.begin()) {
            box.add_right(comma);
        }
        StringBox arg = apply(*p);
        box.add_right(arg);
    }
    return box;
}

void UnicodePrinter::bvisit(const Abs &x)
{
    StringBox box = apply(*x.get_arg());
    box.enclose_abs();
    box_ = box;
}

void UnicodePrinter::bvisit(const Floor &x)
{
    StringBox box = apply(*x.get_arg());
    box.enclose_floor();
    box_ = box;
}

void UnicodePrinter::bvisit(const Ceiling &x)
{
    StringBox box = apply(*x.get_arg());
    box.enclose_ceiling();
    box_ = box;
}

static std::vector<std::string> init_unicode_printer_names()
{
    std::vector<std::string> names = init_str_printer_names();
    names[SYMENGINE_LAMBERTW] = "W";
    names[SYMENGINE_ZETA] = U8("\U0001D701");
    names[SYMENGINE_DIRICHLET_ETA] = U8("\U0001D702");
    names[SYMENGINE_LOWERGAMMA] = U8("\U0001D6FE");
    names[SYMENGINE_UPPERGAMMA] = U8("\u0393");
    names[SYMENGINE_BETA] = U8("B");
    names[SYMENGINE_LOGGAMMA] = U8("log \u0393");
    names[SYMENGINE_GAMMA] = U8("\u0393");
    names[SYMENGINE_PRIMEPI] = U8("\U0001D70B");
    return names;
}

static std::vector<size_t>
init_unicode_printer_lengths(const std::vector<std::string> &names)
{
    std::vector<size_t> lengths;
    for (auto &name : names) {
        lengths.push_back(name.length());
    }
    lengths[SYMENGINE_LAMBERTW] = 1;
    lengths[SYMENGINE_ZETA] = 1;
    lengths[SYMENGINE_DIRICHLET_ETA] = 1;
    lengths[SYMENGINE_LOWERGAMMA] = 1;
    lengths[SYMENGINE_UPPERGAMMA] = 1;
    lengths[SYMENGINE_BETA] = 1;
    lengths[SYMENGINE_LOGGAMMA] = 5;
    lengths[SYMENGINE_GAMMA] = 1;
    lengths[SYMENGINE_PRIMEPI] = 1;
    return lengths;
}

void UnicodePrinter::bvisit(const Function &x)
{
    static const std::vector<std::string> names_ = init_unicode_printer_names();
    static const std::vector<size_t> lengths_
        = init_unicode_printer_lengths(names_);
    StringBox box(names_[x.get_type_code()], lengths_[x.get_type_code()]);
    vec_basic vec = x.get_args();
    StringBox args = apply(vec);
    args.enclose_parens();
    box.add_right(args);
    box_ = box;
}

void UnicodePrinter::bvisit(const FunctionSymbol &x)
{
    StringBox box(x.get_name());
    StringBox args;
    StringBox comma(", ");
    bool first = true;
    for (auto arg : x.get_args()) {
        if (first) {
            first = false;
        } else {
            args.add_right(comma);
        }
        StringBox argbox = apply(arg);
        args.add_right(argbox);
    }
    args.enclose_parens();
    box.add_right(args);
    box_ = box;
}

void UnicodePrinter::bvisit(const Tuple &x)
{
    vec_basic vec = x.get_args();
    StringBox args = apply(vec);
    args.enclose_parens();
    box_ = args;
}

StringBox UnicodePrinter::parenthesizeLT(const RCP<const Basic> &x,
                                         PrecedenceEnum precedenceEnum)
{
    Precedence prec;
    if (prec.getPrecedence(x) < precedenceEnum) {
        auto box = apply(x);
        box.enclose_parens();
        return box;
    } else {
        return apply(x);
    }
}

StringBox UnicodePrinter::parenthesizeLE(const RCP<const Basic> &x,
                                         PrecedenceEnum precedenceEnum)
{
    Precedence prec;
    if (prec.getPrecedence(x) <= precedenceEnum) {
        auto box = apply(x);
        box.enclose_parens();
        return box;
    } else {
        return apply(x);
    }
}

StringBox UnicodePrinter::apply(const RCP<const Basic> &b)
{
    b->accept(*this);
    return box_;
}

StringBox UnicodePrinter::apply(const Basic &b)
{
    b.accept(*this);
    return box_;
}

StringBox UnicodePrinter::print_mul()
{
    return StringBox(U8("\u22C5"), 1);
}

std::string UnicodePrinter::get_imag_symbol()
{
    return U8("\U0001D456");
}

std::string unicode(const Basic &x)
{
    UnicodePrinter printer;
    return printer.apply(x).get_string();
}

} // namespace SymEngine
