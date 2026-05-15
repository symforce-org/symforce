#include <symengine/printers/sbml.h>
#include <symengine/printers.h>

namespace SymEngine
{

static std::vector<std::string> init_sbml_printer_names()
{
    std::vector<std::string> names = init_str_printer_names();
    names[SYMENGINE_LOG] = "ln";
    names[SYMENGINE_ASIN] = "arcsin";
    names[SYMENGINE_ACOS] = "arccos";
    names[SYMENGINE_ASEC] = "arcsec";
    names[SYMENGINE_ACSC] = "arccsc";
    names[SYMENGINE_ATAN] = "arctan";
    names[SYMENGINE_ACOT] = "arccot";
    names[SYMENGINE_ASINH] = "arcsinh";
    names[SYMENGINE_ACSCH] = "arccsch";
    names[SYMENGINE_ACOSH] = "arccosh";
    names[SYMENGINE_ATANH] = "arctanh";
    names[SYMENGINE_ACOTH] = "arccoth";
    names[SYMENGINE_ASECH] = "arcsech";
    return names;
}

void SbmlPrinter::_print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                             const RCP<const Basic> &b)
{
    if (eq(*a, *E)) {
        o << "exp(" << apply(b) << ")";
    } else if (eq(*b, *rational(1, 2))) {
        o << "sqrt(" << apply(a) << ")";
    } else {
        o << parenthesizeLE(a, PrecedenceEnum::Pow);
        o << "^";
        o << parenthesizeLE(b, PrecedenceEnum::Pow);
    }
}

void SbmlPrinter::bvisit(const BooleanAtom &x)
{
    if (x.get_val()) {
        str_ = "true";
    } else {
        str_ = "false";
    }
}

void SbmlPrinter::bvisit(const And &x)
{
    std::ostringstream s;
    const auto &container = x.get_container();
    s << "and(";
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << ", " << apply(*it);
    }
    s << ")";
    str_ = s.str();
}

void SbmlPrinter::bvisit(const Or &x)
{
    std::ostringstream s;
    const auto &container = x.get_container();
    s << "or(";
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << ", " << apply(*it);
    }
    s << ")";
    str_ = s.str();
}

void SbmlPrinter::bvisit(const Xor &x)
{
    std::ostringstream s;
    const auto &container = x.get_container();
    s << "xor(";
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << ", " << apply(*it);
    }
    s << ")";
    str_ = s.str();
}

void SbmlPrinter::bvisit(const Not &x)
{
    std::ostringstream s;
    s << "not(" << *x.get_arg() << ")";
    str_ = s.str();
}

void SbmlPrinter::bvisit(const Piecewise &x)
{
    std::ostringstream s;
    auto vec = x.get_vec();
    auto it = vec.begin();
    s << "piecewise(";
    while (it != vec.end()) {
        s << apply((*it).first);
        s << ", ";
        s << apply((*it).second);
        ++it;
        if (it != vec.end()) {
            s << ", ";
        }
    }
    s << ")";
    str_ = s.str();
}

void SbmlPrinter::bvisit(const Infty &x)
{
    str_ = "inf";
}

void SbmlPrinter::bvisit(const Constant &x)
{
    if (eq(x, *E)) {
        str_ = "exponentiale";
    } else {
        str_ = x.get_name();
        std::transform(str_.begin(), str_.end(), str_.begin(), ::tolower);
    }
}

void SbmlPrinter::bvisit(const Function &x)
{
    static const std::vector<std::string> names_ = init_sbml_printer_names();
    std::ostringstream o;
    vec_basic vec = x.get_args();
    if (x.get_type_code() == SYMENGINE_GAMMA) {
        // sbml only has factorial, no gamma function
        o << "factorial(" << apply(vec) << " - 1)";
    } else if (x.get_type_code() == SYMENGINE_LOG && vec.size() == 2) {
        // sbml log has order of arguments inverted
        o << "log(" << apply(vec[1]) << ", " << apply(vec[0]) << ")";
    } else {
        o << names_[x.get_type_code()];
        o << parenthesize(apply(vec));
    }
    str_ = o.str();
}

std::string sbml(const Basic &x)
{
    SbmlPrinter m;
    return m.apply(x);
}

} // namespace SymEngine
