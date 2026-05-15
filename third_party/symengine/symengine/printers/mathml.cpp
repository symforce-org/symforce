#include <limits>
#include <symengine/printers/mathml.h>
#include <symengine/eval_double.h>
#include <symengine/printers.h>

namespace SymEngine
{

std::vector<std::string> init_mathml_printer_names()
{
    std::vector<std::string> names = init_str_printer_names();
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

void MathMLPrinter::bvisit(const Basic &x)
{
    throw SymEngineException("Error: not supported");
}

void MathMLPrinter::bvisit(const Symbol &x)
{
    s << "<ci>" << x.get_name() << "</ci>";
}

void MathMLPrinter::bvisit(const Integer &x)
{
    s << "<cn type=\"integer\">" << x.as_integer_class() << "</cn>";
}

void MathMLPrinter::bvisit(const Rational &x)
{
    const auto &rational = x.as_rational_class();
    s << "<cn type=\"rational\">" << get_num(rational) << "<sep/>"
      << get_den(rational) << "</cn>";
}

void MathMLPrinter::bvisit(const RealDouble &x)
{
    s << "<cn type=\"real\">" << x << "</cn>";
}

#ifdef HAVE_SYMENGINE_MPFR
void MathMLPrinter::bvisit(const RealMPFR &x)
{
    // TODO: Use bigfloat here
    s << "<cn type=\"real\">" << x << "</cn>";
}
#endif

void MathMLPrinter::bvisit(const ComplexBase &x)
{
    s << "<apply><csymbol cd=\"nums1\">complex_cartesian</csymbol>";
    x.real_part()->accept(*this);
    x.imaginary_part()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Interval &x)
{
    s << "<interval closure=";
    if (x.get_left_open()) {
        if (x.get_right_open()) {
            s << "\"open\">";
        } else {
            s << "\"open-closed\">";
        }
    } else {
        if (x.get_right_open()) {
            s << "\"closed-open\">";
        } else {
            s << "\"closed\">";
        }
    }
    x.get_start()->accept(*this);
    x.get_end()->accept(*this);
    s << "</interval>";
}

void MathMLPrinter::bvisit(const Piecewise &x)
{
    s << "<piecewise>";
    const auto &equations = x.get_vec();
    for (const auto &equation : equations) {
        s << "<piece>";
        equation.first->accept(*this);
        equation.second->accept(*this);
        s << "</piece>";
    }
    s << "</piecewise>";
}

void MathMLPrinter::bvisit(const EmptySet &x)
{
    s << "<emptyset/>";
}

void MathMLPrinter::bvisit(const Complexes &x)
{
    s << "<complexes/>";
}

void MathMLPrinter::bvisit(const Reals &x)
{
    s << "<reals/>";
}

void MathMLPrinter::bvisit(const Rationals &x)
{
    s << "<rationals/>";
}

void MathMLPrinter::bvisit(const Integers &x)
{
    s << "<integers/>";
}

void MathMLPrinter::bvisit(const FiniteSet &x)
{
    s << "<set>";
    const auto &args = x.get_args();
    for (const auto &arg : args) {
        arg->accept(*this);
    }
    s << "</set>";
}

void MathMLPrinter::bvisit(const ConditionSet &x)
{
    s << "<set><bvar>";
    x.get_symbol()->accept(*this);
    s << "</bvar><condition>";
    x.get_condition()->accept(*this);
    s << "</condition>";
    x.get_symbol()->accept(*this);
    s << "</set>";
}

void MathMLPrinter::bvisit(const Contains &x)
{
    s << "<apply><in/>";
    x.get_expr()->accept(*this);
    x.get_set()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const BooleanAtom &x)
{
    if (x.get_val()) {
        s << "<true/>";
    } else {
        s << "<false/>";
    }
}

void MathMLPrinter::bvisit(const And &x)
{
    s << "<apply><and/>";
    const auto &conditions = x.get_args();
    for (const auto &condition : conditions) {
        condition->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Or &x)
{
    s << "<apply><or/>";
    const auto &conditions = x.get_args();
    for (const auto &condition : conditions) {
        condition->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Xor &x)
{
    s << "<apply><xor/>";
    const auto &conditions = x.get_args();
    for (const auto &condition : conditions) {
        condition->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Not &x)
{
    s << "<apply><not/>";
    x.get_arg()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Union &x)
{
    s << "<apply><union/>";
    const auto &sets = x.get_args();
    for (const auto &set : sets) {
        set->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Complement &x)
{
    s << "<apply><setdiff/>";
    x.get_universe()->accept(*this);
    x.get_container()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const ImageSet &x)
{
    s << "<set><bvar>";
    x.get_expr()->accept(*this);
    s << "</bvar><condition><apply><in/>";
    x.get_symbol()->accept(*this);
    x.get_baseset()->accept(*this);
    s << "</apply></condition>";
    x.get_symbol()->accept(*this);
    s << "</set>";
}

void MathMLPrinter::bvisit(const Add &x)
{
    s << "<apply><plus/>";
    auto args = x.get_args();
    for (auto arg : args) {
        arg->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Mul &x)
{
    s << "<apply><times/>";
    auto args = x.get_args();
    for (auto arg : args) {
        arg->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Pow &x)
{
    s << "<apply><power/>";
    x.get_base()->accept(*this);
    x.get_exp()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Constant &x)
{
    s << "<";
    if (eq(x, *pi)) {
        s << "pi/";
    } else if (eq(x, *E)) {
        s << "exponentiale/";
    } else if (eq(x, *EulerGamma)) {
        s << "eulergamma/";
    } else {
        s << "cn type=\"real\">" << eval_double(x) << "</cn";
    }
    s << ">";
}

void MathMLPrinter::bvisit(const Function &x)
{
    static const std::vector<std::string> names_ = init_mathml_printer_names();
    s << "<apply>";
    s << "<" << names_[x.get_type_code()] << "/>";
    const auto &args = x.get_args();
    for (const auto &arg : args) {
        arg->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const UnevaluatedExpr &x)
{
    apply(*x.get_arg());
}

void MathMLPrinter::bvisit(const FunctionSymbol &x)
{
    s << "<apply><ci>" << x.get_name() << "</ci>";
    const auto &args = x.get_args();
    for (const auto &arg : args) {
        arg->accept(*this);
    }
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Equality &x)
{
    s << "<apply><eq/>";
    x.get_arg1()->accept(*this);
    x.get_arg2()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Unequality &x)
{
    s << "<apply><neq/>";
    x.get_arg1()->accept(*this);
    x.get_arg2()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const LessThan &x)
{
    s << "<apply><leq/>";
    x.get_arg1()->accept(*this);
    x.get_arg2()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const StrictLessThan &x)
{
    s << "<apply><lt/>";
    x.get_arg1()->accept(*this);
    x.get_arg2()->accept(*this);
    s << "</apply>";
}

void MathMLPrinter::bvisit(const Derivative &x)
{
    s << "<apply><partialdiff/><bvar>";
    for (const auto &elem : x.get_symbols()) {
        elem->accept(*this);
    }
    s << "</bvar>";
    x.get_arg()->accept(*this);
    s << "</apply>";
}

std::string MathMLPrinter::apply(const Basic &b)
{
    b.accept(*this);
    return s.str();
}

std::string mathml(const Basic &x)
{
    MathMLPrinter m;
    return m.apply(x);
}
} // namespace SymEngine
