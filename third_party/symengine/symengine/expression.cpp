#include <symengine/printers/strprinter.h>
#include <symengine/parser.h>

namespace SymEngine
{

namespace detail
{
std::string poly_print(const Expression &x)
{
    Precedence prec;
    if (prec.getPrecedence(x.get_basic()) == PrecedenceEnum::Add) {
        return "(" + x.get_basic()->__str__() + ")";
    }
    return x.get_basic()->__str__();
}
} // namespace detail

Expression::Expression(const std::string &s)
{
    m_basic = parse(s);
}

} // namespace SymEngine
