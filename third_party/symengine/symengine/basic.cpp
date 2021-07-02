#include <symengine/printers.h>
#include <symengine/subs.h>

namespace SymEngine
{

int Basic::__cmp__(const Basic &o) const
{
    auto a = this->get_type_code();
    auto b = o.get_type_code();
    if (a == b) {
        return this->compare(o);
    } else {
        // We return the order given by the numerical value of the TypeID enum
        // type.
        // The types don't need to be ordered in any given way, they just need
        // to be ordered.
        return a < b ? -1 : 1;
    }
}

std::string Basic::__str__() const
{
    return str(*this);
}

RCP<const Basic> Basic::subs(const map_basic_basic &subs_dict) const
{
    return SymEngine::subs(this->rcp_from_this(), subs_dict);
}

RCP<const Basic> Basic::xreplace(const map_basic_basic &xreplace_dict) const
{
    return SymEngine::xreplace(this->rcp_from_this(), xreplace_dict);
}

const char *get_version()
{
    return SYMENGINE_VERSION;
}

bool is_a_Atom(const Basic &b)
{
    return is_a_Number(b) or is_a<Symbol>(b) or is_a<Constant>(b);
}

} // namespace SymEngine
