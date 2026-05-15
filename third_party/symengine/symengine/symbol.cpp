#include <symengine/constants.h>
#include <symengine/symengine_casts.h>

namespace SymEngine
{

Symbol::Symbol(const std::string &name)
    : name_{name} {SYMENGINE_ASSIGN_TYPEID()}

      hash_t Symbol::__hash__() const
{
    hash_t seed = 0;
    hash_combine(seed, name_);
    return seed;
}

bool Symbol::__eq__(const Basic &o) const
{
    if (is_a<Symbol>(o))
        return name_ == down_cast<const Symbol &>(o).name_;
    return false;
}

int Symbol::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Symbol>(o))
    const Symbol &s = down_cast<const Symbol &>(o);
    if (name_ == s.name_)
        return 0;
    return name_ < s.name_ ? -1 : 1;
}

RCP<const Symbol> Symbol::as_dummy() const
{
    return dummy(name_);
}

size_t Dummy::count_ = 0;

Dummy::Dummy() : Symbol("_Dummy_" + to_string(count_))
{
    SYMENGINE_ASSIGN_TYPEID()
    count_ += 1;
    dummy_index = count_;
}

Dummy::Dummy(const std::string &name) : Symbol("_" + name)
{
    SYMENGINE_ASSIGN_TYPEID()
    count_ += 1;
    dummy_index = count_;
}

hash_t Dummy::__hash__() const
{
    hash_t seed = 0;
    hash_combine(seed, get_name());
    hash_combine(seed, dummy_index);
    return seed;
}

bool Dummy::__eq__(const Basic &o) const
{
    if (is_a<Dummy>(o))
        return ((get_name() == down_cast<const Dummy &>(o).get_name())
                and (dummy_index == down_cast<const Dummy &>(o).get_index()));
    return false;
}

int Dummy::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Dummy>(o))
    const Dummy &s = down_cast<const Dummy &>(o);
    if (get_name() == s.get_name()) {
        if (dummy_index == s.get_index())
            return 0;
        return dummy_index < s.get_index() ? -1 : 1;
    }
    return get_name() < s.get_name() ? -1 : 1;
}

} // namespace SymEngine
