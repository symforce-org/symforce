#include <symengine/data_buffer_element.h>
#include <symengine/add.h>
#include <symengine/dict.h>
#include <symengine/symengine_exception.h>
#include <symengine/visitor.h>

namespace SymEngine
{

DataBufferElement::DataBufferElement(const RCP<const Symbol> &name, const RCP<const Basic> &i)
    : name_(name), i_(i)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(*name, *i))
}

bool DataBufferElement::is_canonical(const Symbol &name, const Basic &i) const
{
    return true;
}

hash_t DataBufferElement::__hash__() const
{
    hash_t seed = SYMENGINE_DATABUFFERELEMENT;
    hash_combine<Symbol>(seed, *name_);
    hash_combine<Basic>(seed, *i_);
    return seed;
}

bool DataBufferElement::__eq__(const Basic &o) const
{
    if (is_a<DataBufferElement>(o) and eq(*name_, *(down_cast<const DataBufferElement &>(o).name_))
        and eq(*i_, *(down_cast<const DataBufferElement &>(o).i_))) {
        return true;
    }

    return false;
}

int DataBufferElement::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<DataBufferElement>(o))
    const DataBufferElement &s = down_cast<const DataBufferElement &>(o);
    int name_cmp = name_->__cmp__(*s.name_);
    if (name_cmp == 0) {
        return i_->__cmp__(*s.i_);
    } else {
        return name_cmp;
    }
}

RCP<const Basic> data_buffer_element(const RCP<const Symbol> &name, const RCP<const Basic> &i)
{
    return make_rcp<const DataBufferElement>(name, i);
}

vec_basic DataBufferElement::get_args() const
{
    return {name_, i_};
}

} // SymEngine
