/**
 *  \file uratpoly.h
 *  Class for sparse Polynomial: URatPoly
 **/
#ifndef SYMENGINE_URATPOLY_H
#define SYMENGINE_URATPOLY_H

#include <symengine/polys/usymenginepoly.h>

namespace SymEngine
{

class URatDict : public ODictWrapper<unsigned int, rational_class, URatDict>
{

public:
    URatDict() SYMENGINE_NOEXCEPT {}
    ~URatDict() SYMENGINE_NOEXCEPT {}
    URatDict(URatDict &&other) SYMENGINE_NOEXCEPT
        : ODictWrapper(std::move(other))
    {
    }
    URatDict(const int &i) : ODictWrapper(i) {}
    URatDict(const map_uint_mpq &p) : ODictWrapper(p) {}
    URatDict(const rational_class &i) : ODictWrapper(i) {}

    URatDict(const URatDict &) = default;
    URatDict &operator=(const URatDict &) = default;

    int compare(const URatDict &other) const
    {
        if (dict_.size() != other.dict_.size())
            return (dict_.size() < other.dict_.size()) ? -1 : 1;
        return unified_compare(dict_, other.dict_);
    }
}; // URatDict

class URatPoly : public USymEnginePoly<URatDict, URatPolyBase, URatPoly>
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_URATPOLY)
    //! Constructor of URatPoly class
    URatPoly(const RCP<const Basic> &var, URatDict &&dict);

    //! \return size of the hash
    hash_t __hash__() const override;
}; // URatPoly

// true & sets `out` to b/a if a exactly divides b, otherwise false & undefined
bool divides_upoly(const URatPoly &a, const URatPoly &b,
                   const Ptr<RCP<const URatPoly>> &res);

} // namespace SymEngine

#endif
