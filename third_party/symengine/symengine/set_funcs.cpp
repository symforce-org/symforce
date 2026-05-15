#ifndef SYMENGINE_SET_FUNCS_H
#define SYMENGINE_SET_FUNCS_H

#include <symengine/visitor.h>
#include <symengine/functions.h>

namespace SymEngine
{

class SupVisitor : public BaseVisitor<SupVisitor>
{
private:
    RCP<const Basic> sup_;

public:
    SupVisitor() {}

    void bvisit(const Basic &x){};

    void bvisit(const Set &x)
    {
        throw SymEngineException(
            "Set not partially ordered: supremum undefined");
    };

    void bvisit(const Reals &x)
    {
        sup_ = infty(1);
    };

    void bvisit(const Rationals &x)
    {
        sup_ = infty(1);
    };

    void bvisit(const Integers &x)
    {
        sup_ = infty(1);
    };

    void bvisit(const Naturals &x)
    {
        sup_ = infty(1);
    };

    void bvisit(const Naturals0 &x)
    {
        sup_ = infty(1);
    };

    void bvisit(const Interval &x)
    {
        sup_ = x.get_end();
    };

    void bvisit(const FiniteSet &x)
    {
        const set_basic &container = x.get_container();
        vec_basic v(container.begin(), container.end());
        sup_ = max(v);
    };

    void bvisit(const Union &x)
    {
        vec_basic suprema;
        for (auto &a : x.get_container()) {
            a->accept(*this);
            suprema.push_back(sup_);
        }
        sup_ = max(suprema);
    };

    void bvisit(const Complement &x)
    {
        throw NotImplementedError("sup for Complement not implemented");
    };

    void bvisit(const ImageSet &x)
    {
        throw NotImplementedError("sup for ImageSet not implemented");
    };

    RCP<const Basic> apply(const Set &s)
    {
        s.accept(*this);
        return sup_;
    };
};

class InfVisitor : public BaseVisitor<InfVisitor>
{
private:
    RCP<const Basic> inf_;

public:
    InfVisitor() {}

    void bvisit(const Basic &x){};

    void bvisit(const Set &x)
    {
        throw SymEngineException(
            "Set not partially ordered: infimum undefined");
    };

    void bvisit(const Reals &x)
    {
        inf_ = infty(-1);
    };

    void bvisit(const Rationals &x)
    {
        inf_ = infty(-1);
    };

    void bvisit(const Integers &x)
    {
        inf_ = infty(-1);
    };

    void bvisit(const Naturals &x)
    {
        inf_ = integer(1);
    };

    void bvisit(const Naturals0 &x)
    {
        inf_ = integer(0);
    };

    void bvisit(const Interval &x)
    {
        inf_ = x.get_start();
    };

    void bvisit(const FiniteSet &x)
    {
        const set_basic &container = x.get_container();
        vec_basic v(container.begin(), container.end());
        inf_ = min(v);
    };

    void bvisit(const Union &x)
    {
        vec_basic infima;
        for (auto &a : x.get_container()) {
            a->accept(*this);
            infima.push_back(inf_);
        }
        inf_ = min(infima);
    };

    void bvisit(const Complement &x)
    {
        throw NotImplementedError("inf for Complement not implemented");
    };

    void bvisit(const ImageSet &x)
    {
        throw NotImplementedError("inf for ImageSet not implemented");
    };

    RCP<const Basic> apply(const Set &s)
    {
        s.accept(*this);
        return inf_;
    };
};

RCP<const Basic> sup(const Set &s)
{
    SupVisitor visitor;
    return visitor.apply(s);
}

RCP<const Basic> inf(const Set &s)
{
    InfVisitor visitor;
    return visitor.apply(s);
}

class BoundaryVisitor : public BaseVisitor<BoundaryVisitor>
{
private:
    RCP<const Set> boundary_;

public:
    BoundaryVisitor() {}

    void bvisit(const Basic &x){};

    void bvisit(const EmptySet &x)
    {
        boundary_ = emptyset();
    };

    void bvisit(const UniversalSet &x)
    {
        boundary_ = emptyset();
    };

    void bvisit(const Complexes &x)
    {
        boundary_ = emptyset();
    };

    void bvisit(const Reals &x)
    {
        boundary_ = emptyset();
    };

    void bvisit(const Rationals &x)
    {
        boundary_ = reals();
    };

    void bvisit(const Integers &x)
    {
        boundary_ = integers();
    };

    void bvisit(const Naturals &x)
    {
        boundary_ = naturals();
    };

    void bvisit(const Naturals0 &x)
    {
        boundary_ = naturals0();
    };

    void bvisit(const Interval &x)
    {
        boundary_ = finiteset({x.get_start(), x.get_end()});
    };

    void bvisit(const FiniteSet &x)
    {
        boundary_ = rcp_static_cast<const Set>(x.rcp_from_this());
    };

    void bvisit(const Union &x)
    {
        set_set boundary_sets;
        const set_set &sets = x.get_container();
        for (auto it = sets.begin(); it != sets.end(); ++it) {
            set_set interior_sets;
            for (auto interit = sets.begin(); interit != sets.end();
                 ++interit) {
                if (it != interit) {
                    interior_sets.insert(interior(**interit));
                }
            }
            boundary_sets.insert(
                set_complement(apply(**it), set_union(interior_sets)));
        }
        boundary_ = set_union(boundary_sets);
    };

    void bvisit(const Complement &x)
    {
        throw NotImplementedError("inf for Complement not implemented");
    };

    void bvisit(const ImageSet &x)
    {
        throw NotImplementedError("inf for ImageSet not implemented");
    };

    RCP<const Set> apply(const Set &s)
    {
        s.accept(*this);
        return boundary_;
    };
};

RCP<const Set> boundary(const Set &s)
{
    BoundaryVisitor visitor;
    return visitor.apply(s);
}

RCP<const Set> interior(const Set &s)
{
    return set_complement(rcp_static_cast<const Set>(s.rcp_from_this()),
                          boundary(s));
}

RCP<const Set> closure(const Set &s)
{
    return s.set_union(boundary(s));
}

} // namespace SymEngine

#endif // SYMENGINE_SET_FUNCS_H
