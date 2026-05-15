/**
 *  \file sets.h
 *
 **/
#ifndef SYMENGINE_SETS_H
#define SYMENGINE_SETS_H

#include <symengine/functions.h>
#include <symengine/complex.h>
#include <symengine/symengine_casts.h>
#include <iterator>
namespace SymEngine
{
class Set;
class BooleanAtom;
class Boolean;
inline bool is_a_Boolean(const Basic &b);
RCP<const BooleanAtom> boolean(bool b);
} // namespace SymEngine
#include <symengine/logic.h>

namespace SymEngine
{
typedef std::set<RCP<const Set>, RCPBasicKeyLess> set_set;
class Set : public Basic
{
public:
    vec_basic get_args() const override = 0;
    virtual RCP<const Set> set_intersection(const RCP<const Set> &o) const = 0;
    virtual RCP<const Set> set_union(const RCP<const Set> &o) const = 0;
    virtual RCP<const Set> set_complement(const RCP<const Set> &o) const = 0;
    virtual RCP<const Boolean> contains(const RCP<const Basic> &a) const = 0;
    bool is_subset(const RCP<const Set> &o) const
    {
        return eq(*this->set_intersection(o), *this);
    }
    bool is_proper_subset(const RCP<const Set> &o) const
    {
        return not eq(*this, *o) and this->is_subset(o);
    }
    bool is_superset(const RCP<const Set> &o) const
    {
        return o->is_subset(rcp_from_this_cast<const Set>());
    }
    bool is_proper_superset(const RCP<const Set> &o) const
    {
        return not eq(*this, *o) and this->is_superset(o);
    }
};

class EmptySet : public Set
{
public:
    EmptySet(){SYMENGINE_ASSIGN_TYPEID()}

    IMPLEMENT_TYPEID(SYMENGINE_EMPTYSET)
        // EmptySet(EmptySet const&) = delete;
        void
        operator=(EmptySet const &)
        = delete;
    const static RCP<const EmptySet> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override
    {
        return boolean(false);
    };
};

class UniversalSet : public Set
{
public:
    UniversalSet()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_UNIVERSALSET)
    // UniversalSet(UniversalSet const&) = delete;
    void operator=(UniversalSet const &) = delete;
    const static RCP<const UniversalSet> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override
    {
        return boolean(true);
    };
};

class FiniteSet : public Set
{
private:
    set_basic container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_FINITESET)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return vec_basic(container_.begin(), container_.end());
    }

    FiniteSet(const set_basic &container);
    static bool is_canonical(const set_basic &container);

    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
    RCP<const Set> create(const set_basic &container) const;

    inline const set_basic &get_container() const
    {
        return this->container_;
    }
};

class Interval : public Set
{
private:
    RCP<const Number> start_;
    RCP<const Number> end_;
    bool left_open_, right_open_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_INTERVAL)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;

    Interval(const RCP<const Number> &start, const RCP<const Number> &end,
             const bool left_open = false, const bool right_open = false);

    RCP<const Set> open() const;
    RCP<const Set> close() const;
    RCP<const Set> Lopen() const;
    RCP<const Set> Ropen() const;

    static bool is_canonical(const RCP<const Number> &start,
                             const RCP<const Number> &end, bool left_open,
                             bool right_open);

    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
    vec_basic get_args() const override;

    inline const RCP<const Number> &get_start() const
    {
        return start_;
    }
    inline const RCP<const Number> &get_end() const
    {
        return end_;
    }
    inline const bool &get_left_open() const
    {
        return this->left_open_;
    }
    inline const bool &get_right_open() const
    {
        return this->right_open_;
    }
};

class Complexes : public Set
{
public:
    Complexes()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_COMPLEXES)
    void operator=(Complexes const &) = delete;
    const static RCP<const Complexes> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
};

class Reals : public Set
{
public:
    Reals()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_REALS)
    void operator=(Reals const &) = delete;
    const static RCP<const Reals> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
};

class Rationals : public Set
{
public:
    Rationals()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_RATIONALS)
    void operator=(Rationals const &) = delete;
    const static RCP<const Rationals> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
};

class Integers : public Set
{
public:
    Integers()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_INTEGERS)
    void operator=(Integers const &) = delete;
    const static RCP<const Integers> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
};

class Naturals : public Set
{
public:
    Naturals()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_NATURALS)
    void operator=(Naturals const &) = delete;
    const static RCP<const Naturals> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
};

class Naturals0 : public Set
{
public:
    Naturals0()
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

public:
    IMPLEMENT_TYPEID(SYMENGINE_NATURALS0)
    void operator=(Naturals0 const &) = delete;
    const static RCP<const Naturals0> &getInstance();
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {};
    }

    template <typename T_, typename... Args>
    friend inline RCP<T_> make_rcp(Args &&...args);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
};

class Union : public Set
{
private:
    set_set container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_UNION)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override;
    Union(const set_set &in);
    static bool is_canonical(const set_set &in);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;

    inline const set_set &get_container() const
    {
        return this->container_;
    }

    RCP<const Set> create(const set_set &in) const;
};

class Intersection : public Set
{
private:
    set_set container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_INTERSECTION)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override;
    Intersection(const set_set &in);
    static bool is_canonical(const set_set &in);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;

    inline const set_set &get_container() const
    {
        return this->container_;
    }

    RCP<const Set> create(const set_set &in) const;
};

class Complement : public Set
{
private:
    // represents universe_ - container_
    RCP<const Set> universe_;
    RCP<const Set> container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_COMPLEMENT)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {universe_, container_};
    }
    Complement(const RCP<const Set> &universe, const RCP<const Set> &container);

    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;

    inline const RCP<const Set> &get_universe() const
    {
        return this->universe_;
    }
    inline const RCP<const Set> &get_container() const
    {
        return this->container_;
    }
};

class ConditionSet : public Set
{
private:
    RCP<const Basic> sym;
    RCP<const Boolean> condition_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_CONDITIONSET)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {sym, condition_};
    }
    ConditionSet(const RCP<const Basic> &sym,
                 const RCP<const Boolean> &condition);
    static bool is_canonical(const RCP<const Basic> &sym,
                             const RCP<const Boolean> &condition);
    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;
    inline const RCP<const Basic> &get_symbol() const
    {
        return this->sym;
    }
    inline const RCP<const Boolean> &get_condition() const
    {
        return this->condition_;
    }
};

class ImageSet : public Set
{
private:
    // represents {expr_ for sym_ in base_}
    RCP<const Basic> sym_;
    RCP<const Basic> expr_;
    RCP<const Set> base_; // base set for all symbols

public:
    IMPLEMENT_TYPEID(SYMENGINE_IMAGESET)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override
    {
        return {sym_, expr_, base_};
    }
    ImageSet(const RCP<const Basic> &sym, const RCP<const Basic> &expr,
             const RCP<const Set> &base);

    static bool is_canonical(const RCP<const Basic> &sym,
                             const RCP<const Basic> &expr,
                             const RCP<const Set> &base);
    RCP<const Set> set_intersection(const RCP<const Set> &o) const override;
    RCP<const Set> set_union(const RCP<const Set> &o) const override;
    RCP<const Set> set_complement(const RCP<const Set> &o) const override;
    RCP<const Boolean> contains(const RCP<const Basic> &a) const override;

    inline const RCP<const Basic> &get_symbol() const
    {
        return this->sym_;
    }
    inline const RCP<const Basic> &get_expr() const
    {
        return this->expr_;
    }
    inline const RCP<const Set> &get_baseset() const
    {
        return this->base_;
    }

    RCP<const Set> create(const RCP<const Basic> &sym,
                          const RCP<const Basic> &expr,
                          const RCP<const Set> &base) const;
};

inline bool is_a_Set(const Basic &b)
{
    return (b.get_type_code() == SYMENGINE_EMPTYSET
            || b.get_type_code() == SYMENGINE_UNIVERSALSET
            || b.get_type_code() == SYMENGINE_FINITESET
            || b.get_type_code() == SYMENGINE_COMPLEMENT
            || b.get_type_code() == SYMENGINE_CONDITIONSET
            || b.get_type_code() == SYMENGINE_INTERVAL
            || b.get_type_code() == SYMENGINE_COMPLEXES
            || b.get_type_code() == SYMENGINE_REALS
            || b.get_type_code() == SYMENGINE_RATIONALS
            || b.get_type_code() == SYMENGINE_INTEGERS
            || b.get_type_code() == SYMENGINE_NATURALS
            || b.get_type_code() == SYMENGINE_NATURALS0
            || b.get_type_code() == SYMENGINE_UNION
            || b.get_type_code() == SYMENGINE_INTERSECTION
            || b.get_type_code() == SYMENGINE_IMAGESET);
}

//! \return RCP<const Complexes>
inline RCP<const Complexes> complexes()
{
    return Complexes::getInstance();
}

//! \return RCP<const Reals>
inline RCP<const Reals> reals()
{
    return Reals::getInstance();
}

//! \return RCP<const Rationals>
inline RCP<const Rationals> rationals()
{
    return Rationals::getInstance();
}

//! \return RCP<const Integers>
inline RCP<const Integers> integers()
{
    return Integers::getInstance();
}

//! \return RCP<const Naturals>
inline RCP<const Naturals> naturals()
{
    return Naturals::getInstance();
}

//! \return RCP<const Naturals>
inline RCP<const Naturals0> naturals0()
{
    return Naturals0::getInstance();
}

//! \return RCP<const EmptySet>
inline RCP<const EmptySet> emptyset()
{
    return EmptySet::getInstance();
}

//! \return RCP<const UniversalSet>
inline RCP<const UniversalSet> universalset()
{
    return UniversalSet::getInstance();
}

//! \return RCP<const Set>
inline RCP<const Set> finiteset(const set_basic &container)
{
    if (FiniteSet::is_canonical(container)) {
        return make_rcp<const FiniteSet>(container);
    }
    return emptyset();
}

//! \return RCP<const Set>
inline RCP<const Set> interval(const RCP<const Number> &start,
                               const RCP<const Number> &end,
                               const bool left_open = false,
                               const bool right_open = false)
{
    if (Interval::is_canonical(start, end, left_open, right_open))
        return make_rcp<const Interval>(start, end, left_open, right_open);
    if (eq(*start, *end) and not(left_open or right_open))
        return finiteset({start});
    return emptyset();
}

// ! \return RCP<const Set>
inline RCP<const Set> imageset(const RCP<const Basic> &sym,
                               const RCP<const Basic> &expr,
                               const RCP<const Set> &base)
{
    if (not is_a_sub<Symbol>(*sym))
        throw SymEngineException("first arg is expected to be a symbol");

    if (eq(*expr, *sym) or eq(*base, *emptyset()))
        return base;

    if (is_a_Number(*expr))
        return finiteset({expr});
    if (is_a_Set(*expr)) {
        for (const auto &s : static_cast<const Set &>(*expr).get_args()) {
            if (not(is_a_Number(*s) or is_a<Constant>(*s)
                    or is_a_Boolean(*s))) {
                return make_rcp<const ImageSet>(sym, expr, base);
            }
        }
        return finiteset({expr});
    }

    if (is_a<FiniteSet>(*base)) {
        map_basic_basic d;
        set_basic temp;
        for (const auto &s :
             down_cast<const FiniteSet &>(*base).get_container()) {
            d[sym] = s;
            temp.insert(expr->subs(d));
            d.clear();
        }
        return finiteset(temp);
    }

    if (is_a<ImageSet>(*base)) {
        const ImageSet &imbase = down_cast<const ImageSet &>(*base);
        map_basic_basic d;
        d[sym] = imbase.get_expr();
        return imageset(imbase.get_symbol(), expand(expr->subs(d)),
                        imbase.get_baseset());
    }

    return make_rcp<const ImageSet>(sym, expr, base);
}

// ! \return RCP<const Set>
RCP<const Set> set_union(const set_set &in);

// ! \return RCP<const Set>
RCP<const Set> set_intersection(const set_set &in);

RCP<const Set> set_complement_helper(const RCP<const Set> &container,
                                     const RCP<const Set> &universe);

// ! \return RCP<const Set>
RCP<const Set> set_complement(const RCP<const Set> &universe,
                              const RCP<const Set> &container);

//! \return RCP<const Set>
RCP<const Set> conditionset(const RCP<const Basic> &sym,
                            const RCP<const Boolean> &condition);

RCP<const Basic> sup(const Set &s);
RCP<const Basic> inf(const Set &s);
RCP<const Set> boundary(const Set &s);
RCP<const Set> interior(const Set &s);
RCP<const Set> closure(const Set &s);

} // namespace SymEngine
#endif
