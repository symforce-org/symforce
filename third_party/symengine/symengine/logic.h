/**
 *  \file logic.h
 *
 **/

#ifndef SYMENGINE_LOGIC_H
#define SYMENGINE_LOGIC_H

#include <symengine/basic.h>
#include <symengine/sets.h>

namespace SymEngine
{
typedef std::set<RCP<const Boolean>, RCPBasicKeyLess> set_boolean;
typedef std::vector<RCP<const Boolean>> vec_boolean;

// Parent class for expressing boolean statements
class Boolean : public Basic
{
public:
    virtual RCP<const Boolean> logical_not() const;
};

// Booleans True and False
class BooleanAtom : public Boolean
{
private:
    bool b_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_BOOLEAN_ATOM)
    BooleanAtom(bool b);
    //! \return the hash
    hash_t __hash__() const override;
    bool get_val() const;
    vec_basic get_args() const override;
    bool __eq__(const Basic &o) const override;
    //! Structural equality comparator
    int compare(const Basic &o) const override;
    RCP<const Boolean> logical_not() const override;
};

extern SYMENGINE_EXPORT RCP<const BooleanAtom> boolTrue;
extern SYMENGINE_EXPORT RCP<const BooleanAtom> boolFalse;

inline RCP<const BooleanAtom> boolean(bool b)
{
    return b ? boolTrue : boolFalse;
}

// Represents `expr_` is inside set `set_`
// `set_` can be any `Set` including `Interval`, `FiniteSet`
class Contains : public Boolean
{
private:
    RCP<const Basic> expr_;
    RCP<const Set> set_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_CONTAINS)
    //! Constructor
    Contains(const RCP<const Basic> &expr, const RCP<const Set> &set);
    hash_t __hash__() const override;
    RCP<const Basic> get_expr() const;
    RCP<const Set> get_set() const;
    vec_basic get_args() const override;
    bool __eq__(const Basic &o) const override;
    RCP<const Basic> create(const RCP<const Basic> &lhs,
                            const RCP<const Set> &rhs) const;
    //! Structural equality comparator
    int compare(const Basic &o) const override;
};

RCP<const Boolean> contains(const RCP<const Basic> &expr,
                            const RCP<const Set> &set);

typedef std::vector<std::pair<RCP<const Basic>, RCP<const Boolean>>>
    PiecewiseVec;

// Represents a piecewise function
// Keeps a vector of (Expr, Condition) pairs
class Piecewise : public Basic
{
private:
    PiecewiseVec vec_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_PIECEWISE)
    //! Constructor
    Piecewise(PiecewiseVec &&vec);
    bool is_canonical(const PiecewiseVec &vec);
    hash_t __hash__() const override;
    const PiecewiseVec &get_vec() const;
    vec_basic get_args() const override;
    bool __eq__(const Basic &o) const override;
    //! Structural equality comparator
    int compare(const Basic &o) const override;
};

// Vec is vector of pairs of RCP<const Basic> and RCP<const Boolean> to
// represent (Expr, Condition) pairs
RCP<const Basic> piecewise(const PiecewiseVec &vec);

class And : public Boolean
{
private:
    set_boolean container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_AND)
    And(const set_boolean &s);
    bool is_canonical(const set_boolean &container_);
    //! \return the hash
    hash_t __hash__() const override;
    vec_basic get_args() const override;
    RCP<const Basic> create(const set_boolean &a) const;
    bool __eq__(const Basic &o) const override;
    //! Structural equality comparator
    int compare(const Basic &o) const override;
    const set_boolean &get_container() const;
    RCP<const Boolean> logical_not() const override;
};

class Or : public Boolean
{
private:
    set_boolean container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_OR)
    Or(const set_boolean &s);
    bool is_canonical(const set_boolean &container_);
    //! \return the hash
    hash_t __hash__() const override;
    vec_basic get_args() const override;
    bool __eq__(const Basic &o) const override;
    //! Structural equality comparator
    int compare(const Basic &o) const override;
    const set_boolean &get_container() const;
    RCP<const Boolean> logical_not() const override;
};

class Not : public Boolean
{
private:
    RCP<const Boolean> arg_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_NOT)
    Not(const RCP<const Boolean> &s);
    bool is_canonical(const RCP<const Boolean> &s);
    //! \return the hash
    hash_t __hash__() const override;
    vec_basic get_args() const override;
    bool __eq__(const Basic &o) const override;
    //! Structural equality comparator
    int compare(const Basic &o) const override;
    RCP<const Boolean> get_arg() const;
    RCP<const Boolean> logical_not() const override;
};

class Xor : public Boolean
{
private:
    vec_boolean container_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_XOR)
    Xor(const vec_boolean &s);
    bool is_canonical(const vec_boolean &container_);
    hash_t __hash__() const override;
    vec_basic get_args() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    const vec_boolean &get_container() const;
};

class Relational : public TwoArgBasic<Boolean>
{
public:
    //! Constructor
    Relational(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
    //! \return `true` if canonical
    virtual bool is_canonical(const RCP<const Basic> &lhs,
                              const RCP<const Basic> &rhs) const;
};

class Equality : public Relational
{
    //! Class for operator `==`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_EQUALITY)
    Equality(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
    RCP<const Basic> create(const RCP<const Basic> &lhs,
                            const RCP<const Basic> &rhs) const override;
    RCP<const Boolean> logical_not() const override;
};

class Unequality : public Relational
{
    //! Class for operator `!=`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_UNEQUALITY)
    Unequality(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
    RCP<const Basic> create(const RCP<const Basic> &lhs,
                            const RCP<const Basic> &rhs) const override;
    RCP<const Boolean> logical_not() const override;
};

class LessThan : public Relational
{
    //! Class for operator `<=`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_LESSTHAN)
    LessThan(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
    RCP<const Basic> create(const RCP<const Basic> &lhs,
                            const RCP<const Basic> &rhs) const override;
    RCP<const Boolean> logical_not() const override;
};

class StrictLessThan : public Relational
{
    //! Class for operator `<`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_STRICTLESSTHAN)
    StrictLessThan(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
    RCP<const Basic> create(const RCP<const Basic> &lhs,
                            const RCP<const Basic> &rhs) const override;
    RCP<const Boolean> logical_not() const override;
};

inline bool is_a_Relational(const Basic &b)
{
    return (b.get_type_code() == SYMENGINE_EQUALITY
            || b.get_type_code() == SYMENGINE_UNEQUALITY
            || b.get_type_code() == SYMENGINE_LESSTHAN
            || b.get_type_code() == SYMENGINE_STRICTLESSTHAN);
}

inline bool is_a_Boolean(const Basic &b)
{
    return (b.get_type_code() == SYMENGINE_BOOLEAN_ATOM
            || b.get_type_code() == SYMENGINE_CONTAINS
            || b.get_type_code() == SYMENGINE_AND
            || b.get_type_code() == SYMENGINE_OR
            || b.get_type_code() == SYMENGINE_NOT
            || b.get_type_code() == SYMENGINE_XOR || is_a_Relational(b));
}

//! Returns the canonicalized Equality object from a single argument
RCP<const Boolean> Eq(const RCP<const Basic> &lhs);
//! Returns the canonicalized Equality object from the two arguments
RCP<const Boolean> Eq(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
//! Returns the canonicalized Unequality object from the arguments
RCP<const Boolean> Ne(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
//! Convenience function returning LessThan object
RCP<const Boolean> Ge(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
//! Convenience function returning StrictLessThan object
RCP<const Boolean> Gt(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
//! Returns the canonicalized LessThan object from the arguments
RCP<const Boolean> Le(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);
//! Returns the canonicalized StrictLessThan object from the arguments
RCP<const Boolean> Lt(const RCP<const Basic> &lhs, const RCP<const Basic> &rhs);

RCP<const Boolean> logical_and(const set_boolean &s);
RCP<const Boolean> logical_nand(const set_boolean &s);
RCP<const Boolean> logical_or(const set_boolean &s);
RCP<const Boolean> logical_not(const RCP<const Boolean> &s);
RCP<const Boolean> logical_nor(const set_boolean &s);
RCP<const Boolean> logical_xor(const vec_boolean &s);
RCP<const Boolean> logical_xnor(const vec_boolean &s);
} // namespace SymEngine

#endif
