/**
 *  \file symbol.h
 *  Class Symbol
 *
 **/
#ifndef SYMENGINE_SYMBOL_H
#define SYMENGINE_SYMBOL_H

#include <symengine/basic.h>

namespace SymEngine
{

class Symbol : public Basic
{
private:
    //! name of Symbol
    std::string name_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_SYMBOL)
    //! Symbol Constructor
    explicit Symbol(const std::string &name);
    //! \return Size of the hash
    hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;

    /*! Comparison operator
     * \param o - Object to be compared with
     * \return `0` if equal, `-1` , `1` according to string compare
     * */
    int compare(const Basic &o) const override;
    //! \return name of the Symbol.
    inline const std::string &get_name() const
    {
        return name_;
    }

    vec_basic get_args() const override
    {
        return {};
    }
    RCP<const Symbol> as_dummy() const;
};

class Dummy : public Symbol
{
private:
    //! Dummy count
    static size_t count_;
    //! Dummy index
    size_t dummy_index;

public:
    IMPLEMENT_TYPEID(SYMENGINE_DUMMY)
    //! Dummy Constructors
    explicit Dummy();
    explicit Dummy(const std::string &name);
    //! \return Size of the hash
    hash_t __hash__() const override;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    bool __eq__(const Basic &o) const override;
    /*! Comparison operator
     * \param o - Object to be compared with
     * \return `0` if equal, `-1` , `1` according to string compare
     * */
    int compare(const Basic &o) const override;
    size_t get_index() const
    {
        return dummy_index;
    }
};

//! inline version to return `Symbol`
inline RCP<const Symbol> symbol(const std::string &name)
{
    return make_rcp<const Symbol>(name);
}

//! inline version to return `Dummy`
inline RCP<const Dummy> dummy()
{
    return make_rcp<const Dummy>();
}

inline RCP<const Dummy> dummy(const std::string &name)
{
    return make_rcp<const Dummy>(name);
}

} // namespace SymEngine

#endif
