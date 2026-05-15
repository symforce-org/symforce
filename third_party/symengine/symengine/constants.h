/**
 *  \file constants.h
 *  Declare all the special constants in this file
 *
 **/

#ifndef SYMENGINE_CONSTANTS_H
#define SYMENGINE_CONSTANTS_H

#include <symengine/integer.h>
#include <symengine/symbol.h>
#include <symengine/infinity.h>
#include <symengine/nan.h>

namespace SymEngine
{

class Constant : public Basic
{
private:
    //! name of Constant
    std::string name_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_CONSTANT)
    //! Constant Constructor
    Constant(const std::string &name);
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
    //! \return name of the Constant.
    inline std::string get_name() const
    {
        return name_;
    }

    vec_basic get_args() const override
    {
        return {};
    }
};

//! inline version to return `Constant`
inline RCP<const Constant> constant(const std::string &name)
{
    return make_rcp<const Constant>(name);
}

// Constant Numbers
extern SYMENGINE_EXPORT RCP<const Integer> zero;
extern SYMENGINE_EXPORT RCP<const Integer> one;
extern SYMENGINE_EXPORT RCP<const Integer> minus_one;
extern SYMENGINE_EXPORT RCP<const Integer> two;
extern SYMENGINE_EXPORT RCP<const Number> I;

// Symbolic Constants
extern SYMENGINE_EXPORT RCP<const Constant> pi;
extern SYMENGINE_EXPORT RCP<const Constant> E;
extern SYMENGINE_EXPORT RCP<const Constant> EulerGamma;
extern SYMENGINE_EXPORT RCP<const Constant> Catalan;
extern SYMENGINE_EXPORT RCP<const Constant> GoldenRatio;

// Infinity
extern SYMENGINE_EXPORT RCP<const Infty> Inf;
extern SYMENGINE_EXPORT RCP<const Infty> NegInf;
extern SYMENGINE_EXPORT RCP<const Infty> ComplexInf;

// Not a Number
extern SYMENGINE_EXPORT RCP<const NaN> Nan;
} // namespace SymEngine

#endif
