/**
 *  \file data_buffer_element.h
 *  Index into an external data buffer
 *  Added by Hayk.
 **/
#ifndef SYMENGINE_DATABUFFERELEMENT_H
#define SYMENGINE_DATABUFFERELEMENT_H

#include <vector>
#include <string>
#include <symengine/basic.h>
#include <symengine/symbol.h>

namespace SymEngine
{

class DataBufferElement : public Basic
{
private:
    RCP<const Symbol> name_;
    RCP<const Basic> i_;
public:
    IMPLEMENT_TYPEID(SYMENGINE_DATABUFFERELEMENT)
    //! DataBufferElement Constructor
    DataBufferElement(const RCP<const Symbol> &name, const RCP<const Basic> &i);

    //! \return Size of the hash
    virtual hash_t __hash__() const;

    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    virtual bool __eq__(const Basic &o) const;
    virtual int compare(const Basic &o) const;

    bool is_canonical(const Symbol& mat, const Basic &i) const;

    inline RCP<const Basic> get_i() const
    {
        return i_;
    }

    inline RCP<const Symbol> get_name() const
    {
        return name_;
    }

    virtual vec_basic get_args() const;
};

//! \return DataBufferElement name[i]
RCP<const Basic> data_buffer_element(const RCP<const Symbol> &name,
                                     const RCP<const Basic> &i);

// Parse an expression to get the set of data buffers encountered and return
// in sorted order.
std::vector<std::string> GetDataBufferNames(const vec_basic &outputs);

} // SymEngine

#endif
