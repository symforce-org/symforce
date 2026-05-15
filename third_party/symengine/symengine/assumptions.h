#ifndef SYMENGINE_ASSUMPTIONS_H
#define SYMENGINE_ASSUMPTIONS_H

#include <symengine/dict.h>
#include <symengine/basic.h>

namespace SymEngine
{

typedef std::unordered_map<RCP<const Basic>, bool, RCPBasicHash, RCPBasicKeyEq>
    umap_basic_bool;

class Assumptions
{
private:
    set_basic complex_symbols_;
    set_basic real_symbols_;
    set_basic rational_symbols_;
    set_basic integer_symbols_;
    umap_basic_bool positive_;
    umap_basic_bool nonnegative_;
    umap_basic_bool negative_;
    umap_basic_bool nonpositive_;
    umap_basic_bool nonzero_;
    umap_basic_bool zero_;

    void set_map(umap_basic_bool &map, const RCP<const Basic> &symbol,
                 bool value);
    tribool from_map(const umap_basic_bool &map,
                     const RCP<const Basic> &symbol) const;

public:
    Assumptions(const set_basic &statements);
    tribool is_complex(const RCP<const Basic> &symbol) const;
    tribool is_real(const RCP<const Basic> &symbol) const;
    tribool is_rational(const RCP<const Basic> &symbol) const;
    tribool is_integer(const RCP<const Basic> &symbol) const;
    tribool is_positive(const RCP<const Basic> &symbol) const;
    tribool is_nonnegative(const RCP<const Basic> &symbol) const;
    tribool is_negative(const RCP<const Basic> &symbol) const;
    tribool is_nonpositive(const RCP<const Basic> &symbol) const;
    tribool is_nonzero(const RCP<const Basic> &symbol) const;
    tribool is_zero(const RCP<const Basic> &symbol) const;
};
} // namespace SymEngine

#endif // SYMENGINE_ASSUMPTIONS_H
