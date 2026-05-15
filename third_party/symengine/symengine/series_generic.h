/**
 *  \file series_generic.h
 *  Class for univariate series.
 *
 **/
#ifndef SYMENGINE_SERIES_GENERIC_H
#define SYMENGINE_SERIES_GENERIC_H

#include <symengine/polys/uintpoly.h>
#include <symengine/polys/uexprpoly.h>
#include <symengine/series.h>

namespace SymEngine
{
//! UnivariateSeries Class
class UnivariateSeries
    : public SeriesBase<UExprDict, Expression, UnivariateSeries>
{
    // UnivariateSeries 1 + 2*x + x**2 + O(x**5) has dict_ = {{0, 1}, {1, 2},
    // {2, 1}} with var_ = "x" and prec_ = 5
public:
    using SeriesBase::mul;
    using SeriesBase::pow;
    IMPLEMENT_TYPEID(SYMENGINE_UNIVARIATESERIES)
    UnivariateSeries(const UExprDict &sp, const std::string varname,
                     const unsigned degree)
        : SeriesBase(std::move(sp), varname, degree)
    {
        SYMENGINE_ASSIGN_TYPEID()
    }

    static RCP<const UnivariateSeries> create(const RCP<const Symbol> &var,
                                              const unsigned int &prec,
                                              const UExprDict &s)
    {
        return make_rcp<const UnivariateSeries>(std::move(s), var->get_name(),
                                                prec);
    }

    static RCP<const UnivariateSeries>
    series(const RCP<const Basic> &t, const std::string &x, unsigned int prec);
    hash_t __hash__() const override;
    int compare(const Basic &o) const override;
    bool operator==(const UnivariateSeries &u) const;
    RCP<const Basic> as_basic() const override;
    umap_int_basic as_dict() const override;
    RCP<const Basic> get_coeff(int) const override;
    static UExprDict var(const std::string &s);

    static Expression convert(const Basic &x);

    static int ldegree(const UExprDict &s);
    static UExprDict mul(const UExprDict &s, const UExprDict &r, unsigned prec);
    static UExprDict pow(const UExprDict &s, int n, unsigned prec);
    static Expression find_cf(const UExprDict &s, const UExprDict &var,
                              int deg);
    static Expression root(Expression &c, unsigned n);
    static UExprDict diff(const UExprDict &s, const UExprDict &var);
    static UExprDict integrate(const UExprDict &s, const UExprDict &var);
    static UExprDict subs(const UExprDict &s, const UExprDict &var,
                          const UExprDict &r, unsigned prec);

    static Expression sin(const Expression &c);
    static Expression cos(const Expression &c);
    static Expression tan(const Expression &c);
    static Expression asin(const Expression &c);
    static Expression acos(const Expression &c);
    static Expression atan(const Expression &c);
    static Expression sinh(const Expression &c);
    static Expression cosh(const Expression &c);
    static Expression tanh(const Expression &c);
    static Expression asinh(const Expression &c);
    static Expression acosh(const Expression &c);
    static Expression atanh(const Expression &c);
    static Expression exp(const Expression &c);
    static Expression log(const Expression &c);
};

inline RCP<const UnivariateSeries>
univariate_series(RCP<const Symbol> i, unsigned int prec, const UExprDict &s)
{
    return make_rcp<const UnivariateSeries>(std::move(s), i->get_name(), prec);
}

} // namespace SymEngine
#endif
