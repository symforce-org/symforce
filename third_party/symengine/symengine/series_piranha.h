#ifndef SYMENGINE_SERIES_PIRANHA_H
#define SYMENGINE_SERIES_PIRANHA_H

#include <symengine/series.h>
#include <symengine/expression.h>

#ifdef HAVE_SYMENGINE_PIRANHA
#include <piranha/monomial.hpp>
#include <piranha/polynomial.hpp>

namespace SymEngine
{

using pp_t = piranha::polynomial<piranha::rational, piranha::monomial<short>>;
// Univariate Rational Coefficient Power SeriesBase using Piranha
class URatPSeriesPiranha
    : public SeriesBase<pp_t, piranha::rational, URatPSeriesPiranha>
{
public:
    URatPSeriesPiranha(const pp_t p, const std::string varname,
                       const unsigned degree);
    IMPLEMENT_TYPEID(SYMENGINE_URATPSERIESPIRANHA)
    virtual int compare(const Basic &o) const;
    virtual hash_t __hash__() const;
    virtual RCP<const Basic> as_basic() const;
    virtual umap_int_basic as_dict() const;
    virtual RCP<const Basic> get_coeff(int) const;

    static RCP<const URatPSeriesPiranha>
    series(const RCP<const Basic> &t, const std::string &x, unsigned int prec);
    static piranha::integer convert(const Integer &x);
    static piranha::rational convert(const rational_class &x);
    static pp_t var(const std::string &s);
    static piranha::rational convert(const Rational &x);
    static piranha::rational convert(const Basic &x);
    static pp_t mul(const pp_t &s, const pp_t &r, unsigned prec);
    static pp_t pow(const pp_t &s, int n, unsigned prec);
    static unsigned ldegree(const pp_t &s);
    static piranha::rational find_cf(const pp_t &s, const pp_t &var,
                                     unsigned deg);
    static piranha::rational root(piranha::rational &c, unsigned n);
    static pp_t diff(const pp_t &s, const pp_t &var);
    static pp_t integrate(const pp_t &s, const pp_t &var);
    static pp_t subs(const pp_t &s, const pp_t &var, const pp_t &r,
                     unsigned prec);
};

using p_expr = piranha::polynomial<Expression, piranha::monomial<int>>;
// Univariate Rational Coefficient Power SeriesBase using Piranha
class UPSeriesPiranha : public SeriesBase<p_expr, Expression, UPSeriesPiranha>
{
public:
    UPSeriesPiranha(const p_expr p, const std::string varname,
                    const unsigned degree);
    IMPLEMENT_TYPEID(SYMENGINE_UPSERIESPIRANHA)
    virtual int compare(const Basic &o) const;
    virtual hash_t __hash__() const;
    virtual RCP<const Basic> as_basic() const;
    virtual umap_int_basic as_dict() const;
    virtual RCP<const Basic> get_coeff(int) const;

    static RCP<const UPSeriesPiranha>
    series(const RCP<const Basic> &t, const std::string &x, unsigned int prec);
    static p_expr var(const std::string &s);
    static Expression convert(const Basic &x);
    static p_expr mul(const p_expr &s, const p_expr &r, unsigned prec);
    static p_expr pow(const p_expr &s, int n, unsigned prec);
    static unsigned ldegree(const p_expr &s);
    static Expression find_cf(const p_expr &s, const p_expr &var, unsigned deg);
    static Expression root(Expression &c, unsigned n);
    static p_expr diff(const p_expr &s, const p_expr &var);
    static p_expr integrate(const p_expr &s, const p_expr &var);
    static p_expr subs(const p_expr &s, const p_expr &var, const p_expr &r,
                       unsigned prec);

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
    static Expression atanh(const Expression &c);
    static Expression exp(const Expression &c);
    static Expression log(const Expression &c);
};
} // namespace SymEngine

#endif // HAVE_SYMENGINE_PIRANHA

#endif // SYMENGINE_SERIES_PIRANHA_H
