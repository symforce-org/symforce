/**
 *  \file expansion.h
 *  Series expansions of expressions using Piranha
 *
 **/

#include <symengine/series_visitor.h>

namespace SymEngine
{

bool needs_symbolic_constants(const RCP<const Basic> &ex,
                              const RCP<const Symbol> &var)
{
    NeedsSymbolicExpansionVisitor v;
    return v.apply(*ex, var);
}

RCP<const SeriesCoeffInterface> series(const RCP<const Basic> &ex,
                                       const RCP<const Symbol> &var,
                                       unsigned int prec)
{
    auto syms = free_symbols(*ex);
#ifdef HAVE_SYMENGINE_PIRANHA
    if (prec == 0)
        return make_rcp<const UPSeriesPiranha>(p_expr{Expression()},
                                               var->get_name(), prec);
    if (not has_symbol(*ex, *var))
        return make_rcp<const UPSeriesPiranha>(p_expr{Expression(ex)},
                                               var->get_name(), prec);
    if (is_a<Symbol>(*ex))
        return make_rcp<const UPSeriesPiranha>(p_expr{Expression(ex)},
                                               var->get_name(), prec);

    if (syms.size() == 1) {
        if (needs_symbolic_constants(ex, var))
            return UPSeriesPiranha::series(ex, var->get_name(), prec);
#ifdef HAVE_SYMENGINE_FLINT
        return URatPSeriesFlint::series(ex, var->get_name(), prec);
#else
        return URatPSeriesPiranha::series(ex, var->get_name(), prec);
#endif
    }

    return UPSeriesPiranha::series(ex, var->get_name(), prec);
#elif defined(HAVE_SYMENGINE_FLINT)
    //! TODO: call generic impl. where needed
    if (prec == 0)
        return URatPSeriesFlint::series(integer(0), var->get_name(), prec);

    if (syms.size() > 1)
        return UnivariateSeries::series(ex, var->get_name(), prec);

    if (needs_symbolic_constants(ex, var))
        return UnivariateSeries::series(ex, var->get_name(), prec);
    return URatPSeriesFlint::series(ex, var->get_name(), prec);
#else
    return UnivariateSeries::series(ex, var->get_name(), prec);
#endif
}

RCP<const SeriesCoeffInterface> series_invfunc(const RCP<const Basic> &ex,
                                               const RCP<const Symbol> &var,
                                               unsigned int prec)
{
#ifdef HAVE_SYMENGINE_PIRANHA
    if (prec == 0)
        return make_rcp<const UPSeriesPiranha>(p_expr{Expression()},
                                               var->get_name(), prec);
    if (is_a<Symbol>(*ex))
        return make_rcp<const UPSeriesPiranha>(p_expr{Expression(ex)},
                                               var->get_name(), prec);

    return make_rcp<const UPSeriesPiranha>(
        UPSeriesPiranha::series_reverse(
            UPSeriesPiranha::series(ex, var->get_name(), prec)->get_poly(),
            p_expr(var->get_name()), prec),
        var->get_name(), prec);

#else
    throw SymEngineException("Series reversion is supported only with Piranha");
#endif
}

} // namespace SymEngine
