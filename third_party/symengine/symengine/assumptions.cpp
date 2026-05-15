#include <symengine/logic.h>
#include <symengine/assumptions.h>
#include <symengine/number.h>

namespace SymEngine
{

Assumptions::Assumptions(const set_basic &statements)
{
    // Convert a set of statements into a faster and easier internal form
    for (const auto &s : statements) {
        if (is_a<Contains>(*s)) {
            const Contains &contains = down_cast<const Contains &>(*s);
            const auto expr = contains.get_expr();
            const auto set = contains.get_set();
            if (is_a<Symbol>(*expr)) {
                if (is_a<Complexes>(*set)) {
                    complex_symbols_.insert(expr);
                } else if (is_a<Reals>(*set)) {
                    complex_symbols_.insert(expr);
                    real_symbols_.insert(expr);
                } else if (is_a<Rationals>(*set)) {
                    complex_symbols_.insert(expr);
                    real_symbols_.insert(expr);
                    rational_symbols_.insert(expr);
                } else if (is_a<Integers>(*set)) {
                    complex_symbols_.insert(expr);
                    real_symbols_.insert(expr);
                    rational_symbols_.insert(expr);
                    integer_symbols_.insert(expr);
                }
            }
        } else if (is_a<LessThan>(*s)) {
            const LessThan &less_than = down_cast<const LessThan &>(*s);
            const auto &arg1 = less_than.get_arg1();
            const auto &arg2 = less_than.get_arg2();
            if (is_a<Symbol>(*arg2) and is_a_Number(*arg1)) {
                real_symbols_.insert(arg2);
                if (down_cast<const Number &>(*arg1).is_positive()) {
                    set_map(nonnegative_, arg2, true);
                    set_map(positive_, arg2, true);
                    set_map(negative_, arg2, false);
                    set_map(nonpositive_, arg2, false);
                    set_map(nonzero_, arg2, true);
                    set_map(zero_, arg2, false);
                } else if (down_cast<const Number &>(*arg1).is_zero()) {
                    set_map(nonnegative_, arg2, true);
                    set_map(negative_, arg2, false);
                }
            } else if (is_a<Symbol>(*arg1) and is_a_Number(*arg2)) {
                real_symbols_.insert(arg1);
                if (down_cast<const Number &>(*arg2).is_negative()) {
                    set_map(nonnegative_, arg1, false);
                    set_map(positive_, arg1, false);
                    set_map(negative_, arg1, true);
                    set_map(nonpositive_, arg1, true);
                    set_map(nonzero_, arg1, true);
                    set_map(zero_, arg1, false);
                } else if (down_cast<const Number &>(*arg2).is_zero()) {
                    set_map(nonpositive_, arg1, true);
                    set_map(positive_, arg1, false);
                }
            }
        } else if (is_a<StrictLessThan>(*s)) {
            const StrictLessThan &strictly_less_than
                = down_cast<const StrictLessThan &>(*s);
            const auto arg1 = strictly_less_than.get_arg1();
            const auto arg2 = strictly_less_than.get_arg2();
            if (is_a<Symbol>(*arg2) and is_a_Number(*arg1)) {
                real_symbols_.insert(arg2);
                if (not down_cast<const Number &>(*arg1).is_negative()) {
                    set_map(nonnegative_, arg2, true);
                    set_map(positive_, arg2, true);
                    set_map(negative_, arg2, false);
                    set_map(nonpositive_, arg2, false);
                    set_map(nonzero_, arg2, true);
                    set_map(zero_, arg2, false);
                }
            } else if (is_a<Symbol>(*arg1) and is_a_Number(*arg2)) {
                real_symbols_.insert(arg1);
                if (not down_cast<const Number &>(*arg2).is_positive()) {
                    set_map(nonnegative_, arg1, false);
                    set_map(positive_, arg1, false);
                    set_map(negative_, arg1, true);
                    set_map(nonpositive_, arg1, true);
                    set_map(nonzero_, arg1, true);
                    set_map(zero_, arg1, false);
                }
            }
        } else if (is_a<Equality>(*s)) {
            const Equality &equals = down_cast<const Equality &>(*s);
            const auto arg1 = equals.get_arg1();
            const auto arg2 = equals.get_arg2();
            if (is_a_Number(*arg1) and is_a<Symbol>(*arg2)) {
                complex_symbols_.insert(arg2);
                if (down_cast<const Number &>(*arg1).is_zero()) {
                    set_map(zero_, arg2, true);
                    real_symbols_.insert(arg2);
                    rational_symbols_.insert(arg2);
                    integer_symbols_.insert(arg2);
                    set_map(positive_, arg2, false);
                    set_map(negative_, arg2, false);
                    set_map(nonpositive_, arg2, true);
                    set_map(nonnegative_, arg2, true);
                    set_map(nonzero_, arg2, false);
                } else {
                    set_map(zero_, arg2, false);
                    set_map(nonzero_, arg2, true);
                }
            }
        } else if (is_a<Unequality>(*s)) {
            const Unequality &uneq = down_cast<const Unequality &>(*s);
            const auto arg1 = uneq.get_arg1();
            const auto arg2 = uneq.get_arg2();
            if (is_a_Number(*arg1) and is_a<Symbol>(*arg2)) {
                if (down_cast<const Number &>(*arg1).is_zero()) {
                    set_map(zero_, arg2, false);
                    set_map(nonzero_, arg2, true);
                }
            }
        }
    }
}

void Assumptions::set_map(umap_basic_bool &map, const RCP<const Basic> &symbol,
                          bool value)
{
    // Set element in map to true or false. Check for consistency within map
    tribool old_value = from_map(map, symbol);
    if ((is_true(old_value) and not value) or (is_false(old_value) and value)) {
        throw SymEngineException("Symbol " + symbol->__str__()
                                 + " have inconsistent positive/negativeness");
    }
    map[symbol] = value;
}

tribool Assumptions::from_map(const umap_basic_bool &map,
                              const RCP<const Basic> &symbol) const
{
    auto it = map.find(symbol);
    if (it != map.end()) {
        return (tribool)((*it).second);
    } else {
        return tribool::indeterminate;
    }
}

tribool Assumptions::is_complex(const RCP<const Basic> &symbol) const
{
    bool cmplx = complex_symbols_.find(symbol) != complex_symbols_.end();
    return cmplx ? tribool::tritrue : tribool::indeterminate;
}

tribool Assumptions::is_real(const RCP<const Basic> &symbol) const
{
    bool real = real_symbols_.find(symbol) != real_symbols_.end();
    return real ? tribool::tritrue : tribool::indeterminate;
}

tribool Assumptions::is_rational(const RCP<const Basic> &symbol) const
{
    bool rational = rational_symbols_.find(symbol) != rational_symbols_.end();
    return rational ? tribool::tritrue : tribool::indeterminate;
}

tribool Assumptions::is_integer(const RCP<const Basic> &symbol) const
{
    bool integer = integer_symbols_.find(symbol) != integer_symbols_.end();
    return integer ? tribool::tritrue : tribool::indeterminate;
}

tribool Assumptions::is_positive(const RCP<const Basic> &symbol) const
{
    return from_map(positive_, symbol);
}

tribool Assumptions::is_nonnegative(const RCP<const Basic> &symbol) const
{
    return from_map(nonnegative_, symbol);
}

tribool Assumptions::is_negative(const RCP<const Basic> &symbol) const
{
    return from_map(negative_, symbol);
}

tribool Assumptions::is_nonpositive(const RCP<const Basic> &symbol) const
{
    return from_map(nonpositive_, symbol);
}

tribool Assumptions::is_nonzero(const RCP<const Basic> &symbol) const
{
    return from_map(nonzero_, symbol);
}

tribool Assumptions::is_zero(const RCP<const Basic> &symbol) const
{
    return from_map(zero_, symbol);
}

} // namespace SymEngine
