#include <symengine/simplify.h>
#include <symengine/refine.h>

namespace SymEngine
{

void SimplifyVisitor::bvisit(const OneArgFunction &x)
{
    auto farg = x.get_arg();
    auto newarg = apply(farg);
    result_ = x.create(newarg);
}

void SimplifyVisitor::bvisit(const Pow &x)
{
    auto e = apply(x.get_exp());
    auto base = apply(x.get_base());
    auto pair = simplify_pow(e, base);
    result_ = pow(pair.second, pair.first);
}

std::pair<RCP<const Basic>, RCP<const Basic>>
SimplifyVisitor::simplify_pow(const RCP<const Basic> &e,
                              const RCP<const Basic> &b)
{
    if (is_a<Csc>(*b) and eq(*e, *minus_one)) {
        // csc(expr) ** -1 = sin(expr)
        return std::make_pair(
            one, sin(down_cast<const OneArgFunction &>(*b).get_arg()));
    } else if (is_a<Sec>(*b) and eq(*e, *minus_one)) {
        // sec(expr) ** -1 = cos(expr)
        return std::make_pair(
            one, cos(down_cast<const OneArgFunction &>(*b).get_arg()));
    } else if (is_a<Cot>(*b) and eq(*e, *minus_one)) {
        // cot(expr) ** -1 = tan(expr)
        return std::make_pair(
            one, tan(down_cast<const OneArgFunction &>(*b).get_arg()));
    } else {
        return std::make_pair(e, b);
    }
}

void SimplifyVisitor::bvisit(const Mul &x)
{
    map_basic_basic map;
    for (const auto &p : x.get_dict()) {
        auto base = apply(p.first);
        auto newpair = simplify_pow(p.second, base);
        Mul::dict_add_term(map, newpair.first, newpair.second);
    }
    result_ = Mul::from_dict(x.get_coef(), std::move(map));
}

RCP<const Basic> simplify(const RCP<const Basic> &x,
                          const Assumptions *assumptions)
{
    auto expr = refine(x, assumptions);
    SimplifyVisitor b;
    return b.apply(expr);
}

} // namespace SymEngine
