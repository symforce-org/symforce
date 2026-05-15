#ifndef SYMENGINE_SIMPLIFY_H
#define SYMENGINE_SIMPLIFY_H

#include <symengine/visitor.h>
#include <symengine/basic.h>
#include <symengine/assumptions.h>

namespace SymEngine
{

class SimplifyVisitor : public BaseVisitor<SimplifyVisitor, TransformVisitor>
{
private:
    std::pair<RCP<const Basic>, RCP<const Basic>>
    simplify_pow(const RCP<const Basic> &e, const RCP<const Basic> &b);

public:
    using TransformVisitor::bvisit;

    SimplifyVisitor() : BaseVisitor<SimplifyVisitor, TransformVisitor>() {}

    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const OneArgFunction &x);
};

RCP<const Basic> simplify(const RCP<const Basic> &x,
                          const Assumptions *assumptions = nullptr);

} // namespace SymEngine

#endif
