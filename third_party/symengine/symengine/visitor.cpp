#include <symengine/visitor.h>
#include <symengine/polys/basic_conversions.h>
#include <symengine/sets.h>

#define ACCEPT(CLASS)                                                          \
    void CLASS::accept(Visitor &v) const                                       \
    {                                                                          \
        v.visit(*this);                                                        \
    }

namespace SymEngine
{

#define SYMENGINE_ENUM(TypeID, Class) ACCEPT(Class)
#include "symengine/type_codes.inc"
#undef SYMENGINE_ENUM

void preorder_traversal(const Basic &b, Visitor &v)
{
    b.accept(v);
    for (const auto &p : b.get_args())
        preorder_traversal(*p, v);
}

void postorder_traversal(const Basic &b, Visitor &v)
{
    for (const auto &p : b.get_args())
        postorder_traversal(*p, v);
    b.accept(v);
}

void preorder_traversal_stop(const Basic &b, StopVisitor &v)
{
    b.accept(v);
    if (v.stop_)
        return;
    for (const auto &p : b.get_args()) {
        preorder_traversal_stop(*p, v);
        if (v.stop_)
            return;
    }
}

void postorder_traversal_stop(const Basic &b, StopVisitor &v)
{
    for (const auto &p : b.get_args()) {
        postorder_traversal_stop(*p, v);
        if (v.stop_)
            return;
    }
    b.accept(v);
}

bool has_symbol(const Basic &b, const Basic &x)
{
    // We are breaking a rule when using ptrFromRef() here, but since
    // HasSymbolVisitor is only instantiated and freed from here, the `x` can
    // never go out of scope, so this is safe.
    HasSymbolVisitor v(ptrFromRef(x));
    return v.apply(b);
}

RCP<const Basic> coeff(const Basic &b, const Basic &x, const Basic &n)
{
    if (!(is_a<Symbol>(x) || is_a<FunctionSymbol>(x))) {
        throw NotImplementedError("Not implemented for non (Function)Symbols.");
    }
    CoeffVisitor v(ptrFromRef(x), ptrFromRef(n));
    return v.apply(b);
}

class FreeSymbolsVisitor : public BaseVisitor<FreeSymbolsVisitor>
{
public:
    set_basic s;
    uset_basic v;

    void bvisit(const Symbol &x)
    {
        s.insert(x.rcp_from_this());
    }

    void bvisit(const Subs &x)
    {
        set_basic set_ = free_symbols(*x.get_arg());
        for (const auto &p : x.get_variables()) {
            set_.erase(p);
        }
        s.insert(set_.begin(), set_.end());
        for (const auto &p : x.get_point()) {
            auto iter = v.insert(p->rcp_from_this());
            if (iter.second) {
                p->accept(*this);
            }
        }
    }

    void bvisit(const Basic &x)
    {
        for (const auto &p : x.get_args()) {
            auto iter = v.insert(p->rcp_from_this());
            if (iter.second) {
                p->accept(*this);
            }
        }
    }

    set_basic apply(const Basic &b)
    {
        b.accept(*this);
        return s;
    }

    set_basic apply(const MatrixBase &m)
    {
        for (unsigned i = 0; i < m.nrows(); i++) {
            for (unsigned j = 0; j < m.ncols(); j++) {
                m.get(i, j)->accept(*this);
            }
        }
        return s;
    }
};

set_basic free_symbols(const MatrixBase &m)
{
    FreeSymbolsVisitor visitor;
    return visitor.apply(m);
}

set_basic free_symbols(const Basic &b)
{
    FreeSymbolsVisitor visitor;
    return visitor.apply(b);
}

set_basic function_symbols(const Basic &b)
{
    return atoms<FunctionSymbol>(b);
}

RCP<const Basic> TransformVisitor::apply(const RCP<const Basic> &x)
{
    x->accept(*this);
    return result_;
}

void TransformVisitor::bvisit(const Basic &x)
{
    result_ = x.rcp_from_this();
}

void TransformVisitor::bvisit(const Add &x)
{
    vec_basic newargs;
    for (const auto &a : x.get_args()) {
        newargs.push_back(apply(a));
    }
    result_ = add(newargs);
}

void TransformVisitor::bvisit(const Mul &x)
{
    vec_basic newargs;
    for (const auto &a : x.get_args()) {
        newargs.push_back(apply(a));
    }
    result_ = mul(newargs);
}

void TransformVisitor::bvisit(const Pow &x)
{
    auto base_ = x.get_base(), exp_ = x.get_exp();
    auto newarg1 = apply(base_), newarg2 = apply(exp_);
    if (base_ != newarg1 or exp_ != newarg2) {
        result_ = pow(newarg1, newarg2);
    } else {
        result_ = x.rcp_from_this();
    }
}

void TransformVisitor::bvisit(const OneArgFunction &x)
{
    auto farg = x.get_arg();
    auto newarg = apply(farg);
    if (eq(*newarg, *farg)) {
        result_ = x.rcp_from_this();
    } else {
        result_ = x.create(newarg);
    }
}

void TransformVisitor::bvisit(const MultiArgFunction &x)
{
    auto fargs = x.get_args();
    vec_basic newargs;
    for (const auto &a : fargs) {
        newargs.push_back(apply(a));
    }
    auto nbarg = x.create(newargs);
    result_ = nbarg;
}

void preorder_traversal_local_stop(const Basic &b, LocalStopVisitor &v)
{
    b.accept(v);
    if (v.stop_ or v.local_stop_)
        return;
    for (const auto &p : b.get_args()) {
        preorder_traversal_local_stop(*p, v);
        if (v.stop_)
            return;
    }
}

void CountOpsVisitor::apply(const Basic &b)
{
    unsigned count_now = count;
    auto it = v.find(b.rcp_from_this());
    if (it == v.end()) {
        b.accept(*this);
        insert(v, b.rcp_from_this(), count - count_now);
    } else {
        count += it->second;
    }
}

void CountOpsVisitor::bvisit(const Mul &x)
{
    if (neq(*(x.get_coef()), *one)) {
        count++;
        apply(*x.get_coef());
    }

    for (const auto &p : x.get_dict()) {
        if (neq(*p.second, *one)) {
            count++;
            apply(*p.second);
        }
        apply(*p.first);
        count++;
    }
    count--;
}

void CountOpsVisitor::bvisit(const Add &x)
{
    if (neq(*(x.get_coef()), *zero)) {
        count++;
        apply(*x.get_coef());
    }

    unsigned i = 0;
    for (const auto &p : x.get_dict()) {
        if (neq(*p.second, *one)) {
            count++;
            apply(*p.second);
        }
        apply(*p.first);
        count++;
        i++;
    }
    count--;
}

void CountOpsVisitor::bvisit(const Pow &x)
{
    count++;
    apply(*x.get_exp());
    apply(*x.get_base());
}

void CountOpsVisitor::bvisit(const Number &x)
{
}

void CountOpsVisitor::bvisit(const ComplexBase &x)
{
    if (neq(*x.real_part(), *zero)) {
        count++;
    }

    if (neq(*x.imaginary_part(), *one)) {
        count++;
    }
}

void CountOpsVisitor::bvisit(const Symbol &x)
{
}

void CountOpsVisitor::bvisit(const Constant &x)
{
}

void CountOpsVisitor::bvisit(const Basic &x)
{
    count++;
    for (const auto &p : x.get_args()) {
        apply(*p);
    }
}

unsigned count_ops(const vec_basic &a)
{
    CountOpsVisitor v;
    for (auto &p : a) {
        v.apply(*p);
    }
    return v.count;
}

} // SymEngine
