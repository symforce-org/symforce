#ifndef SYMENGINE_SBML_H
#define SYMENGINE_SBML_H

#include <symengine/visitor.h>
#include <symengine/printers/strprinter.h>

namespace SymEngine
{

class SbmlPrinter : public BaseVisitor<SbmlPrinter, StrPrinter>
{
public:
    using StrPrinter::apply;
    using StrPrinter::bvisit;
    static const std::vector<std::string> names_;
    void _print_pow(std::ostringstream &o, const RCP<const Basic> &a,
                    const RCP<const Basic> &b) override;
    void bvisit(const BooleanAtom &x);
    void bvisit(const And &x);
    void bvisit(const Or &x);
    void bvisit(const Xor &x);
    void bvisit(const Not &x);
    void bvisit(const Piecewise &x);
    void bvisit(const Infty &x);
    void bvisit(const Constant &x);
    void bvisit(const Function &x);
};

} // namespace SymEngine

#endif
