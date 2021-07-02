#ifndef SYMENGINE_LLVM_DOUBLE_H
#define SYMENGINE_LLVM_DOUBLE_H

#include <symengine/basic.h>
#include <symengine/visitor.h>
#include <float.h>

#ifdef HAVE_SYMENGINE_LLVM

// Forward declare llvm types
namespace llvm
{
class Module;
class Value;
class Type;
class Function;
class ExecutionEngine;
class MemoryBufferRef;
class LLVMContext;
class Pass;
namespace legacy
{
class FunctionPassManager;
}
}

namespace SymEngine
{

class IRBuilder;

class LLVMVisitor : public BaseVisitor<LLVMVisitor>
{
protected:
    vec_basic symbols;
    std::vector<llvm::Value *> symbol_ptrs;
    std::map<RCP<const Basic>, llvm::Value *, RCPBasicKeyLess>
        replacement_symbol_ptrs;
    llvm::Value *result_;
    std::shared_ptr<llvm::LLVMContext> context;
    std::shared_ptr<llvm::ExecutionEngine> executionengine;
    std::shared_ptr<llvm::legacy::FunctionPassManager> fpm;
    intptr_t func;

    // Following are invalid after the init call.
    IRBuilder *builder;
    llvm::Module *mod;
    std::string membuffer;
    llvm::Function *get_function_type(llvm::LLVMContext *);
    virtual llvm::Type *get_float_type(llvm::LLVMContext *) = 0;

public:
    llvm::Value *apply(const Basic &b);
    void init(const vec_basic &x, const Basic &b,
              const bool symbolic_cse = false, unsigned opt_level = 3);
    void init(const vec_basic &x, const Basic &b, const bool symbolic_cse,
              const std::vector<llvm::Pass *> &passes, unsigned opt_level = 3);
    void init(const vec_basic &inputs, const vec_basic &outputs,
              const bool symbolic_cse = false, unsigned opt_level = 3);
    void init(const vec_basic &inputs, const vec_basic &outputs,
              const bool symbolic_cse, const std::vector<llvm::Pass *> &passes,
              unsigned opt_level = 3);

    static std::vector<llvm::Pass *> create_default_passes(int optlevel);

    // Helper functions
    void set_double(double d);
    llvm::Function *get_external_function(const std::string &name,
                                          size_t nargs = 1);
    llvm::Function *get_powi();

    void bvisit(const Integer &x);
    void bvisit(const Rational &x);
    void bvisit(const RealDouble &x);
#ifdef HAVE_SYMENGINE_MPFR
    void bvisit(const RealMPFR &x);
#endif
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const Log &x);
    void bvisit(const Abs &x);
    void bvisit(const Symbol &x);
    void bvisit(const Constant &x);
    void bvisit(const Basic &);
    void bvisit(const Sin &x);
    void bvisit(const Cos &x);
    void bvisit(const Piecewise &x);
    void bvisit(const BooleanAtom &x);
    void bvisit(const And &x);
    void bvisit(const Or &x);
    void bvisit(const Xor &x);
    void bvisit(const Not &x);
    void bvisit(const Equality &x);
    void bvisit(const Unequality &x);
    void bvisit(const LessThan &x);
    void bvisit(const StrictLessThan &x);
    void bvisit(const Max &x);
    void bvisit(const Min &x);
    void bvisit(const Contains &x);
    void bvisit(const Infty &x);
    void bvisit(const Floor &x);
    void bvisit(const Ceiling &x);
    void bvisit(const Truncate &x);
    void bvisit(const Sign &x);
    // Return the compiled function as a binary string which can be loaded using
    // `load`
    const std::string &dumps() const;
    // Load a previously compiled function from a string
    void loads(const std::string &s);
    void bvisit(const UnevaluatedExpr &x);
};

class LLVMDoubleVisitor : public LLVMVisitor
{
public:
    double call(const std::vector<double> &vec) const;
    void call(double *outs, const double *inps) const;
    llvm::Type *get_float_type(llvm::LLVMContext *) override;
    void visit(const Tan &x) override;
    void visit(const ASin &x) override;
    void visit(const ACos &x) override;
    void visit(const ATan &x) override;
    void visit(const ATan2 &x) override;
    void visit(const Sinh &x) override;
    void visit(const Cosh &x) override;
    void visit(const Tanh &x) override;
    void visit(const ASinh &x) override;
    void visit(const ACosh &x) override;
    void visit(const ATanh &x) override;
    void visit(const Gamma &x) override;
    void visit(const LogGamma &x) override;
    void visit(const Erf &x) override;
    void visit(const Erfc &x) override;
};

class LLVMFloatVisitor : public LLVMVisitor
{
public:
    float call(const std::vector<float> &vec) const;
    void call(float *outs, const float *inps) const;
    llvm::Type *get_float_type(llvm::LLVMContext *) override;
    void visit(const Tan &x) override;
    void visit(const ASin &x) override;
    void visit(const ACos &x) override;
    void visit(const ATan &x) override;
    void visit(const ATan2 &x) override;
    void visit(const Sinh &x) override;
    void visit(const Cosh &x) override;
    void visit(const Tanh &x) override;
    void visit(const ASinh &x) override;
    void visit(const ACosh &x) override;
    void visit(const ATanh &x) override;
    void visit(const Gamma &x) override;
    void visit(const LogGamma &x) override;
    void visit(const Erf &x) override;
    void visit(const Erfc &x) override;
};

#if SYMENGINE_SIZEOF_LONG_DOUBLE > 8 && defined(__x86_64__) || defined(__i386__)
#define SYMENGINE_HAVE_LLVM_LONG_DOUBLE 1
class LLVMLongDoubleVisitor : public LLVMVisitor
{
public:
    long double call(const std::vector<long double> &vec) const;
    void call(long double *outs, const long double *inps) const;
    llvm::Type *get_float_type(llvm::LLVMContext *) override;
    void visit(const Tan &x) override;
    void visit(const ASin &x) override;
    void visit(const ACos &x) override;
    void visit(const ATan &x) override;
    void visit(const ATan2 &x) override;
    void visit(const Sinh &x) override;
    void visit(const Cosh &x) override;
    void visit(const Tanh &x) override;
    void visit(const ASinh &x) override;
    void visit(const ACosh &x) override;
    void visit(const ATanh &x) override;
    void visit(const Gamma &x) override;
    void visit(const LogGamma &x) override;
    void visit(const Erf &x) override;
    void visit(const Erfc &x) override;
    void visit(const Integer &x) override;
    void visit(const Rational &x) override;
    void convert_from_mpfr(const Basic &x);
    void visit(const Constant &x) override;
#ifdef HAVE_SYMENGINE_MPFR
    void visit(const RealMPFR &x) override;
#endif
};
#endif

} // namespace SymEngine
#endif
#endif // SYMENGINE_LAMBDA_DOUBLE_H
