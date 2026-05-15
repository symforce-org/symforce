#ifndef SYMENGINE_TEST_VISITORS_H
#define SYMENGINE_TEST_VISITORS_H

#include <symengine/visitor.h>

namespace SymEngine
{

class ZeroVisitor : public BaseVisitor<ZeroVisitor>
{
private:
    tribool is_zero_;
    const Assumptions *assumptions_;

    void error();

public:
    ZeroVisitor(const Assumptions *assumptions) : assumptions_(assumptions) {}

    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);
    void bvisit(const Constant &x);
    void bvisit(const Abs &x);
    void bvisit(const Conjugate &x);
    void bvisit(const Sign &x);
    void bvisit(const PrimePi &x);

    tribool apply(const Basic &b);
};

class PositiveVisitor : public BaseVisitor<PositiveVisitor>
{
private:
    tribool is_positive_;
    const Assumptions *assumptions_;

    void error();

public:
    PositiveVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions){};
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Constant &x);
    void bvisit(const Add &x);
    void bvisit(const Basic &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);

    tribool apply(const Basic &b);
};

class NonPositiveVisitor : public BaseVisitor<NonPositiveVisitor>
{
private:
    tribool is_nonpositive_;
    const Assumptions *assumptions_;

    void error();

public:
    NonPositiveVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions){};
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Constant &x);
    void bvisit(const Basic &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);

    tribool apply(const Basic &b);
};

class NegativeVisitor : public BaseVisitor<NegativeVisitor>
{
private:
    tribool is_negative_;
    const Assumptions *assumptions_;

    void error();

public:
    NegativeVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions){};
    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);
    void bvisit(const Constant &x);

    tribool apply(const Basic &b);
};

class NonNegativeVisitor : public BaseVisitor<NonNegativeVisitor>
{
private:
    tribool is_nonnegative_;
    const Assumptions *assumptions_;

    void error();

public:
    NonNegativeVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions){};
    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);
    void bvisit(const Constant &x);

    tribool apply(const Basic &b);
};

class IntegerVisitor : public BaseVisitor<IntegerVisitor>
{
private:
    tribool is_integer_;
    const Assumptions *assumptions_;

public:
    IntegerVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions){};
    void bvisit(const Basic &x)
    {
        is_integer_ = tribool::indeterminate;
    };
    void bvisit(const Symbol &x);
    void bvisit(const Integer &x)
    {
        is_integer_ = tribool::tritrue;
    };
    void bvisit(const Number &x)
    {
        is_integer_ = tribool::trifalse;
    };
    void bvisit(const Set &x)
    {
        is_integer_ = tribool::trifalse;
    };
    void bvisit(const Relational &x)
    {
        is_integer_ = tribool::trifalse;
    };
    void bvisit(const Boolean &x)
    {
        is_integer_ = tribool::trifalse;
    };
    void bvisit(const Constant &x);
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Conjugate &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const KroneckerDelta &x)
    {
        is_integer_ = tribool::tritrue;
    };

    tribool apply(const Basic &b);
};

class RealVisitor : public BaseVisitor<RealVisitor>
{
private:
    tribool is_real_;
    const Assumptions *assumptions_;

    void check_power(const RCP<const Basic> &base, const RCP<const Basic> &exp);

public:
    RealVisitor(const Assumptions *assumptions) : assumptions_(assumptions){};
    void bvisit(const Basic &x)
    {
        is_real_ = tribool::indeterminate;
    };
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Set &x)
    {
        is_real_ = tribool::trifalse;
    };
    void bvisit(const Relational &x)
    {
        is_real_ = tribool::trifalse;
    };
    void bvisit(const Boolean &x)
    {
        is_real_ = tribool::trifalse;
    };
    void bvisit(const Constant &x);
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);

    tribool apply(const Basic &b);
};

class ComplexVisitor : public BaseVisitor<ComplexVisitor>
{
private:
    tribool is_complex_;
    const Assumptions *assumptions_;

    void check_power(const Basic &base, const Basic &exp);
    void complex_arg_not_zero(const OneArgFunction &x, const Basic &not_zero);
    void complex_arg_not_pm(const OneArgFunction &x, bool one);

public:
    ComplexVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions){};
    void bvisit(const Basic &x)
    {
        is_complex_ = tribool::indeterminate;
    };
    void bvisit(const Symbol &x);
    void bvisit(const Number &x);
    void bvisit(const Integer &x)
    {
        is_complex_ = tribool::tritrue;
    }
    void bvisit(const Rational &x)
    {
        is_complex_ = tribool::tritrue;
    }
    void bvisit(const Set &x)
    {
        is_complex_ = tribool::trifalse;
    };
    void bvisit(const Relational &x)
    {
        is_complex_ = tribool::trifalse;
    };
    void bvisit(const Boolean &x)
    {
        is_complex_ = tribool::trifalse;
    };
    void bvisit(const Constant &x)
    {
        is_complex_ = tribool::tritrue;
    };
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const Cos &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Sin &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const ASin &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const ACos &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const ATan &x);
    void bvisit(const ATanh &x);
    void bvisit(const ACot &x);
    void bvisit(const ACoth &x);
    void bvisit(const Sinh &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Cosh &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Tan &x);
    void bvisit(const Cot &x);
    void bvisit(const Sec &x);
    void bvisit(const ASec &x);
    void bvisit(const ASech &x);
    void bvisit(const Csc &x);
    void bvisit(const ACsc &x);
    void bvisit(const ACsch &x);
    void bvisit(const Log &x);
    void bvisit(const Sign &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Floor &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Ceiling &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Abs &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const Conjugate &x)
    {
        x.get_arg()->accept(*this);
    };
    void bvisit(const KroneckerDelta &x)
    {
        is_complex_ = tribool::tritrue;
    };

    tribool apply(const Basic &b);
};

class PolynomialVisitor : public BaseVisitor<PolynomialVisitor>
{
private:
    bool is_polynomial_ = true;
    bool variables_allowed_ = true;
    const set_basic &variables_;

    void check_power(const Basic &base, const Basic &exp);

public:
    PolynomialVisitor(const set_basic &variables) : variables_(variables) {}
    void bvisit(const Basic &x);
    void bvisit(const Number &x){};
    void bvisit(const Constant &x){};
    void bvisit(const Symbol &x);
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const Set &x)
    {
        is_polynomial_ = false;
    };
    void bvisit(const Relational &x)
    {
        is_polynomial_ = false;
    };

    bool apply(const Basic &b);
};

/**
 * @brief Check if expression is a polynomial
 * @param b Basic
 * @param variables Set of symbols for variables in polynomial
 * @returns True if b is a polynomial in variables and false otherwise
 *
 * Check if b is a polynomial in variables. If no variables are specified
 * all free symbols in b are considered to be variables. All symbols that
 * are not variables will be considered to be constants.
 */
bool is_polynomial(const Basic &b, const set_basic &variables = {});

class RationalVisitor : public BaseVisitor<RationalVisitor>
{
private:
    bool rational_; // are we testing for rational or irrational?
    tribool is_rational_;
    bool neither_ = false; // Neither rational or irrational (i.e. not real)

public:
    RationalVisitor(bool rational) : rational_{rational} {}
    void bvisit(const Basic &x)
    {
        is_rational_ = tribool::indeterminate;
    };
    void bvisit(const Symbol &x)
    {
        is_rational_ = tribool::indeterminate;
    };
    void bvisit(const Integer &x)
    {
        is_rational_ = tribool::tritrue;
    };
    void bvisit(const Rational &x)
    {
        is_rational_ = tribool::tritrue;
    };
    void bvisit(const Number &x);
    void bvisit(const Set &x)
    {
        is_rational_ = tribool::trifalse;
        neither_ = true;
    };
    void bvisit(const Relational &x)
    {
        is_rational_ = tribool::trifalse;
        neither_ = true;
    };
    void bvisit(const Boolean &x)
    {
        is_rational_ = tribool::trifalse;
        neither_ = true;
    };
    void bvisit(const Constant &x);
    void bvisit(const Add &x);

    tribool apply(const Basic &b);
};

class FiniteVisitor : public BaseVisitor<FiniteVisitor>
{
private:
    tribool is_finite_;
    const Assumptions *assumptions_;

    void error();

public:
    FiniteVisitor(const Assumptions *assumptions) : assumptions_(assumptions){};

    void bvisit(const Basic &x);
    void bvisit(const Symbol &x);
    void bvisit(const Infty &x);
    void bvisit(const NaN &x);
    void bvisit(const Number &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);
    void bvisit(const Constant &x);

    tribool apply(const Basic &b);
};

class AlgebraicVisitor : public BaseVisitor<AlgebraicVisitor>
{
private:
    tribool is_algebraic_;
    const Assumptions *assumptions_;

    void trans_nonzero_and_algebraic(const Basic &b);
    void error();

public:
    AlgebraicVisitor(const Assumptions *assumptions) : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x);
    void bvisit(const Add &x);
    void bvisit(const Symbol &x);
    void bvisit(const Integer &x);
    void bvisit(const Rational &x);
    void bvisit(const Set &x);
    void bvisit(const Relational &x);
    void bvisit(const Boolean &x);
    void bvisit(const Constant &x);
    void bvisit(const TrigFunction &x);
    void bvisit(const HyperbolicFunction &x);
    void bvisit(const LambertW &x);

    tribool apply(const Basic &b);
};

} // namespace SymEngine

#endif // SYMENGINE_TEST_VISITORS_H
