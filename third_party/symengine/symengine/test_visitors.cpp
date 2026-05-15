#include <symengine/test_visitors.h>

namespace SymEngine
{

void ZeroVisitor::error()
{
    throw SymEngineException(
        "Only numeric types allowed for is_zero/is_nonzero");
}

void ZeroVisitor::bvisit(const Basic &x)
{
    is_zero_ = tribool::indeterminate;
}

void ZeroVisitor::bvisit(const Set &x)
{
    error();
}

void ZeroVisitor::bvisit(const Relational &x)
{
    error();
}

void ZeroVisitor::bvisit(const Boolean &x)
{
    error();
}

void ZeroVisitor::bvisit(const Constant &x)
{
    is_zero_ = tribool::trifalse;
}

void ZeroVisitor::bvisit(const Abs &x)
{
    x.get_arg()->accept(*this);
}

void ZeroVisitor::bvisit(const Conjugate &x)
{
    x.get_arg()->accept(*this);
}

void ZeroVisitor::bvisit(const Sign &x)
{
    x.get_arg()->accept(*this);
}

void ZeroVisitor::bvisit(const PrimePi &x)
{
    // First prime is 2 so pi(x) is zero for x < 2
    is_zero_ = is_negative(*sub(x.get_arg(), integer(2)));
}

void ZeroVisitor::bvisit(const Number &x)
{
    if (bool(x.is_zero())) {
        is_zero_ = tribool::tritrue;
    } else {
        is_zero_ = tribool::trifalse;
    }
}

void ZeroVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_zero_ = assumptions_->is_zero(x.rcp_from_this());
    } else {
        is_zero_ = tribool::indeterminate;
    }
}

tribool ZeroVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_zero_;
}

tribool is_zero(const Basic &b, const Assumptions *assumptions)
{
    ZeroVisitor visitor(assumptions);
    return visitor.apply(b);
}

tribool is_nonzero(const Basic &b, const Assumptions *assumptions)
{
    ZeroVisitor visitor(assumptions);
    return not_tribool(visitor.apply(b));
}

void PositiveVisitor::error()
{
    throw SymEngineException("Only numeric types allowed for is_positive");
}

void PositiveVisitor::bvisit(const Constant &x)
{
    is_positive_ = tribool::tritrue;
}

void PositiveVisitor::bvisit(const Basic &x)
{
    is_positive_ = tribool::indeterminate;
}

void PositiveVisitor::bvisit(const Set &x)
{
    error();
}

void PositiveVisitor::bvisit(const Relational &x)
{
    error();
}

void PositiveVisitor::bvisit(const Boolean &x)
{
    error();
}

void PositiveVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_positive_ = assumptions_->is_positive(x.rcp_from_this());
    } else {
        is_positive_ = tribool::indeterminate;
    }
}

void PositiveVisitor::bvisit(const Number &x)
{
    if (is_a_Complex(x)) {
        is_positive_ = tribool::trifalse;
    } else if (bool(x.is_positive())) {
        is_positive_ = tribool::tritrue;
    } else {
        is_positive_ = tribool::trifalse;
    }
}

void PositiveVisitor::bvisit(const Add &x)
{
    // True if all are positive
    // False if all are negative
    auto coef = x.get_coef();
    auto dict = x.get_dict();

    bool can_be_true = true;
    bool can_be_false = true;
    if (coef->is_positive()) {
        can_be_false = false;
    } else if (coef->is_negative()) {
        can_be_true = false;
    }
    NegativeVisitor neg_visitor(assumptions_);
    for (const auto &p : dict) {
        if (not can_be_true and not can_be_false) {
            is_positive_ = tribool::indeterminate;
            return;
        }
        p.first->accept(*this);
        if ((p.second->is_positive() and is_true(is_positive_))
            or (p.second->is_negative()
                and is_true(neg_visitor.apply(*p.first)))) {
            // key * value is positive
            can_be_false = false;
        } else if ((p.second->is_negative() and is_true(is_positive_))
                   or (p.second->is_positive()
                       and is_true(neg_visitor.apply(*p.first)))) {
            // key * value is negative
            can_be_true = false;
        } else {
            can_be_true = false;
            can_be_false = false;
        }
    }
    if (can_be_true) {
        is_positive_ = tribool::tritrue;
    } else if (can_be_false) {
        is_positive_ = tribool::trifalse;
    } else {
        is_positive_ = tribool::indeterminate;
    }
}

tribool PositiveVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_positive_;
}

tribool is_positive(const Basic &b, const Assumptions *assumptions)
{
    PositiveVisitor visitor(assumptions);
    return visitor.apply(b);
}

void NonPositiveVisitor::error()
{
    throw SymEngineException("Only numeric types allowed for is_negative");
}

void NonPositiveVisitor::bvisit(const Constant &x)
{
    is_nonpositive_ = tribool::trifalse;
}

void NonPositiveVisitor::bvisit(const Basic &x)
{
    is_nonpositive_ = tribool::indeterminate;
}

void NonPositiveVisitor::bvisit(const Set &x)
{
    error();
}

void NonPositiveVisitor::bvisit(const Relational &x)
{
    error();
}

void NonPositiveVisitor::bvisit(const Boolean &x)
{
    error();
}

void NonPositiveVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_nonpositive_ = assumptions_->is_nonpositive(x.rcp_from_this());
    } else {
        is_nonpositive_ = tribool::indeterminate;
    }
}

void NonPositiveVisitor::bvisit(const Number &x)
{
    if (is_a_Complex(x)) {
        is_nonpositive_ = tribool::trifalse;
    } else if (bool(x.is_positive())) {
        is_nonpositive_ = tribool::trifalse;
    } else {
        is_nonpositive_ = tribool::tritrue;
    }
}

tribool NonPositiveVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_nonpositive_;
}

tribool is_nonpositive(const Basic &b, const Assumptions *assumptions)
{
    NonPositiveVisitor visitor(assumptions);
    return visitor.apply(b);
}

void NegativeVisitor::error()
{
    throw SymEngineException("Only numeric types allowed for is_negative");
}

void NegativeVisitor::bvisit(const Basic &x)
{
    is_negative_ = tribool::indeterminate;
}

void NegativeVisitor::bvisit(const Set &x)
{
    error();
}

void NegativeVisitor::bvisit(const Relational &x)
{
    error();
}

void NegativeVisitor::bvisit(const Boolean &x)
{
    error();
}

void NegativeVisitor::bvisit(const Constant &x)
{
    is_negative_ = tribool::trifalse;
}

void NegativeVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_negative_ = assumptions_->is_negative(x.rcp_from_this());
    } else {
        is_negative_ = tribool::indeterminate;
    }
}

void NegativeVisitor::bvisit(const Number &x)
{
    if (is_a_Complex(x)) {
        is_negative_ = tribool::trifalse;
    } else if (bool(x.is_negative())) {
        is_negative_ = tribool::tritrue;
    } else {
        is_negative_ = tribool::trifalse;
    }
}

tribool NegativeVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_negative_;
}

tribool is_negative(const Basic &b, const Assumptions *assumptions)
{
    NegativeVisitor visitor(assumptions);
    return visitor.apply(b);
}

void NonNegativeVisitor::error()
{
    throw SymEngineException("Only numeric types allowed for is_nonnegative");
}

void NonNegativeVisitor::bvisit(const Basic &x)
{
    is_nonnegative_ = tribool::indeterminate;
}

void NonNegativeVisitor::bvisit(const Set &x)
{
    error();
}

void NonNegativeVisitor::bvisit(const Relational &x)
{
    error();
}

void NonNegativeVisitor::bvisit(const Boolean &x)
{
    error();
}

void NonNegativeVisitor::bvisit(const Constant &x)
{
    is_nonnegative_ = tribool::tritrue;
}

void NonNegativeVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_nonnegative_ = assumptions_->is_nonnegative(x.rcp_from_this());
    } else {
        is_nonnegative_ = tribool::indeterminate;
    }
}

void NonNegativeVisitor::bvisit(const Number &x)
{
    if (is_a_Complex(x)) {
        is_nonnegative_ = tribool::trifalse;
    } else if (bool(x.is_negative())) {
        is_nonnegative_ = tribool::trifalse;
    } else {
        is_nonnegative_ = tribool::tritrue;
    }
}

tribool NonNegativeVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_nonnegative_;
}

tribool is_nonnegative(const Basic &b, const Assumptions *assumptions)
{
    NonNegativeVisitor visitor(assumptions);
    return visitor.apply(b);
}

void IntegerVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_integer_ = assumptions_->is_integer(x.rcp_from_this());
    } else {
        is_integer_ = tribool::indeterminate;
    }
}

void IntegerVisitor::bvisit(const Constant &x)
{
    if (eq(x, *pi) or eq(x, *E) or eq(x, *EulerGamma) or eq(x, *Catalan)
        or eq(x, *GoldenRatio)) {
        is_integer_ = tribool::trifalse;
    } else {
        is_integer_ = tribool::indeterminate;
    }
}

void IntegerVisitor::bvisit(const Add &x)
{
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        if (not is_true(is_integer_)) {
            is_integer_ = tribool::indeterminate;
            return;
        }
    }
}

void IntegerVisitor::bvisit(const Mul &x)
{
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        if (not is_true(is_integer_)) {
            is_integer_ = tribool::indeterminate;
            return;
        }
    }
}

tribool IntegerVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_integer_;
}

tribool is_integer(const Basic &b, const Assumptions *assumptions)
{
    IntegerVisitor visitor(assumptions);
    return visitor.apply(b);
}

void RealVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_real_ = assumptions_->is_real(x.rcp_from_this());
    } else {
        is_real_ = tribool::indeterminate;
    }
}

void RealVisitor::bvisit(const Number &x)
{
    if (is_a_Complex(x) or is_a<Infty>(x) or is_a<NaN>(x)) {
        is_real_ = tribool::trifalse;
    } else {
        is_real_ = tribool::tritrue;
    }
}

void RealVisitor::bvisit(const Constant &x)
{
    if (eq(x, *pi) or eq(x, *E) or eq(x, *EulerGamma) or eq(x, *Catalan)
        or eq(x, *GoldenRatio)) {
        is_real_ = tribool::tritrue;
    } else {
        is_real_ = tribool::indeterminate;
    }
}

void RealVisitor::bvisit(const Add &x)
{
    tribool b = tribool::tritrue;
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        b = andwk_tribool(b, is_real_);
        if (is_indeterminate(b)) {
            break;
        }
    }
    is_real_ = b;
}

void RealVisitor::check_power(const RCP<const Basic> &base,
                              const RCP<const Basic> &exp)
{
    if (is_true(is_zero(*exp, assumptions_))) {
        // exp == 0 => true
        is_real_ = tribool::tritrue;
        return;
    }
    base->accept(*this);
    if (is_true(is_real_)) {
        if (is_true(is_integer(*exp, assumptions_))) {
            // base is real and exp is integer => true
            is_real_ = tribool::tritrue;
        } else if (is_true(is_nonnegative(*base, assumptions_))) {
            // base >= 0 and exp is real => true
            exp->accept(*this);
            if (is_false(is_real_)) {
                is_real_ = tribool::indeterminate;
            }
        } else {
            is_real_ = tribool::indeterminate;
        }
    } else if (is_false(is_real_) && is_true(is_complex(*base, assumptions_))
               && is_true(is_zero(*sub(exp, integer(1)), assumptions_))) {
        // base is not real but complex and exp = 1 => false
        is_real_ = tribool::trifalse;
    } else {
        is_real_ = tribool::indeterminate;
    }
}

void RealVisitor::bvisit(const Mul &x)
{
    unsigned non_real = 0;
    tribool b = tribool_from_bool(!x.get_coef()->is_complex());
    if (is_false(b)) {
        non_real++;
    }
    for (const auto &p : x.get_dict()) {
        this->check_power(p.first, p.second);
        if (is_false(is_real_)) {
            non_real++;
            if (non_real > 1) {
                is_real_ = tribool::indeterminate;
                return;
            }
        }
        b = andwk_tribool(b, is_real_);
        if (is_indeterminate(b)) {
            is_real_ = tribool::indeterminate;
            return;
        }
    }
    if (non_real == 1) {
        is_real_ = tribool::trifalse;
    } else {
        is_real_ = b;
    }
}

void RealVisitor::bvisit(const Pow &x)
{
    this->check_power(x.get_base(), x.get_exp());
}

tribool RealVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_real_;
}

tribool is_real(const Basic &b, const Assumptions *assumptions)
{
    RealVisitor visitor(assumptions);
    return visitor.apply(b);
}

void ComplexVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_complex_ = assumptions_->is_complex(x.rcp_from_this());
    } else {
        is_complex_ = tribool::indeterminate;
    }
}

void ComplexVisitor::bvisit(const Number &x)
{
    if (is_a<Infty>(x) or is_a<NaN>(x)) {
        is_complex_ = tribool::trifalse;
    } else {
        is_complex_ = tribool::tritrue;
    }
}

void ComplexVisitor::bvisit(const Add &x)
{
    tribool b = tribool::tritrue;
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        b = andwk_tribool(b, is_complex_);
        if (is_indeterminate(b) or is_false(b))
            return;
    }
}

void ComplexVisitor::bvisit(const Mul &x)
{
    tribool b = tribool::tritrue;
    for (const auto &p : x.get_dict()) {
        this->check_power(*p.first, *p.second);
        b = andwk_tribool(b, is_complex_);
        if (is_indeterminate(b) or is_false(b))
            return;
    }
}

void ComplexVisitor::check_power(const Basic &base, const Basic &exp)
{
    base.accept(*this);
    if (is_true(is_complex_)) {
        exp.accept(*this);
    }
}

void ComplexVisitor::bvisit(const Pow &x)
{
    check_power(*x.get_base(), *x.get_exp());
}

void ComplexVisitor::bvisit(const Log &x)
{
    complex_arg_not_zero(x, *x.get_arg());
}

void ComplexVisitor::bvisit(const Tan &x)
{
    complex_arg_not_zero(x, *cos(x.get_arg()));
}

void ComplexVisitor::complex_arg_not_zero(const OneArgFunction &x,
                                          const Basic &not_zero)
{
    // Check if function argument is complex and then if 'not_zero' is not zero
    x.get_arg()->accept(*this);
    if (is_true(is_complex_)) {
        tribool zero = is_zero(not_zero);
        if (not is_false(zero)) {
            is_complex_ = not_tribool(zero);
        }
    }
}

void ComplexVisitor::complex_arg_not_pm(const OneArgFunction &x, bool one)
{
    // Check if function argument is complex but not plus/minus 1 (one=True) or
    // i (one=False)
    x.get_arg()->accept(*this);
    if (not is_true(is_complex_))
        return;
    RCP<const Number> i1;
    if (one)
        i1 = integer(1);
    else
        i1 = Complex::from_two_nums(*integer(0), *integer(1));
    tribool zi1 = is_zero(*sub(x.get_arg(), i1));
    if (not is_false(zi1)) {
        is_complex_ = not_tribool(zi1);
        return;
    }
    RCP<const Number> mi1;
    if (one)
        mi1 = integer(-1);
    else
        mi1 = Complex::from_two_nums(*integer(0), *integer(-1));
    tribool zmi1 = is_zero(*sub(x.get_arg(), mi1));
    is_complex_ = not_tribool(zmi1);
}

void ComplexVisitor::bvisit(const ATan &x)
{
    complex_arg_not_pm(x, false);
}

void ComplexVisitor::bvisit(const ATanh &x)
{
    complex_arg_not_pm(x, true);
}

void ComplexVisitor::bvisit(const ACot &x)
{
    complex_arg_not_pm(x, false);
}

void ComplexVisitor::bvisit(const ACoth &x)
{
    complex_arg_not_pm(x, true);
}

void ComplexVisitor::bvisit(const Cot &x)
{
    complex_arg_not_zero(x, *sin(x.get_arg()));
}

void ComplexVisitor::bvisit(const Sec &x)
{
    complex_arg_not_zero(x, *cos(x.get_arg()));
}

void ComplexVisitor::bvisit(const ASec &x)
{
    complex_arg_not_zero(x, *x.get_arg());
}

void ComplexVisitor::bvisit(const ASech &x)
{
    complex_arg_not_zero(x, *x.get_arg());
}

void ComplexVisitor::bvisit(const Csc &x)
{
    complex_arg_not_zero(x, *sin(x.get_arg()));
}

void ComplexVisitor::bvisit(const ACsc &x)
{
    complex_arg_not_zero(x, *x.get_arg());
}

void ComplexVisitor::bvisit(const ACsch &x)
{
    complex_arg_not_zero(x, *x.get_arg());
}

tribool ComplexVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_complex_;
}

tribool is_complex(const Basic &b, const Assumptions *assumptions)
{
    ComplexVisitor visitor(assumptions);
    return visitor.apply(b);
}

void PolynomialVisitor::bvisit(const Basic &x)
{
    auto old_allowed = variables_allowed_;
    variables_allowed_ = false;
    for (const auto &p : x.get_args()) {
        p->accept(*this);
        if (!is_polynomial_) {
            variables_allowed_ = old_allowed;
            return;
        }
    }
    variables_allowed_ = old_allowed;
}

void PolynomialVisitor::bvisit(const Add &x)
{
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        if (!is_polynomial_)
            return;
    }
}

void PolynomialVisitor::bvisit(const Mul &x)
{
    for (const auto &p : x.get_dict()) {
        this->check_power(*p.first, *p.second);
        if (!is_polynomial_)
            return;
    }
}

void PolynomialVisitor::check_power(const Basic &base, const Basic &exp)
{
    if (variables_allowed_) {
        variables_allowed_ = false;
        exp.accept(*this);
        if (!is_polynomial_) {
            variables_allowed_ = true;
            return;
        }
        base.accept(*this);
        variables_allowed_ = true;
        if (!is_polynomial_) {
            is_polynomial_ = true;
            base.accept(*this);
            is_polynomial_ = is_polynomial_ and is_a<Integer>(exp)
                             and down_cast<const Integer &>(exp).is_positive();
        }
    } else {
        base.accept(*this);
        if (!is_polynomial_)
            return;
        exp.accept(*this);
    }
}

void PolynomialVisitor::bvisit(const Pow &x)
{
    check_power(*x.get_base(), *x.get_exp());
}

void PolynomialVisitor::bvisit(const Symbol &x)
{
    if (variables_allowed_)
        return;

    if (variables_.empty()) { // All symbols are variables
        is_polynomial_ = false;
    } else {
        for (const auto &elem : variables_) {
            if (x.__eq__(*elem)) {
                is_polynomial_ = false;
                return;
            }
        }
    }
}

bool PolynomialVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_polynomial_;
}

bool is_polynomial(const Basic &b, const set_basic &variables)
{
    PolynomialVisitor visitor(variables);
    return visitor.apply(b);
}

void RationalVisitor::bvisit(const Number &x)
{
    is_rational_ = tribool::trifalse;
    if (is_a_Complex(x) or is_a<Infty>(x) or is_a<NaN>(x)) {
        neither_ = true;
    }
}

void RationalVisitor::bvisit(const Constant &x)
{
    if (eq(x, *pi) or eq(x, *E) or eq(x, *GoldenRatio)) {
        // It is currently (2021) not known whether Catalan's constant
        // or Euler's constant are rational or irrational
        is_rational_ = tribool::trifalse;
    } else {
        is_rational_ = tribool::indeterminate;
    }
}

void RationalVisitor::bvisit(const Add &x)
{
    tribool b = tribool::tritrue;
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        b = andwk_tribool(b, is_rational_);
        if (is_indeterminate(b))
            return;
    }
}

tribool RationalVisitor::apply(const Basic &b)
{
    b.accept(*this);
    tribool result = is_rational_;
    if (not rational_ and not neither_) {
        result = not_tribool(result);
    }
    return result;
}

tribool is_rational(const Basic &b)
{
    RationalVisitor visitor(true);
    return visitor.apply(b);
}

tribool is_irrational(const Basic &b)
{
    RationalVisitor visitor(false);
    return visitor.apply(b);
}

void FiniteVisitor::error()
{
    throw SymEngineException(
        "Only numeric types allowed for is_finite/is_infinite");
}

void FiniteVisitor::bvisit(const Basic &x)
{
    is_finite_ = tribool::indeterminate;
}

void FiniteVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_finite_ = assumptions_->is_complex(x.rcp_from_this());
    } else {
        is_finite_ = tribool::indeterminate;
    }
}

void FiniteVisitor::bvisit(const Number &x)
{
    is_finite_ = tribool::tritrue;
}

void FiniteVisitor::bvisit(const Infty &x)
{
    is_finite_ = tribool::trifalse;
}

void FiniteVisitor::bvisit(const NaN &x)
{
    error();
}

void FiniteVisitor::bvisit(const Set &x)
{
    error();
}

void FiniteVisitor::bvisit(const Relational &x)
{
    error();
}

void FiniteVisitor::bvisit(const Boolean &x)
{
    error();
}

void FiniteVisitor::bvisit(const Constant &x)
{
    is_finite_ = tribool::tritrue;
}

tribool FiniteVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_finite_;
}

tribool is_finite(const Basic &b, const Assumptions *assumptions)
{
    FiniteVisitor visitor(assumptions);
    return visitor.apply(b);
}

tribool is_infinite(const Basic &b, const Assumptions *assumptions)
{
    FiniteVisitor visitor(assumptions);
    return not_tribool(visitor.apply(b));
}

tribool is_even(const Basic &b, const Assumptions *assumptions)
{
    return is_integer(*div(b.rcp_from_this(), integer(2)), assumptions);
}

tribool is_odd(const Basic &b, const Assumptions *assumptions)
{
    return is_integer(*div(add(b.rcp_from_this(), integer(1)), integer(2)),
                      assumptions);
}

void AlgebraicVisitor::error()
{
    throw SymEngineException(
        "Only numeric types allowed for is_algebraic/is_transcendental");
}

void AlgebraicVisitor::bvisit(const Basic &x)
{
    is_algebraic_ = tribool::indeterminate;
}

void AlgebraicVisitor::bvisit(const Set &x)
{
    error();
}

void AlgebraicVisitor::bvisit(const Relational &x)
{
    error();
}

void AlgebraicVisitor::bvisit(const Boolean &x)
{
    error();
}

void AlgebraicVisitor::bvisit(const Add &x)
{
    // algebraic + algebraic = algebraic
    // algebraic + transcendental = transcendental
    // algebraic + transcendental + transcendental = indeterminate
    tribool current = tribool::tritrue;
    for (const auto &arg : x.get_args()) {
        arg->accept(*this);
        if (is_false(current) and is_false(is_algebraic_)) {
            is_algebraic_ = tribool::indeterminate;
            return;
        }
        current = andwk_tribool(current, is_algebraic_);
        if (is_indeterminate(current)) {
            is_algebraic_ = current;
            return;
        }
    }
    is_algebraic_ = current;
}

void AlgebraicVisitor::bvisit(const Symbol &x)
{
    if (assumptions_) {
        is_algebraic_ = assumptions_->is_rational(x.rcp_from_this());
        if (is_false(is_algebraic_)) {
            is_algebraic_ = tribool::indeterminate;
        }
    } else {
        is_algebraic_ = tribool::indeterminate;
    }
}

void AlgebraicVisitor::bvisit(const Constant &x)
{
    if (eq(x, *pi) or eq(x, *E)) {
        is_algebraic_ = tribool::trifalse;
    } else if (eq(x, *GoldenRatio)) {
        is_algebraic_ = tribool::tritrue;
    } else {
        // It is unknown (2021) whether EulerGamma or Catalan are algebraic or
        // transcendental
        is_algebraic_ = tribool::indeterminate;
    }
}

void AlgebraicVisitor::bvisit(const Integer &x)
{
    is_algebraic_ = tribool::tritrue;
}

void AlgebraicVisitor::bvisit(const Rational &x)
{
    is_algebraic_ = tribool::tritrue;
}

void AlgebraicVisitor::trans_nonzero_and_algebraic(const Basic &b)
{
    // transcendental if b is algebraic and nonzero
    b.accept(*this);
    if (is_true(is_algebraic_) and is_true(is_nonzero(b))) {
        is_algebraic_ = tribool::trifalse;
    } else {
        is_algebraic_ = tribool::indeterminate;
    }
}

void AlgebraicVisitor::bvisit(const TrigFunction &x)
{
    // x algebraic and not 0 => sin(x) transcendental
    trans_nonzero_and_algebraic(*x.get_arg());
}

void AlgebraicVisitor::bvisit(const HyperbolicFunction &x)
{
    // x algebraic and not 0 => sinh(x) transcendental
    trans_nonzero_and_algebraic(*x.get_arg());
}

void AlgebraicVisitor::bvisit(const LambertW &x)
{
    // x algebraic and not 0 => W(x) transcendental
    trans_nonzero_and_algebraic(*x.get_arg());
}

tribool AlgebraicVisitor::apply(const Basic &b)
{
    b.accept(*this);
    return is_algebraic_;
}

tribool is_algebraic(const Basic &b, const Assumptions *assumptions)
{
    AlgebraicVisitor visitor(assumptions);
    return visitor.apply(b);
}

tribool is_transcendental(const Basic &b, const Assumptions *assumptions)
{
    AlgebraicVisitor visitor(assumptions);
    return not_tribool(visitor.apply(b));
}

} // namespace SymEngine
