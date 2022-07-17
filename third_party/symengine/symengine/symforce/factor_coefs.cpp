#include <symengine/symforce/factor_coefs.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/number.h>
#include <symengine/constants.h>
#include <symengine/visitor.h>

namespace SymEngine
{

    /**
     * Factors out leading coefficient if all coefficients have the same
     * magnitude. Returns leading coefficient.
     *
     * If not all coeficients have the same magnitude, returns a RCP of
     * a nullptr, and leaves the dict is a trashed state.
     */
    RCP<const Number> factor_leading_coef(add_operands_map& dict) {
        RCP<const Number> leading_coef;
        for (auto& term_coef : dict) {
            const RCP<const Number>& coef = term_coef.second;
            if (leading_coef.is_null()) {
                leading_coef = coef;
            }
            if (eq(*leading_coef, *coef)) {
                term_coef.second = one;
            } else if (eq(*leading_coef, *neg(coef))) {
                term_coef.second = minus_one;
            } else {
                return RCP<const Number>();
            }
        }
        return leading_coef;
    }

    RCP<const Basic> factor_add(const RCP<const Add>& a) {
        const RCP<const Number>& coef = a->get_coef();
        add_operands_map dict_copy = a->get_dict();
        const RCP<const Number> leading_coef = factor_leading_coef(dict_copy);
        if (leading_coef.is_null()) {
            return a;
        }
        // leading_coef because the coefs in dict can never be zero
        // This can probably be made more efficient by replacing mul with Mul
        // need to check to make sure it would be canonical though.
        // For example, would need to be carefu if leading_coef is 1.
        return mul(leading_coef, make_rcp<const Add>(coef->div(*leading_coef), std::move(dict_copy)));
    }


    class ConstFactorVisitor : public BaseVisitor<ConstFactorVisitor, TransformVisitor> {
    public:
        virtual RCP<const Basic> apply(const RCP<const Basic> &orig_expr) {
            // I can only assume this check is to save the trouble of recuring
            // through, as an atom's application will always be unchanged.
            if (is_a_Atom(*orig_expr)) {
                return orig_expr;
            }

            orig_expr->accept(*this);
            // now, result_ is the new_expr

            if (is_a<Add>(*result_)) {
                return factor_add(rcp_static_cast<const Add>(result_));
            }

            return result_;
        }
    };

    RCP<const Basic> factor_coefs(const RCP<const Basic>& b) {
        ConstFactorVisitor visitor;
        return visitor.apply(b);
    }

}

