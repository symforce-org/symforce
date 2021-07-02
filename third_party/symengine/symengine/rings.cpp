#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/rings.h>
#include <symengine/monomials.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{

void expr2poly(const RCP<const Basic> &p, umap_basic_num &syms, umap_vec_mpz &P)
{
    if (is_a<Add>(*p)) {
        auto n = syms.size();
        const umap_basic_num &d = down_cast<const Add &>(*p).get_dict();
        vec_int exp;
        integer_class coef;
        for (const auto &p : d) {
            if (not is_a<Integer>(*p.second))
                throw NotImplementedError("Not Implemented");
            coef = down_cast<const Integer &>(*p.second).as_integer_class();
            exp.assign(n, 0); // Initialize to [0]*n
            if (is_a<Mul>(*p.first)) {
                const map_basic_basic &term
                    = down_cast<const Mul &>(*p.first).get_dict();
                for (const auto &q : term) {
                    RCP<const Basic> sym = q.first;
                    if (not is_a<Integer>(*syms.at(sym)))
                        throw NotImplementedError("Not Implemented");
                    int i = numeric_cast<int>(
                        down_cast<const Integer &>(*syms.at(sym)).as_int());
                    if (is_a<Integer>(*q.second)) {
                        exp[i] = numeric_cast<int>(
                            down_cast<const Integer &>(*q.second).as_int());
                    } else {
                        throw SymEngineException(
                            "Cannot convert symbolic exponents to sparse "
                            "polynomials with integer exponents.");
                    }
                }
            } else if (is_a<Pow>(*p.first)) {
                RCP<const Basic> sym
                    = down_cast<const Pow &>(*p.first).get_base();
                RCP<const Basic> exp_
                    = down_cast<const Pow &>(*p.first).get_exp();
                if (not is_a<Integer>(*syms.at(sym)))
                    throw NotImplementedError("Not Implemented");
                int i = numeric_cast<int>(
                    down_cast<const Integer &>(*syms.at(sym)).as_int());
                if (not is_a<Integer>(*exp_))
                    throw NotImplementedError("Not Implemented");
                exp[i] = numeric_cast<int>(
                    down_cast<const Integer &>(*exp_).as_int());
            } else if (is_a<Symbol>(*p.first)) {
                RCP<const Basic> sym = p.first;
                if (not is_a<Integer>(*syms.at(sym)))
                    throw NotImplementedError("Not Implemented");
                int i = numeric_cast<int>(
                    down_cast<const Integer &>(*syms.at(sym)).as_int());
                exp[i] = 1;
            } else {
                throw NotImplementedError("Not Implemented");
            }

            P[exp] = coef;
        }
    } else {
        throw NotImplementedError("Not Implemented");
    }
}

void poly_mul(const umap_vec_mpz &A, const umap_vec_mpz &B, umap_vec_mpz &C)
{
    vec_int exp;
    auto n = A.begin()->first.size();
    exp.assign(n, 0); // Initialize to [0]*n
    /*
    std::cout << "A: " << A.load_factor() << " " << A.bucket_count() << " " <<
    A.size() << " "
        << A.max_bucket_count() << std::endl;
    std::cout << "B: " << B.load_factor() << " " << B.bucket_count() << " " <<
    B.size() << " "
        << B.max_bucket_count() << std::endl;
    std::cout << "C: " << C.load_factor() << " " << C.bucket_count() << " " <<
    C.size() << " "
        << C.max_bucket_count() << std::endl;
        */
    for (const auto &a : A) {
        for (const auto &b : B) {
            monomial_mul(a.first, b.first, exp);
            mp_addmul(C[exp], a.second, b.second);
        }
    }
    /*
    std::cout << "C: " << C.load_factor() << " " << C.bucket_count() << " " <<
    C.size() << " "
        << C.max_bucket_count() << std::endl;
    for (const std::size_t n=0; n < C.bucket_count(); n++) {
        std::cout << n << ": " << C.bucket_size(n) << "|";
        for (auto it = C.begin(n); it != C.end(n); ++it)
            std::cout << " " << it->first << myhash2(it->first) %
    C.bucket_count();
        std::cout << std::endl;
    }
    */
}

} // SymEngine
