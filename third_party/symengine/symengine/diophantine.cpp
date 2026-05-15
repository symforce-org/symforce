#include <symengine/diophantine.h>
#include <symengine/constants.h>
#include <symengine/add.h>
#include <symengine/mul.h>

namespace SymEngine
{

bool order(const DenseMatrix &t, const std::vector<DenseMatrix> &basis,
           unsigned k)
{
    bool eq = true;

    for (unsigned j = 0; j < t.ncols(); j++) {
        SYMENGINE_ASSERT(is_a<Integer>(*t.get(0, j)));
        integer_class t_
            = down_cast<const Integer &>(*t.get(0, j)).as_integer_class();

        SYMENGINE_ASSERT(is_a<Integer>(*basis[k].get(0, j)));
        integer_class b_ = down_cast<const Integer &>(*basis[k].get(0, j))
                               .as_integer_class();

        if (t_ < b_) {
            return false;
        }
        if (t_ > b_) {
            eq = false;
        }
    }

    return not eq;
}

bool is_minimum(const DenseMatrix &t, const std::vector<DenseMatrix> &basis,
                unsigned n)
{
    if (n == 0) {
        return true;
    }

    return not order(t, basis, n - 1) and is_minimum(t, basis, n - 1);
}

// Solve the diophantine system Ax = 0 and return a basis set for solutions
// Reference:
// Evelyne Contejean, Herve Devie. An Efficient Incremental Algorithm for
// Solving
// Systems of Linear Diophantine Equations. Information and computation,
// 113(1):143-172,
// August 1994.
void homogeneous_lde(std::vector<DenseMatrix> &basis, const DenseMatrix &A)
{
    unsigned p = A.nrows();
    unsigned q = A.ncols();

    SYMENGINE_ASSERT(p > 0 and q > 1);

    DenseMatrix row_zero(1, q);
    zeros(row_zero);

    DenseMatrix col_zero(p, 1);
    zeros(col_zero);

    std::vector<DenseMatrix> P;
    P.push_back(row_zero);

    std::vector<std::vector<bool>> Frozen(q, std::vector<bool>(q, true));
    for (unsigned j = 0; j < q; j++) {
        Frozen[0][j] = false;
    }

    std::vector<bool> F(q, false);

    DenseMatrix t, transpose, product, T;
    RCP<const Integer> dot;

    product = DenseMatrix(p, 1);
    transpose = DenseMatrix(q, 1);

    while (P.size() > 0) {
        auto n = P.size() - 1;
        t = P[n];
        P.pop_back();

        t.transpose(transpose);
        A.mul_matrix(transpose, product);

        if (product.eq(col_zero) and not t.eq(row_zero)) {
            basis.push_back(t);
        } else {
            for (unsigned i = 0; i < q; i++) {
                F[i] = Frozen[n][i];
            }

            T = t;
            for (unsigned i = 0; i < q; i++) {
                SYMENGINE_ASSERT(is_a<Integer>(*T.get(0, i)));
                T.set(0, i,
                      down_cast<const Integer &>(*T.get(0, i)).addint(*one));

                if (i > 0) {
                    SYMENGINE_ASSERT(is_a<Integer>(*T.get(0, i - 1)));
                    T.set(0, i - 1,
                          down_cast<const Integer &>(*T.get(0, i - 1))
                              .subint(*one));
                }

                dot = zero;
                for (unsigned j = 0; j < p; j++) {
                    SYMENGINE_ASSERT(is_a<Integer>(*product.get(j, 0)));
                    RCP<const Integer> p_j0
                        = rcp_static_cast<const Integer>(product.get(j, 0));

                    SYMENGINE_ASSERT(is_a<Integer>(*A.get(j, i)));
                    RCP<const Integer> A_ji
                        = rcp_static_cast<const Integer>(A.get(j, i));

                    dot = dot->addint(*p_j0->mulint(*A_ji));
                }

                if (F[i] == false
                    and ((dot->is_negative()
                          and is_minimum(T, basis,
                                         numeric_cast<unsigned>(basis.size())))
                         or t.eq(row_zero))) {
                    P.push_back(T);
                    n = n + 1;

                    for (unsigned j = 0; j < q; j++) {
                        Frozen[n - 1][j] = F[j];
                    }

                    F[i] = true;
                }
            }
        }
    }
}
} // namespace SymEngine
