#include <symengine/matrix.h>
#include <symengine/number.h>
#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/pow.h>
#include <symengine/subs.h>
#include <symengine/symengine_exception.h>
#include <symengine/polys/uexprpoly.h>
#include <symengine/solve.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

// Constructors
DenseMatrix::DenseMatrix() {}

DenseMatrix::DenseMatrix(unsigned row, unsigned col) : row_(row), col_(col)
{
    m_ = std::vector<RCP<const Basic>>(row * col);
}

DenseMatrix::DenseMatrix(unsigned row, unsigned col, const vec_basic &l)
    : m_{l}, row_(row), col_(col){SYMENGINE_ASSERT(m_.size() == row * col)}

      DenseMatrix::DenseMatrix(const vec_basic &column_elements)
    : m_(column_elements), row_(static_cast<unsigned>(column_elements.size())),
      col_(1)
{
}

// Resize the Matrix
void DenseMatrix::resize(unsigned row, unsigned col)
{
    row_ = row;
    col_ = col;
    m_.resize(row * col);
}

// Get and set elements
RCP<const Basic> DenseMatrix::get(unsigned i, unsigned j) const
{
    SYMENGINE_ASSERT(i < row_ and j < col_);
    return m_[i * col_ + j];
}

void DenseMatrix::set(unsigned i, unsigned j, const RCP<const Basic> &e)
{
    SYMENGINE_ASSERT(i < row_ and j < col_);
    m_[i * col_ + j] = e;
}

// Flat (row major) view of Matrix
vec_basic DenseMatrix::as_vec_basic() const
{
    return m_;
}

RCP<const Basic> DenseMatrix::trace() const
{
    SYMENGINE_ASSERT(row_ == col_);
    unsigned offset = 0;
    vec_basic diag;
    for (unsigned i = 0; i < row_; i++) {
        diag.push_back(m_[offset]);
        offset += row_ + 1;
    }
    auto sum = add(diag);
    return sum;
}

unsigned DenseMatrix::rank() const
{
    throw NotImplementedError("Not Implemented");
}

bool DenseMatrix::is_lower() const
{
    auto A = *this;
    unsigned n = A.nrows();
    for (unsigned i = 1; i < n; ++i) {
        for (unsigned j = 0; j < i; ++j) {
            if (not is_number_and_zero(*A.get(i, j))) {
                return false;
            }
        }
    }
    return true;
}

bool DenseMatrix::is_upper() const
{
    auto A = *this;
    unsigned n = A.nrows();
    for (unsigned i = 0; i < n - 1; ++i) {
        for (unsigned j = i + 1; j < n; ++j) {
            if (not is_number_and_zero(*A.get(i, j))) {
                return false;
            }
        }
    }
    return true;
}

tribool DenseMatrix::is_zero() const
{
    auto A = *this;
    tribool cur = tribool::tritrue;
    for (auto &e : m_) {
        cur = and_tribool(cur, SymEngine::is_zero(*e));
        if (is_false(cur)) {
            return cur;
        }
    }
    return cur;
}

tribool DenseMatrix::is_diagonal() const
{
    auto A = *this;
    if (not A.is_square()) {
        return tribool::trifalse;
    }
    unsigned ncols = A.ncols();
    unsigned offset;
    tribool cur = tribool::tritrue;
    for (unsigned i = 0; i < ncols; i++) {
        offset = i * ncols;
        for (unsigned j = 0; j < ncols; j++) {
            if (j != i) {
                auto &e = m_[offset];
                cur = and_tribool(cur, SymEngine::is_zero(*e));
                if (is_false(cur)) {
                    return cur;
                }
            }
            offset++;
        }
    }
    return cur;
}

tribool DenseMatrix::is_real(const Assumptions *assumptions) const
{
    RealVisitor visitor(assumptions);
    tribool cur = tribool::tritrue;
    for (auto &e : m_) {
        cur = and_tribool(cur, visitor.apply(*e));
        if (is_false(cur)) {
            return cur;
        }
    }
    return cur;
}

tribool DenseMatrix::is_symmetric() const
{
    auto A = *this;
    if (not A.is_square()) {
        return tribool::trifalse;
    }
    unsigned ncols = A.ncols();
    tribool cur = tribool::tritrue;
    for (unsigned i = 0; i < ncols; i++) {
        for (unsigned j = 0; j <= i; j++) {
            if (j != i) {
                auto &e = m_[i * ncols + j];
                auto &e2 = m_[j * ncols + i];
                cur = and_tribool(cur, SymEngine::is_zero(*sub(e, e2)));
            }
            if (is_false(cur)) {
                return cur;
            }
        }
    }
    return cur;
}

tribool DenseMatrix::is_hermitian() const
{
    auto A = *this;
    if (not A.is_square()) {
        return tribool::trifalse;
    }
    unsigned ncols = A.ncols();
    tribool cur = tribool::tritrue;
    for (unsigned i = 0; i < ncols; i++) {
        for (unsigned j = 0; j <= i; j++) {
            auto &e = m_[i * ncols + j];
            if (j != i) {
                auto &e2 = m_[j * ncols + i];
                cur = and_tribool(
                    cur, SymEngine::is_zero(*sub(e, SymEngine::conjugate(e2))));
            } else {
                cur = and_tribool(cur, SymEngine::is_real(*e));
            }
            if (is_false(cur)) {
                return cur;
            }
        }
    }
    return cur;
}

tribool DenseMatrix::is_weakly_diagonally_dominant() const
{
    auto A = *this;
    if (not A.is_square()) {
        return tribool::trifalse;
    }

    unsigned ncols = A.ncols();
    RCP<const Basic> diag;
    RCP<const Basic> sum;
    tribool diagdom = tribool::tritrue;
    for (unsigned i = 0; i < ncols; i++) {
        sum = zero;
        for (unsigned j = 0; j < ncols; j++) {
            auto &e = m_[i * ncols + j];
            if (i == j) {
                diag = abs(e);
            } else {
                sum = add(sum, abs(e));
            }
        }
        diagdom = and_tribool(diagdom, is_nonnegative(*sub(diag, sum)));
        if (is_false(diagdom)) {
            return diagdom;
        }
    }
    return diagdom;
}

tribool DenseMatrix::is_strictly_diagonally_dominant() const
{
    auto A = *this;
    if (not A.is_square()) {
        return tribool::trifalse;
    }

    unsigned ncols = A.ncols();
    RCP<const Basic> diag;
    RCP<const Basic> sum;
    tribool diagdom = tribool::tritrue;
    for (unsigned i = 0; i < ncols; i++) {
        sum = zero;
        for (unsigned j = 0; j < ncols; j++) {
            auto &e = m_[i * ncols + j];
            if (i == j) {
                diag = abs(e);
            } else {
                sum = add(sum, abs(e));
            }
        }
        diagdom = and_tribool(diagdom, is_positive(*sub(diag, sum)));
        if (is_false(diagdom)) {
            return diagdom;
        }
    }
    return diagdom;
}

tribool DenseMatrix::shortcut_to_posdef() const
{
    tribool is_diagonal_positive = tribool::tritrue;
    unsigned offset = 0;
    for (unsigned i = 0; i < row_; i++) {
        is_diagonal_positive
            = and_tribool(is_diagonal_positive, is_positive(*m_[offset]));
        if (is_false(is_diagonal_positive))
            return is_diagonal_positive;
        offset += row_ + 1;
    }
    if (is_true(and_tribool(is_diagonal_positive,
                            this->is_strictly_diagonally_dominant())))
        return tribool::tritrue;
    return tribool::indeterminate;
}

tribool DenseMatrix::is_positive_definite_GE()
{
    auto size = row_;
    for (unsigned i = 0; i < size; i++) {
        auto ispos = is_positive(*m_[i * size + i]);
        if (!is_true(ispos))
            return ispos;
        for (unsigned j = i + 1; j < size; j++) {
            for (unsigned k = i + 1; k < size; k++) {
                m_[j * size + k] = sub(mul(m_[i * size + i], m_[j * size + k]),
                                       mul(m_[j * size + i], m_[i * size + k]));
            }
        }
    }
    return tribool::tritrue;
}

tribool DenseMatrix::is_positive_definite() const
{

    auto A = *this;
    std::unique_ptr<DenseMatrix> B;
    const DenseMatrix *H;
    if (!is_true(A.is_hermitian())) {
        if (!A.is_square())
            return tribool::trifalse;
        DenseMatrix tmp1 = DenseMatrix(A.row_, A.col_);
        B = std::unique_ptr<DenseMatrix>(new DenseMatrix(A.row_, A.col_));
        A.conjugate_transpose(tmp1);
        add_dense_dense(A, tmp1, *B.get());
        H = B.get();
    } else {
        H = this;
    }

    tribool shortcut = H->shortcut_to_posdef();
    if (!is_indeterminate(shortcut))
        return shortcut;

    if (!B) {
        B = std::unique_ptr<DenseMatrix>(new DenseMatrix(A));
    }
    return B->is_positive_definite_GE();
}

tribool DenseMatrix::is_negative_definite() const
{
    auto res = DenseMatrix(row_, col_);
    mul_dense_scalar(*this, integer(-1), res);
    return res.is_positive_definite();
}

RCP<const Basic> DenseMatrix::det() const
{
    return det_bareis(*this);
}

void DenseMatrix::inv(MatrixBase &result) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        inverse_pivoted_LU(*this, r);
    }
}

void DenseMatrix::add_matrix(const MatrixBase &other, MatrixBase &result) const
{
    SYMENGINE_ASSERT(row_ == result.nrows() and col_ == result.ncols());

    if (is_a<DenseMatrix>(other) and is_a<DenseMatrix>(result)) {
        const DenseMatrix &o = down_cast<const DenseMatrix &>(other);
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        add_dense_dense(*this, o, r);
    }
}

void DenseMatrix::mul_matrix(const MatrixBase &other, MatrixBase &result) const
{
    SYMENGINE_ASSERT(row_ == result.nrows()
                     and other.ncols() == result.ncols());

    if (is_a<DenseMatrix>(other) and is_a<DenseMatrix>(result)) {
        const DenseMatrix &o = down_cast<const DenseMatrix &>(other);
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        mul_dense_dense(*this, o, r);
    }
}

void DenseMatrix::elementwise_mul_matrix(const MatrixBase &other,
                                         MatrixBase &result) const
{
    SYMENGINE_ASSERT(row_ == result.nrows() and col_ == result.ncols()
                     and row_ == other.nrows() and col_ == other.ncols());

    if (is_a<DenseMatrix>(other) and is_a<DenseMatrix>(result)) {
        const DenseMatrix &o = down_cast<const DenseMatrix &>(other);
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        elementwise_mul_dense_dense(*this, o, r);
    }
}

// Add a scalar
void DenseMatrix::add_scalar(const RCP<const Basic> &k,
                             MatrixBase &result) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        add_dense_scalar(*this, k, r);
    }
}

// Multiply by a scalar
void DenseMatrix::mul_scalar(const RCP<const Basic> &k,
                             MatrixBase &result) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        mul_dense_scalar(*this, k, r);
    }
}

// Matrix conjugate
void DenseMatrix::conjugate(MatrixBase &result) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        conjugate_dense(*this, r);
    }
}

// Matrix transpose
void DenseMatrix::transpose(MatrixBase &result) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        transpose_dense(*this, r);
    }
}

// Matrix conjugate transpose
void DenseMatrix::conjugate_transpose(MatrixBase &result) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        conjugate_transpose_dense(*this, r);
    }
}

// Extract out a submatrix
void DenseMatrix::submatrix(MatrixBase &result, unsigned row_start,
                            unsigned col_start, unsigned row_end,
                            unsigned col_end, unsigned row_step,
                            unsigned col_step) const
{
    if (is_a<DenseMatrix>(result)) {
        DenseMatrix &r = down_cast<DenseMatrix &>(result);
        submatrix_dense(*this, r, row_start, col_start, row_end, col_end,
                        row_step, col_step);
    }
}

// LU factorization
void DenseMatrix::LU(MatrixBase &L, MatrixBase &U) const
{
    if (is_a<DenseMatrix>(L) and is_a<DenseMatrix>(U)) {
        DenseMatrix &L_ = down_cast<DenseMatrix &>(L);
        DenseMatrix &U_ = down_cast<DenseMatrix &>(U);
        SymEngine::LU(*this, L_, U_);
    }
}

// LDL factorization
void DenseMatrix::LDL(MatrixBase &L, MatrixBase &D) const
{
    if (is_a<DenseMatrix>(L) and is_a<DenseMatrix>(D)) {
        DenseMatrix &L_ = down_cast<DenseMatrix &>(L);
        DenseMatrix &D_ = down_cast<DenseMatrix &>(D);
        SymEngine::LDL(*this, L_, D_);
    }
}

// Solve Ax = b using LU factorization
void DenseMatrix::LU_solve(const MatrixBase &b, MatrixBase &x) const
{
    if (is_a<DenseMatrix>(b) and is_a<DenseMatrix>(x)) {
        const DenseMatrix &b_ = down_cast<const DenseMatrix &>(b);
        DenseMatrix &x_ = down_cast<DenseMatrix &>(x);
        SymEngine::LU_solve(*this, b_, x_);
    }
}

// Fraction free LU factorization
void DenseMatrix::FFLU(MatrixBase &LU) const
{
    if (is_a<DenseMatrix>(LU)) {
        DenseMatrix &LU_ = down_cast<DenseMatrix &>(LU);
        fraction_free_LU(*this, LU_);
    }
}

// Fraction free LDU factorization
void DenseMatrix::FFLDU(MatrixBase &L, MatrixBase &D, MatrixBase &U) const
{
    if (is_a<DenseMatrix>(L) and is_a<DenseMatrix>(D)
        and is_a<DenseMatrix>(U)) {
        DenseMatrix &L_ = down_cast<DenseMatrix &>(L);
        DenseMatrix &D_ = down_cast<DenseMatrix &>(D);
        DenseMatrix &U_ = down_cast<DenseMatrix &>(U);
        fraction_free_LDU(*this, L_, D_, U_);
    }
}

// QR factorization
void DenseMatrix::QR(MatrixBase &Q, MatrixBase &R) const
{
    if (is_a<DenseMatrix>(Q) and is_a<DenseMatrix>(R)) {
        DenseMatrix &Q_ = down_cast<DenseMatrix &>(Q);
        DenseMatrix &R_ = down_cast<DenseMatrix &>(R);
        SymEngine::QR(*this, Q_, R_);
    }
}

// Cholesky decomposition
void DenseMatrix::cholesky(MatrixBase &L) const
{
    if (is_a<DenseMatrix>(L)) {
        DenseMatrix &L_ = down_cast<DenseMatrix &>(L);
        SymEngine::cholesky(*this, L_);
    }
}

// ---------------------------- Jacobian -------------------------------------//

void jacobian(const DenseMatrix &A, const DenseMatrix &x, DenseMatrix &result,
              bool diff_cache)
{
    SYMENGINE_ASSERT(A.col_ == 1);
    SYMENGINE_ASSERT(x.col_ == 1);
    SYMENGINE_ASSERT(A.row_ == result.nrows() and x.row_ == result.ncols());
    bool error = false;
#pragma omp parallel for
    for (unsigned i = 0; i < result.row_; i++) {
        for (unsigned j = 0; j < result.col_; j++) {
            if (is_a<Symbol>(*(x.m_[j]))) {
                const RCP<const Symbol> x_
                    = rcp_static_cast<const Symbol>(x.m_[j]);
                result.m_[i * result.col_ + j] = A.m_[i]->diff(x_, diff_cache);
            } else {
                error = true;
                break;
            }
        }
    }
    if (error) {
        throw SymEngineException(
            "'x' must contain Symbols only. "
            "Use sjacobian for SymPy style differentiation");
    }
}

void sjacobian(const DenseMatrix &A, const DenseMatrix &x, DenseMatrix &result,
               bool diff_cache)
{
    SYMENGINE_ASSERT(A.col_ == 1);
    SYMENGINE_ASSERT(x.col_ == 1);
    SYMENGINE_ASSERT(A.row_ == result.nrows() and x.row_ == result.ncols());
#pragma omp parallel for
    for (unsigned i = 0; i < result.row_; i++) {
        for (unsigned j = 0; j < result.col_; j++) {
            if (is_a<Symbol>(*(x.m_[j]))) {
                const RCP<const Symbol> x_
                    = rcp_static_cast<const Symbol>(x.m_[j]);
                result.m_[i * result.col_ + j] = A.m_[i]->diff(x_, diff_cache);
            } else {
                // TODO: Use a dummy symbol
                const RCP<const Symbol> x_ = symbol("x_");
                result.m_[i * result.col_ + j] = ssubs(
                    ssubs(A.m_[i], {{x.m_[j], x_}})->diff(x_, diff_cache),
                    {{x_, x.m_[j]}});
            }
        }
    }
}

// ---------------------------- Diff -------------------------------------//

void diff(const DenseMatrix &A, const RCP<const Symbol> &x, DenseMatrix &result,
          bool diff_cache)
{
    SYMENGINE_ASSERT(A.row_ == result.nrows() and A.col_ == result.ncols());
#pragma omp parallel for
    for (unsigned i = 0; i < result.row_; i++) {
        for (unsigned j = 0; j < result.col_; j++) {
            result.m_[i * result.col_ + j]
                = A.m_[i * result.col_ + j]->diff(x, diff_cache);
        }
    }
}

void sdiff(const DenseMatrix &A, const RCP<const Basic> &x, DenseMatrix &result,
           bool diff_cache)
{
    SYMENGINE_ASSERT(A.row_ == result.nrows() and A.col_ == result.ncols());
#pragma omp parallel for
    for (unsigned i = 0; i < result.row_; i++) {
        for (unsigned j = 0; j < result.col_; j++) {
            if (is_a<Symbol>(*x)) {
                const RCP<const Symbol> x_ = rcp_static_cast<const Symbol>(x);
                result.m_[i * result.col_ + j]
                    = A.m_[i * result.col_ + j]->diff(x_, diff_cache);
            } else {
                // TODO: Use a dummy symbol
                const RCP<const Symbol> x_ = symbol("_x");
                result.m_[i * result.col_ + j]
                    = ssubs(ssubs(A.m_[i * result.col_ + j], {{x, x_}})
                                ->diff(x_, diff_cache),
                            {{x_, x}});
            }
        }
    }
}

// ----------------------------- Matrix Conjugate ----------------------------//
void conjugate_dense(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(B.col_ == A.col_ and B.row_ == A.row_);

    for (unsigned i = 0; i < A.row_; i++)
        for (unsigned j = 0; j < A.col_; j++)
            B.m_[i * B.col_ + j] = conjugate(A.m_[i * A.col_ + j]);
}

// ----------------------------- Matrix Transpose ----------------------------//
void transpose_dense(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(B.row_ == A.col_ and B.col_ == A.row_);

    for (unsigned i = 0; i < A.row_; i++)
        for (unsigned j = 0; j < A.col_; j++)
            B.m_[j * B.col_ + i] = A.m_[i * A.col_ + j];
}

// ----------------------------- Matrix Conjugate Transpose -----------------//
void conjugate_transpose_dense(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(B.row_ == A.col_ and B.col_ == A.row_);

    for (unsigned i = 0; i < A.row_; i++)
        for (unsigned j = 0; j < A.col_; j++)
            B.m_[j * B.col_ + i] = conjugate(A.m_[i * A.col_ + j]);
}

// ------------------------------- Submatrix ---------------------------------//
void submatrix_dense(const DenseMatrix &A, DenseMatrix &B, unsigned row_start,
                     unsigned col_start, unsigned row_end, unsigned col_end,
                     unsigned row_step, unsigned col_step)
{
    SYMENGINE_ASSERT(row_end >= row_start and col_end >= col_start);
    SYMENGINE_ASSERT(row_end < A.row_);
    SYMENGINE_ASSERT(col_end < A.col_);
    SYMENGINE_ASSERT(B.row_ == row_end - row_start + 1
                     and B.col_ == col_end - col_start + 1);

    unsigned row = B.row_, col = B.col_;

    for (unsigned i = 0; i < row; i += row_step)
        for (unsigned j = 0; j < col; j += col_step)
            B.m_[i * col + j] = A.m_[(row_start + i) * A.col_ + col_start + j];
}

// ------------------------------- Matrix Addition ---------------------------//
void add_dense_dense(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_ and A.row_ == C.row_
                     and A.col_ == C.col_);

    unsigned row = A.row_, col = A.col_;

    for (unsigned i = 0; i < row; i++) {
        for (unsigned j = 0; j < col; j++) {
            C.m_[i * col + j] = add(A.m_[i * col + j], B.m_[i * col + j]);
        }
    }
}

void add_dense_scalar(const DenseMatrix &A, const RCP<const Basic> &k,
                      DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned row = A.row_, col = A.col_;

    for (unsigned i = 0; i < row; i++) {
        for (unsigned j = 0; j < col; j++) {
            B.m_[i * col + j] = add(A.m_[i * col + j], k);
        }
    }
}

// ------------------------------- Matrix Multiplication ---------------------//
void mul_dense_dense(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    SYMENGINE_ASSERT(A.col_ == B.row_ and C.row_ == A.row_
                     and C.col_ == B.col_);

    unsigned row = A.row_, col = B.col_;

    if (&A != &C and &B != &C) {
        for (unsigned r = 0; r < row; r++) {
            for (unsigned c = 0; c < col; c++) {
                C.m_[r * col + c] = zero; // Integer Zero
                for (unsigned k = 0; k < A.col_; k++)
                    C.m_[r * col + c]
                        = add(C.m_[r * col + c],
                              mul(A.m_[r * A.col_ + k], B.m_[k * col + c]));
            }
        }
    } else {
        DenseMatrix tmp = DenseMatrix(A.row_, B.col_);
        mul_dense_dense(A, B, tmp);
        C = tmp;
    }
}

void elementwise_mul_dense_dense(const DenseMatrix &A, const DenseMatrix &B,
                                 DenseMatrix &C)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_ and A.row_ == C.row_
                     and A.col_ == C.col_);

    unsigned row = A.row_, col = A.col_;

    for (unsigned i = 0; i < row; i++) {
        for (unsigned j = 0; j < col; j++) {
            C.m_[i * col + j] = mul(A.m_[i * col + j], B.m_[i * col + j]);
        }
    }
}

void mul_dense_scalar(const DenseMatrix &A, const RCP<const Basic> &k,
                      DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.col_ == B.col_ and A.row_ == B.row_);

    unsigned row = A.row_, col = A.col_;

    for (unsigned i = 0; i < row; i++) {
        for (unsigned j = 0; j < col; j++) {
            B.m_[i * col + j] = mul(A.m_[i * col + j], k);
        }
    }
}

// ---------------------------- Joining Operations ---------------------------//
void DenseMatrix::row_join(const DenseMatrix &B)
{
    this->col_insert(B, col_);
}

void DenseMatrix::col_join(const DenseMatrix &B)
{
    this->row_insert(B, row_);
}

void DenseMatrix::row_insert(const DenseMatrix &B, unsigned pos)
{
    SYMENGINE_ASSERT(col_ == B.col_ and pos <= row_)

    unsigned row = row_, col = col_;
    this->resize(row_ + B.row_, col_);

    for (unsigned i = row; i-- > pos;) {
        for (unsigned j = col; j-- > 0;) {
            this->m_[(i + B.row_) * col + j] = this->m_[i * col + j];
        }
    }

    for (unsigned i = 0; i < B.row_; i++) {
        for (unsigned j = 0; j < col; j++) {
            this->m_[(i + pos) * col + j] = B.m_[i * col + j];
        }
    }
}

void DenseMatrix::col_insert(const DenseMatrix &B, unsigned pos)
{
    SYMENGINE_ASSERT(row_ == B.row_ and pos <= col_)

    unsigned row = row_, col = col_;
    this->resize(row_, col_ + B.col_);

    for (unsigned i = row; i-- > 0;) {
        for (unsigned j = col; j-- > 0;) {
            if (j >= pos) {
                this->m_[i * (col + B.col_) + j + B.col_]
                    = this->m_[i * col + j];
            } else {
                this->m_[i * (col + B.col_) + j] = this->m_[i * col + j];
            }
        }
    }

    for (unsigned i = 0; i < row; i++) {
        for (unsigned j = 0; j < B.col_; j++) {
            this->m_[i * (col + B.col_) + j + pos] = B.m_[i * B.col_ + j];
        }
    }
}

void DenseMatrix::row_del(unsigned k)
{
    SYMENGINE_ASSERT(k < row_)

    if (row_ == 1)
        this->resize(0, 0);
    else {
        for (unsigned i = k; i < row_ - 1; i++) {
            row_exchange_dense(*this, i, i + 1);
        }
        this->resize(row_ - 1, col_);
    }
}

void DenseMatrix::col_del(unsigned k)
{
    SYMENGINE_ASSERT(k < col_)

    if (col_ == 1)
        this->resize(0, 0);
    else {
        unsigned row = row_, col = col_, m = 0;

        for (unsigned i = 0; i < row; i++) {
            for (unsigned j = 0; j < col; j++) {
                if (j != k) {
                    this->m_[m] = this->m_[i * col + j];
                    m++;
                }
            }
        }
        this->resize(row_, col_ - 1);
    }
}

// -------------------------------- Row Operations ---------------------------//
void row_exchange_dense(DenseMatrix &A, unsigned i, unsigned j)
{
    SYMENGINE_ASSERT(i != j and i < A.row_ and j < A.row_);

    unsigned col = A.col_;

    for (unsigned k = 0; k < A.col_; k++)
        std::swap(A.m_[i * col + k], A.m_[j * col + k]);
}

void row_mul_scalar_dense(DenseMatrix &A, unsigned i, RCP<const Basic> &c)
{
    SYMENGINE_ASSERT(i < A.row_);

    unsigned col = A.col_;

    for (unsigned j = 0; j < A.col_; j++)
        A.m_[i * col + j] = mul(c, A.m_[i * col + j]);
}

void row_add_row_dense(DenseMatrix &A, unsigned i, unsigned j,
                       RCP<const Basic> &c)
{
    SYMENGINE_ASSERT(i != j and i < A.row_ and j < A.row_);

    unsigned col = A.col_;

    for (unsigned k = 0; k < A.col_; k++)
        A.m_[i * col + k] = add(A.m_[i * col + k], mul(c, A.m_[j * col + k]));
}

void permuteFwd(DenseMatrix &A, permutelist &pl)
{
    for (auto &p : pl) {
        row_exchange_dense(A, p.first, p.second);
    }
}

// ----------------------------- Column Operations ---------------------------//
void column_exchange_dense(DenseMatrix &A, unsigned i, unsigned j)
{
    SYMENGINE_ASSERT(i != j and i < A.col_ and j < A.col_);

    unsigned col = A.col_;

    for (unsigned k = 0; k < A.row_; k++)
        std::swap(A.m_[k * col + i], A.m_[k * col + j]);
}

// ------------------------------ Gaussian Elimination -----------------------//
void pivoted_gaussian_elimination(const DenseMatrix &A, DenseMatrix &B,
                                  permutelist &pl)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned row = A.row_, col = A.col_;
    unsigned index = 0, i, j, k;
    B.m_ = A.m_;

    RCP<const Basic> scale;

    for (i = 0; i < col - 1; i++) {
        if (index == row)
            break;

        k = pivot(B, index, i);
        if (k == row)
            continue;
        if (k != index) {
            row_exchange_dense(B, k, index);
            pl.push_back({k, index});
        }

        scale = div(one, B.m_[index * col + i]);
        row_mul_scalar_dense(B, index, scale);

        for (j = i + 1; j < row; j++) {
            for (k = i + 1; k < col; k++)
                B.m_[j * col + k]
                    = sub(B.m_[j * col + k],
                          mul(B.m_[j * col + i], B.m_[i * col + k]));
            B.m_[j * col + i] = zero;
        }

        index++;
    }
}

// Algorithm 1, page 12, Nakos, G. C., Turner, P. R., Williams, R. M. (1997).
// Fraction-free algorithms for linear and polynomial equations.
// ACM SIGSAM Bulletin, 31(3), 11–19. doi:10.1145/271130.271133.
void fraction_free_gaussian_elimination(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned col = A.col_;
    B.m_ = A.m_;

    for (unsigned i = 0; i < col - 1; i++)
        for (unsigned j = i + 1; j < A.row_; j++) {
            for (unsigned k = i + 1; k < col; k++) {
                B.m_[j * col + k]
                    = sub(mul(B.m_[i * col + i], B.m_[j * col + k]),
                          mul(B.m_[j * col + i], B.m_[i * col + k]));
                if (i > 0)
                    B.m_[j * col + k]
                        = div(B.m_[j * col + k], B.m_[i * col - col + i - 1]);
            }
            B.m_[j * col + i] = zero;
        }
}

// Pivoted version of `fraction_free_gaussian_elimination`
void pivoted_fraction_free_gaussian_elimination(const DenseMatrix &A,
                                                DenseMatrix &B, permutelist &pl)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned col = A.col_, row = A.row_;
    unsigned index = 0, i, k, j;
    B.m_ = A.m_;

    for (i = 0; i < col - 1; i++) {
        if (index == row)
            break;

        k = pivot(B, index, i);
        if (k == row)
            continue;
        if (k != index) {
            row_exchange_dense(B, k, index);
            pl.push_back({k, index});
        }

        for (j = i + 1; j < row; j++) {
            for (k = i + 1; k < col; k++) {
                B.m_[j * col + k]
                    = sub(mul(B.m_[i * col + i], B.m_[j * col + k]),
                          mul(B.m_[j * col + i], B.m_[i * col + k]));
                if (i > 0)
                    B.m_[j * col + k]
                        = div(B.m_[j * col + k], B.m_[i * col - col + i - 1]);
            }
            B.m_[j * col + i] = zero;
        }

        index++;
    }
}

void pivoted_gauss_jordan_elimination(const DenseMatrix &A, DenseMatrix &B,
                                      permutelist &pl)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned row = A.row_, col = A.col_;
    unsigned index = 0, i, j, k;
    RCP<const Basic> scale;
    B.m_ = A.m_;

    for (i = 0; i < col; i++) {
        if (index == row)
            break;

        k = pivot(B, index, i);
        if (k == row)
            continue;
        if (k != index) {
            row_exchange_dense(B, k, index);
            pl.push_back({k, index});
        }

        scale = div(one, B.m_[index * col + i]);
        row_mul_scalar_dense(B, index, scale);

        for (j = 0; j < row; j++) {
            if (j == index)
                continue;

            scale = mul(minus_one, B.m_[j * col + i]);
            row_add_row_dense(B, j, index, scale);
        }

        index++;
    }
}

// Algorithm 2, page 13, Nakos, G. C., Turner, P. R., Williams, R. M. (1997).
// Fraction-free algorithms for linear and polynomial equations.
// ACM SIGSAM Bulletin, 31(3), 11–19. doi:10.1145/271130.271133.
void fraction_free_gauss_jordan_elimination(const DenseMatrix &A,
                                            DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned row = A.row_, col = A.col_;
    unsigned i, j, k;
    RCP<const Basic> d;

    B.m_ = A.m_;

    for (i = 0; i < col; i++) {
        if (i > 0)
            d = B.m_[i * col - col + i - 1];
        for (j = 0; j < row; j++)
            if (j != i)
                for (k = 0; k < col; k++) {
                    if (k != i) {
                        B.m_[j * col + k]
                            = sub(mul(B.m_[i * col + i], B.m_[j * col + k]),
                                  mul(B.m_[j * col + i], B.m_[i * col + k]));
                        if (i > 0)
                            B.m_[j * col + k] = div(B.m_[j * col + k], d);
                    }
                }
        for (j = 0; j < row; j++)
            if (j != i)
                B.m_[j * col + i] = zero;
    }
}

void pivoted_fraction_free_gauss_jordan_elimination(const DenseMatrix &A,
                                                    DenseMatrix &B,
                                                    permutelist &pl)
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_);

    unsigned row = A.row_, col = A.col_;
    unsigned index = 0, i, k, j;
    RCP<const Basic> d;

    B.m_ = A.m_;

    for (i = 0; i < col; i++) {
        if (index == row)
            break;

        k = pivot(B, index, i);
        if (k == row)
            continue;
        if (k != index) {
            row_exchange_dense(B, k, index);
            pl.push_back({k, index});
        }

        for (j = 0; j < row; j++) {
            if (j == index)
                continue;
            for (k = 0; k < col; k++) {
                if (k == i)
                    continue;
                B.m_[j * col + k]
                    = sub(mul(B.m_[index * col + i], B.m_[j * col + k]),
                          mul(B.m_[j * col + i], B.m_[index * col + k]));
                if (index == 0)
                    continue;
                B.m_[j * col + k] = div(B.m_[j * col + k], d);
            }
        }

        d = B.m_[index * col + i];

        for (j = 0; j < row; j++) {
            if (j == index)
                continue;
            B.m_[j * col + i] = zero;
        }
        index++;
    }
}

unsigned pivot(DenseMatrix &B, unsigned r, unsigned c)
{
    for (unsigned k = r; k < B.row_; k++) {
        if (!is_true(is_zero(*(B.m_[k * B.col_ + c])))) {
            return k;
        }
    }
    return B.row_;
}

void reduced_row_echelon_form(const DenseMatrix &A, DenseMatrix &b,
                              vec_uint &pivot_cols, bool normalize_last)
{
    permutelist pl;
    if (normalize_last) {
        pivoted_fraction_free_gauss_jordan_elimination(A, b, pl);
    } else {
        pivoted_gauss_jordan_elimination(A, b, pl);
    }
    unsigned row = 0;
    for (unsigned col = 0; col < b.col_ && row < b.row_; col++) {
        if (is_true(is_zero(*b.get(row, col))))
            continue;
        pivot_cols.push_back(col);
        if (row == 0 and normalize_last) {
            RCP<const Basic> m = div(one, b.get(row, col));
            b.mul_scalar(m, b);
        }
        row++;
    }
}

// --------------------------- Solve Ax = b  ---------------------------------//
// Assuming A is a diagonal square matrix
void diagonal_solve(const DenseMatrix &A, const DenseMatrix &b, DenseMatrix &x)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);
    SYMENGINE_ASSERT(x.row_ == A.col_ and x.col_ == b.col_);

    const unsigned sys = b.col_;

    // No checks are done to see if the diagonal entries are zero
    for (unsigned k = 0; k < sys; k++) {
        for (unsigned i = 0; i < A.col_; i++) {
            x.m_[i * sys + k] = div(b.m_[i * sys + k], A.m_[i * A.col_ + i]);
        }
    }
}

void back_substitution(const DenseMatrix &U, const DenseMatrix &b,
                       DenseMatrix &x)
{
    SYMENGINE_ASSERT(U.row_ == U.col_);
    SYMENGINE_ASSERT(b.row_ == U.row_);
    SYMENGINE_ASSERT(x.row_ == U.col_ and x.col_ == b.col_);

    const unsigned col = U.col_;
    const unsigned sys = b.col_;
    x.m_ = b.m_;

    for (unsigned k = 0; k < sys; k++) {
        for (int i = col - 1; i >= 0; i--) {
            for (unsigned j = i + 1; j < col; j++)
                x.m_[i * sys + k]
                    = sub(x.m_[i * sys + k],
                          mul(U.m_[i * col + j], x.m_[j * sys + k]));
            x.m_[i * sys + k] = div(x.m_[i * sys + k], U.m_[i * col + i]);
        }
    }
}

void forward_substitution(const DenseMatrix &A, const DenseMatrix &b,
                          DenseMatrix &x)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);
    SYMENGINE_ASSERT(b.row_ == A.row_);
    SYMENGINE_ASSERT(x.row_ == A.col_ and x.col_ == b.col_);

    unsigned col = A.col_;
    const unsigned sys = b.col_;
    x.m_ = b.m_;

    for (unsigned k = 0; k < b.col_; k++) {
        for (unsigned i = 0; i < col - 1; i++) {
            for (unsigned j = i + 1; j < col; j++) {
                x.m_[j * sys + k]
                    = sub(mul(A.m_[i * col + i], x.m_[j * sys + k]),
                          mul(A.m_[j * col + i], x.m_[i * sys + k]));
                if (i > 0)
                    x.m_[j * sys + k]
                        = div(x.m_[j * sys + k], A.m_[(i - 1) * col + i - 1]);
            }
        }
    }
}

void fraction_free_gaussian_elimination_solve(const DenseMatrix &A,
                                              const DenseMatrix &b,
                                              DenseMatrix &x)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);
    SYMENGINE_ASSERT(b.row_ == A.row_ and x.row_ == A.row_);
    SYMENGINE_ASSERT(x.col_ == b.col_);

    int i, j, k, col = A.col_, bcol = b.col_;
    DenseMatrix A_ = DenseMatrix(A.row_, A.col_, A.m_);
    DenseMatrix b_ = DenseMatrix(b.row_, b.col_, b.m_);

    for (i = 0; i < col - 1; i++)
        for (j = i + 1; j < col; j++) {
            for (k = 0; k < bcol; k++) {
                b_.m_[j * bcol + k]
                    = sub(mul(A_.m_[i * col + i], b_.m_[j * bcol + k]),
                          mul(A_.m_[j * col + i], b_.m_[i * bcol + k]));
                if (i > 0)
                    b_.m_[j * bcol + k] = div(b_.m_[j * bcol + k],
                                              A_.m_[i * col - col + i - 1]);
            }

            for (k = i + 1; k < col; k++) {
                A_.m_[j * col + k]
                    = sub(mul(A_.m_[i * col + i], A_.m_[j * col + k]),
                          mul(A_.m_[j * col + i], A_.m_[i * col + k]));
                if (i > 0)
                    A_.m_[j * col + k]
                        = div(A_.m_[j * col + k], A_.m_[i * col - col + i - 1]);
            }
            A_.m_[j * col + i] = zero;
        }

    for (i = 0; i < col * bcol; i++)
        x.m_[i] = zero; // Integer zero;

    for (k = 0; k < bcol; k++) {
        for (i = col - 1; i >= 0; i--) {
            for (j = i + 1; j < col; j++)
                b_.m_[i * bcol + k]
                    = sub(b_.m_[i * bcol + k],
                          mul(A_.m_[i * col + j], x.m_[j * bcol + k]));
            x.m_[i * bcol + k] = div(b_.m_[i * bcol + k], A_.m_[i * col + i]);
        }
    }
}

void fraction_free_gauss_jordan_solve(const DenseMatrix &A,
                                      const DenseMatrix &b, DenseMatrix &x,
                                      bool pivot)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);
    SYMENGINE_ASSERT(b.row_ == A.row_ and x.row_ == A.row_);
    SYMENGINE_ASSERT(x.col_ == b.col_);

    unsigned i, j, k, p, col = A.col_, bcol = b.col_;
    RCP<const Basic> d;
    DenseMatrix A_ = DenseMatrix(A.row_, A.col_, A.m_);
    DenseMatrix b_ = DenseMatrix(b.row_, b.col_, b.m_);

    for (i = 0; i < col; i++) {
        if (i > 0)
            d = A_.m_[i * col - col + i - 1];
        if (pivot) {
            p = i;
            while (p < col and eq(*A_.m_[p * col + i], *zero)) {
                p++;
            }
            SYMENGINE_ASSERT(p != col);
            if (p != i) {
                // pivot A
                for (k = i; k < col; k++) {
                    std::swap(A_.m_[p * col + k], A_.m_[i * col + k]);
                }
                for (k = 0; k < bcol; k++) {
                    std::swap(b_.m_[p * bcol + k], b_.m_[i * bcol + k]);
                }
            }
        }
        for (j = 0; j < col; j++)
            if (j != i) {
                for (k = 0; k < bcol; k++) {
                    b_.m_[j * bcol + k]
                        = sub(mul(A_.m_[i * col + i], b_.m_[j * bcol + k]),
                              mul(A_.m_[j * col + i], b_.m_[i * bcol + k]));
                    if (i > 0)
                        b_.m_[j * bcol + k] = div(b_.m_[j * bcol + k], d);
                }

                for (k = 0; k < col; k++) {
                    if (k != i) {
                        A_.m_[j * col + k]
                            = sub(mul(A_.m_[i * col + i], A_.m_[j * col + k]),
                                  mul(A_.m_[j * col + i], A_.m_[i * col + k]));
                        if (i > 0)
                            A_.m_[j * col + k] = div(A_.m_[j * col + k], d);
                    }
                }
            }

        for (j = 0; j < col; j++)
            if (j != i)
                A_.m_[j * col + i] = zero;
    }

    // No checks are done to see if the diagonal entries are zero
    for (k = 0; k < bcol; k++)
        for (i = 0; i < col; i++)
            x.m_[i * bcol + k] = div(b_.m_[i * bcol + k], A_.m_[i * col + i]);
}

void fraction_free_LU_solve(const DenseMatrix &A, const DenseMatrix &b,
                            DenseMatrix &x)
{
    DenseMatrix LU = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix x_ = DenseMatrix(b.nrows(), b.ncols());

    fraction_free_LU(A, LU);
    forward_substitution(LU, b, x_);
    back_substitution(LU, x_, x);
}

void LU_solve(const DenseMatrix &A, const DenseMatrix &b, DenseMatrix &x)
{
    DenseMatrix L = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix U = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix x_ = DenseMatrix(b.nrows(), b.ncols());

    LU(A, L, U);
    forward_substitution(L, b, x_);
    back_substitution(U, x_, x);
}

void pivoted_LU_solve(const DenseMatrix &A, const DenseMatrix &b,
                      DenseMatrix &x)
{
    DenseMatrix L = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix U = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix x_ = DenseMatrix(b);
    permutelist pl;

    pivoted_LU(A, L, U, pl);
    permuteFwd(x_, pl);
    forward_substitution(L, x_, x_);
    back_substitution(U, x_, x);
}

void LDL_solve(const DenseMatrix &A, const DenseMatrix &b, DenseMatrix &x)
{
    DenseMatrix L = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix D = DenseMatrix(A.nrows(), A.ncols());
    DenseMatrix x_ = DenseMatrix(b.nrows(), b.ncols());

    if (not is_symmetric_dense(A))
        throw SymEngineException("Matrix must be symmetric");

    LDL(A, L, D);
    forward_substitution(L, b, x);
    diagonal_solve(D, x, x_);
    transpose_dense(L, D);
    back_substitution(D, x_, x);
}

// --------------------------- Matrix Decomposition --------------------------//

// Algorithm 3, page 14, Nakos, G. C., Turner, P. R., Williams, R. M. (1997).
// Fraction-free algorithms for linear and polynomial equations.
// ACM SIGSAM Bulletin, 31(3), 11–19. doi:10.1145/271130.271133.
// This algorithms is not a true factorization of the matrix A(i.e. A != LU))
// but can be used to solve linear systems by applying forward and backward
// substitutions respectively.
void fraction_free_LU(const DenseMatrix &A, DenseMatrix &LU)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and LU.row_ == LU.col_
                     and A.row_ == LU.row_);

    unsigned n = A.row_;
    unsigned i, j, k;

    LU.m_ = A.m_;

    for (i = 0; i < n - 1; i++)
        for (j = i + 1; j < n; j++)
            for (k = i + 1; k < n; k++) {
                LU.m_[j * n + k] = sub(mul(LU.m_[i * n + i], LU.m_[j * n + k]),
                                       mul(LU.m_[j * n + i], LU.m_[i * n + k]));
                if (i)
                    LU.m_[j * n + k]
                        = div(LU.m_[j * n + k], LU.m_[i * n - n + i - 1]);
            }
}

// SymPy LUDecomposition algorithm, in
// sympy.matrices.matrices.Matrix.LUdecomposition
// with no pivoting
void LU(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &U)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and L.row_ == L.col_
                     and U.row_ == U.col_);
    SYMENGINE_ASSERT(A.row_ == L.row_ and A.row_ == U.row_);

    unsigned n = A.row_;
    unsigned i, j, k;
    RCP<const Basic> scale;

    U.m_ = A.m_;

    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++)
            for (k = 0; k < i; k++)
                U.m_[i * n + j] = sub(U.m_[i * n + j],
                                      mul(U.m_[i * n + k], U.m_[k * n + j]));

        for (i = j; i < n; i++) {
            for (k = 0; k < j; k++)
                U.m_[i * n + j] = sub(U.m_[i * n + j],
                                      mul(U.m_[i * n + k], U.m_[k * n + j]));
        }

        scale = div(one, U.m_[j * n + j]);

        for (i = j + 1; i < n; i++)
            U.m_[i * n + j] = mul(U.m_[i * n + j], scale);
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            L.m_[i * n + j] = U.m_[i * n + j];
            U.m_[i * n + j] = zero; // Integer zero
        }
        L.m_[i * n + i] = one; // Integer one
        for (j = i + 1; j < n; j++)
            L.m_[i * n + j] = zero; // Integer zero
    }
}

// SymPy LUDecomposition_Simple algorithm, in
// sympy.matrices.matrices.Matrix.LUdecomposition_Simple with pivoting.
// P must be an initialized matrix and will be permuted.
void pivoted_LU(const DenseMatrix &A, DenseMatrix &LU, permutelist &pl)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and LU.row_ == LU.col_);
    SYMENGINE_ASSERT(A.row_ == LU.row_);

    unsigned n = A.row_;
    unsigned i, j, k;
    RCP<const Basic> scale;
    int pivot;

    LU.m_ = A.m_;

    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++)
            for (k = 0; k < i; k++)
                LU.m_[i * n + j] = sub(LU.m_[i * n + j],
                                       mul(LU.m_[i * n + k], LU.m_[k * n + j]));
        pivot = -1;
        for (i = j; i < n; i++) {
            for (k = 0; k < j; k++) {
                LU.m_[i * n + j] = sub(LU.m_[i * n + j],
                                       mul(LU.m_[i * n + k], LU.m_[k * n + j]));
            }
            if (pivot == -1 and !is_true(is_zero(*LU.m_[i * n + j])))
                pivot = i;
        }
        if (pivot == -1)
            throw SymEngineException("Matrix is rank deficient");
        if (pivot - j != 0) { // row must be swapped
            row_exchange_dense(LU, pivot, j);
            pl.push_back({pivot, j});
        }
        scale = div(one, LU.m_[j * n + j]);

        for (i = j + 1; i < n; i++)
            LU.m_[i * n + j] = mul(LU.m_[i * n + j], scale);
    }
}

// SymPy LUDecomposition algorithm, in
// sympy.matrices.matrices.Matrix.LUdecomposition_Simple with pivoting.
// P must be an initialized matrix and will be permuted.
void pivoted_LU(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &U,
                permutelist &pl)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and L.row_ == L.col_
                     and U.row_ == U.col_);
    SYMENGINE_ASSERT(A.row_ == L.row_ and A.row_ == U.row_);

    pivoted_LU(A, U, pl);
    unsigned n = A.col_;
    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < i; j++) {
            L.m_[i * n + j] = U.m_[i * n + j];
            U.m_[i * n + j] = zero; // Integer zero
        }
        L.m_[i * n + i] = one; // Integer one
        for (unsigned j = i + 1; j < n; j++)
            L.m_[i * n + j] = zero; // Integer zero
    }
}

// SymPy's fraction free LU decomposition, without pivoting
// sympy.matrices.matrices.MatrixBase.LUdecompositionFF
// W. Zhou & D.J. Jeffrey, "Fraction-free matrix factors: new forms for LU and
// QR factors".
// Frontiers in Computer Science in China, Vol 2, no. 1, pp. 67-80, 2008.
void fraction_free_LDU(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &D,
                       DenseMatrix &U)
{
    SYMENGINE_ASSERT(A.row_ == L.row_ and A.row_ == U.row_);
    SYMENGINE_ASSERT(A.col_ == L.col_ and A.col_ == U.col_);

    unsigned row = A.row_, col = A.col_;
    unsigned i, j, k;
    RCP<const Basic> old = integer(1);

    U.m_ = A.m_;

    // Initialize L
    for (i = 0; i < row; i++)
        for (j = 0; j < row; j++)
            if (i != j)
                L.m_[i * col + j] = zero;
            else
                L.m_[i * col + i] = one;

    // Initialize D
    for (i = 0; i < row * row; i++)
        D.m_[i] = zero; // Integer zero

    for (k = 0; k < row - 1; k++) {
        L.m_[k * col + k] = U.m_[k * col + k];
        D.m_[k * col + k] = mul(old, U.m_[k * col + k]);

        for (i = k + 1; i < row; i++) {
            L.m_[i * col + k] = U.m_[i * col + k];
            for (j = k + 1; j < col; j++)
                U.m_[i * col + j]
                    = div(sub(mul(U.m_[k * col + k], U.m_[i * col + j]),
                              mul(U.m_[k * col + j], U.m_[i * col + k])),
                          old);
            U.m_[i * col + k] = zero; // Integer zero
        }

        old = U.m_[k * col + k];
    }

    D.m_[row * col - col + row - 1] = old;
}

// SymPy's QRecomposition in sympy.matrices.matrices.MatrixBase.QRdecomposition
// Rank check is not performed
void QR(const DenseMatrix &A, DenseMatrix &Q, DenseMatrix &R)
{
    unsigned row = A.row_;
    unsigned col = A.col_;

    SYMENGINE_ASSERT(Q.row_ == row and Q.col_ == col and R.row_ == col
                     and R.col_ == col);

    unsigned i, j, k;
    RCP<const Basic> t;
    std::vector<RCP<const Basic>> tmp(row);

    // Initialize Q
    for (i = 0; i < row * col; i++)
        Q.m_[i] = zero;

    // Initialize R
    for (i = 0; i < col * col; i++)
        R.m_[i] = zero;

    for (j = 0; j < col; j++) {
        // Use submatrix for this
        for (k = 0; k < row; k++)
            tmp[k] = A.m_[k * col + j];

        for (i = 0; i < j; i++) {
            t = zero;
            for (k = 0; k < row; k++)
                t = add(t, mul(A.m_[k * col + j], Q.m_[k * col + i]));
            for (k = 0; k < row; k++)
                tmp[k] = expand(sub(tmp[k], mul(Q.m_[k * col + i], t)));
        }

        // calculate norm
        t = zero;
        for (k = 0; k < row; k++)
            t = add(t, pow(tmp[k], integer(2)));

        t = pow(t, div(one, integer(2)));

        R.m_[j * col + j] = t;
        for (k = 0; k < row; k++)
            Q.m_[k * col + j] = div(tmp[k], t);

        for (i = 0; i < j; i++) {
            t = zero;
            for (k = 0; k < row; k++)
                t = add(t, mul(Q.m_[k * col + i], A.m_[k * col + j]));
            R.m_[i * col + j] = t;
        }
    }
}

// SymPy's LDL decomposition, Assuming A is a symmetric, square, positive
// definite non singular matrix
void LDL(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &D)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);
    SYMENGINE_ASSERT(L.row_ == A.row_ and L.col_ == A.row_);
    SYMENGINE_ASSERT(D.row_ == A.row_ and D.col_ == A.row_);

    unsigned col = A.col_;
    unsigned i, k, j;
    RCP<const Basic> sum;
    RCP<const Basic> i2 = integer(2);

    // Initialize D
    for (i = 0; i < col; i++)
        for (j = 0; j < col; j++)
            D.m_[i * col + j] = zero; // Integer zero

    // Initialize L
    for (i = 0; i < col; i++)
        for (j = 0; j < col; j++)
            L.m_[i * col + j] = (i != j) ? zero : one;

    for (i = 0; i < col; i++) {
        for (j = 0; j < i; j++) {
            sum = zero;
            for (k = 0; k < j; k++)
                sum = add(sum, mul(mul(L.m_[i * col + k], L.m_[j * col + k]),
                                   D.m_[k * col + k]));
            L.m_[i * col + j]
                = mul(div(one, D.m_[j * col + j]), sub(A.m_[i * col + j], sum));
        }
        sum = zero;
        for (k = 0; k < i; k++)
            sum = add(sum, mul(pow(L.m_[i * col + k], i2), D.m_[k * col + k]));
        D.m_[i * col + i] = sub(A.m_[i * col + i], sum);
    }
}

// SymPy's cholesky decomposition
void cholesky(const DenseMatrix &A, DenseMatrix &L)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);
    SYMENGINE_ASSERT(L.row_ == A.row_ and L.col_ == A.row_);

    unsigned col = A.col_;
    unsigned i, j, k;
    RCP<const Basic> sum;
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> half = div(one, i2);

    // Initialize L
    for (i = 0; i < col; i++)
        for (j = 0; j < col; j++)
            L.m_[i * col + j] = zero;

    for (i = 0; i < col; i++) {
        for (j = 0; j < i; j++) {
            sum = zero;
            for (k = 0; k < j; k++)
                sum = add(sum, mul(L.m_[i * col + k], L.m_[j * col + k]));

            L.m_[i * col + j]
                = mul(div(one, L.m_[j * col + j]), sub(A.m_[i * col + j], sum));
        }
        sum = zero;
        for (k = 0; k < i; k++)
            sum = add(sum, pow(L.m_[i * col + k], i2));
        L.m_[i * col + i] = pow(sub(A.m_[i * col + i], sum), half);
    }
}

// Matrix Queries
bool is_symmetric_dense(const DenseMatrix &A)
{
    if (A.col_ != A.row_)
        return false;

    unsigned col = A.col_;
    bool sym = true;

    for (unsigned i = 0; i < col; i++)
        for (unsigned j = i + 1; j < col; j++)
            if (not eq(*(A.m_[j * col + i]), *(A.m_[i * col + j]))) {
                sym = false;
                break;
            }

    return sym;
}

// ----------------------------- Determinant ---------------------------------//
RCP<const Basic> det_bareis(const DenseMatrix &A)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);

    unsigned n = A.row_;

    if (n == 1) {
        return A.m_[0];
    } else if (n == 2) {
        // If A = [[a, b], [c, d]] then det(A) = ad - bc
        return sub(mul(A.m_[0], A.m_[3]), mul(A.m_[1], A.m_[2]));
    } else if (n == 3) {
        // if A = [[a, b, c], [d, e, f], [g, h, i]] then
        // det(A) = (aei + bfg + cdh) - (ceg + bdi + afh)
        return sub(add(add(mul(mul(A.m_[0], A.m_[4]), A.m_[8]),
                           mul(mul(A.m_[1], A.m_[5]), A.m_[6])),
                       mul(mul(A.m_[2], A.m_[3]), A.m_[7])),
                   add(add(mul(mul(A.m_[2], A.m_[4]), A.m_[6]),
                           mul(mul(A.m_[1], A.m_[3]), A.m_[8])),
                       mul(mul(A.m_[0], A.m_[5]), A.m_[7])));
    } else {

        if (A.is_lower() or A.is_upper()) {
            RCP<const Basic> det = A.m_[0];
            for (unsigned i = 1; i < n; ++i) {
                det = mul(det, A.m_[i * n + i]);
            }
            return det;
        }

        DenseMatrix B = DenseMatrix(n, n, A.m_);
        unsigned i, sign = 1;
        RCP<const Basic> d;

        for (unsigned k = 0; k < n - 1; k++) {
            if (is_true(is_zero(*B.m_[k * n + k]))) {
                for (i = k + 1; i < n; i++)
                    if (!is_true(is_zero(*B.m_[i * n + k]))) {
                        row_exchange_dense(B, i, k);
                        sign *= -1;
                        break;
                    }
                if (i == n)
                    return zero;
            }

            for (i = k + 1; i < n; i++) {
                for (unsigned j = k + 1; j < n; j++) {
                    d = sub(mul(B.m_[k * n + k], B.m_[i * n + j]),
                            mul(B.m_[i * n + k], B.m_[k * n + j]));
                    if (k > 0)
                        d = div(d, B.m_[(k - 1) * n + k - 1]);
                    B.m_[i * n + j] = d;
                }
            }
        }

        return (sign == 1) ? B.m_[n * n - 1] : mul(minus_one, B.m_[n * n - 1]);
    }
}

// Returns the coefficients of characterisitc polynomials of leading principal
// minors of Matrix A as elements in `polys`. Principal leading minor of kth
// order is the submatrix of A obtained by deleting last n-k rows and columns
// from A. Here `n` is the dimension of the square matrix A.
void berkowitz(const DenseMatrix &A, std::vector<DenseMatrix> &polys)
{
    SYMENGINE_ASSERT(A.row_ == A.col_);

    unsigned col = A.col_;
    unsigned i, k, l, m;

    std::vector<DenseMatrix> items;
    std::vector<DenseMatrix> transforms;
    std::vector<RCP<const Basic>> items_;

    for (unsigned n = col; n > 1; n--) {
        items.clear();
        k = n - 1;
        DenseMatrix T = DenseMatrix(n + 1, n);
        DenseMatrix C = DenseMatrix(k, 1);

        // Initialize T and C
        for (i = 0; i < n * (n + 1); i++)
            T.m_[i] = zero;
        for (i = 0; i < k; i++)
            C.m_[i] = A.m_[i * col + k];
        items.push_back(C);

        for (i = 0; i < n - 2; i++) {
            DenseMatrix B = DenseMatrix(k, 1);
            for (l = 0; l < k; l++) {
                B.m_[l] = zero;
                for (m = 0; m < k; m++)
                    B.m_[l]
                        = add(B.m_[l], mul(A.m_[l * col + m], items[i].m_[m]));
            }
            items.push_back(B);
        }

        items_.clear();
        for (i = 0; i < n - 1; i++) {
            RCP<const Basic> element = zero;
            for (l = 0; l < k; l++)
                element = add(element, mul(A.m_[k * col + l], items[i].m_[l]));
            items_.push_back(mul(minus_one, element));
        }
        items_.insert(items_.begin(), mul(minus_one, A.m_[k * col + k]));
        items_.insert(items_.begin(), one);

        for (i = 0; i < n; i++) {
            for (l = 0; l < n - i + 1; l++)
                T.m_[(i + l) * n + i] = items_[l];
        }

        transforms.push_back(T);
    }

    polys.push_back(DenseMatrix(2, 1, {one, mul(A.m_[0], minus_one)}));

    for (i = 0; i < col - 1; i++) {
        unsigned t_row = transforms[col - 2 - i].nrows();
        unsigned t_col = transforms[col - 2 - i].ncols();
        DenseMatrix B = DenseMatrix(t_row, 1);

        for (l = 0; l < t_row; l++) {
            B.m_[l] = zero;
            for (m = 0; m < t_col; m++) {
                B.m_[l] = add(B.m_[l],
                              mul(transforms[col - 2 - i].m_[l * t_col + m],
                                  polys[i].m_[m]));
                B.m_[l] = expand(B.m_[l]);
            }
        }
        polys.push_back(B);
    }
}

RCP<const Basic> det_berkowitz(const DenseMatrix &A)
{
    std::vector<DenseMatrix> polys;

    berkowitz(A, polys);
    DenseMatrix poly = polys[polys.size() - 1];

    if (polys.size() % 2 == 1)
        return mul(minus_one, poly.get(poly.nrows() - 1, 0));

    return poly.get(poly.nrows() - 1, 0);
}

void char_poly(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(B.ncols() == 1 and B.nrows() == A.nrows() + 1);
    SYMENGINE_ASSERT(A.nrows() == A.ncols());

    std::vector<DenseMatrix> polys;

    berkowitz(A, polys);
    B = polys[polys.size() - 1];
}

void inverse_fraction_free_LU(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and B.row_ == B.col_
                     and B.row_ == A.row_);

    unsigned n = A.row_, i;
    DenseMatrix LU = DenseMatrix(n, n);
    DenseMatrix e = DenseMatrix(n, 1);
    DenseMatrix x = DenseMatrix(n, 1);
    DenseMatrix x_ = DenseMatrix(n, 1);

    // Initialize matrices
    for (i = 0; i < n * n; i++) {
        LU.m_[i] = zero;
        B.m_[i] = zero;
    }
    for (i = 0; i < n; i++) {
        e.m_[i] = zero;
        x.m_[i] = zero;
        x_.m_[i] = zero;
    }

    fraction_free_LU(A, LU);

    // We solve AX_{i} = e_{i} for i = 1, 2, .. n and combine the row vectors
    // X_{1}, X_{2}, ... X_{n} to form the inverse of A. Here, e_{i}'s are the
    // elements of the standard basis.
    for (unsigned j = 0; j < n; j++) {
        e.m_[j] = one;

        forward_substitution(LU, e, x_);
        back_substitution(LU, x_, x);

        for (i = 0; i < n; i++)
            B.m_[i * n + j] = x.m_[i];

        e.m_[j] = zero;
    }
}

void inverse_LU(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and B.row_ == B.col_
                     and B.row_ == A.row_);
    DenseMatrix e = DenseMatrix(A.row_, A.col_);
    eye(e);
    LU_solve(A, e, B);
}

void inverse_pivoted_LU(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and B.row_ == B.col_
                     and B.row_ == A.row_);
    DenseMatrix e = DenseMatrix(A.row_, A.col_);
    eye(e);
    pivoted_LU_solve(A, e, B);
}

void inverse_gauss_jordan(const DenseMatrix &A, DenseMatrix &B)
{
    SYMENGINE_ASSERT(A.row_ == A.col_ and B.row_ == B.col_
                     and B.row_ == A.row_);

    unsigned n = A.row_;
    DenseMatrix e = DenseMatrix(n, n);

    // Initialize matrices
    for (unsigned i = 0; i < n; i++)
        for (unsigned j = 0; j < n; j++) {
            if (i != j) {
                e.m_[i * n + j] = zero;
            } else {
                e.m_[i * n + i] = one;
            }
            B.m_[i * n + j] = zero;
        }

    fraction_free_gauss_jordan_solve(A, e, B);
}

// ----------------------- Vector-specific Methods --------------------------//

void dot(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    if (A.col_ == B.row_) {
        if (B.col_ != 1) {
            DenseMatrix tmp1 = DenseMatrix(A.col_, A.row_);
            A.transpose(tmp1);
            DenseMatrix tmp2 = DenseMatrix(B.col_, B.row_);
            B.transpose(tmp2);
            C.resize(tmp1.row_, tmp2.col_);
            mul_dense_dense(tmp1, tmp2, C);
        } else {
            C.resize(A.row_, B.col_);
            mul_dense_dense(A, B, C);
        }
        C.resize(1, C.row_ * C.col_);
    } else if (A.col_ == B.col_) {
        DenseMatrix tmp2 = DenseMatrix(B.col_, B.row_);
        B.transpose(tmp2);
        dot(A, tmp2, C);
    } else if (A.row_ == B.row_) {
        DenseMatrix tmp1 = DenseMatrix(A.col_, A.row_);
        A.transpose(tmp1);
        dot(tmp1, B, C);
    } else {
        throw SymEngineException("Dimensions incorrect for dot product");
    }
}

void cross(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
    SYMENGINE_ASSERT((A.row_ * A.col_ == 3 and B.row_ * B.col_ == 3)
                     and (A.row_ == C.row_ and A.col_ == C.col_));

    C.m_[0] = sub(mul(A.m_[1], B.m_[2]), mul(A.m_[2], B.m_[1]));
    C.m_[1] = sub(mul(A.m_[2], B.m_[0]), mul(A.m_[0], B.m_[2]));
    C.m_[2] = sub(mul(A.m_[0], B.m_[1]), mul(A.m_[1], B.m_[0]));
}

RCP<const Set> eigen_values(const DenseMatrix &A)
{
    unsigned n = A.nrows();
    if (A.is_lower() or A.is_upper()) {
        RCP<const Set> eigenvals = emptyset();
        set_basic x;
        for (unsigned i = 0; i < n; ++i) {
            x.insert(A.get(i, i));
        }
        eigenvals = finiteset(x);
        return eigenvals;
    }

    DenseMatrix B = DenseMatrix(A.nrows() + 1, 1);
    char_poly(A, B);
    map_int_Expr coeffs;
    auto nr = A.nrows();
    for (unsigned i = 0; i <= nr; i++)
        insert(coeffs, nr - i, B.get(i, 0));
    auto lambda = symbol("lambda");
    return solve_poly(uexpr_poly(lambda, coeffs), lambda);
}

// ------------------------- NumPy-like functions ----------------------------//

// Mimic `eye` function in NumPy
void eye(DenseMatrix &A, int k)
{
    if ((k >= 0 and (unsigned) k >= A.col_) or k + A.row_ <= 0) {
        zeros(A);
    }

    vec_basic v = vec_basic(k > 0 ? A.col_ - k : A.row_ + k, one);

    diag(A, v, k);
}

// Create diagonal matrices directly
void diag(DenseMatrix &A, vec_basic &v, int k)
{
    SYMENGINE_ASSERT(v.size() > 0);

    unsigned k_ = std::abs(k);

    if (k >= 0) {
        for (unsigned i = 0; i < A.row_; i++) {
            for (unsigned j = 0; j < A.col_; j++) {
                if (j != (unsigned)k) {
                    A.m_[i * A.col_ + j] = zero;
                } else {
                    A.m_[i * A.col_ + j] = v[k - k_];
                }
            }
            k++;
        }
    } else {
        k = -k;

        for (unsigned j = 0; j < A.col_; j++) {
            for (unsigned i = 0; i < A.row_; i++) {
                if (i != (unsigned)k) {
                    A.m_[i * A.col_ + j] = zero;
                } else {
                    A.m_[i * A.col_ + j] = v[k - k_];
                }
            }
            k++;
        }
    }
}

// Create a matrix filled with ones
void ones(DenseMatrix &A)
{
    for (unsigned i = 0; i < A.row_ * A.col_; i++) {
        A.m_[i] = one;
    }
}

// Create a matrix filled with zeros
void zeros(DenseMatrix &A)
{
    for (unsigned i = 0; i < A.row_ * A.col_; i++) {
        A.m_[i] = zero;
    }
}

} // namespace SymEngine
