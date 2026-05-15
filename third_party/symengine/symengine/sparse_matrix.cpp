#include <numeric>
#include <symengine/matrix.h>
#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/mul.h>
#include <symengine/constants.h>
#include <symengine/symengine_exception.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{
// ----------------------------- CSRMatrix ------------------------------------
CSRMatrix::CSRMatrix() {}

CSRMatrix::CSRMatrix(unsigned row, unsigned col) : row_(row), col_(col)
{
    p_ = std::vector<unsigned>(row + 1, 0);
    SYMENGINE_ASSERT(is_canonical());
}

CSRMatrix::CSRMatrix(unsigned row, unsigned col, const std::vector<unsigned> &p,
                     const std::vector<unsigned> &j, const vec_basic &x)
    : p_{p}, j_{j}, x_{x}, row_(row), col_(col)
{
    SYMENGINE_ASSERT(is_canonical());
}

CSRMatrix::CSRMatrix(unsigned row, unsigned col, std::vector<unsigned> &&p,
                     std::vector<unsigned> &&j, vec_basic &&x)
    : p_{std::move(p)}, j_{std::move(j)}, x_{std::move(x)}, row_(row), col_(col)
{
    SYMENGINE_ASSERT(is_canonical());
}

CSRMatrix &CSRMatrix::operator=(CSRMatrix &&other)
{
    col_ = other.col_;
    row_ = other.row_;
    p_ = std::move(other.p_);
    j_ = std::move(other.j_);
    x_ = std::move(other.x_);
    return *this;
}

std::tuple<std::vector<unsigned>, std::vector<unsigned>, vec_basic>
CSRMatrix::as_vectors() const
{
    auto p = p_, j = j_;
    auto x = x_;
    return std::make_tuple(std::move(p), std::move(j), std::move(x));
}

bool CSRMatrix::eq(const MatrixBase &other) const
{
    unsigned row = this->nrows();
    if (row != other.nrows() or this->ncols() != other.ncols())
        return false;

    if (is_a<CSRMatrix>(other)) {
        const CSRMatrix &o = down_cast<const CSRMatrix &>(other);

        if (this->p_[row] != o.p_[row])
            return false;

        for (unsigned i = 0; i <= row; i++)
            if (this->p_[i] != o.p_[i])
                return false;

        for (unsigned i = 0; i < this->p_[row]; i++)
            if ((this->j_[i] != o.j_[i]) or neq(*this->x_[i], *(o.x_[i])))
                return false;

        return true;
    } else {
        return this->MatrixBase::eq(other);
    }
}

bool CSRMatrix::is_canonical() const
{
    if (p_.size() != row_ + 1 or j_.size() != p_[row_] or x_.size() != p_[row_])
        return false;

    if (p_[row_] != 0) // Zero matrix is in canonical format
        return csr_has_canonical_format(p_, j_, row_);
    return true;
}

// Get and set elements
RCP<const Basic> CSRMatrix::get(unsigned i, unsigned j) const
{
    SYMENGINE_ASSERT(i < row_ and j < col_);

    unsigned row_start = p_[i];
    unsigned row_end = p_[i + 1];
    unsigned k;

    if (row_start == row_end) {
        return zero;
    }

    while (row_start < row_end) {
        k = (row_start + row_end) / 2;
        if (j_[k] == j) {
            return x_[k];
        } else if (j_[k] < j) {
            row_start = k + 1;
        } else {
            row_end = k;
        }
    }

    return zero;
}

void CSRMatrix::set(unsigned i, unsigned j, const RCP<const Basic> &e)
{
    SYMENGINE_ASSERT(i < row_ and j < col_);

    unsigned k = p_[i];
    unsigned row_end = p_[i + 1];
    unsigned end = p_[i + 1];
    unsigned mid;

    while (k < end) {
        mid = (k + end) / 2;
        if (mid == k) {
            if (j_[k] < j) {
                k++;
            }
            break;
        } else if (j_[mid] >= j and j_[mid - 1] < j) {
            k = mid;
            break;
        } else if (j_[mid - 1] >= j) {
            end = mid - 1;
        } else {
            k = mid + 1;
        }
    }

    if (!is_true(is_zero(*e))) {
        if (k < row_end and j_[k] == j) {
            x_[k] = e;
        } else { // j_[k] > j or k is the last non-zero element
            x_.insert(x_.begin() + k, e);
            j_.insert(j_.begin() + k, j);
            for (unsigned l = i + 1; l <= row_; l++)
                p_[l]++;
        }
    } else {                              // e is zero
        if (k < row_end and j_[k] == j) { // remove existing non-zero element
            x_.erase(x_.begin() + k);
            j_.erase(j_.begin() + k);
            for (unsigned l = i + 1; l <= row_; l++)
                p_[l]--;
        }
    }
}

tribool CSRMatrix::is_real(const Assumptions *assumptions) const
{
    RealVisitor visitor(assumptions);
    tribool cur = tribool::tritrue;
    for (auto &e : x_) {
        cur = and_tribool(cur, visitor.apply(*e));
        if (is_false(cur)) {
            return cur;
        }
    }
    return cur;
}

unsigned CSRMatrix::rank() const
{
    throw NotImplementedError("Not Implemented");
}

RCP<const Basic> CSRMatrix::det() const
{
    throw NotImplementedError("Not Implemented");
}

void CSRMatrix::inv(MatrixBase &result) const
{
    throw NotImplementedError("Not Implemented");
}

void CSRMatrix::add_matrix(const MatrixBase &other, MatrixBase &result) const
{
    throw NotImplementedError("Not Implemented");
}

void CSRMatrix::mul_matrix(const MatrixBase &other, MatrixBase &result) const
{
    throw NotImplementedError("Not Implemented");
}

void CSRMatrix::elementwise_mul_matrix(const MatrixBase &other,
                                       MatrixBase &result) const
{
    if (is_a<CSRMatrix>(result)) {
        auto &o = down_cast<const CSRMatrix &>(other);
        auto &r = down_cast<CSRMatrix &>(result);
        csr_binop_csr_canonical(*this, o, r, mul);
    }
}

// Add a scalar
void CSRMatrix::add_scalar(const RCP<const Basic> &k, MatrixBase &result) const
{
    throw NotImplementedError("Not Implemented");
}

// Multiply by a scalar
void CSRMatrix::mul_scalar(const RCP<const Basic> &k, MatrixBase &result) const
{
    throw NotImplementedError("Not Implemented");
}

// Matrix conjugate
void CSRMatrix::conjugate(MatrixBase &result) const
{
    if (is_a<CSRMatrix>(result)) {
        auto &r = down_cast<CSRMatrix &>(result);
        std::vector<unsigned> p(p_), j(j_);
        vec_basic x(x_.size());
        for (unsigned i = 0; i < x_.size(); ++i) {
            x[i] = SymEngine::conjugate(x_[i]);
        }
        r = CSRMatrix(col_, row_, std::move(p), std::move(j), std::move(x));
    } else {
        throw NotImplementedError("Not Implemented");
    }
}

// Matrix transpose
void CSRMatrix::transpose(MatrixBase &result) const
{
    if (is_a<CSRMatrix>(result)) {
        auto &r = down_cast<CSRMatrix &>(result);
        r = this->transpose();
    } else {
        throw NotImplementedError("Not Implemented");
    }
}

CSRMatrix CSRMatrix::transpose(bool conjugate) const
{
    const auto nnz = j_.size();
    std::vector<unsigned> p(col_ + 1, 0), j(nnz), tmp(col_, 0);
    vec_basic x(nnz);

    for (unsigned i = 0; i < nnz; ++i)
        p[j_[i] + 1]++;
    std::partial_sum(p.begin(), p.end(), p.begin());

    for (unsigned ri = 0; ri < row_; ++ri) {
        for (unsigned i = p_[ri]; i < p_[ri + 1]; ++i) {
            const auto ci = j_[i];
            const unsigned k = p[ci] + tmp[ci];
            j[k] = ri;
            if (conjugate) {
                x[k] = SymEngine::conjugate(x_[i]);
            } else {
                x[k] = x_[i];
            }
            tmp[ci]++;
        }
    }
    return CSRMatrix(col_, row_, std::move(p), std::move(j), std::move(x));
}

// Matrix conjugate transpose
void CSRMatrix::conjugate_transpose(MatrixBase &result) const
{
    if (is_a<CSRMatrix>(result)) {
        auto &r = down_cast<CSRMatrix &>(result);
        r = this->transpose(true);
    } else {
        throw NotImplementedError("Not Implemented");
    }
}

// Extract out a submatrix
void CSRMatrix::submatrix(MatrixBase &result, unsigned row_start,
                          unsigned col_start, unsigned row_end,
                          unsigned col_end, unsigned row_step,
                          unsigned col_step) const
{
    throw NotImplementedError("Not Implemented");
}

// LU factorization
void CSRMatrix::LU(MatrixBase &L, MatrixBase &U) const
{
    throw NotImplementedError("Not Implemented");
}

// LDL factorization
void CSRMatrix::LDL(MatrixBase &L, MatrixBase &D) const
{
    throw NotImplementedError("Not Implemented");
}

// Solve Ax = b using LU factorization
void CSRMatrix::LU_solve(const MatrixBase &b, MatrixBase &x) const
{
    throw NotImplementedError("Not Implemented");
}

// Fraction free LU factorization
void CSRMatrix::FFLU(MatrixBase &LU) const
{
    throw NotImplementedError("Not Implemented");
}

// Fraction free LDU factorization
void CSRMatrix::FFLDU(MatrixBase &L, MatrixBase &D, MatrixBase &U) const
{
    throw NotImplementedError("Not Implemented");
}

// QR factorization
void CSRMatrix::QR(MatrixBase &Q, MatrixBase &R) const
{
    throw NotImplementedError("Not Implemented");
}

// Cholesky decomposition
void CSRMatrix::cholesky(MatrixBase &L) const
{
    throw NotImplementedError("Not Implemented");
}

void CSRMatrix::csr_sum_duplicates(std::vector<unsigned> &p_,
                                   std::vector<unsigned> &j_, vec_basic &x_,
                                   unsigned row_)
{
    unsigned nnz = 0;
    unsigned row_end = 0;
    unsigned jj = 0, j = 0;
    RCP<const Basic> x = zero;

    for (unsigned i = 0; i < row_; i++) {
        jj = row_end;
        row_end = p_[i + 1];

        while (jj < row_end) {
            j = j_[jj];
            x = x_[jj];
            jj++;

            while (jj < row_end and j_[jj] == j) {
                x = add(x, x_[jj]);
                jj++;
            }

            j_[nnz] = j;
            x_[nnz] = x;
            nnz++;
        }
        p_[i + 1] = nnz;
    }

    // Resize to discard unnecessary elements
    j_.resize(nnz);
    x_.resize(nnz);
}

void CSRMatrix::csr_sort_indices(std::vector<unsigned> &p_,
                                 std::vector<unsigned> &j_, vec_basic &x_,
                                 unsigned row_)
{
    std::vector<std::pair<unsigned, RCP<const Basic>>> temp;

    for (unsigned i = 0; i < row_; i++) {
        unsigned row_start = p_[i];
        unsigned row_end = p_[i + 1];

        temp.clear();

        for (unsigned jj = row_start; jj < row_end; jj++) {
            temp.push_back(std::make_pair(j_[jj], x_[jj]));
        }

        std::sort(temp.begin(), temp.end(),
                  [](const std::pair<unsigned, RCP<const Basic>> &x,
                     const std::pair<unsigned, RCP<const Basic>> &y) {
                      return x.first < y.first;
                  });

        for (unsigned jj = row_start, n = 0; jj < row_end; jj++, n++) {
            j_[jj] = temp[n].first;
            x_[jj] = temp[n].second;
        }
    }
}

// Assumes that the indices are sorted
bool CSRMatrix::csr_has_duplicates(const std::vector<unsigned> &p_,
                                   const std::vector<unsigned> &j_,
                                   unsigned row_)
{
    for (unsigned i = 0; i < row_; i++) {
        for (unsigned j = p_[i]; j + 1 < p_[i + 1]; j++) {
            if (j_[j] == j_[j + 1])
                return true;
        }
    }
    return false;
}

bool CSRMatrix::csr_has_sorted_indices(const std::vector<unsigned> &p_,
                                       const std::vector<unsigned> &j_,
                                       unsigned row_)
{
    for (unsigned i = 0; i < row_; i++) {
        for (unsigned jj = p_[i]; jj < p_[i + 1] - 1; jj++) {
            if (j_[jj] > j_[jj + 1])
                return false;
        }
    }
    return true;
}

bool CSRMatrix::csr_has_canonical_format(const std::vector<unsigned> &p_,
                                         const std::vector<unsigned> &j_,
                                         unsigned row_)
{
    for (unsigned i = 0; i < row_; i++) {
        if (p_[i] > p_[i + 1])
            return false;
    }

    return csr_has_sorted_indices(p_, j_, row_)
           and not csr_has_duplicates(p_, j_, row_);
}

CSRMatrix CSRMatrix::from_coo(unsigned row, unsigned col,
                              const std::vector<unsigned> &i,
                              const std::vector<unsigned> &j,
                              const vec_basic &x)
{
    // cast is okay, because CSRMatrix indices are unsigned.
    unsigned nnz = numeric_cast<unsigned>(x.size());
    std::vector<unsigned> p_ = std::vector<unsigned>(row + 1, 0);
    std::vector<unsigned> j_ = std::vector<unsigned>(nnz);
    vec_basic x_ = vec_basic(nnz);

    for (unsigned n = 0; n < nnz; n++) {
        p_[i[n]]++;
    }

    // cumsum the nnz per row to get p
    unsigned temp;
    for (unsigned i = 0, cumsum = 0; i < row; i++) {
        temp = p_[i];
        p_[i] = cumsum;
        cumsum += temp;
    }
    p_[row] = nnz;

    // write j, x into j_, x_
    unsigned row_, dest_;
    for (unsigned n = 0; n < nnz; n++) {
        row_ = i[n];
        dest_ = p_[row_];

        j_[dest_] = j[n];
        x_[dest_] = x[n];

        p_[row_]++;
    }

    for (unsigned i = 0, last = 0; i <= row; i++) {
        std::swap(p_[i], last);
    }

    // sort indices
    csr_sort_indices(p_, j_, x_, row);
    // Remove duplicates
    csr_sum_duplicates(p_, j_, x_, row);

    CSRMatrix B
        = CSRMatrix(row, col, std::move(p_), std::move(j_), std::move(x_));
    return B;
}

CSRMatrix CSRMatrix::jacobian(const vec_basic &exprs, const vec_sym &x,
                              bool diff_cache)
{
    const unsigned nrows = static_cast<unsigned>(exprs.size());
    const unsigned ncols = static_cast<unsigned>(x.size());
    std::vector<unsigned> p(1, 0), j;
    vec_basic elems;
    p.reserve(nrows + 1);
    j.reserve(nrows);
    elems.reserve(nrows);
    for (unsigned ri = 0; ri < nrows; ++ri) {
        p.push_back(p.back());
        for (unsigned ci = 0; ci < ncols; ++ci) {
            auto elem = exprs[ri]->diff(x[ci], diff_cache);
            if (!is_true(is_zero(*elem))) {
                p.back()++;
                j.push_back(ci);
                elems.emplace_back(std::move(elem));
            }
        }
    }
    return CSRMatrix(nrows, ncols, std::move(p), std::move(j),
                     std::move(elems));
}

CSRMatrix CSRMatrix::jacobian(const DenseMatrix &A, const DenseMatrix &x,
                              bool diff_cache)
{
    SYMENGINE_ASSERT(A.col_ == 1);
    SYMENGINE_ASSERT(x.col_ == 1);
    vec_sym syms;
    syms.reserve(x.row_);
    for (const auto &dx : x.m_) {
        if (!is_a<Symbol>(*dx)) {
            throw SymEngineException("'x' must contain Symbols only");
        }
        syms.push_back(rcp_static_cast<const Symbol>(dx));
    }
    return CSRMatrix::jacobian(A.m_, syms, diff_cache);
}

void csr_matmat_pass1(const CSRMatrix &A, const CSRMatrix &B, CSRMatrix &C)
{
    // method that uses O(n) temp storage
    std::vector<unsigned> mask(A.col_, -1);
    C.p_[0] = 0;

    unsigned nnz = 0;
    for (unsigned i = 0; i < A.row_; i++) {
        // npy_intp row_nnz = 0;
        unsigned row_nnz = 0;

        for (unsigned jj = A.p_[i]; jj < A.p_[i + 1]; jj++) {
            unsigned j = A.j_[jj];
            for (unsigned kk = B.p_[j]; kk < B.p_[j + 1]; kk++) {
                unsigned k = B.j_[kk];
                if (mask[k] != i) {
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }

        unsigned next_nnz = nnz + row_nnz;

        // Addition overflow: http://www.cplusplus.com/articles/DE18T05o/
        if (next_nnz < nnz) {
            throw std::overflow_error("nnz of the result is too large");
        }

        nnz = next_nnz;
        C.p_[i + 1] = nnz;
    }
}

// Pass 2 computes CSR entries for matrix C = A*B using the
// row pointer Cp[] computed in Pass 1.
void csr_matmat_pass2(const CSRMatrix &A, const CSRMatrix &B, CSRMatrix &C)
{
    std::vector<int> next(A.col_, -1);
    vec_basic sums(A.col_, zero);

    unsigned nnz = 0;

    C.p_[0] = 0;

    for (unsigned i = 0; i < A.row_; i++) {
        int head = -2;
        unsigned length = 0;

        unsigned jj_start = A.p_[i];
        unsigned jj_end = A.p_[i + 1];
        for (unsigned jj = jj_start; jj < jj_end; jj++) {
            unsigned j = A.j_[jj];
            RCP<const Basic> v = A.x_[jj];

            unsigned kk_start = B.p_[j];
            unsigned kk_end = B.p_[j + 1];
            for (unsigned kk = kk_start; kk < kk_end; kk++) {
                unsigned k = B.j_[kk];

                sums[k] = add(sums[k], mul(v, B.x_[kk]));

                if (next[k] == -1) {
                    next[k] = head;
                    head = k;
                    length++;
                }
            }
        }

        for (unsigned jj = 0; jj < length; jj++) {

            if (!is_true(is_zero(*sums[head]))) {
                C.j_[nnz] = head;
                C.x_[nnz] = sums[head];
                nnz++;
            }

            unsigned temp = head;
            head = next[head];

            next[temp] = -1; // clear arrays
            sums[temp] = zero;
        }

        C.p_[i + 1] = nnz;
    }
}

// Extract main diagonal of CSR matrix A
void csr_diagonal(const CSRMatrix &A, DenseMatrix &D)
{
    unsigned N = std::min(A.row_, A.col_);

    SYMENGINE_ASSERT(D.nrows() == N and D.ncols() == 1);

    unsigned row_start;
    unsigned row_end;
    RCP<const Basic> diag;

    for (unsigned i = 0; i < N; i++) {
        row_start = A.p_[i];
        row_end = A.p_[i + 1];
        diag = zero;
        unsigned jj;

        while (row_start <= row_end) {
            jj = (row_start + row_end) / 2;
            if (A.j_[jj] == i) {
                diag = A.x_[jj];
                break;
            } else if (A.j_[jj] < i) {
                row_start = jj + 1;
            } else {
                row_end = jj - 1;
            }
        }

        D.set(i, 0, diag);
    }
}

// Scale the rows of a CSR matrix *in place*
// A[i, :] *= X[i]
void csr_scale_rows(CSRMatrix &A, const DenseMatrix &X)
{
    SYMENGINE_ASSERT(A.row_ == X.nrows() and X.ncols() == 1);

    for (unsigned i = 0; i < A.row_; i++) {
        if (is_true(is_zero(*X.get(i, 0))))
            throw SymEngineException("Scaling factor can't be zero");
        for (unsigned jj = A.p_[i]; jj < A.p_[i + 1]; jj++)
            A.x_[jj] = mul(A.x_[jj], X.get(i, 0));
    }
}

// Scale the columns of a CSR matrix *in place*
// A[:, i] *= X[i]
void csr_scale_columns(CSRMatrix &A, const DenseMatrix &X)
{
    SYMENGINE_ASSERT(A.col_ == X.nrows() and X.ncols() == 1);

    const unsigned nnz = A.p_[A.row_];
    unsigned i;

    for (i = 0; i < A.col_; i++) {
        if (is_true(is_zero(*X.get(i, 0))))
            throw SymEngineException("Scaling factor can't be zero");
    }

    for (i = 0; i < nnz; i++)
        A.x_[i] = mul(A.x_[i], X.get(A.j_[i], 0));
}

// Compute C = A (binary_op) B for CSR matrices that are in the
// canonical CSR format. Matrix dimensions of A and B should be the
// same. C will be in canonical format as well.
void csr_binop_csr_canonical(
    const CSRMatrix &A, const CSRMatrix &B, CSRMatrix &C,
    RCP<const Basic> (&bin_op)(const RCP<const Basic> &,
                               const RCP<const Basic> &))
{
    SYMENGINE_ASSERT(A.row_ == B.row_ and A.col_ == B.col_ and C.row_ == A.row_
                     and C.col_ == A.col_);

    // Method that works for canonical CSR matrices
    C.p_[0] = 0;
    C.j_.clear();
    C.x_.clear();
    unsigned nnz = 0;
    unsigned A_pos, B_pos, A_end, B_end;

    for (unsigned i = 0; i < A.row_; i++) {
        A_pos = A.p_[i];
        B_pos = B.p_[i];
        A_end = A.p_[i + 1];
        B_end = B.p_[i + 1];

        // while not finished with either row
        while (A_pos < A_end and B_pos < B_end) {
            unsigned A_j = A.j_[A_pos];
            unsigned B_j = B.j_[B_pos];

            if (A_j == B_j) {
                RCP<const Basic> result = bin_op(A.x_[A_pos], B.x_[B_pos]);
                if (!is_true(is_zero(*result))) {
                    C.j_.push_back(A_j);
                    C.x_.push_back(result);
                    nnz++;
                }
                A_pos++;
                B_pos++;
            } else if (A_j < B_j) {
                RCP<const Basic> result = bin_op(A.x_[A_pos], zero);
                if (!is_true(is_zero(*result))) {
                    C.j_.push_back(A_j);
                    C.x_.push_back(result);
                    nnz++;
                }
                A_pos++;
            } else {
                // B_j < A_j
                RCP<const Basic> result = bin_op(zero, B.x_[B_pos]);
                if (!is_true(is_zero(*result))) {
                    C.j_.push_back(B_j);
                    C.x_.push_back(result);
                    nnz++;
                }
                B_pos++;
            }
        }

        // tail
        while (A_pos < A_end) {
            RCP<const Basic> result = bin_op(A.x_[A_pos], zero);
            if (!is_true(is_zero(*result))) {
                C.j_.push_back(A.j_[A_pos]);
                C.x_.push_back(result);
                nnz++;
            }
            A_pos++;
        }
        while (B_pos < B_end) {
            RCP<const Basic> result = bin_op(zero, B.x_[B_pos]);
            if (!is_true(is_zero(*result))) {
                C.j_.push_back(B.j_[B_pos]);
                C.x_.push_back(result);
                nnz++;
            }
            B_pos++;
        }

        C.p_[i + 1] = nnz;
    }

    // It's enough to check for duplicates as the column indices
    // remain sorted after the above operations
    if (CSRMatrix::csr_has_duplicates(C.p_, C.j_, A.row_))
        CSRMatrix::csr_sum_duplicates(C.p_, C.j_, C.x_, A.row_);
}

} // namespace SymEngine
