#ifndef SYMENGINE_MATRIX_H
#define SYMENGINE_MATRIX_H

#include <symengine/basic.h>
#include <symengine/sets.h>

namespace SymEngine
{

// Base class for matrices
class MatrixBase
{
public:
    virtual ~MatrixBase(){};

    bool is_square() const
    {
        return ncols() == nrows();
    }

    // Below methods should be implemented by the derived classes. If not
    // applicable, raise an exception

    // Get the # of rows and # of columns
    virtual unsigned nrows() const = 0;
    virtual unsigned ncols() const = 0;
    virtual bool eq(const MatrixBase &other) const;

    // Get and set elements
    virtual RCP<const Basic> get(unsigned i, unsigned j) const = 0;
    virtual void set(unsigned i, unsigned j, const RCP<const Basic> &e) = 0;

    // Print Matrix, very mundane version, should be overriden derived
    // class if better printing is available
    virtual std::string __str__() const;

    virtual unsigned rank() const = 0;
    virtual RCP<const Basic> det() const = 0;
    virtual void inv(MatrixBase &result) const = 0;

    // Matrix addition
    virtual void add_matrix(const MatrixBase &other,
                            MatrixBase &result) const = 0;

    // Matrix Multiplication
    virtual void mul_matrix(const MatrixBase &other,
                            MatrixBase &result) const = 0;

    // Matrix elementwise Multiplication
    virtual void elementwise_mul_matrix(const MatrixBase &other,
                                        MatrixBase &result) const = 0;

    // Add a scalar
    virtual void add_scalar(const RCP<const Basic> &k,
                            MatrixBase &result) const = 0;

    // Multiply by a scalar
    virtual void mul_scalar(const RCP<const Basic> &k,
                            MatrixBase &result) const = 0;

    // Matrix conjugate
    virtual void conjugate(MatrixBase &result) const = 0;

    // Matrix transpose
    virtual void transpose(MatrixBase &result) const = 0;

    // Matrix conjugate transpose
    virtual void conjugate_transpose(MatrixBase &result) const = 0;

    // Extract out a submatrix
    virtual void submatrix(MatrixBase &result, unsigned row_start,
                           unsigned col_start, unsigned row_end,
                           unsigned col_end, unsigned row_step = 1,
                           unsigned col_step = 1) const = 0;
    // LU factorization
    virtual void LU(MatrixBase &L, MatrixBase &U) const = 0;

    // LDL factorization
    virtual void LDL(MatrixBase &L, MatrixBase &D) const = 0;

    // Fraction free LU factorization
    virtual void FFLU(MatrixBase &LU) const = 0;

    // Fraction free LDU factorization
    virtual void FFLDU(MatrixBase &L, MatrixBase &D, MatrixBase &U) const = 0;

    // QR factorization
    virtual void QR(MatrixBase &Q, MatrixBase &R) const = 0;

    // Cholesky decomposition
    virtual void cholesky(MatrixBase &L) const = 0;

    // Solve Ax = b using LU factorization
    virtual void LU_solve(const MatrixBase &b, MatrixBase &x) const = 0;
};

typedef std::vector<std::pair<int, int>> permutelist;

class CSRMatrix;

// ----------------------------- Dense Matrix --------------------------------//
class DenseMatrix : public MatrixBase
{
public:
    // Constructors
    DenseMatrix();
    DenseMatrix(const DenseMatrix &) = default;
    DenseMatrix(unsigned row, unsigned col);
    DenseMatrix(unsigned row, unsigned col, const vec_basic &l);
    DenseMatrix(const vec_basic &column_elements);
    DenseMatrix &operator=(const DenseMatrix &other) = default;
    // Resize
    void resize(unsigned i, unsigned j);

    // Should implement all the virtual methods from MatrixBase
    // and throw an exception if a method is not applicable.

    // Get and set elements
    virtual RCP<const Basic> get(unsigned i, unsigned j) const;
    virtual void set(unsigned i, unsigned j, const RCP<const Basic> &e);
    virtual vec_basic as_vec_basic() const;

    virtual unsigned nrows() const
    {
        return row_;
    }
    virtual unsigned ncols() const
    {
        return col_;
    }

    virtual bool is_lower() const;
    virtual bool is_upper() const;
    virtual tribool is_zero() const;
    virtual tribool is_diagonal() const;
    virtual tribool is_real() const;
    virtual tribool is_symmetric() const;
    virtual tribool is_hermitian() const;
    virtual tribool is_weakly_diagonally_dominant() const;
    virtual tribool is_strictly_diagonally_dominant() const;
    virtual tribool is_positive_definite() const;
    virtual tribool is_negative_definite() const;

    RCP<const Basic> trace() const;
    virtual unsigned rank() const;
    virtual RCP<const Basic> det() const;
    virtual void inv(MatrixBase &result) const;

    // Matrix addition
    virtual void add_matrix(const MatrixBase &other, MatrixBase &result) const;

    // Matrix multiplication
    virtual void mul_matrix(const MatrixBase &other, MatrixBase &result) const;

    // Matrix elementwise Multiplication
    virtual void elementwise_mul_matrix(const MatrixBase &other,
                                        MatrixBase &result) const;

    // Add a scalar
    virtual void add_scalar(const RCP<const Basic> &k,
                            MatrixBase &result) const;

    // Multiply by a scalar
    virtual void mul_scalar(const RCP<const Basic> &k,
                            MatrixBase &result) const;

    // Matrix conjugate
    virtual void conjugate(MatrixBase &result) const;

    // Matrix transpose
    virtual void transpose(MatrixBase &result) const;

    // Matrix conjugate transpose
    virtual void conjugate_transpose(MatrixBase &result) const;

    // Extract out a submatrix
    virtual void submatrix(MatrixBase &result, unsigned row_start,
                           unsigned col_start, unsigned row_end,
                           unsigned col_end, unsigned row_step = 1,
                           unsigned col_step = 1) const;

    // LU factorization
    virtual void LU(MatrixBase &L, MatrixBase &U) const;

    // LDL factorization
    virtual void LDL(MatrixBase &L, MatrixBase &D) const;

    // Solve Ax = b using LU factorization
    virtual void LU_solve(const MatrixBase &b, MatrixBase &x) const;

    // Fraction free LU factorization
    virtual void FFLU(MatrixBase &LU) const;

    // Fraction free LDU factorization
    virtual void FFLDU(MatrixBase &L, MatrixBase &D, MatrixBase &U) const;

    // QR factorization
    virtual void QR(MatrixBase &Q, MatrixBase &R) const;

    // Cholesky decomposition
    virtual void cholesky(MatrixBase &L) const;

    // Return the Jacobian of the matrix
    friend void jacobian(const DenseMatrix &A, const DenseMatrix &x,
                         DenseMatrix &result, bool diff_cache);
    // Return the Jacobian of the matrix using sdiff
    friend void sjacobian(const DenseMatrix &A, const DenseMatrix &x,
                          DenseMatrix &result, bool diff_cache);

    // Differentiate the matrix element-wise
    friend void diff(const DenseMatrix &A, const RCP<const Symbol> &x,
                     DenseMatrix &result, bool diff_cache);
    // Differentiate the matrix element-wise using SymPy compatible diff
    friend void sdiff(const DenseMatrix &A, const RCP<const Basic> &x,
                      DenseMatrix &result, bool diff_cache);

    // Friend functions related to Matrix Operations
    friend void add_dense_dense(const DenseMatrix &A, const DenseMatrix &B,
                                DenseMatrix &C);
    friend void add_dense_scalar(const DenseMatrix &A,
                                 const RCP<const Basic> &k, DenseMatrix &B);
    friend void mul_dense_dense(const DenseMatrix &A, const DenseMatrix &B,
                                DenseMatrix &C);
    friend void elementwise_mul_dense_dense(const DenseMatrix &A,
                                            const DenseMatrix &B,
                                            DenseMatrix &C);
    friend void mul_dense_scalar(const DenseMatrix &A,
                                 const RCP<const Basic> &k, DenseMatrix &C);
    friend void conjugate_dense(const DenseMatrix &A, DenseMatrix &B);
    friend void transpose_dense(const DenseMatrix &A, DenseMatrix &B);
    friend void conjugate_transpose_dense(const DenseMatrix &A, DenseMatrix &B);
    friend void submatrix_dense(const DenseMatrix &A, DenseMatrix &B,
                                unsigned row_start, unsigned col_start,
                                unsigned row_end, unsigned col_end,
                                unsigned row_step, unsigned col_step);
    void row_join(const DenseMatrix &B);
    void col_join(const DenseMatrix &B);
    void row_insert(const DenseMatrix &B, unsigned pos);
    void col_insert(const DenseMatrix &B, unsigned pos);
    void row_del(unsigned k);
    void col_del(unsigned k);

    // Row operations
    friend void row_exchange_dense(DenseMatrix &A, unsigned i, unsigned j);
    friend void row_mul_scalar_dense(DenseMatrix &A, unsigned i,
                                     RCP<const Basic> &c);
    friend void row_add_row_dense(DenseMatrix &A, unsigned i, unsigned j,
                                  RCP<const Basic> &c);
    friend void permuteFwd(DenseMatrix &A, permutelist &pl);

    // Column operations
    friend void column_exchange_dense(DenseMatrix &A, unsigned i, unsigned j);

    // Gaussian elimination
    friend void pivoted_gaussian_elimination(const DenseMatrix &A,
                                             DenseMatrix &B,
                                             permutelist &pivotlist);
    friend void fraction_free_gaussian_elimination(const DenseMatrix &A,
                                                   DenseMatrix &B);
    friend void pivoted_fraction_free_gaussian_elimination(
        const DenseMatrix &A, DenseMatrix &B, permutelist &pivotlist);
    friend void pivoted_gauss_jordan_elimination(const DenseMatrix &A,
                                                 DenseMatrix &B,
                                                 permutelist &pivotlist);
    friend void fraction_free_gauss_jordan_elimination(const DenseMatrix &A,
                                                       DenseMatrix &B);
    friend void pivoted_fraction_free_gauss_jordan_elimination(
        const DenseMatrix &A, DenseMatrix &B, permutelist &pivotlist);
    friend unsigned pivot(DenseMatrix &B, unsigned r, unsigned c);

    friend void reduced_row_echelon_form(const DenseMatrix &A, DenseMatrix &B,
                                         vec_uint &pivot_cols,
                                         bool normalize_last);

    // Ax = b
    friend void diagonal_solve(const DenseMatrix &A, const DenseMatrix &b,
                               DenseMatrix &x);
    friend void back_substitution(const DenseMatrix &U, const DenseMatrix &b,
                                  DenseMatrix &x);
    friend void forward_substitution(const DenseMatrix &A, const DenseMatrix &b,
                                     DenseMatrix &x);
    friend void fraction_free_gaussian_elimination_solve(const DenseMatrix &A,
                                                         const DenseMatrix &b,
                                                         DenseMatrix &x);
    friend void fraction_free_gauss_jordan_solve(const DenseMatrix &A,
                                                 const DenseMatrix &b,
                                                 DenseMatrix &x);

    // Matrix Decomposition
    friend void fraction_free_LU(const DenseMatrix &A, DenseMatrix &LU);
    friend void LU(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &U);
    friend void pivoted_LU(const DenseMatrix &A, DenseMatrix &LU,
                           permutelist &pl);
    friend void pivoted_LU(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &U,
                           permutelist &pl);
    friend void fraction_free_LDU(const DenseMatrix &A, DenseMatrix &L,
                                  DenseMatrix &D, DenseMatrix &U);
    friend void QR(const DenseMatrix &A, DenseMatrix &Q, DenseMatrix &R);
    friend void LDL(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &D);
    friend void cholesky(const DenseMatrix &A, DenseMatrix &L);

    // Matrix queries
    friend bool is_symmetric_dense(const DenseMatrix &A);

    // Determinant
    friend RCP<const Basic> det_bareis(const DenseMatrix &A);
    friend void berkowitz(const DenseMatrix &A,
                          std::vector<DenseMatrix> &polys);

    // Inverse
    friend void inverse_fraction_free_LU(const DenseMatrix &A, DenseMatrix &B);
    friend void inverse_LU(const DenseMatrix &A, DenseMatrix &B);
    friend void inverse_pivoted_LU(const DenseMatrix &A, DenseMatrix &B);
    friend void inverse_gauss_jordan(const DenseMatrix &A, DenseMatrix &B);

    // Vector-specific methods
    friend void dot(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);
    friend void cross(const DenseMatrix &A, const DenseMatrix &B,
                      DenseMatrix &C);

    // NumPy-like functions
    friend void eye(DenseMatrix &A, int k);
    friend void diag(DenseMatrix &A, vec_basic &v, int k);
    friend void ones(DenseMatrix &A);
    friend void zeros(DenseMatrix &A);

    friend CSRMatrix;

private:
    // Matrix elements are stored in row-major order
    vec_basic m_;
    // Stores the dimension of the Matrix
    unsigned row_;
    unsigned col_;

    tribool shortcut_to_posdef() const;
    tribool is_positive_definite_GE();
};

// ----------------------------- Sparse Matrices -----------------------------//
class CSRMatrix : public MatrixBase
{
public:
    CSRMatrix();
    CSRMatrix(unsigned row, unsigned col);
    CSRMatrix(unsigned row, unsigned col, const std::vector<unsigned> &p,
              const std::vector<unsigned> &j, const vec_basic &x);
    CSRMatrix(unsigned row, unsigned col, std::vector<unsigned> &&p,
              std::vector<unsigned> &&j, vec_basic &&x);
    CSRMatrix &operator=(CSRMatrix &&other);
    CSRMatrix(const CSRMatrix &) = default;
    std::tuple<std::vector<unsigned>, std::vector<unsigned>, vec_basic>
    as_vectors() const;

    bool is_canonical() const;

    virtual bool eq(const MatrixBase &other) const;

    // Get and set elements
    virtual RCP<const Basic> get(unsigned i, unsigned j) const;
    virtual void set(unsigned i, unsigned j, const RCP<const Basic> &e);

    virtual unsigned nrows() const
    {
        return row_;
    }
    virtual unsigned ncols() const
    {
        return col_;
    }
    virtual unsigned rank() const;
    virtual RCP<const Basic> det() const;
    virtual void inv(MatrixBase &result) const;

    // Matrix addition
    virtual void add_matrix(const MatrixBase &other, MatrixBase &result) const;

    // Matrix Multiplication
    virtual void mul_matrix(const MatrixBase &other, MatrixBase &result) const;

    // Matrix elementwise Multiplication
    virtual void elementwise_mul_matrix(const MatrixBase &other,
                                        MatrixBase &result) const;

    // Add a scalar
    virtual void add_scalar(const RCP<const Basic> &k,
                            MatrixBase &result) const;

    // Multiply by a scalar
    virtual void mul_scalar(const RCP<const Basic> &k,
                            MatrixBase &result) const;

    // Matrix conjugate
    virtual void conjugate(MatrixBase &result) const;

    // Matrix transpose
    virtual void transpose(MatrixBase &result) const;
    CSRMatrix transpose(bool conjugate = false) const;

    // Matrix conjugate transpose
    virtual void conjugate_transpose(MatrixBase &result) const;

    // Extract out a submatrix
    virtual void submatrix(MatrixBase &result, unsigned row_start,
                           unsigned col_start, unsigned row_end,
                           unsigned col_end, unsigned row_step = 1,
                           unsigned col_step = 1) const;

    // LU factorization
    virtual void LU(MatrixBase &L, MatrixBase &U) const;

    // LDL factorization
    virtual void LDL(MatrixBase &L, MatrixBase &D) const;

    // Solve Ax = b using LU factorization
    virtual void LU_solve(const MatrixBase &b, MatrixBase &x) const;

    // Fraction free LU factorization
    virtual void FFLU(MatrixBase &LU) const;

    // Fraction free LDU factorization
    virtual void FFLDU(MatrixBase &L, MatrixBase &D, MatrixBase &U) const;

    // QR factorization
    virtual void QR(MatrixBase &Q, MatrixBase &R) const;

    // Cholesky decomposition
    virtual void cholesky(MatrixBase &L) const;

    static void csr_sum_duplicates(std::vector<unsigned> &p_,
                                   std::vector<unsigned> &j_, vec_basic &x_,
                                   unsigned row_);

    static void csr_sort_indices(std::vector<unsigned> &p_,
                                 std::vector<unsigned> &j_, vec_basic &x_,
                                 unsigned row_);

    static bool csr_has_sorted_indices(const std::vector<unsigned> &p_,
                                       const std::vector<unsigned> &j_,
                                       unsigned row_);

    static bool csr_has_duplicates(const std::vector<unsigned> &p_,
                                   const std::vector<unsigned> &j_,
                                   unsigned row_);

    static bool csr_has_canonical_format(const std::vector<unsigned> &p_,
                                         const std::vector<unsigned> &j_,
                                         unsigned row_);

    static CSRMatrix from_coo(unsigned row, unsigned col,
                              const std::vector<unsigned> &i,
                              const std::vector<unsigned> &j,
                              const vec_basic &x);
    static CSRMatrix jacobian(const vec_basic &exprs, const vec_sym &x,
                              bool diff_cache = true);
    static CSRMatrix jacobian(const DenseMatrix &A, const DenseMatrix &x,
                              bool diff_cache = true);

    friend void csr_matmat_pass1(const CSRMatrix &A, const CSRMatrix &B,
                                 CSRMatrix &C);
    friend void csr_matmat_pass2(const CSRMatrix &A, const CSRMatrix &B,
                                 CSRMatrix &C);
    friend void csr_diagonal(const CSRMatrix &A, DenseMatrix &D);
    friend void csr_scale_rows(CSRMatrix &A, const DenseMatrix &X);
    friend void csr_scale_columns(CSRMatrix &A, const DenseMatrix &X);

    friend void csr_binop_csr_canonical(
        const CSRMatrix &A, const CSRMatrix &B, CSRMatrix &C,
        RCP<const Basic> (&bin_op)(const RCP<const Basic> &,
                                   const RCP<const Basic> &));

private:
    std::vector<unsigned> p_;
    std::vector<unsigned> j_;
    vec_basic x_;
    // Stores the dimension of the Matrix
    unsigned row_;
    unsigned col_;
};

// Return the Jacobian of the matrix
void jacobian(const DenseMatrix &A, const DenseMatrix &x, DenseMatrix &result,
              bool diff_cache = true);
// Return the Jacobian of the matrix using sdiff
void sjacobian(const DenseMatrix &A, const DenseMatrix &x, DenseMatrix &result,
               bool diff_cache = true);

// Differentiate all the elements
void diff(const DenseMatrix &A, const RCP<const Symbol> &x, DenseMatrix &result,
          bool diff_cache = true);
// Differentiate all the elements using SymPy compatible diff
void sdiff(const DenseMatrix &A, const RCP<const Basic> &x, DenseMatrix &result,
           bool diff_cache = true);

// Get submatrix from a DenseMatrix
void submatrix_dense(const DenseMatrix &A, DenseMatrix &B, unsigned row_start,
                     unsigned col_start, unsigned row_end, unsigned col_end,
                     unsigned row_step = 1, unsigned col_step = 1);

// Row operations
void row_exchange_dense(DenseMatrix &A, unsigned i, unsigned j);
void row_mul_scalar_dense(DenseMatrix &A, unsigned i, RCP<const Basic> &c);
void row_add_row_dense(DenseMatrix &A, unsigned i, unsigned j,
                       RCP<const Basic> &c);

// Column operations
void column_exchange_dense(DenseMatrix &A, unsigned i, unsigned j);

// Vector-specific methods
void dot(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);
void cross(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);

// Matrix Factorization
void LU(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &U);
void LDL(const DenseMatrix &A, DenseMatrix &L, DenseMatrix &D);
void QR(const DenseMatrix &A, DenseMatrix &Q, DenseMatrix &R);
void cholesky(const DenseMatrix &A, DenseMatrix &L);

// Inverse
void inverse_fraction_free_LU(const DenseMatrix &A, DenseMatrix &B);

void inverse_gauss_jordan(const DenseMatrix &A, DenseMatrix &B);

// Solving Ax = b
void fraction_free_LU_solve(const DenseMatrix &A, const DenseMatrix &b,
                            DenseMatrix &x);

void fraction_free_gauss_jordan_solve(const DenseMatrix &A,
                                      const DenseMatrix &b, DenseMatrix &x);

void LU_solve(const DenseMatrix &A, const DenseMatrix &b, DenseMatrix &x);
void pivoted_LU_solve(const DenseMatrix &A, const DenseMatrix &b,
                      DenseMatrix &x);

void LDL_solve(const DenseMatrix &A, const DenseMatrix &b, DenseMatrix &x);

// Determinant
RCP<const Basic> det_berkowitz(const DenseMatrix &A);

// Characteristic polynomial: Only the coefficients of monomials in decreasing
// order of monomial powers is returned, i.e. if `B = transpose([1, -2, 3])`
// then the corresponding polynomial is `x**2 - 2x + 3`.
void char_poly(const DenseMatrix &A, DenseMatrix &B);

// returns a finiteset of eigenvalues of a matrix
RCP<const Set> eigen_values(const DenseMatrix &A);

// Mimic `eye` function in NumPy
void eye(DenseMatrix &A, int k = 0);

// Create diagonal matrices directly
void diag(DenseMatrix &A, vec_basic &v, int k = 0);

// Create a matrix filled with ones
void ones(DenseMatrix &A);

// Create a matrix filled with zeros
void zeros(DenseMatrix &A);

// Reduced row echelon form and returns the cols with pivots
void reduced_row_echelon_form(const DenseMatrix &A, DenseMatrix &B,
                              vec_uint &pivot_cols,
                              bool normalize_last = false);

// Returns true if `b` is exactly the type T.
// Here T can be a DenseMatrix, CSRMatrix, etc.
template <class T>
inline bool is_a(const MatrixBase &b)
{
    return typeid(T) == typeid(b);
}

// Test two matrices for equality
inline bool operator==(const SymEngine::MatrixBase &lhs,
                       const SymEngine::MatrixBase &rhs)
{
    return lhs.eq(rhs);
}

// Test two matrices for equality
inline bool operator!=(const SymEngine::MatrixBase &lhs,
                       const SymEngine::MatrixBase &rhs)
{
    return not lhs.eq(rhs);
}

} // SymEngine

// Print Matrix
inline std::ostream &operator<<(std::ostream &out,
                                const SymEngine::MatrixBase &A)
{
    return out << A.__str__();
}

#endif
