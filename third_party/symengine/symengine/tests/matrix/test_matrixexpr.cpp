#include "catch.hpp"
#include <symengine/matrix_expressions.h>
#include <symengine/rational.h>
#include <symengine/complex.h>
#include <symengine/add.h>

using SymEngine::add;
using SymEngine::Complex;
using SymEngine::ConjugateMatrix;
using SymEngine::diagonal_matrix;
using SymEngine::DiagonalMatrix;
using SymEngine::DomainError;
using SymEngine::down_cast;
using SymEngine::eq;
using SymEngine::hadamard_product;
using SymEngine::HadamardProduct;
using SymEngine::identity_matrix;
using SymEngine::IdentityMatrix;
using SymEngine::immutable_dense_matrix;
using SymEngine::ImmutableDenseMatrix;
using SymEngine::integer;
using SymEngine::is_a;
using SymEngine::is_false;
using SymEngine::is_indeterminate;
using SymEngine::is_real;
using SymEngine::is_symmetric;
using SymEngine::is_true;
using SymEngine::make_rcp;
using SymEngine::matrix_add;
using SymEngine::matrix_mul;
using SymEngine::matrix_symbol;
using SymEngine::MatrixAdd;
using SymEngine::MatrixMul;
using SymEngine::one;
using SymEngine::Rational;
using SymEngine::RCP;
using SymEngine::symbol;
using SymEngine::Trace;
using SymEngine::Transpose;
using SymEngine::vec_basic;
using SymEngine::zero;
using SymEngine::ZeroMatrix;

TEST_CASE("Test IdentityMatrix", "[IdentityMatrix]")
{
    auto n1 = integer(1);
    auto n2 = integer(2);
    auto x = symbol("x");

    auto I1 = identity_matrix(n1);
    auto I2 = identity_matrix(n2);
    auto Ix = identity_matrix(x);
    REQUIRE(!eq(*I1, *I2));
    REQUIRE(eq(*I1, *I1));
    REQUIRE(I1->__hash__() != I2->__hash__());
    REQUIRE(I1->compare(*I2) == -1);
    REQUIRE(I2->compare(*I1) == 1);
    REQUIRE(I1->compare(*I1) == 0);
    REQUIRE(I1->get_args().size() == 1);
    CHECK_THROWS_AS(identity_matrix(integer(-1)), DomainError);
    auto rat1 = Rational::from_two_ints(*integer(1), *integer(2));
    CHECK_THROWS_AS(identity_matrix(rat1), DomainError);

    REQUIRE(I1->__str__() == "I");
    REQUIRE(!down_cast<const IdentityMatrix &>(*I1).is_canonical(integer(-1)));
    REQUIRE(!down_cast<const IdentityMatrix &>(*I1).is_canonical(rat1));
}

TEST_CASE("Test ZeroMatrix", "[ZeroMatrix]")
{
    auto n1 = integer(1);
    auto n2 = integer(2);
    auto x = symbol("x");

    auto Z1 = zero_matrix(n1, n1);
    auto Z2 = zero_matrix(n2, n1);
    auto Zx = zero_matrix(x, x);
    REQUIRE(!eq(*Z1, *Z2));
    REQUIRE(eq(*Z1, *Z1));
    REQUIRE(Z1->__hash__() != Z2->__hash__());
    REQUIRE(Z1->compare(*Z2) == -1);
    REQUIRE(Z2->compare(*Z1) == 1);
    REQUIRE(Z1->compare(*Z1) == 0);
    REQUIRE(Z1->get_args().size() == 2);
    CHECK_THROWS_AS(zero_matrix(integer(-1), integer(1)), DomainError);
    CHECK_THROWS_AS(zero_matrix(integer(1), integer(-1)), DomainError);
    auto rat1 = Rational::from_two_ints(*integer(1), *integer(2));
    CHECK_THROWS_AS(zero_matrix(rat1, integer(1)), DomainError);
    CHECK_THROWS_AS(zero_matrix(integer(1), rat1), DomainError);

    REQUIRE(Z1->__str__() == "0");
    REQUIRE(!down_cast<const ZeroMatrix &>(*Z1).is_canonical(integer(2),
                                                             integer(-1)));
    REQUIRE(!down_cast<const ZeroMatrix &>(*Z1).is_canonical(integer(-1),
                                                             integer(1)));
    REQUIRE(!down_cast<const ZeroMatrix &>(*Z1).is_canonical(rat1, integer(2)));
    REQUIRE(!down_cast<const ZeroMatrix &>(*Z1).is_canonical(integer(2), rat1));
}

TEST_CASE("Test MatrixSymbol", "[MatrixSymbol]")
{
    auto n1 = integer(1);
    auto n2 = integer(2);
    auto x = symbol("x");

    auto A = matrix_symbol("A");
    auto B = matrix_symbol("B");
    REQUIRE(!eq(*A, *B));
    REQUIRE(eq(*A, *A));
    REQUIRE(A->__hash__() != B->__hash__());
    REQUIRE(A->compare(*B) == -1);
    REQUIRE(B->compare(*A) == 1);
    REQUIRE(A->compare(*A) == 0);
    REQUIRE(A->get_args().size() == 0);
}

TEST_CASE("Test DiagonalMatrix", "[DiagonalMatrix]")
{
    auto i1 = integer(1);
    auto i2 = integer(23);
    auto diag1 = diagonal_matrix({i1, i2});
    auto diag2 = diagonal_matrix({i2, i1});
    REQUIRE(!eq(*diag1, *diag2));
    REQUIRE(eq(*diag1, *diag1));
    REQUIRE(!eq(*diag1, *i1));
    REQUIRE(diag1->__hash__() != diag2->__hash__());
    REQUIRE(diag1->compare(*diag2) == -1);
    REQUIRE(diag2->compare(*diag1) == 1);
    REQUIRE(diag2->compare(*diag2) == 0);
    REQUIRE(diag1->get_args().size() == 2);
    REQUIRE(!down_cast<const DiagonalMatrix &>(*diag1).is_canonical({}));

    auto diag3 = diagonal_matrix({zero, zero, zero});
    auto z3 = zero_matrix(integer(3), integer(3));
    REQUIRE(eq(*diag3, *z3));

    auto diag4 = diagonal_matrix({one, one, one, one});
    auto ident4 = identity_matrix(integer(4));
    REQUIRE(eq(*diag4, *ident4));
}

TEST_CASE("Test ImmutableDenseMatrix", "[ImmutableDenseMatrix]")
{
    auto A1 = immutable_dense_matrix(
        2, 2, {integer(2), integer(23), integer(5), integer(9)});
    auto A2 = immutable_dense_matrix(
        2, 2, {integer(2), integer(23), integer(5), integer(10)});
    auto A3 = immutable_dense_matrix(1, 2, {one, zero});
    auto A4 = immutable_dense_matrix(2, 1, {one, zero});

    REQUIRE(!eq(*A1, *A2));
    REQUIRE(eq(*A1, *A1));
    REQUIRE(!eq(*A1, *A3));
    REQUIRE(A1->compare(*A2) == -1);
    REQUIRE(A3->compare(*A2) == -1);
    REQUIRE(A2->compare(*A3) == 1);
    REQUIRE(A4->compare(*A2) == -1);
    REQUIRE(A2->compare(*A4) == 1);
    REQUIRE(A1->__hash__() != A2->__hash__());
    REQUIRE(
        !down_cast<const ImmutableDenseMatrix &>(*A1).is_canonical(0, 0, {}));

    auto A5 = immutable_dense_matrix(2, 2, {zero, zero, zero, zero});
    auto Z2 = zero_matrix(integer(2), integer(2));
    REQUIRE(eq(*A5, *Z2));

    auto A6 = immutable_dense_matrix(2, 2, {one, zero, zero, one});
    auto I2 = identity_matrix(integer(2));
    REQUIRE(eq(*A6, *I2));

    auto A7
        = immutable_dense_matrix(2, 2, {integer(2), zero, zero, symbol("x")});
    auto D2 = diagonal_matrix({integer(2), symbol("x")});
    REQUIRE(eq(*A7, *D2));

    auto A8 = immutable_dense_matrix(1, 1, {integer(2)});
    auto D1 = diagonal_matrix({integer(2)});
    REQUIRE(eq(*A8, *D1));
}

TEST_CASE("Test Trace", "[Trace]")
{
    auto n1 = integer(1);
    auto n2 = integer(2);
    auto x = symbol("x");

    auto Z1 = zero_matrix(n1, n1);
    auto Z2 = zero_matrix(n2, n2);
    auto Z3 = zero_matrix(n2, n1);
    auto Z4 = zero_matrix(x, n2);
    auto Z5 = zero_matrix(x, n1);
    REQUIRE(eq(*trace(Z1), *zero));
    REQUIRE(eq(*trace(Z2), *zero));
    CHECK_THROWS_AS(trace(Z3), DomainError);
    auto tr1 = trace(Z4);
    auto tr2 = trace(Z5);
    REQUIRE(is_a<Trace>(*tr1));
    REQUIRE(tr1->get_args().size() == 1);

    auto I1 = identity_matrix(x);
    REQUIRE(eq(*trace(I1), *x));

    REQUIRE(tr1->compare(*tr1) == 0);
    REQUIRE(eq(*tr1, *tr1));
    REQUIRE(!eq(*tr1, *tr2));
    REQUIRE(tr1->__hash__() != tr2->__hash__());
    REQUIRE(tr1->compare(*tr2) == 1);
    REQUIRE(tr2->compare(*tr1) == -1);

    auto D1 = diagonal_matrix({integer(2), integer(23)});
    REQUIRE(eq(*trace(D1), *integer(25)));

    auto A1 = immutable_dense_matrix(
        2, 2, {integer(2), integer(23), integer(5), integer(9)});
    auto A2 = immutable_dense_matrix(1, 2, {integer(5), integer(9)});
    REQUIRE(eq(*trace(A1), *integer(11)));
    CHECK_THROWS_AS(trace(A2), DomainError);

    auto S1 = matrix_symbol("A");
    auto MA1 = matrix_add({S1, A1});
    auto correct = add(make_rcp<const Trace>(S1), integer(11));
    REQUIRE(eq(*trace(MA1), *correct));
}

TEST_CASE("Test ConjugateMatrix", "[ConjugateMatrix]")
{
    auto n1 = integer(1);
    auto n2 = integer(2);
    auto x = symbol("x");
    auto c1 = Complex::from_two_nums(*one, *one);
    auto c2 = Complex::from_two_nums(*one, *integer(-1));

    auto Z1 = zero_matrix(n1, n1);
    auto Z2 = zero_matrix(n2, n1);
    REQUIRE(eq(*conjugate_matrix(Z1), *Z1));
    REQUIRE(eq(*conjugate_matrix(Z2), *Z2));

    auto I1 = identity_matrix(x);
    REQUIRE(eq(*conjugate_matrix(I1), *I1));

    auto A = matrix_symbol("A");
    auto B = matrix_symbol("B");
    auto conj1 = conjugate_matrix(A);
    auto conj2 = conjugate_matrix(B);
    REQUIRE(conj1->compare(*conj1) == 0);
    REQUIRE(eq(*conj1, *conj1));
    REQUIRE(!eq(*conj1, *conj2));
    REQUIRE(conj1->__hash__() != conj2->__hash__());
    REQUIRE(conj1->compare(*conj2) == -1);
    REQUIRE(conj2->compare(*conj1) == 1);

    auto D1 = diagonal_matrix({integer(2), integer(23)});
    auto D2 = diagonal_matrix({integer(2), c1});
    auto D3 = diagonal_matrix({integer(2), c2});
    REQUIRE(eq(*conjugate_matrix(D1), *D1));
    REQUIRE(eq(*conjugate_matrix(D2), *D3));

    auto A1 = immutable_dense_matrix(
        2, 2, {integer(2), integer(23), integer(5), integer(9)});
    auto A2 = immutable_dense_matrix(2, 2, {c1, integer(23), c1, integer(9)});
    auto A3 = immutable_dense_matrix(2, 2, {c2, integer(23), c2, integer(9)});
    REQUIRE(eq(*conjugate_matrix(A1), *A1));
    REQUIRE(eq(*conjugate_matrix(A2), *A3));

    auto A_conj = conjugate_matrix(A);
    REQUIRE(is_a<ConjugateMatrix>(*A_conj));
    REQUIRE(eq(*down_cast<const ConjugateMatrix &>(*A_conj).get_arg(), *A));
    REQUIRE(eq(*conjugate_matrix(A_conj), *A));

    auto MA1 = matrix_add({A, B});
    REQUIRE(eq(*conjugate_matrix(MA1),
               *matrix_add({conjugate_matrix(A), conjugate_matrix(B)})));

    auto HP1 = hadamard_product({A, B});
    REQUIRE(eq(*conjugate_matrix(HP1),
               *hadamard_product({conjugate_matrix(A), conjugate_matrix(B)})));
}

TEST_CASE("Test Transpose", "[Transpose]")
{
    auto n1 = integer(1);
    auto n2 = integer(2);
    auto x = symbol("x");

    auto Z1 = zero_matrix(n1, n1);
    auto Z2 = zero_matrix(n2, n1);
    auto Z3 = zero_matrix(n1, n2);
    REQUIRE(eq(*transpose(Z1), *Z1));
    REQUIRE(eq(*transpose(Z2), *Z3));

    auto I1 = identity_matrix(x);
    REQUIRE(eq(*transpose(I1), *I1));

    auto A = matrix_symbol("A");
    auto B = matrix_symbol("B");
    auto AT = transpose(A);
    auto BT = transpose(B);
    REQUIRE(AT->compare(*AT) == 0);
    REQUIRE(eq(*AT, *AT));
    REQUIRE(!eq(*AT, *BT));
    REQUIRE(AT->__hash__() != BT->__hash__());
    REQUIRE(AT->compare(*BT) == -1);
    REQUIRE(BT->compare(*AT) == 1);

    auto D1 = diagonal_matrix({integer(2), integer(23)});
    REQUIRE(eq(*transpose(D1), *D1));

    auto A1 = immutable_dense_matrix(
        2, 2, {integer(2), integer(23), integer(5), integer(9)});
    auto A1T = immutable_dense_matrix(
        2, 2, {integer(2), integer(5), integer(23), integer(9)});
    REQUIRE(eq(*transpose(A1), *A1T));

    REQUIRE(is_a<Transpose>(*AT));
    REQUIRE(eq(*down_cast<const Transpose &>(*AT).get_arg(), *A));
    REQUIRE(eq(*transpose(AT), *A));

    auto MA1 = matrix_add({A, B});
    REQUIRE(eq(*transpose(MA1), *matrix_add({transpose(A), transpose(B)})));

    auto HP1 = hadamard_product({A, B});
    REQUIRE(
        eq(*transpose(HP1), *hadamard_product({transpose(A), transpose(B)})));

    auto H1 = transpose(conjugate_matrix(A));
    auto H2 = conjugate_matrix(transpose(A));
    REQUIRE(eq(*H1, *H2));
}

TEST_CASE("Test MatrixAdd", "[MatrixAdd]")
{
    auto i1 = integer(3);
    auto i2 = integer(5);
    auto Z1 = zero_matrix(i1, i1);
    auto Z2 = zero_matrix(i2, i2);
    auto Z3 = zero_matrix(i2, i1);
    auto I1 = identity_matrix(i1);
    auto I2 = identity_matrix(i2);
    auto D1 = diagonal_matrix({integer(2), integer(23), integer(-2)});
    auto D2 = diagonal_matrix({integer(-1), integer(5), integer(0)});
    auto D3 = diagonal_matrix({integer(1), integer(28), integer(-2)});
    auto D4 = diagonal_matrix({integer(2), integer(10)});
    auto A1 = immutable_dense_matrix(
        2, 2, {integer(1), integer(2), integer(3), integer(4)});
    auto A2 = immutable_dense_matrix(
        2, 2, {integer(2), integer(4), integer(6), integer(9)});
    auto A3 = immutable_dense_matrix(
        2, 2, {integer(3), integer(6), integer(9), integer(13)});
    auto A4 = immutable_dense_matrix(
        2, 2, {integer(3), integer(2), integer(3), integer(14)});
    auto S1 = matrix_symbol("S1");
    auto S2 = matrix_symbol("S2");

    auto sum = matrix_add({Z1, I1});
    REQUIRE(eq(*sum, *I1));
    sum = matrix_add({Z1, I1, Z1, Z1});
    REQUIRE(eq(*sum, *I1));
    sum = matrix_add({I1, I1});
    auto vec = vec_basic({I1, I1});
    REQUIRE(eq(*sum, *make_rcp<const MatrixAdd>(vec)));
    sum = matrix_add({Z1, I1, D1, Z1});
    vec = vec_basic({I1, D1});
    REQUIRE(eq(*sum, *make_rcp<const MatrixAdd>(vec)));
    auto sum2 = matrix_add({sum, D2});
    vec = vec_basic({I1, D3});
    REQUIRE(eq(*sum2, *make_rcp<const MatrixAdd>(vec)));
    REQUIRE(sum2->__hash__() == make_rcp<const MatrixAdd>(vec)->__hash__());
    REQUIRE(!eq(*sum, *sum2));
    REQUIRE(sum->compare(*sum) == 0);
    REQUIRE(sum2->compare(*sum) == -1);
    REQUIRE(sum->compare(*sum2) == 1);
    sum = matrix_add({Z1, D1});
    REQUIRE(eq(*sum, *D1));
    sum = matrix_add({D1, D2});
    REQUIRE(eq(*sum, *D3));
    sum = matrix_add({Z1, Z1, Z1});
    REQUIRE(eq(*sum, *Z1));
    sum = matrix_add({I1});
    REQUIRE(eq(*sum, *I1));
    sum = matrix_add({A1, A2});
    REQUIRE(eq(*sum, *A3));
    sum = matrix_add({D4, A1});
    REQUIRE(eq(*sum, *A4));
    sum = matrix_add({A1, D4});
    REQUIRE(eq(*sum, *A4));
    sum = matrix_add({matrix_mul({i1, S1}), matrix_mul({i2, S2})});
    auto terms = sum->get_args();
    REQUIRE(terms.size() == 2);
    auto f1 = terms[0]->get_args();
    REQUIRE(f1.size() == 2);
    REQUIRE(eq(*f1[0], *i1));
    REQUIRE(eq(*f1[1], *S1));
    auto f2 = terms[1]->get_args();
    REQUIRE(f2.size() == 2);
    REQUIRE(eq(*f2[0], *i2));
    REQUIRE(eq(*f2[1], *S2));

    CHECK_THROWS_AS(matrix_add({Z1, Z2}), DomainError);
    CHECK_THROWS_AS(matrix_add({Z2, D1}), DomainError);
    CHECK_THROWS_AS(matrix_add({D1, Z2, D1}), DomainError);
    CHECK_THROWS_AS(matrix_add({D1, I2}), DomainError);
    CHECK_THROWS_AS(matrix_add({Z2, Z3}), DomainError);
    CHECK_THROWS_AS(matrix_add({}), DomainError);

    vec_basic dummyvec{S1, I1};
    RCP<const MatrixAdd> x = make_rcp<MatrixAdd>(dummyvec);
    REQUIRE(!x->is_canonical({D1}));
    REQUIRE(!x->is_canonical({D1, Z1}));
    REQUIRE(!x->is_canonical({A1, A2}));
    REQUIRE(!x->is_canonical({A1, D4}));
}

TEST_CASE("Test HadamardProduct", "[HadamardProduct]")
{
    auto i1 = integer(3);
    auto i2 = integer(5);
    auto Z1 = zero_matrix(i1, i1);
    auto Z2 = zero_matrix(i2, i2);
    auto Z3 = zero_matrix(i2, i1);
    auto I1 = identity_matrix(i1);
    auto I2 = identity_matrix(i2);
    auto D1 = diagonal_matrix({integer(2), integer(23), integer(-2)});
    auto D2 = diagonal_matrix({integer(-1), integer(5), integer(0)});
    auto D3 = diagonal_matrix({integer(-2), integer(115), integer(0)});
    auto D4 = diagonal_matrix({integer(10), integer(20)});
    auto D5 = diagonal_matrix({integer(10), integer(80)});
    auto A1 = immutable_dense_matrix(
        2, 2, {integer(1), integer(2), integer(3), integer(4)});
    auto A2 = immutable_dense_matrix(
        2, 2, {integer(2), integer(4), integer(6), integer(9)});
    auto A3 = immutable_dense_matrix(
        2, 2, {integer(2), integer(8), integer(18), integer(36)});
    auto S1 = matrix_symbol("S1");

    auto prod = hadamard_product({Z1, I1});
    REQUIRE(eq(*prod, *Z1));
    prod = hadamard_product({I1, Z1, Z1});
    REQUIRE(eq(*prod, *Z1));
    prod = hadamard_product({I1, I1});
    REQUIRE(eq(*prod, *I1));
    prod = hadamard_product({I1, D1});
    auto vec = vec_basic({I1, D1});
    REQUIRE(eq(*prod, *make_rcp<const HadamardProduct>(vec)));
    REQUIRE(prod->__hash__()
            == make_rcp<const HadamardProduct>(vec)->__hash__());
    auto prod2 = hadamard_product({I1, D2});
    REQUIRE(prod->compare(*prod2) == 1);
    REQUIRE(prod2->compare(*prod) == -1);
    REQUIRE(prod->compare(*prod) == 0);
    REQUIRE(!eq(*prod, *I1));
    prod = hadamard_product({prod, prod2});
    vec = vec_basic({D3, I1});
    REQUIRE(!eq(*prod, *make_rcp<const HadamardProduct>(vec)));
    prod = hadamard_product({D1, D2});
    REQUIRE(eq(*prod, *D3));
    prod = hadamard_product({I1});
    REQUIRE(eq(*prod, *I1));
    prod = hadamard_product({A1, A2});
    REQUIRE(eq(*prod, *A3));
    prod = hadamard_product({D4, A1});
    REQUIRE(eq(*prod, *D5));

    CHECK_THROWS_AS(hadamard_product({Z1, Z2}), DomainError);
    CHECK_THROWS_AS(hadamard_product({Z2, D1}), DomainError);
    CHECK_THROWS_AS(hadamard_product({D1, Z2, D1}), DomainError);
    CHECK_THROWS_AS(hadamard_product({D1, I2}), DomainError);
    CHECK_THROWS_AS(hadamard_product({Z2, Z3}), DomainError);
    CHECK_THROWS_AS(hadamard_product({}), DomainError);

    vec_basic dummyvec{S1, D2};
    RCP<const HadamardProduct> x = make_rcp<HadamardProduct>(dummyvec);
    REQUIRE(!x->is_canonical({D2}));
    REQUIRE(!x->is_canonical({Z1, D2}));
    REQUIRE(!x->is_canonical({A1, A2}));
    REQUIRE(!x->is_canonical({A1, D4}));
}

TEST_CASE("Test MatrixMul", "[MatrixMul]")
{
    auto i1 = integer(2);
    auto i2 = integer(5);
    auto Z1 = zero_matrix(i1, i1);
    auto Z2 = zero_matrix(i2, i2);
    auto Z3 = zero_matrix(i2, i1);
    auto I1 = identity_matrix(i1);
    auto I2 = identity_matrix(i2);
    auto D1 = diagonal_matrix({integer(2), integer(3), integer(-2)});
    auto D2 = diagonal_matrix({integer(-1), integer(2), integer(5)});
    auto D3 = diagonal_matrix({integer(-2), integer(6), integer(-10)});
    auto D4 = diagonal_matrix({integer(4), integer(36), integer(100)});
    auto D5 = diagonal_matrix({integer(2), integer(-3)});
    auto A1 = immutable_dense_matrix(
        2, 2, {integer(1), integer(2), integer(3), integer(4)});
    auto A2 = immutable_dense_matrix(
        2, 2, {integer(1), integer(2), integer(3), integer(4)});
    auto A3 = immutable_dense_matrix(
        2, 2, {integer(2), integer(4), integer(-9), integer(-12)});
    auto A4 = immutable_dense_matrix(
        2, 2, {integer(2), integer(-6), integer(6), integer(-12)});
    auto A5 = immutable_dense_matrix(
        2, 2, {integer(-16), integer(-20), integer(-30), integer(-36)});
    auto A6 = immutable_dense_matrix(
        2, 2, {integer(14), integer(20), integer(-45), integer(-66)});
    auto S1 = matrix_symbol("S1");

    auto prod = matrix_mul({Z1, I1});
    REQUIRE(eq(*prod, *Z1));
    prod = matrix_mul({I1, I1, Z1, Z1});
    REQUIRE(eq(*prod, *Z1));
    prod = matrix_mul({I1, I1});
    REQUIRE(eq(*prod, *I1));
    prod = matrix_mul({I1, A1});
    REQUIRE(eq(*prod, *A1));
    prod = matrix_mul({A1, I1});
    REQUIRE(eq(*prod, *A1));
    prod = matrix_mul({I1, A1, I1});
    REQUIRE(eq(*prod, *A1));
    prod = matrix_mul({D1, D2});
    REQUIRE(eq(*prod, *D3));
    auto vec = vec_basic({D1, S1, D2});
    prod = matrix_mul(vec);
    REQUIRE(eq(*prod, *make_rcp<const MatrixMul>(one, vec)));
    prod = matrix_mul({D1, D2, D3});
    REQUIRE(eq(*prod, *D4));
    prod = matrix_mul({D5, A2});
    REQUIRE(eq(*prod, *A3));
    prod = matrix_mul({A2, D5});
    REQUIRE(eq(*prod, *A4));
    prod = matrix_mul({A2, A3});
    REQUIRE(eq(*prod, *A5));
    prod = matrix_mul({A3, A2});
    REQUIRE(eq(*prod, *A6));
    prod = matrix_mul({A3});
    REQUIRE(eq(*prod, *A3));
    prod = matrix_mul({S1, A1});
    vec = vec_basic({S1, A1});
    REQUIRE(eq(*prod, *make_rcp<const MatrixMul>(one, vec)));
    auto prod2 = matrix_mul({prod, S1});
    REQUIRE(!eq(*prod, *prod2));
    REQUIRE(prod->compare(*prod) == 0);
    REQUIRE(prod2->compare(*prod) == 1);
    REQUIRE(prod->compare(*prod2) == -1);
    REQUIRE(prod->__hash__()
            == make_rcp<const MatrixMul>(one, vec)->__hash__());
    prod = matrix_mul({i1, S1, i2});
    REQUIRE(
        eq(*down_cast<const MatrixMul &>(*prod).get_scalar(), *integer(10)));

    CHECK_THROWS_AS(matrix_mul({Z1, Z2}), DomainError);
    CHECK_THROWS_AS(matrix_mul({Z2, D1}), DomainError);
    CHECK_THROWS_AS(matrix_mul({}), DomainError);

    vec_basic dummyvec{S1, A1};
    RCP<const MatrixMul> x = make_rcp<MatrixMul>(i1, dummyvec);
    REQUIRE(x->is_canonical(i1, {D1}));
    REQUIRE(!x->is_canonical(i1, {D1, Z1}));
    REQUIRE(!x->is_canonical(i1, {A1, A2}));
    REQUIRE(!x->is_canonical(i1, {A1, D4}));
}

TEST_CASE("Test is_zero", "[is_zero]")
{
    auto x = symbol("x");
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z5 = zero_matrix(n5, n5);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto D2 = diagonal_matrix({integer(0), integer(0)});
    auto D3 = diagonal_matrix({integer(0), integer(0), symbol("x")});
    auto Dense1 = immutable_dense_matrix(2, 2, {zero, zero, zero, zero});
    auto Dense2 = immutable_dense_matrix(2, 2, {zero, zero, zero, integer(1)});
    auto Dense3 = immutable_dense_matrix(2, 2, {zero, zero, zero, x});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_false(is_zero(*I5)));
    REQUIRE(is_true(is_zero(*Z5)));
    REQUIRE(is_false(is_zero(*D1)));
    REQUIRE(is_true(is_zero(*D2)));
    REQUIRE(is_indeterminate(is_zero(*D3)));
    REQUIRE(is_true(is_zero(*Dense1)));
    REQUIRE(is_false(is_zero(*Dense2)));
    REQUIRE(is_indeterminate(is_zero(*Dense3)));
    REQUIRE(is_indeterminate(is_zero(*S1)));
}

TEST_CASE("Test is_real", "[is_real]")
{
    auto x = symbol("x");
    auto c1 = Complex::from_two_nums(*one, *one);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z5 = zero_matrix(n5, n5);
    auto D1 = diagonal_matrix({integer(0), integer(0), symbol("x")});
    auto D2 = diagonal_matrix({integer(23), integer(0)});
    auto D3 = diagonal_matrix({integer(23), c1, integer(0)});
    auto Dense1 = immutable_dense_matrix(1, 1, {integer(1)});
    auto Dense2 = immutable_dense_matrix(
        2, 2, {integer(1), integer(1), integer(1), c1});
    auto Dense3
        = immutable_dense_matrix(2, 2, {integer(1), integer(1), integer(1), x});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_true(is_real(*I5)));
    REQUIRE(is_true(is_real(*Z5)));
    REQUIRE(is_indeterminate(is_real(*D1)));
    REQUIRE(is_true(is_real(*D2)));
    REQUIRE(is_false(is_real(*D3)));
    REQUIRE(is_true(is_real(*Dense1)));
    REQUIRE(is_false(is_real(*Dense2)));
    REQUIRE(is_indeterminate(is_real(*Dense3)));
    REQUIRE(is_indeterminate(is_real(*S1)));
}

TEST_CASE("Test is_symmetric", "[is_symmetric]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto I1 = identity_matrix(n2);
    auto A1 = matrix_add({D1, I1});
    auto Dense1 = immutable_dense_matrix(1, 1, {integer(1)});
    auto Dense2 = immutable_dense_matrix(
        2, 2, {integer(1), integer(2), integer(2), integer(3)});
    auto Dense3
        = immutable_dense_matrix(2, 2, {integer(1), x, integer(2), integer(3)});
    auto Dense4 = immutable_dense_matrix(
        2, 2, {integer(1), integer(0), integer(2), integer(3)});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_true(is_symmetric(*I5)));
    REQUIRE(is_false(is_symmetric(*Z52)));
    REQUIRE(is_true(is_symmetric(*Z5)));
    REQUIRE(is_true(is_symmetric(*Zx)));
    REQUIRE(is_indeterminate(is_symmetric(*Zxy)));
    REQUIRE(is_true(is_symmetric(*D1)));
    REQUIRE(is_true(is_symmetric(*A1)));
    REQUIRE(is_true(is_symmetric(*Dense1)));
    REQUIRE(is_true(is_symmetric(*Dense2)));
    REQUIRE(is_indeterminate(is_symmetric(*Dense3)));
    REQUIRE(is_false(is_symmetric(*Dense4)));
    REQUIRE(is_indeterminate(is_symmetric(*S1)));
}

TEST_CASE("Test is_square", "[is_square]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto I1 = identity_matrix(n2);
    auto A1 = matrix_add({D1, I1});
    auto Dense1 = immutable_dense_matrix(2, 2, {integer(1), x, y, integer(2)});
    auto Dense2 = immutable_dense_matrix(2, 1, {integer(1), x});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_true(is_square(*I5)));
    REQUIRE(is_false(is_square(*Z52)));
    REQUIRE(is_true(is_square(*Z5)));
    REQUIRE(is_true(is_square(*Zx)));
    REQUIRE(is_indeterminate(is_square(*Zxy)));
    REQUIRE(is_true(is_square(*D1)));
    REQUIRE(is_true(is_square(*A1)));
    REQUIRE(is_true(is_square(*Dense1)));
    REQUIRE(is_false(is_square(*Dense2)));
    REQUIRE(is_indeterminate(is_square(*S1)));
}

TEST_CASE("Test is_diagonal", "[is_diagonal]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto I1 = identity_matrix(n2);
    auto A1 = matrix_add({D1, I1});
    auto H1 = hadamard_product({I1, D1});
    auto S1 = matrix_symbol("S1");
    auto H2 = hadamard_product({S1, D1});
    auto Dense1 = immutable_dense_matrix(1, 1, {integer(1)});
    auto Dense2 = immutable_dense_matrix(
        2, 2, {integer(1), integer(0), integer(0), integer(2)});
    auto Dense3
        = immutable_dense_matrix(2, 2, {integer(1), x, integer(0), integer(2)});
    auto Dense4
        = immutable_dense_matrix(2, 2, {integer(1), x, integer(3), integer(2)});

    REQUIRE(is_true(is_diagonal(*I5)));
    REQUIRE(is_false(is_diagonal(*Z52)));
    REQUIRE(is_true(is_diagonal(*Z5)));
    REQUIRE(is_true(is_diagonal(*Zx)));
    REQUIRE(is_indeterminate(is_diagonal(*Zxy)));
    REQUIRE(is_true(is_diagonal(*D1)));
    REQUIRE(is_true(is_diagonal(*A1)));
    REQUIRE(is_true(is_diagonal(*H1)));
    REQUIRE(is_true(is_diagonal(*H2)));
    REQUIRE(is_true(is_diagonal(*Dense1)));
    REQUIRE(is_true(is_diagonal(*Dense2)));
    REQUIRE(is_indeterminate(is_diagonal(*Dense3)));
    REQUIRE(is_false(is_diagonal(*Dense4)));
    REQUIRE(is_indeterminate(is_diagonal(*S1)));
}

TEST_CASE("Test is_lower", "[is_lower]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto I1 = identity_matrix(n2);
    auto A1 = matrix_add({D1, I1});
    auto Dense1 = immutable_dense_matrix(
        2, 2, {integer(1), integer(0), integer(2), integer(3)});
    auto Dense2
        = immutable_dense_matrix(2, 2, {x, integer(0), integer(2), integer(3)});
    auto Dense3
        = immutable_dense_matrix(2, 2, {integer(1), x, integer(2), integer(3)});
    auto Dense4 = immutable_dense_matrix(1, 1, {x});
    auto Dense5
        = immutable_dense_matrix(3, 3,
                                 {integer(1), integer(0), integer(0),
                                  integer(2), integer(3), integer(2), x, x, x});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_true(is_lower(*I5)));
    REQUIRE(is_false(is_lower(*Z52)));
    REQUIRE(is_true(is_lower(*Z5)));
    REQUIRE(is_true(is_lower(*Zx)));
    REQUIRE(is_indeterminate(is_lower(*Zxy)));
    REQUIRE(is_true(is_lower(*D1)));
    REQUIRE(is_true(is_lower(*A1)));
    REQUIRE(is_true(is_lower(*Dense1)));
    REQUIRE(is_true(is_lower(*Dense2)));
    REQUIRE(is_indeterminate(is_lower(*Dense3)));
    REQUIRE(is_true(is_lower(*Dense4)));
    REQUIRE(is_false(is_lower(*Dense5)));
    REQUIRE(is_indeterminate(is_lower(*S1)));
}

TEST_CASE("Test is_upper", "[is_upper]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto I1 = identity_matrix(n2);
    auto A1 = matrix_add({D1, I1});
    auto Dense1 = immutable_dense_matrix(
        2, 2, {integer(2), integer(3), integer(0), integer(1)});
    auto Dense2
        = immutable_dense_matrix(2, 2, {x, integer(1), integer(0), integer(3)});
    auto Dense3 = immutable_dense_matrix(2, 2, {integer(1), x, x, integer(3)});
    auto Dense4 = immutable_dense_matrix(1, 1, {x});
    auto Dense5
        = immutable_dense_matrix(3, 3,
                                 {integer(1), integer(0), integer(0),
                                  integer(2), integer(3), integer(2), x, x, x});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_true(is_upper(*I5)));
    REQUIRE(is_false(is_upper(*Z52)));
    REQUIRE(is_true(is_upper(*Z5)));
    REQUIRE(is_true(is_upper(*Zx)));
    REQUIRE(is_indeterminate(is_upper(*Zxy)));
    REQUIRE(is_true(is_upper(*D1)));
    REQUIRE(is_true(is_upper(*A1)));
    REQUIRE(is_true(is_upper(*Dense1)));
    REQUIRE(is_true(is_upper(*Dense2)));
    REQUIRE(is_indeterminate(is_upper(*Dense3)));
    REQUIRE(is_true(is_upper(*Dense4)));
    REQUIRE(is_false(is_upper(*Dense5)));
    REQUIRE(is_indeterminate(is_upper(*S1)));
}

TEST_CASE("Test is_toeplitz", "[is_toeplitz]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto D2 = diagonal_matrix({integer(23), integer(23)});
    auto D3 = diagonal_matrix({x, y, integer(23)});
    auto D4 = diagonal_matrix({x});
    auto Dense1 = immutable_dense_matrix(1, 1, {integer(1)});
    auto Dense2 = immutable_dense_matrix(
        5, 1, {integer(1), integer(2), integer(3), x, y});
    auto Dense3 = immutable_dense_matrix(2, 2, {x, integer(1), integer(0), x});
    auto Dense4 = immutable_dense_matrix(2, 2, {x, integer(1), integer(0), y});
    auto Dense5
        = immutable_dense_matrix(3, 3,
                                 {one, zero, integer(5), integer(2), one, zero,
                                  integer(4), integer(2), integer(-1)});
    auto Dense6
        = immutable_dense_matrix(4, 2,
                                 {one, zero, integer(2), one, integer(4),
                                  integer(2), integer(5), integer(4)});
    auto S1 = matrix_symbol("S1");

    REQUIRE(is_true(is_toeplitz(*I5)));
    REQUIRE(is_true(is_toeplitz(*Z52)));
    REQUIRE(is_true(is_toeplitz(*Z5)));
    REQUIRE(is_true(is_toeplitz(*Zx)));
    REQUIRE(is_true(is_toeplitz(*Zxy)));
    REQUIRE(is_false(is_toeplitz(*D1)));
    REQUIRE(is_true(is_toeplitz(*D2)));
    REQUIRE(is_indeterminate(is_toeplitz(*D3)));
    REQUIRE(is_true(is_toeplitz(*D4)));
    REQUIRE(is_true(is_toeplitz(*Dense1)));
    REQUIRE(is_true(is_toeplitz(*Dense2)));
    REQUIRE(is_true(is_toeplitz(*Dense3)));
    REQUIRE(is_indeterminate(is_toeplitz(*Dense4)));
    REQUIRE(is_false(is_toeplitz(*Dense5)));
    REQUIRE(is_true(is_toeplitz(*Dense6)));
    REQUIRE(is_indeterminate(is_toeplitz(*S1)));
}

TEST_CASE("Test size", "[size]")
{
    auto x = symbol("x");
    auto y = symbol("y");
    auto n2 = integer(2);
    auto n5 = integer(5);
    auto I5 = identity_matrix(n5);
    auto Z52 = zero_matrix(n5, n2);
    auto Z5 = zero_matrix(n5, n5);
    auto Zx = zero_matrix(x, x);
    auto Zxy = zero_matrix(x, y);
    auto D1 = diagonal_matrix({integer(0), integer(23)});
    auto Dense1 = immutable_dense_matrix(1, 2, {one, one});
    auto A = matrix_symbol("A");
    auto ADD1 = matrix_add({A, D1});
    auto ADD2 = matrix_add({A, D1, A});
    auto HAD1 = hadamard_product({A, A, Dense1});
    auto MUL1 = matrix_mul({A, D1, A});
    auto MUL2 = matrix_mul({A, D1});

    auto sz = size(*I5);
    REQUIRE(eq(*sz.first, *n5));
    REQUIRE(eq(*sz.second, *n5));
    sz = size(*Z52);
    REQUIRE(eq(*sz.first, *n5));
    REQUIRE(eq(*sz.second, *n2));
    sz = size(*Z5);
    REQUIRE(eq(*sz.first, *n5));
    REQUIRE(eq(*sz.second, *n5));
    sz = size(*Zx);
    REQUIRE(eq(*sz.first, *x));
    REQUIRE(eq(*sz.second, *x));
    sz = size(*Zxy);
    REQUIRE(eq(*sz.first, *x));
    REQUIRE(eq(*sz.second, *y));
    sz = size(*D1);
    REQUIRE(eq(*sz.first, *n2));
    REQUIRE(eq(*sz.second, *n2));
    sz = size(*Dense1);
    REQUIRE(eq(*sz.first, *integer(1)));
    REQUIRE(eq(*sz.second, *integer(2)));
    sz = size(*ADD1);
    REQUIRE(eq(*sz.first, *n2));
    REQUIRE(eq(*sz.second, *n2));
    sz = size(*ADD2);
    REQUIRE(eq(*sz.first, *n2));
    REQUIRE(eq(*sz.second, *n2));
    sz = size(*HAD1);
    REQUIRE(eq(*sz.first, *integer(1)));
    REQUIRE(eq(*sz.second, *integer(2)));
    sz = size(*MUL1);
    REQUIRE(sz.first.is_null());
    REQUIRE(sz.second.is_null());
    sz = size(*MUL2);
    REQUIRE(sz.first.is_null());
    REQUIRE(eq(*sz.second, *integer(2)));
}
