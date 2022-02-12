/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <sym/ops/lie_group_ops.h>
#include <sym/ops/storage_ops.h>
#include <sym/util/epsilon.h>
#include <sym/util/type_ops.h>
#include <sym/util/typedefs.h>

namespace sym {

namespace internal {

/**
 * Helpers to get TangentDim at runtime for dynamic vectors and matrices
 */
template <typename T>
struct TangentDimHelper {
  static int32_t TangentDim(const T& x) {
    return sym::LieGroupOps<T>::TangentDim();
  }
};

template <typename Scalar, int Rows, int Cols>
struct TangentDimHelper<Eigen::Matrix<Scalar, Rows, Cols>> {
  static int32_t TangentDim(const Eigen::Matrix<Scalar, Rows, Cols>& x) {
    return x.rows() * x.cols();
  }
};

}  // namespace internal

template <typename T>
static constexpr T Square(T v) {
  return v * v;
}

template <typename T, typename Tl>
static constexpr T Clamp(T x, Tl min, Tl max) {
  return (x < min) ? min : ((x > max) ? max : x);
}

// ensure self-adjoint property of symmetric matrices (correction from numerical errors)
template <typename Scalar>
MatrixX<Scalar> Symmetrize(const MatrixX<Scalar>& mat) {
  return (mat + mat.transpose()) / 2;
}

/**
 * Interpolation between Lie group elements a and b.  Result is a linear interpolation by
 * coefficient t (in [0, 1]) in the tangent space around a
 */
template <typename T>
class Interpolator {
 public:
  using Scalar = typename sym::StorageOps<T>::Scalar;

  explicit Interpolator(const Scalar epsilon = kDefaultEpsilon<Scalar>) : epsilon_(epsilon) {}

  T operator()(const T& a, const T& b, const Scalar t) {
    return sym::LieGroupOps<T>::Retract(
        a, t * sym::LieGroupOps<T>::LocalCoordinates(a, b, epsilon_), epsilon_);
  }

 private:
  const Scalar epsilon_;
};

/**
 * Interpolation between Lie group elements a and b.  Result is a linear interpolation by
 * coefficient t (in [0, 1]) in the tangent space around a
 *
 * This function version will not always be usable for passing into things that expect a functor
 * callable with three arguments; for those applications, use sym::Interpolator<T>{}
 */
template <typename T>
T Interpolate(const T& a, const T& b, const typename StorageOps<T>::Scalar t,
              const typename StorageOps<T>::Scalar epsilon =
                  kDefaultEpsilon<typename StorageOps<T>::Scalar>) {
  return Interpolator<T>(epsilon)(a, b, t);
}

/**
 * Compute the numerical derivative of a function using a central difference approximation
 *
 * Args:
 *   f: The function to differentiate
 *   x: Input at which to calculate the derivative
 *   epsilon: Epsilon for Lie Group operations
 *   delta: Derivative step size
 *
 * TODO(aaron): Add a higher-order approximation to the derivative either as an option or as the
 * default
 */
template <typename F, typename X>
auto NumericalDerivative(const F f, const X& x,
                         const typename sym::StorageOps<X>::Scalar epsilon =
                             kDefaultEpsilon<typename StorageOps<X>::Scalar>,
                         const typename sym::StorageOps<X>::Scalar delta = 1e-2f) {
  using Scalar = typename sym::StorageOps<X>::Scalar;
  using Y = typename std::result_of<F(X)>::type;
  using JacobianMat =
      Eigen::Matrix<Scalar, sym::LieGroupOps<Y>::TangentDim(), sym::LieGroupOps<X>::TangentDim()>;
  static_assert(std::is_same<typename sym::StorageOps<Y>::Scalar, Scalar>::value,
                "X and Y must have same scalar type");

  const Y f0 = f(x);
  typename sym::LieGroupOps<X>::TangentVec dx =
      sym::LieGroupOps<X>::TangentVec::Zero(internal::TangentDimHelper<X>::TangentDim(x));

  JacobianMat J = JacobianMat::Zero(internal::TangentDimHelper<Y>::TangentDim(f0),
                                    internal::TangentDimHelper<X>::TangentDim(x));
  for (size_t i = 0; i < internal::TangentDimHelper<X>::TangentDim(x); i++) {
    dx(i) = delta;
    const typename sym::LieGroupOps<Y>::TangentVec y_plus = sym::LieGroupOps<Y>::LocalCoordinates(
        f0, f(sym::LieGroupOps<X>::Retract(x, dx, epsilon)), epsilon);
    dx(i) = -delta;
    const typename sym::LieGroupOps<Y>::TangentVec y_minus = sym::LieGroupOps<Y>::LocalCoordinates(
        f0, f(sym::LieGroupOps<X>::Retract(x, dx, epsilon)), epsilon);
    dx(i) = 0.0f;
    J.col(i) = (y_plus - y_minus) / (2.0f * delta);
  }

  return J;
}

template <typename T>
std::enable_if_t<kIsEigenType<T>, bool> IsApprox(const T& a, const T& b,
                                                 const typename StorageOps<T>::Scalar epsilon) {
  return a.isApprox(b, epsilon);
}

template <typename T>
std::enable_if_t<!kIsEigenType<T>, bool> IsApprox(const T& a, const T& b,
                                                  const typename StorageOps<T>::Scalar epsilon) {
  return a.IsApprox(b, epsilon);
}

}  // namespace sym
