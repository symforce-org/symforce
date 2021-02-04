#pragma once

#include <geo/ops/lie_group_ops.h>
#include <geo/ops/storage_ops.h>
#include <sym/util/typedefs.h>

namespace sym {

namespace internal {

/**
 * Helpers to get TangentDim at runtime for dynamic vectors and matrices
 */
template <typename T>
struct TangentDimHelper {
  static int32_t TangentDim(const T& x) {
    return geo::LieGroupOps<T>::TangentDim();
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
  using Scalar = typename geo::StorageOps<T>::Scalar;

  explicit Interpolator(const Scalar epsilon = 1e-8f) : epsilon_(epsilon) {}

  T operator()(const T& a, const T& b, const Scalar t) {
    return geo::LieGroupOps<T>::Retract(
        a, t * geo::LieGroupOps<T>::LocalCoordinates(a, b, epsilon_), epsilon_);
  }

 private:
  const Scalar epsilon_;
};

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
                         const typename geo::StorageOps<X>::Scalar epsilon = 1e-8f,
                         const typename geo::StorageOps<X>::Scalar delta = 1e-2f) {
  using Scalar = typename geo::StorageOps<X>::Scalar;
  using Y = typename std::result_of<F(X)>::type;
  using JacobianMat =
      Eigen::Matrix<Scalar, geo::LieGroupOps<Y>::TangentDim(), geo::LieGroupOps<X>::TangentDim()>;
  static_assert(std::is_same<typename geo::StorageOps<Y>::Scalar, Scalar>::value,
                "X and Y must have same scalar type");

  const Y f0 = f(x);
  typename geo::LieGroupOps<X>::TangentVec dx =
      geo::LieGroupOps<X>::TangentVec::Zero(internal::TangentDimHelper<X>::TangentDim(x));

  JacobianMat J = JacobianMat::Zero(internal::TangentDimHelper<Y>::TangentDim(f0),
                                    internal::TangentDimHelper<X>::TangentDim(x));
  for (size_t i = 0; i < internal::TangentDimHelper<X>::TangentDim(x); i++) {
    dx(i) = delta;
    const typename geo::LieGroupOps<Y>::TangentVec y_plus = geo::LieGroupOps<Y>::LocalCoordinates(
        f0, f(geo::LieGroupOps<X>::Retract(x, dx, epsilon)), epsilon);
    dx(i) = -delta;
    const typename geo::LieGroupOps<Y>::TangentVec y_minus = geo::LieGroupOps<Y>::LocalCoordinates(
        f0, f(geo::LieGroupOps<X>::Retract(x, dx, epsilon)), epsilon);
    dx(i) = 0.0f;
    J.col(i) = (y_plus - y_minus) / (2.0f * delta);
  }

  return J;
}

}  // namespace sym
