/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./marginalization.h"

#include <optional>

#include <Eigen/Cholesky>

#include "./assert.h"
#include "./dense_linearizer.h"
#include "./tic_toc.h"

namespace sym {

template <typename ScalarType>
auto MarginalizationFactor<ScalarType>::FromLcmType(const LcmType& msg) -> MarginalizationFactor {
  return MarginalizationFactor{
      .H = msg.H,
      .rhs = msg.rhs,
      .c = msg.c,
      .linearization_values = Values<Scalar>(msg.linearization_values),
      .index = msg.index,
  };
}

template <typename ScalarType>
auto MarginalizationFactor<ScalarType>::GetLcmType() const -> LcmType {
  return LcmType{H, rhs, c, linearization_values.GetLcmType(), index};
}

// Explicit instantiations
template struct MarginalizationFactor<double>;
template struct MarginalizationFactor<float>;

std::vector<Key> ComputeMarginalizationKeyOrder(
    const std::unordered_set<Key>& keys_to_optimize,
    const std::unordered_set<Key>& keys_to_marginalize) {
  std::vector<Key> key_order(keys_to_marginalize.begin(), keys_to_marginalize.end());
  for (const Key& key : keys_to_optimize) {
    if (keys_to_marginalize.find(key) == keys_to_marginalize.end()) {
      key_order.push_back(key);
    }
  }
  return key_order;
}

template <typename Scalar>
std::variant<MarginalizationFactor<Scalar>, Eigen::ComputationInfo> ComputeSchurComplement(
    const MatrixX<Scalar>& H, const VectorX<Scalar>& rhs, const Scalar c, const int delimiter) {
  SYM_TIME_SCOPE("ComputeSchurComplement");
  SYM_ASSERT_EQ(H.rows(), H.cols(), "Hessian must be square");
  SYM_ASSERT_EQ(H.rows(), rhs.rows(), "Hessian must match rhs");
  SYM_ASSERT_GT(delimiter, 0, "Delimiter must be positive");
  SYM_ASSERT_LT(delimiter, H.rows(), "Delimiter must be less than linear system dimensions");

  // Extract the sub-matrices.
  // H = | A B | = | A   B |
  //     | C D |   | B.T D |
  // rhs = | rhs_u rhs_l |.T
  // Use auto for Eigen's block expression type, avoiding a copy with an explicit assignment to
  // a full matrix type.
  const auto& A = H.topLeftCorner(delimiter, delimiter);
  // The matrix is lower-only, so we take the transpose of the lower left corner.
  const auto& B = H.bottomLeftCorner(H.rows() - delimiter, delimiter).transpose();
  const auto& D = H.bottomRightCorner(H.rows() - delimiter, H.cols() - delimiter);

  const VectorX<Scalar> rhs_u = rhs.segment(0, delimiter);
  const VectorX<Scalar> rhs_l = rhs.segment(delimiter, rhs.size() - delimiter);

  // Factorize A = L * L.T
  const auto llt = A.template selfadjointView<Eigen::Lower>().llt();
  if (llt.info() != Eigen::ComputationInfo::Success) {
    return llt.info();
  }

  // Z = L^{-1} * B
  const MatrixX<Scalar> Z = llt.matrixL().solve(B);
  // Rank update does:
  // this = this + alpha * input * input.T
  //
  // Ultimately, we want:
  // S = D - B.T * A^{-1} * B
  //   = D - B.T * (L * L.T)^{-1} * B
  //   = D - B.T * (L^{-1}.T * L^{-1}) * B
  //   = D - (B.T * L^{-1}.T) * (L^{-1} * B)
  //   = D - (L^{-1} * B).T * (L^{-1} * B)
  //   = D - Z.T * Z
  MatrixX<Scalar> S = D;
  S.template selfadjointView<Eigen::Lower>().rankUpdate(Z.transpose(), -1.0);

  MarginalizationFactor<Scalar> marginalization_factor{};
  // Copy over to the upper triangle. Doing just the lower above allows us to save on the wasted
  // computation.
  marginalization_factor.H = S.template selfadjointView<Eigen::Lower>();

  // Solve for z = A^{-1} rhs_u
  const VectorX<Scalar> z = llt.solve(rhs_u);
  marginalization_factor.rhs = rhs_l - B.transpose() * z;

  // c' = c - rhs_u.T * A^{-1} rhs_u = c - rhs_u.T * z
  marginalization_factor.c = c - rhs_u.dot(z);

  return marginalization_factor;
}

template <typename Scalar>
std::variant<std::pair<MarginalizationFactor<Scalar>, std::vector<Key>>, Eigen::ComputationInfo>
Marginalize(const std::vector<Factor<Scalar>>& factors, const Values<Scalar>& values,
            const std::unordered_set<Key>& keys_to_optimize,
            const std::unordered_set<Key>& keys_to_marginalize) {
  SYM_TIME_SCOPE("Marginalize");
  std::vector<Key> key_order =
      ComputeMarginalizationKeyOrder(keys_to_optimize, keys_to_marginalize);
  SYM_ASSERT_EQ(key_order.size(), keys_to_optimize.size());
  DenseLinearizer<Scalar> linearizer("marginalization_linearizer", factors, key_order);
  DenseLinearization<Scalar> linearization{};
  {
    SYM_TIME_SCOPE("Marginalize::Relinearize");
    linearizer.Relinearize(values, linearization);
  }
  SYM_ASSERT(linearizer.IsInitialized());

  // Compute the delimiter into the system between the marginalized and non-marginalized keys.
  const std::unordered_map<key_t, index_entry_t>& state_index = linearizer.StateIndex();
  SYM_ASSERT_EQ(state_index.size(), keys_to_optimize.size());
  int delimiter = 0;
  for (const Key& key : keys_to_marginalize) {
    delimiter += state_index.at(key.GetLcmType()).tangent_dim;
  }

  // c = b.T * b, which is initially the squared norm of the residual at the linearization point.
  const Scalar c = linearization.residual.squaredNorm();

  auto marginalization_factor_or_info =
      ComputeSchurComplement(linearization.hessian_lower, linearization.rhs, c, delimiter);

  if (std::holds_alternative<Eigen::ComputationInfo>(marginalization_factor_or_info)) {
    return std::get<Eigen::ComputationInfo>(marginalization_factor_or_info);
  }

  auto marginalization_factor =
      std::move(std::get<MarginalizationFactor<Scalar>>(marginalization_factor_or_info));

  // We want the marginalization factor to keep a record of which keys remain, so we remove the
  // marginalized keys from the key order.
  key_order.erase(std::remove_if(key_order.begin(), key_order.end(),
                                 [&](const Key& key) {
                                   return keys_to_marginalize.find(key) !=
                                          keys_to_marginalize.end();
                                 }),
                  key_order.end());

  // Create a values object that contains just the remaining variables from the marginalization.
  marginalization_factor.linearization_values.UpdateOrSet(values.CreateIndex(key_order), values);

  // Create the index we'll use for LocalCoordinates
  marginalization_factor.index =
      marginalization_factor.linearization_values.CreateIndex(key_order).entries;

  SYM_ASSERT_EQ(key_order.size(), keys_to_optimize.size() - keys_to_marginalize.size());
  SYM_ASSERT_EQ(marginalization_factor.linearization_values.Keys().size(), key_order.size());

  return std::make_pair(std::move(marginalization_factor), std::move(key_order));
}

template <typename Scalar>
class MarginalizationFactorLinearizationFunctor {
 public:
  explicit MarginalizationFactorLinearizationFunctor(MarginalizationFactor<Scalar> f)
      : factor_(std::move(f)), delta_([this]() {
          int32_t tangent_dim = 0;
          for (const auto& entry : factor_.index) {
            tangent_dim += entry.tangent_dim;
          }
          return VectorX<Scalar>(tangent_dim);
        }()) {}

  void operator()(const Values<Scalar>& values, const std::vector<index_entry_t>& keys_to_func,
                  VectorX<Scalar>* residual, MatrixX<Scalar>* jacobian, MatrixX<Scalar>* hessian,
                  VectorX<Scalar>* rhs) {
    // We need to perform the linear update, accounting for the delta of optimization
    // variables since we computed the marginalization.
    //
    // Implement this here, since we'd need a LocalCoordinates method that doesn't allocate, and
    // gets the tangent dim from somewhere, and takes a vector<index_entry_t> instead of index_t,
    // all of which are different from what it currently does.
    {
      SYM_ASSERT_EQ(keys_to_func.size(), factor_.index.size());
      int32_t tangent_idx = 0;
      for (int i = 0; i < static_cast<int>(keys_to_func.size()); ++i) {
        LocalCoordinatesByType<Scalar>(
            keys_to_func[i].type,
            factor_.linearization_values.Data().data() + factor_.index[i].offset,
            values.Data().data() + keys_to_func[i].offset, delta_.data() + tangent_idx,
            kDefaultEpsilon<Scalar>, factor_.index[i].tangent_dim);
        tangent_idx += factor_.index[i].tangent_dim;
      }
    }

    H_delta_ = factor_.H * delta_;

    if (rhs != nullptr) {
      *rhs = H_delta_ + factor_.rhs;
    }

    const Scalar c_updated = delta_.dot(H_delta_) + 2 * factor_.rhs.dot(delta_) + factor_.c;

    // We don't directly have a Jacobian or residual. The residual of the system would've been
    // some MxN matrix, but since we can compute and store only the quadratic form (NxN), we
    // have "lost" information.
    if (jacobian != nullptr) {
      // Computing a plausible jacobian for this factor is expensive, but this is only required
      // when include_jacobians is on, which is generally for debugging purposes.
      //
      // We use the LLT decomposition to compute a Jacobian such that H = J.T * J.
      if (!llt_) {
        llt_.emplace(factor_.H);
      } else {
        llt_->compute(factor_.H);
      }

      // We don't have a way to expose this failure to the user to be handled, so just assert
      // here.
      SYM_ASSERT(llt_->info() == Eigen::Success, "LLT decomposition failed");

      *jacobian = llt_->matrixL().transpose();

      if (residual != nullptr) {
        // Compute a residual consistent with the jacobian, instead of a 1D residual that we
        // do otherwise
        if (rhs != nullptr) {
          *residual = llt_->matrixL().solve(*rhs);
        } else {
          *residual = llt_->matrixL().solve(H_delta_ + factor_.rhs);
        }
      }
    }
    if (residual != nullptr && jacobian == nullptr) {
      // The error is computed as 0.5 * residual.T * residual. Our cost function is:
      // e(x) ~= 0.5 * dx.T * H * dx + rhs.T * dx + 0.5 * c
      // Therefore, we are storing the squared norm of the residual already. We return:
      // r = [sqrt(c)], such that 0.5 * r.T * r = 0.5 * c, as expected.
      *residual = VectorX<Scalar>::Constant(1, std::sqrt(c_updated));
    }
    if (hessian) {
      *hessian = factor_.H;
    }
  }

 private:
  MarginalizationFactor<Scalar> factor_;

  // Space for intermediate computations
  VectorX<Scalar> delta_;
  VectorX<Scalar> H_delta_;
  std::optional<Eigen::LLT<MatrixX<Scalar>>> llt_;
};

template <typename Scalar>
Factor<Scalar> CreateMarginalizationFactor(MarginalizationFactor<Scalar> marginalization_factor,
                                           const std::vector<Key>& keys_to_func) {
  return Factor<Scalar>(
      MarginalizationFactorLinearizationFunctor<Scalar>{std::move(marginalization_factor)},
      keys_to_func);
}

template std::variant<MarginalizationFactor<float>, Eigen::ComputationInfo>
ComputeSchurComplement<float>(const MatrixX<float>& H, const VectorX<float>& rhs, const float c,
                              const int delimiter);
template std::variant<MarginalizationFactor<double>, Eigen::ComputationInfo>
ComputeSchurComplement<double>(const MatrixX<double>& H, const VectorX<double>& rhs, const double c,
                               const int delimiter);

template std::variant<std::pair<MarginalizationFactor<float>, std::vector<Key>>,
                      Eigen::ComputationInfo>
Marginalize(const std::vector<Factor<float>>& factors, const Values<float>& values,
            const std::unordered_set<Key>& keys_to_optimize,
            const std::unordered_set<Key>& keys_to_marginalize);
template std::variant<std::pair<MarginalizationFactor<double>, std::vector<Key>>,
                      Eigen::ComputationInfo>
Marginalize(const std::vector<Factor<double>>& factors, const Values<double>& values,
            const std::unordered_set<Key>& keys_to_optimize,
            const std::unordered_set<Key>& keys_to_marginalize);

template Factor<float> CreateMarginalizationFactor<float>(
    MarginalizationFactor<float> marginalization_factor, const std::vector<Key>& keys_to_func);
template Factor<double> CreateMarginalizationFactor<double>(
    MarginalizationFactor<double> marginalization_factor, const std::vector<Key>& keys_to_func);

}  // namespace sym
