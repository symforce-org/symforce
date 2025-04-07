/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <sym/util/typedefs.h>

#include "../linearization.h"
#include "../values.h"

namespace sym {
namespace internal {

/**
 * Base class that stores multiple {Values, Linearization} blocks for the LevenbergMarquardtSolver
 *
 * We have three of these blocks. For a given iteration, we need the Values and Linearization
 * before and after the update; we also need a third block to store the current best Values.  We
 * support allowing uphill steps, so it's possible that the best Values we've encountered is neither
 * of the blocks we're using for the current iteration. This class also manages which of the three
 * underlying blocks are currently in use for which purpose (New, Init, or Best).
 *
 * This base class is templated on (1) a class implementing functions required to update the state
 * blocks (used via CRTP), (2) the underlying datatype of the state blocks (typically this will be
 * `Values<Scalar>`, but can be a used-defined type in special cases), and (3) the matrix type of
 * the linearization (sparse or dense).
 */
template <class Derived, typename _ValuesType, typename MatrixType>
class LevenbergMarquardtStateBase {
 public:
  using Scalar = typename MatrixType::Scalar;
  using ValuesType = _ValuesType;

  /**
   * Single values with linearization.  The full State contains three of these
   */
  class StateBlock {
   public:
    double Error() const {
      if (!have_cached_error_) {
        cached_error_ = linearization_.Error();
        have_cached_error_ = true;
      }
      return cached_error_;
    }

    void ResetLinearization() {
      linearization_.Reset();
    }

    template <typename LinearizeFunc>
    void Relinearize(const LinearizeFunc& func) {
      func(values, linearization_);
      linearization_.SetInitialized(true);
      have_cached_error_ = false;
    }

    Linearization<MatrixType>& GetLinearization() {
      return linearization_;
    }

    const Linearization<MatrixType>& GetLinearization() const {
      return linearization_;
    }

    ValuesType values{};

   private:
    Linearization<MatrixType> linearization_{};
    mutable bool have_cached_error_{false};
    mutable double cached_error_{0};
  };

  // Reset the state
  void Reset(const ValuesType& values) {
    New().values = values;
    Init().values = {};
    Free().values = {};
    New().ResetLinearization();
    Init().ResetLinearization();
    Free().ResetLinearization();

    best_values_are_valid_ = false;
  }

  bool BestIsValid() const {
    return best_values_are_valid_;
  }

  // Set the new state to be the initial state and increment the iteration.
  void Step() {
    SwapNewAndInit();
  }

  // Swap the new and initial state
  void SwapNewAndInit() {
    const auto tmp = init_idx_;
    init_idx_ = new_idx_;
    new_idx_ = tmp;
  }

  StateBlock& Init() {
    return state_blocks_[init_idx_];
  }

  const StateBlock& Init() const {
    return state_blocks_[init_idx_];
  }

  StateBlock& New() {
    return state_blocks_[new_idx_];
  }

  const StateBlock& New() const {
    return state_blocks_[new_idx_];
  }

  const StateBlock& Best() const {
    return state_blocks_[best_idx_];
  }

  void SetBestToNew() {
    best_values_are_valid_ = true;

    if (best_idx_ == new_idx_) {
      return;
    }
    if (best_idx_ != init_idx_) {
      free_idx_ = best_idx_;
    }
    best_idx_ = new_idx_;
  }

  void SetBestToInit() {
    best_values_are_valid_ = true;

    if (best_idx_ == init_idx_) {
      return;
    }
    if (best_idx_ != new_idx_) {
      free_idx_ = best_idx_;
    }
    best_idx_ = init_idx_;
  }

  void SetInitToNotBest() {
    if (best_idx_ != init_idx_) {
      return;
    }
    init_idx_ = free_idx_;
    // best_idx_ == init_idx_, so this is just a swap between free_idx_ and init_idx_
    free_idx_ = best_idx_;
  }

  // Saves the index for the optimized keys. The full index for the values is computed separately
  // only when `GetLcmType` is called.
  void SetIndex(const index_t& index) {
    index_ = index;
  }

  // ----------------------------------------------------------------------------
  // Functions requiring implementation by the derived class.
  // ----------------------------------------------------------------------------

  // Updates New block by applying `update` to Init block
  void UpdateNewFromInit(const VectorX<Scalar>& update, const Scalar epsilon) {
    static_cast<Derived*>(this)->UpdateNewFromInitImpl(update, epsilon);
  }

  // Returns a serializable type for storage in the LM debug message of the given state block.
  sym::values_t GetLcmType(const StateBlock& state_block) const {
    return static_cast<const Derived*>(this)->GetLcmTypeImpl(state_block.values);
  }

 protected:
  // Optional index of the optimized keys for the associated ValuesType. If ValuesType ==
  // Values<Scalar>, then this is used in `UpdateNewFromInitImpl` to efficiently retract the values.
  // If ValuesType is a user-defined type, then this can either be ignored or used for user-defined
  // purposes.
  index_t index_{};

 private:
  StateBlock& Free() {
    return state_blocks_[free_idx_];
  }

  const StateBlock& Free() const {
    return state_blocks_[free_idx_];
  }

  // memory allocation
  std::array<StateBlock, 3> state_blocks_{};

  // current memory use indices
  int init_idx_{0};
  int new_idx_{1};
  int best_idx_{0};
  // free points to the state block that is not currently being pointed to by new or init
  int free_idx_{2};

  // Does the Best block contain values?
  bool best_values_are_valid_{false};
};

template <typename MatrixType>
class LevenbergMarquardtState
    : public LevenbergMarquardtStateBase<LevenbergMarquardtState<MatrixType>,
                                         Values<typename MatrixType::Scalar>, MatrixType> {
 public:
  using Scalar = typename MatrixType::Scalar;
  using ValuesType = typename LevenbergMarquardtState::ValuesType;

  void UpdateNewFromInitImpl(const VectorX<Scalar>& update, const Scalar epsilon) {
    SYM_ASSERT_EQ(update.rows(), this->index_.tangent_dim,
                  "SetIndex() must be called before UpdateNewFromInit() with the correct index");
    const auto& init_values = this->Init().values;
    auto& new_values = this->New().values;

    if (new_values.NumEntries() == 0) {
      // If the state_ blocks are empty the first time, copy in the full structure
      new_values = init_values;
    } else {
      // Otherwise just copy the keys being optimized
      new_values.Update(this->index_, init_values);
    }

    // Apply the update
    new_values.Retract(this->index_, update.data(), epsilon);
  }

  // Returns the full index (optimized keys + non-optimized keys) + the data of the given values.
  // On the first call, caches the full index to avoid recomputing it on subsequent calls.
  sym::values_t GetLcmTypeImpl(const ValuesType& values) const {
    if (full_index_cached_.entries.size() == 0) {
      full_index_cached_ = values.CreateIndex(/* sort_by_offset = */ false);
    }
    return sym::values_t{full_index_cached_, values.template Cast<double>().Data()};
  }

 private:
  // Index of the Values, including both optimized variables and constants. We assume the structure
  // of the Values does not change.
  mutable index_t full_index_cached_{};
};

}  // namespace internal
}  // namespace sym
