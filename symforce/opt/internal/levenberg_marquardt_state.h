/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <sym/util/typedefs.h>

#include "../linearization.h"
#include "../values.h"

namespace sym {
namespace internal {

/**
 * Class that stores multiple {Values, Linearization} blocks for the LevenbergMarquardtSolver
 *
 * We have three of these blocks.  For a given iteration, we need the Values and Linearization
 * before and after the update; we also need a third block to store the current best Values.  We
 * support allowing uphill steps, so it's possible that the best Values we've encountered is neither
 * of the blocks we're using for the current iteration.
 *
 * This class also manages which of the three underlying blocks are currently in use for which
 * purpose (New, Init, or Best).
 */
template <typename ScalarType>
class LevenbergMarquardtState {
 public:
  using Scalar = ScalarType;

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
      func(values, &linearization_);
      linearization_.SetInitialized(true);
      have_cached_error_ = false;
    }

    const Linearization<Scalar>& GetLinearization() const {
      return linearization_;
    }

    Values<Scalar> values{};

   private:
    Linearization<Scalar> linearization_{};
    mutable bool have_cached_error_{false};
    mutable double cached_error_{0};
  };

  // Reset the state
  void Reset(const Values<Scalar>& values) {
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

}  // namespace internal
}  // namespace sym
