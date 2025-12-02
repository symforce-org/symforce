/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

namespace caspar {

void zero(float* start, float* end);

void copy(const float* start, const float* end, float* target);

void alpha_from_num_denum(const float* alpha_numerator, const float* alpha_denumerator,
                          float* alpha, float* neg_alpha);

void beta_from_num_denum(const float* beta_num, const float* beta_denum, float* beta);

float sum(const float* start, const float* end, float* target_ptr, float* scratch_ptr,
          bool copy_to_host = true);

float read_cumem(const float* const data);
}  // namespace caspar
