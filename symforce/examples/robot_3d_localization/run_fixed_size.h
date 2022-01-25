#pragma once

#include <symforce/opt/factor.h>

namespace robot_3d_localization {

void RunFixed();

template <typename Scalar>
sym::Factor<Scalar> BuildFixedFactor();

extern template sym::Factor<double> BuildFixedFactor<double>();
extern template sym::Factor<float> BuildFixedFactor<float>();

}  // namespace robot_3d_localization
