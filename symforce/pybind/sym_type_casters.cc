#include "./sym_type_casters.h"

namespace pybind11 {
namespace detail {

constexpr descr<4> handle_sym_type_name<sym::Rot2d>::name;
constexpr descr<4> handle_sym_type_name<sym::Rot3d>::name;
constexpr descr<5> handle_sym_type_name<sym::Pose2d>::name;
constexpr descr<5> handle_sym_type_name<sym::Pose3d>::name;

}  // namespace detail
}  // namespace pybind11
