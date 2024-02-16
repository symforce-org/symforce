#pragma once

#include <Eigen/Core>
#include <iostream>
#include <lcm/lcm_json.hpp>
#include <string>
#include <type_traits>

namespace lcm {

// Formatting for eigen types, split into its own header so other lcmtypes aren't forced to include
// eigen.

template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value, bool>>
void format_json(std::ostream &stream, const T &item, uint16_t indent)
{
    if (item.cols() == 1) {
        Eigen::IOFormat vector_fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, "", ", ", "", "",
                                   "[", "]");
        stream << item.format(vector_fmt);
    } else {
        Eigen::IOFormat matrix_fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ",\n",
                                   std::string(indent + kLcmIndentSpaces, ' ') + "[", "]", "[\n",
                                   "\n" + std::string(indent, ' ') + "]");
        stream << item.format(matrix_fmt);
    }
}

template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value, bool>>
void format_json(std::ostream &stream, const T &item, uint16_t indent)
{
    // Treat a quat as a Matrix4
    format_json(stream, item.coeffs(), indent);
}

}  // namespace lcm
