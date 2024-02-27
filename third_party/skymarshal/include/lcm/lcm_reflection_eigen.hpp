#pragma once

#include <Eigen/Core>
#include <iostream>
#include <lcm/lcm_reflection.hpp>
#include <string>
#include <type_traits>

namespace lcm {

// Formatting for eigen types, split into its own header so other lcmtypes aren't forced to include
// eigen.

// Formatting for matrices and vectors
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value, bool>>
void format_json(std::ostream &stream, const T &item, uint16_t indent)
{
    if (item.cols() == 1) {
        // Show vectors in one line with only one set of sq brackets
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

// Formatting for quaternions
template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value, bool>>
void format_json(std::ostream &stream, const T &item, uint16_t indent)
{
    // Treat a quat as a Matrix4
    format_json(stream, item.coeffs(), indent);
}

// Storing for matrices and vectors
// Doesn't currently support resizing for Matrix and Vector Xd types
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value, bool>>
uint16_t store_field(const char *fields[], uint16_t field_size, T &item, const char *value)
{
    const uint8_t dim = item.cols() == 1 ? 1 : 2;
    if (field_size < dim) {
        // Vectors need one index, matrices need 2
        return dim;
    }
    size_t row_index;
    if (fields[0] == nullptr || !parse_index(fields[0], row_index)) {
        return 1;
    }
    if (row_index >= item.rows()) {
        // Index out of range
        return 1;
    }
    size_t col_index = 0;
    if (dim == 2) {
        if (fields[1] == nullptr || !parse_index(fields[1], col_index)) {
            return 2;
        }
        if (col_index >= item.cols()) {
            // Index out of range
            return 2;
        }
    }

    uint16_t ret = store_field(fields + dim, field_size - dim, item(row_index, col_index), value);
    return ret == 0 ? ret : ret + dim;
}

// Storing for quaternions (not a matrix subclass)
template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value, bool>>
uint16_t store_field(const char *fields[], uint16_t field_size, T &item, const char *value)
{
    if (field_size == 0 || fields[0] == nullptr) {
        // Not enough fields for this type
        return 1;
    }
    size_t index;
    if (!parse_index(fields[0], index)) {
        return 1;
    }
    if (index >= 4) {
        // Index out of range of quaternion
        return 1;
    }
    uint16_t ret = store_field(fields + 1, field_size - 1, item.coeffs()(index), value);
    return ret == 0 ? ret : ret + 1;
}

}  // namespace lcm
