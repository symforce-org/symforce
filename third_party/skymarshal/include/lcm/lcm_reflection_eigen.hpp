#pragma once

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <lcm/lcm_reflection.hpp>
#include <string>
#include <type_traits>

namespace lcm {

// Reflection for eigen types, split into its own header so other lcmtypes aren't forced to include
// eigen.

inline Eigen::IOFormat _get_vector_format()
{
    // Show vectors in one line with only one set of sq brackets
    return Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[",
                           "]");
}

inline Eigen::IOFormat _get_matrix_pretty_format(uint32_t indent)
{
    return Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ",\n",
                           std::string(indent + kLcmIndentSpaces, ' ') + "[", "]", "[\n",
                           "\n" + std::string(indent, ' ') + "]");
}

// for matrices and vectors
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value, bool>>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, const FormatSettings &settings)
{
    if (num_fields == 0) {
        if (settings.eigen_no_nested_arrays) {
            // Wrap all Eigen types in object with "data" field. This is consistent with the
            // underlying representation. See comment on "FormatSettings.eigen_no_nested_arrays"
            // for more detail.
            stream << "{\n";
            uint32_t new_indent = settings.indent + kLcmIndentSpaces;
            stream << std::string(new_indent, ' ') << "\"data\": ";

            // Both vectors and matrices: flatten to 1D array
            // Uses Eigen's native column-major storage order for consistency with eigen_lcm
            // encoding. This printout is thus consistent with the underlying representation.
            const auto *data_ptr = item.data();
            stream << "[";
            for (int i = 0; i < item.size(); ++i) {
                if (i > 0) {
                    stream << ", ";
                }
                const auto default_precision{stream.precision()};
                constexpr auto max_precision{std::numeric_limits<typename T::Scalar>::digits10 + 1};
                stream << std::setprecision(max_precision) << data_ptr[i]
                       << std::setprecision(default_precision);
            }
            stream << "]";

            stream << "\n" << std::string(settings.indent, ' ') << "}";
            return 0;
        } else {  // The "pretty" format
            if (item.cols() == 1) {
                stream << item.format(_get_vector_format());
            } else {
                stream << item.format(_get_matrix_pretty_format(settings.indent));
            }
            return 0;
        }
    }
    const uint8_t dim = item.cols() == 1 ? 1 : 2;
    uint32_t row_index = field_indices[0];
    if (row_index >= item.rows()) {
        return 1;
    }

    uint32_t col_index = 0;
    bool has_col_index = dim == 1;
    if (dim == 2 && num_fields > 1) {
        col_index = field_indices[1];
        if (col_index >= item.cols()) {
            return 2;
        }
        has_col_index = true;
    }

    if (has_col_index) {
        show_field(stream, nullptr, 0, item(row_index, col_index), settings);
    } else {
        if (dim == 2) {
            stream << item.row(row_index).format(_get_vector_format());
        } else {
            show_field(stream, nullptr, 0, item(row_index), settings);
        }
    }
    return 0;
}

// Doesn't currently support resizing for Matrix and Vector Xd types
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value, bool>>
uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields, T &item,
                     const char *const value)
{
    const uint8_t dim = item.cols() == 1 ? 1 : 2;
    if (num_fields < dim) {
        // Vectors need one index, matrices need 2
        return dim;
    }
    uint32_t row_index = field_indices[0];
    if (row_index >= item.rows()) {
        return 1;
    }

    uint32_t col_index = 0;
    if (dim == 2) {
        col_index = field_indices[1];
        if (col_index >= item.cols()) {
            return 2;
        }
    }

    uint32_t ret =
        store_field(field_indices + dim, num_fields - dim, item(row_index, col_index), value);
    return ret == 0 ? ret : ret + dim;
}

// quaternion methods (just forwards to function for Vector4)
template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value, bool>>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, const FormatSettings &settings)
{
    return show_field(stream, field_indices, num_fields, item.coeffs(), settings);
}

template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value, bool>>
uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields, T &item,
                     const char *const value)
{
    return store_field(field_indices, num_fields, item.coeffs(), value);
}

template <typename T>
std::enable_if_t<is_eigen_v<T>, SchemaType> get_schema()
{
    return EigenType{};
}

}  // namespace lcm
