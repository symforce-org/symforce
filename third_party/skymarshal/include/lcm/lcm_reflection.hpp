#pragma once

#include <stdlib.h>

#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace lcm {

constexpr uint32_t kLcmIndentSpaces = 2;

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, typename std::conditional<false, typename T::pointer, void>::type>
    : public std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

// Parse a positive integer index, return true if entire input was parsed successfully.
inline bool parse_index(const char *const str, uint32_t &out)
{
    if (str == nullptr) {
        return false;
    }
    char *endptr;
    errno = 0;
    uint64_t index = strtoull(str, &endptr, 10);
    if (errno != 0 || *endptr != '\0') {
        // Invalid integer index
        return false;
    } else if (index > std::numeric_limits<uint32_t>::max()) {
        // Out of range of output type
        return false;
    }
    out = index;
    return true;
}

////
// translate_fields family of functions
// fields[] and field_indices_out[] are both passed in as arrays of length num_fields.
// The functions will translate the string value of each field and output the corresponding
// field index (for structs) or array index (for lists / eigen types).
// Because this only operates on the static type info, array bounds are not checked.
// Return value is 0 if the operation succeeded.
// If the operation failed, return value is equal to 1 + the index of the first invalid field.
////

// for arithmetic types and bools
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
uint32_t translate_fields(const char *const fields[], uint32_t field_indices_out[],
                          uint32_t num_fields)
{
    // Nothing to translate
    return num_fields == 0 ? 0 : 1;
}

// for enums
template <typename T, std::enable_if_t<std::is_enum<typename T::option_t>::value, bool> = true>
uint32_t translate_fields(const char *const fields[], uint32_t field_indices_out[],
                          uint32_t num_fields)
{
    // Nothing to translate
    return num_fields == 0 ? 0 : 1;
}

// for lcm structs
template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::show_field)>::value, bool> = true>
uint32_t translate_fields(const char *const fields[], uint32_t field_indices_out[],
                          uint32_t num_fields)
{
    if (num_fields == 0) {
        return 0;
    } else if (fields[0] == nullptr) {
        return 1;
    }
    return T::translate_fields(fields, field_indices_out, num_fields);
}

// for eigen matrices and vectors
template <typename T, std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value,
                                       bool> = true>
uint32_t translate_fields(const char *const fields[], uint32_t field_indices_out[],
                          uint32_t num_fields)
{
    uint32_t index;
    if (num_fields == 0) {
        return 0;
    } else if (num_fields > 2) {
        return num_fields;
    } else if (!parse_index(fields[0], index)) {
        return 1;
    }
    field_indices_out[0] = index;
    if (num_fields == 2) {
        // We don't know if this is a matrix or a vector so we parse out the second dimension
        // regardless and allow downstream code to do the error handling.
        if (!parse_index(fields[1], index)) {
            return 2;
        }
        field_indices_out[1] = index;
    }
    return 0;
}

// for eigen quaternions
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value,
                           bool> = true>
uint32_t translate_fields(const char *const fields[], uint32_t field_indices_out[],
                          uint32_t num_fields)
{
    return translate_fields<std::remove_reference_t<decltype(static_cast<T *>(nullptr)->coeffs())>>(
        fields, field_indices_out, num_fields);
}

// for arrays and vectors (must appear last due to recursion)
template <typename T, std::enable_if_t<is_iterable_v<T>, bool> = true>
uint32_t translate_fields(const char *const fields[], uint32_t field_indices_out[],
                          uint32_t num_fields)
{
    uint32_t index;
    if (num_fields == 0) {
        return 0;
    } else if (!parse_index(fields[0], index)) {
        return 1;
    }
    field_indices_out[0] = index;
    uint32_t ret =
        translate_fields<std::remove_reference_t<decltype((*static_cast<T *>(nullptr))[0])>>(
            fields + 1, field_indices_out + 1, num_fields - 1);
    return ret == 0 ? ret : ret + 1;
}

// for strings (specialization of the above for iterables)
template <>
inline uint32_t translate_fields<std::string, true>(const char *const fields[],
                                                    uint32_t field_indices_out[],
                                                    uint32_t num_fields)
{
    // Nothing to translate
    return num_fields == 0 ? 0 : 1;
}

////
// show_field family of functions
// field_indices[] is an array of length num_fields that contains either indices within
// an array or eigen type, or indices of struct fields within struct types.
// These functions print out the representation of the object type at the given field path
// to stream (with the given indent number of spaces). Printing structs or lists will recursively
// print the inner values with appropriate indentation.
// Return value is 0 if the operation succeeded.
// If the operation failed, return value is equal to 1 + the index of the first invalid field.
////

// for arithmetic types and bools
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, uint32_t)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (std::is_same<T, bool>::value) {
        // Force bool to be printed as a word
        stream << std::boolalpha << item;
    } else {
        // Force char to be printed as int
        stream << +item;
    }
    return 0;
}

// for enums
// Non-templated helper func to reduce code size
inline void _show_field_enum(std::ostream &stream, const std::string &item_name,
                             const std::string &item_value, const char *const item_typename,
                             uint32_t indent)
{
    stream << "{\n";
    uint32_t new_indent = indent + kLcmIndentSpaces;
    stream << std::string(new_indent, ' ') << "\"name\": \"" << item_name << "\",\n";
    stream << std::string(new_indent, ' ') << "\"value\": " << item_value;
    stream << ",\n";
    stream << std::string(new_indent, ' ') << "\"_enum_\": \"" << item_typename << "\"\n";
    stream << std::string(indent, ' ') << "}";
}

template <typename T, std::enable_if_t<std::is_enum<typename T::option_t>::value, bool> = true>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, uint32_t indent)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    _show_field_enum(stream, item.string_value(), std::to_string(item.int_value()),
                     item.getTypeName(), indent);
    return 0;
}

// for lcm structs
inline void _show_field_struct(std::ostream &stream, const char *const item_typename,
                               uint32_t indent)
{
    stream << std::string(indent + kLcmIndentSpaces, ' ') << "\"_struct_\": \"" << item_typename
           << "\"\n";
    stream << std::string(indent, ' ') << "}";
}
template <
    typename T,
    std::enable_if_t<std::is_member_function_pointer<decltype(&T::show_field)>::value, bool> = true>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, uint32_t indent)
{
    if (num_fields > 0) {
        return item.show_field(stream, field_indices, num_fields, indent);
    }
    stream << "{\n";
    uint32_t new_indent = indent + kLcmIndentSpaces;
    for (uint32_t i = 0; i < item.fields().size(); i++) {
        stream << std::string(new_indent, ' ') << "\"" << item.fields()[i] << "\": ";
        item.show_field(stream, &i, 1, new_indent);
        stream << ",\n";
    }
    _show_field_struct(stream, item.getTypeName(), indent);
    return 0;
}

// Forward declare the eigen specializations, however they are defined in lcm_reflection_eigen.hpp
template <typename T, std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value,
                                       bool> = true>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, uint32_t indent);
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value,
                           bool> = true>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, uint32_t indent);

// for arrays and vectors (must appear last due to recursion)
template <typename T, std::enable_if_t<is_iterable_v<T>, bool> = true>
uint32_t show_field(std::ostream &stream, const uint32_t field_indices[], uint32_t num_fields,
                    const T &item, uint32_t indent)
{
    if (num_fields > 0) {
        if (field_indices[0] >= item.size()) {
            return 1;
        }
        uint32_t ret =
            show_field(stream, field_indices + 1, num_fields - 1, item[field_indices[0]], indent);
        return ret == 0 ? ret : ret + 1;
    }

    stream << "[";
    if (item.size() > 0)
        stream << "\n";
    uint32_t new_indent = indent + kLcmIndentSpaces;
    for (size_t i = 0; i < item.size(); ++i) {
        stream << std::string(new_indent, ' ');
        show_field(stream, nullptr, 0, item[i], new_indent);
        if (i + 1 < item.size())
            stream << ",";
        stream << "\n";
    }
    if (item.size() > 0)
        stream << std::string(indent, ' ');
    stream << "]";
    return 0;
}

// for strings (specialization of the above for iterables)
template <>
inline uint32_t show_field<std::string, true>(std::ostream &stream, const uint32_t field_indices[],
                                              uint32_t num_fields, const std::string &item,
                                              uint32_t)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    stream << "\"" << item << "\"";
    return 0;
}

////
// store_field family of functions
// field_indices[] is an array of length num_fields that contains either indices within
// an array or eigen type, or indices of struct fields within struct types.
// These functions allow dynamically writing strings to specified primitive fields. A given
// path is only valid if it ends in a primitive or enum type.
// Setting entire structs or lists via json format is not yet supported.
// Return value is 0 if the operation succeeded.
// If the operation failed, return value is equal to 1 + the index of the first invalid field,
// 1 + num_fields if there are not enough fields, or 2 + num_fields if the value is invalid.
////

// for signed integral types
template <typename T,
          std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    } else if (value == nullptr) {
        return 2;
    }
    char *endptr;
    errno = 0;
    int64_t v = strtoll(value, &endptr, 0);
    if (errno != 0 || *endptr != '\0') {
        // Invalid string conversion
        return 2;
    } else if (v > std::numeric_limits<T>::max() || v < std::numeric_limits<T>::min()) {
        // Value out of range of type
        return 2;
    }
    item = v;
    return 0;
}

// for unsigned integral types
template <typename T,
          std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value, bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    } else if (value == nullptr) {
        return 2;
    }
    char *endptr;
    errno = 0;
    uint64_t v = strtoull(value, &endptr, 0);
    if (errno != 0 || *endptr != '\0') {
        // Invalid string conversion
        return 2;
    } else if (v > std::numeric_limits<T>::max()) {
        // Value out of range of type
        return 2;
    }
    item = v;
    return 0;
}

// for floating types
template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    } else if (value == nullptr) {
        return 2;
    }
    char *endptr;
    errno = 0;
    double v = strtod(value, &endptr);
    if (errno != 0 || *endptr != '\0') {
        // Invalid string conversion
        return 2;
    } else if (v > std::numeric_limits<T>::max() || v < std::numeric_limits<T>::lowest()) {
        // Value out of range of type
        return 2;
    }
    item = v;
    return 0;
}

// for booleans
template <>
__attribute__((nodiscard)) inline uint32_t store_field<bool, true>(const uint32_t field_indices[],
                                                                   uint32_t num_fields, bool &item,
                                                                   const char *const value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    } else if (value == nullptr) {
        return 2;
    }
    // Be relatively strict about what values are accepted for boolean
    std::string value_str = value;
    for (auto &c : value_str) {
        c = tolower(c);
    }

    if (value_str == "1" || value_str == "true") {
        item = true;
        return 0;
    } else if (value_str == "0" || value_str == "false") {
        item = false;
        return 0;
    }
    return 2;
}

// for enums
template <typename T, std::enable_if_t<std::is_enum<typename T::option_t>::value, bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    } else if (value == nullptr) {
        return 2;
    }
    for (const auto &v : T::values()) {
        if (strcmp(value, T(v).string_value()) == 0) {
            item = v;
            return 0;
        }
    }
    return 2;
}

// for lcm structs
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::store_field)>::value,
                           bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields == 0) {
        return 1;
    }
    return item.store_field(field_indices, num_fields, value);
}

// Forward declare the eigen specializations, however they are defined in lcm_reflection_eigen.hpp
template <typename T, std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value,
                                       bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value);
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value,
                           bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value);

// for arrays, lists. value of nullptr is special here, it means to delete at that index
// forward declared because they recursively reference each other.
template <typename T,
          std::enable_if_t<is_iterable_v<T> &&
                               std::is_member_function_pointer<decltype(&T::clear)>::value,
                           bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value);

template <
    typename T,
    std::enable_if_t<is_iterable_v<T> && std::is_member_function_pointer<decltype(&T::fill)>::value,
                     bool> = true>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value);

// for vectors (handles resizing)
template <
    typename T,
    std::enable_if_t<
        is_iterable_v<T> && std::is_member_function_pointer<decltype(&T::clear)>::value, bool>>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields == 0) {
        // Not enough fields for this type
        return 1;
    } else if (field_indices[0] >= item.max_size()) {
        return 1;
    }

    if (num_fields == 1 && value == nullptr) {
        // Handle deleting at this index only if fields[] ends here. Otherwise it could actually
        // refer to deleting within a sub-list.
        if (field_indices[0] >= item.size()) {
            return 1;
        }
        item.erase(item.begin() + field_indices[0]);
        return 0;
    } else if (field_indices[0] >= item.size()) {
        // Create a temporary object to store the value to. We want to avoid resizing if the
        // store operation on the subtype fails.
        typename T::value_type new_item{};
        uint32_t ret = store_field(field_indices + 1, num_fields - 1, new_item, value);
        if (ret != 0) {
            return ret + 1;
        }
        // Vector needs to be resized
        item.resize(field_indices[0]);
        item.push_back(std::move(new_item));
        return 0;
    }
    uint32_t ret = store_field(field_indices + 1, num_fields - 1, item[field_indices[0]], value);
    return ret == 0 ? ret : ret + 1;
}

// for arrays (no resizing)
template <typename T,
          std::enable_if_t<
              is_iterable_v<T> && std::is_member_function_pointer<decltype(&T::fill)>::value, bool>>
__attribute__((nodiscard)) uint32_t store_field(const uint32_t field_indices[], uint32_t num_fields,
                                                T &item, const char *const value)
{
    if (num_fields == 0) {
        // Not enough fields for this type
        return 1;
    } else if (field_indices[0] >= item.size()) {
        return 1;
    }
    uint32_t ret = store_field(field_indices + 1, num_fields - 1, item[field_indices[0]], value);
    return ret == 0 ? ret : ret + 1;
}

// for strings (specialization of the above for iterables)
template <>
__attribute__((nodiscard)) inline uint32_t store_field<std::string, true>(
    const uint32_t field_indices[], uint32_t num_fields, std::string &item, const char *const value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    } else if (value == nullptr) {
        return 2;
    }
    item = value;
    return 0;
}

}  // namespace lcm
