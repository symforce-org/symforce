#pragma once

#include <stdlib.h>

#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace lcm {

constexpr uint16_t kLcmIndentSpaces = 2;

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, typename std::conditional<false, typename T::pointer, void>::type>
    : public std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

// Formatting for int8 and uint8 types
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value && sizeof(T) == 1, bool> = true>
void format_json(std::ostream &stream, const T &item, uint16_t)
{
    // ostream tries to print this as ascii if you don't cast it
    stream << static_cast<int16_t>(item);
}

// Formatting for all other numerical types
template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value && (sizeof(T) > 1), bool> = true>
void format_json(std::ostream &stream, const T &item, uint16_t)
{
    stream << item;
}

// Formatting for booleans
template <>
inline void format_json<bool, true>(std::ostream &stream, const bool &item, uint16_t)
{
    // ostream normally tries to print this as a number
    stream << (item ? "true" : "false");
}

// Formatting for enums
template <typename T, std::enable_if_t<std::is_enum<typename T::option_t>::value, bool> = true>
void format_json(std::ostream &stream, const T &item, uint16_t indent)
{
    stream << "{\n";
    uint16_t new_indent = indent + kLcmIndentSpaces;
    stream << std::string(new_indent, ' ') << "\"name\": \"" << item.string_value() << "\",\n";
    stream << std::string(new_indent, ' ') << "\"value\": ";
    format_json(stream, item.int_value(), 0);
    stream << ",\n";
    stream << std::string(new_indent, ' ') << "\"_enum_\": \"" << item.getTypeName() << "\"\n";
    stream << std::string(indent, ' ') << "}";
}

// Formatting for normal lcm structs
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::format_field)>::value,
                           bool> = true>
void format_json(std::ostream &stream, const T &item, uint16_t indent)
{
    stream << "{\n";
    uint16_t new_indent = indent + kLcmIndentSpaces;
    for (uint16_t i = 0; i < item.fields().size(); i++) {
        stream << std::string(new_indent, ' ') << "\"" << item.fields()[i] << "\": ";
        item.format_field(stream, i, new_indent);
        stream << ",\n";
    }
    stream << std::string(new_indent, ' ') << "\"_struct_\": \"" << item.getTypeName() << "\"\n";
    stream << std::string(indent, ' ') << "}";
}

// Forward declare the eigen specializations, however they are defined in lcm_reflection_eigen.hpp
template <typename T, std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value,
                                       bool> = true>
void format_json(std::ostream &stream, const T &item, uint16_t indent);
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value,
                           bool> = true>
void format_json(std::ostream &stream, const T &item, uint16_t indent);

// Formatting for arrays, lists
// NOTE(jerry): This must be the last template function to appear in the header because it
// recursively references format_json(). (except for string which is a full specialization of this)
template <typename T, std::enable_if_t<is_iterable_v<T>, bool> = true>
void format_json(std::ostream &stream, const T &list, uint16_t indent)
{
    stream << "[";
    if (list.size() > 0)
        stream << "\n";
    uint16_t new_indent = indent + kLcmIndentSpaces;
    for (size_t i = 0; i < list.size(); ++i) {
        stream << std::string(new_indent, ' ');
        format_json(stream, list[i], new_indent);
        if (i + 1 < list.size())
            stream << ",";
        stream << "\n";
    }
    if (list.size() > 0)
        stream << std::string(indent, ' ');
    stream << "]";
}

// Formatting for strings
template <>
inline void format_json<std::string, true>(std::ostream &stream, const std::string &item, uint16_t)
{
    stream << "\"" << item << "\"";
}

// Store methods allow dynamically writing strings to specified fields, and supports struct
// fieldnames as well as indexing into lists and eigen types.
// Setting entire structs or lists via json format is not yet supported.
// Fieldpath is an already split up spec that can contain either field names or list indices.
// Return value is 0 if the operation succeeded.
// If the operation failed, return value is equal to 1 + the index of the first invalid field,
// 1 + num_fields if there are not enough fields, or 2 + num_fields if the value is invalid.

// Storing for signed integral types
template <typename T,
          std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (value == nullptr) {
        return 2;
    }
    char *endptr;
    errno = 0;
    int64_t v = strtoll(value, &endptr, 0);
    if (errno != 0 || *endptr != '\0') {
        // Invalid string conversion
        return 2;
    }
    if (v > std::numeric_limits<T>::max() || v < std::numeric_limits<T>::min()) {
        // Value out of range of type
        return 2;
    }
    item = v;
    return 0;
}

// Storing for unsigned integral types
template <typename T,
          std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value, bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (value == nullptr) {
        return 2;
    }
    char *endptr;
    errno = 0;
    uint64_t v = strtoull(value, &endptr, 0);
    if (errno != 0 || *endptr != '\0') {
        // Invalid string conversion
        return 2;
    }
    if (v > std::numeric_limits<T>::max()) {
        // Value out of range of type
        return 2;
    }
    item = v;
    return 0;
}

// Storing for floating types
template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (value == nullptr) {
        return 2;
    }
    char *endptr;
    errno = 0;
    double v = strtod(value, &endptr);
    if (errno != 0 || *endptr != '\0') {
        // Invalid string conversion
        return 2;
    }
    if (v > std::numeric_limits<T>::max() || v < std::numeric_limits<T>::lowest()) {
        // Value out of range of type
        return 2;
    }
    item = v;
    return 0;
}

// Storing for booleans
template <>
__attribute__((nodiscard)) inline uint16_t store_field<bool, true>(const char *fields[],
                                                                   uint16_t num_fields, bool &item,
                                                                   const char *value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (value == nullptr) {
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

// Storing for enums
template <typename T, std::enable_if_t<std::is_enum<typename T::option_t>::value, bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (value == nullptr) {
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

// Storing for normal lcm structs
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::store_field)>::value,
                           bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    return item.store_field(fields, num_fields, value);
}

// Forward declare the eigen specializations, however they are defined in lcm_reflection_eigen.hpp
template <typename T, std::enable_if_t<std::is_member_function_pointer<decltype(&T::format)>::value,
                                       bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value);
template <typename T,
          std::enable_if_t<std::is_member_function_pointer<decltype(&T::toRotationMatrix)>::value,
                           bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value);

// Storing for arrays, lists. value of nullptr is special here, it means to delete at that index
// NOTE(jerry): This must be the last template function to appear in the header because it
// recursively references store_field(). (except for string which is a full specialization of
// this)
template <typename T,
          std::enable_if_t<is_iterable_v<T> &&
                               std::is_member_function_pointer<decltype(&T::clear)>::value,
                           bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value);

template <
    typename T,
    std::enable_if_t<is_iterable_v<T> && std::is_member_function_pointer<decltype(&T::fill)>::value,
                     bool> = true>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value);

inline bool parse_index(const char *str, size_t &out)
{
    char *endptr;
    errno = 0;
    uint64_t index = strtoull(str, &endptr, 10);
    if (errno != 0 || *endptr != '\0') {
        // Invalid integer index
        return false;
    }
    if (index > std::numeric_limits<size_t>::max()) {
        // Out of range of output type
        return false;
    }
    out = index;
    return true;
}

// Storing for vectors (handles resizing)
template <
    typename T,
    std::enable_if_t<
        is_iterable_v<T> && std::is_member_function_pointer<decltype(&T::clear)>::value, bool>>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    if (num_fields == 0 || fields[0] == nullptr) {
        // Not enough fields for this type
        return 1;
    }

    size_t index;
    if (!parse_index(fields[0], index)) {
        return 1;
    }
    if (index >= item.max_size()) {
        // Index is larger than max vector size
        return 1;
    }
    if (num_fields == 1 && value == nullptr) {
        // Handle deleting at this index only if fields[] ends here. Otherwise it could actually
        // refer to deleting within a sub-list.
        if (index >= item.size()) {
            return 1;
        }
        item.erase(item.begin() + index);
        return 0;
    }
    if (index >= item.size()) {
        // Create a temporary object to store the value to. We want to avoid resizing if the
        // store operation on the subtype fails.
        typename T::value_type new_item{};
        uint16_t ret = store_field(fields + 1, num_fields - 1, new_item, value);
        if (ret != 0) {
            return ret + 1;
        }
        // Vector needs to be resized
        item.resize(index);
        item.emplace_back(std::move(new_item));
        return 0;
    }
    uint16_t ret = store_field(fields + 1, num_fields - 1, item[index], value);
    return ret == 0 ? ret : ret + 1;
}

// Storing for arrays (no resizing)
template <typename T,
          std::enable_if_t<
              is_iterable_v<T> && std::is_member_function_pointer<decltype(&T::fill)>::value, bool>>
__attribute__((nodiscard)) uint16_t store_field(const char *fields[], uint16_t num_fields, T &item,
                                                const char *value)
{
    if (num_fields == 0 || fields[0] == nullptr) {
        // Not enough fields for this type
        return 1;
    }
    size_t index;
    if (!parse_index(fields[0], index)) {
        return 1;
    }
    if (index >= item.size()) {
        // Index out of range of array (array is not resizable)
        return 1;
    }
    uint16_t ret = store_field(fields + 1, num_fields - 1, item[index], value);
    return ret == 0 ? ret : ret + 1;
}

// Storing for strings
template <>
__attribute__((nodiscard)) inline uint16_t store_field<std::string, true>(const char *fields[],
                                                                          uint16_t num_fields,
                                                                          std::string &item,
                                                                          const char *value)
{
    if (num_fields != 0) {
        // Too many fields for this type
        return 1;
    }
    if (value == nullptr) {
        return 2;
    }
    item = value;
    return 0;
}

}  // namespace lcm
