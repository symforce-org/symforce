#pragma once

#include <iostream>
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

// Forward declare the eigen specializations, however they are defined in lcm_json_eigen.hpp
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

}  // namespace lcm
