/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./key.h"

#include <tuple>

#include "./internal/hash_combine.h"

namespace sym {

constexpr char Key::kInvalidLetter;
constexpr Key::subscript_t Key::kInvalidSub;
constexpr Key::superscript_t Key::kInvalidSuper;

bool Key::LexicalLessThan(const Key& a, const Key& b) {
  return std::make_tuple(a.Letter(), a.Sub(), a.Super()) <
         std::make_tuple(b.Letter(), b.Sub(), b.Super());
}

std::ostream& operator<<(std::ostream& os, const sym::Key& key) {
  if (key.Letter() == sym::Key::kInvalidLetter) {
    os << "NULLKEY";
    return os;
  }

  os << key.Letter();

  if (key.Sub() != sym::Key::kInvalidSub) {
    os << '_';
    if (key.Sub() < 0) {
      os << 'n';
    }
    os << std::abs(key.Sub());
  }

  if (key.Super() != sym::Key::kInvalidSuper) {
    os << '_';
    if (key.Super() < 0) {
      os << 'n';
    }
    os << std::abs(key.Super());
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const sym::key_t& key) {
  os << sym::Key(key);
  return os;
}

}  // namespace sym

namespace std {

std::size_t hash<sym::Key>::operator()(const sym::Key& key) const {
  std::size_t ret = 0;
  sym::internal::hash_combine(ret, key.Letter(), key.Sub(), key.Super());
  return ret;
}

std::size_t hash<sym::key_t>::operator()(const sym::key_t& key) const {
  std::size_t ret = 0;
  sym::internal::hash_combine(ret, key.letter, key.subscript, key.superscript);
  return ret;
}

}  // namespace std
